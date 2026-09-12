from __future__ import annotations

import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import tarfile

import pytest

from luapyre import LuaRuntime


_TOOL = Path(__file__).resolve().parents[1] / "tools" / "official_551.py"
_SPEC = importlib.util.spec_from_file_location("luapyre_official_551_tool", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
suite = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = suite
_SPEC.loader.exec_module(suite)


def _tar(path: Path, members: list[tuple[str, bytes | None, str]]) -> None:
    with tarfile.open(path, "w:gz") as tf:
        for name, data, kind in members:
            info = tarfile.TarInfo(name)
            if kind == "dir":
                info.type = tarfile.DIRTYPE
                info.size = 0
                tf.addfile(info)
            elif kind == "file":
                payload = data or b""
                info.size = len(payload)
                tf.addfile(info, io.BytesIO(payload))
            elif kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = (data or b"target").decode("ascii")
                tf.addfile(info)
            else:
                raise AssertionError(kind)


def _suite_root(tmp_path: Path) -> Path:
    root = tmp_path / suite.SUITE_ROOT
    root.mkdir()
    return root


def test_verify_archive_accepts_exact_hash_and_rejects_mismatch(tmp_path):
    archive = tmp_path / "archive.tar.gz"
    archive.write_bytes(b"fixture")
    digest = hashlib.sha256(b"fixture").hexdigest()
    suite.verify_archive(archive, expected=digest)
    with pytest.raises(suite.SuiteError, match="checksum mismatch"):
        suite.verify_archive(archive, expected="0" * 64)


def test_safe_extract_accepts_regular_suite_tree(tmp_path, monkeypatch):
    archive = tmp_path / "suite.tar.gz"
    _tar(
        archive,
        [
            (suite.SUITE_ROOT, None, "dir"),
            (f"{suite.SUITE_ROOT}/all.lua", b"return 42\n", "file"),
            (f"{suite.SUITE_ROOT}/sub", None, "dir"),
            (f"{suite.SUITE_ROOT}/sub/one.lua", b"return 'one'\n", "file"),
        ],
    )
    monkeypatch.setattr(suite, "verify_archive", lambda path: None)
    root = suite.safe_extract(archive, tmp_path / "out")
    assert root.name == suite.SUITE_ROOT
    assert suite.list_lua_files(root) == ["all.lua", "sub/one.lua"]
    assert (root / "all.lua").read_bytes() == b"return 42\n"


def test_safe_extract_rejects_path_traversal(tmp_path, monkeypatch):
    archive = tmp_path / "suite.tar.gz"
    _tar(archive, [("../escape.lua", b"bad", "file")])
    monkeypatch.setattr(suite, "verify_archive", lambda path: None)
    with pytest.raises(suite.SuiteError, match="unsafe path"):
        suite.safe_extract(archive, tmp_path / "out")
    assert not (tmp_path / "escape.lua").exists()


def test_safe_extract_rejects_links(tmp_path, monkeypatch):
    archive = tmp_path / "suite.tar.gz"
    _tar(
        archive,
        [
            (suite.SUITE_ROOT, None, "dir"),
            (f"{suite.SUITE_ROOT}/link", b"/etc/passwd", "symlink"),
        ],
    )
    monkeypatch.setattr(suite, "verify_archive", lambda path: None)
    with pytest.raises(suite.SuiteError, match="links and devices are rejected"):
        suite.safe_extract(archive, tmp_path / "out")


def test_safe_extract_enforces_uncompressed_size_limit(tmp_path, monkeypatch):
    archive = tmp_path / "suite.tar.gz"
    _tar(
        archive,
        [
            (suite.SUITE_ROOT, None, "dir"),
            (f"{suite.SUITE_ROOT}/large.lua", b"x" * 32, "file"),
        ],
    )
    monkeypatch.setattr(suite, "verify_archive", lambda path: None)
    with pytest.raises(suite.SuiteError, match="extracted-size limit"):
        suite.safe_extract(archive, tmp_path / "out", max_extracted_bytes=16)


def test_resolve_suite_file_stays_under_root(tmp_path):
    root = _suite_root(tmp_path)
    (root / "ok.lua").write_text("return 1", encoding="utf-8")
    (root / "asset.txt").write_text("asset", encoding="utf-8")
    assert suite.resolve_suite_file(root, "ok.lua") == (root / "ok.lua").resolve()
    assert suite.resolve_suite_asset(root, "asset.txt") == (root / "asset.txt").resolve()
    with pytest.raises(suite.SuiteError, match="unsafe path"):
        suite.resolve_suite_file(root, "../outside.lua")
    with pytest.raises(suite.SuiteError, match="unsafe path"):
        suite.resolve_suite_asset(root, "../outside.txt")


def test_suite_capabilities_are_read_only_and_not_default_runtime_globals(tmp_path):
    root = _suite_root(tmp_path)
    (root / "dep.lua").write_text("return {value=41}", encoding="utf-8")
    (root / "answer.lua").write_text("return 42", encoding="utf-8")
    (tmp_path / "outside.lua").write_text("return 99", encoding="utf-8")

    ordinary = LuaRuntime()
    assert ordinary.get("print") is not None
    assert ordinary.get("require") is not None and ordinary.get("package") is not None
    assert ordinary.get("loadfile") is None and ordinary.get("dofile") is None

    lua = suite.make_suite_runtime(root)
    result = lua.execute(
        """
        local dep = require 'dep'
        assert(dep.value == 41)
        assert(require 'dep' == dep)

        local f = assert(loadfile('answer.lua'))
        assert(f() == 42)
        assert(dofile('answer.lua') == 42)

        package.preload.synthetic = function () return {value=7} end
        assert(require('synthetic').value == 7)

        local escaped, err = loadfile('../outside.lua')
        assert(escaped == nil and type(err) == 'string')
        print('captured', dep.value)
        return dofile('answer.lua')
        """,
        chunkname="=suite-capability-test",
    )
    assert result == 42


def test_suite_require_supports_nested_module_names(tmp_path):
    root = _suite_root(tmp_path)
    nested = root / "nested"
    nested.mkdir()
    (nested / "module.lua").write_text("return {value=23}", encoding="utf-8")
    lua = suite.make_suite_runtime(root)
    assert lua.execute("return require('nested.module').value") == 23


def test_static_classification_and_report(tmp_path):
    root = _suite_root(tmp_path)
    (root / "safe.lua").write_text("return string.upper('ok')", encoding="utf-8")
    (root / "loader.lua").write_text("return require 'safe'", encoding="utf-8")
    (root / "host.lua").write_text("return io.open('x')", encoding="utf-8")
    (root / "internal.lua").write_text("return T.querytab({})", encoding="utf-8")
    (root / "big.lua").write_text("return 1", encoding="utf-8")

    categories = {
        item.name: item.category
        for item in suite.classify_suite(root)
    }
    assert categories == {
        "big.lua": "stress",
        "host.lua": "sandbox-host",
        "internal.lua": "internal-c",
        "loader.lua": "harness-assisted",
        "safe.lua": "sandbox-safe",
    }

    report = suite.build_report(root)
    assert report["target"] == "Lua 5.5.1"
    assert report["suite_sha256"] == suite.SUITE_SHA256
    assert report["lua_files"] == 5
    assert report["classification_counts"] == {
        "harness-assisted": 1,
        "internal-c": 1,
        "sandbox-host": 1,
        "sandbox-safe": 1,
        "stress": 1,
    }


def test_run_files_uses_fresh_runtime_per_file(tmp_path):
    root = _suite_root(tmp_path)
    (root / "one.lua").write_text("x = 10; return x", encoding="utf-8")
    (root / "two.lua").write_text("assert(x == nil); return 20", encoding="utf-8")
    results = suite.run_files(root, ["one.lua", "two.lua"])
    assert [result.status for result in results] == ["pass", "pass"]
