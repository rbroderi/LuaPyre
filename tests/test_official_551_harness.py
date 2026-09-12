from __future__ import annotations

import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import tarfile

import pytest


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
    root = tmp_path / suite.SUITE_ROOT
    root.mkdir()
    (root / "ok.lua").write_text("return 1", encoding="utf-8")
    assert suite.resolve_suite_file(root, "ok.lua") == (root / "ok.lua").resolve()
    with pytest.raises(suite.SuiteError, match="unsafe path"):
        suite.resolve_suite_file(root, "../outside.lua")
