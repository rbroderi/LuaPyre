from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile
from urllib.request import Request, urlopen


SUITE_URL = "https://www.lua.org/tests/lua-5.5.1-tests.tar.gz"
SUITE_SHA256 = "da07b543872dc0bb2ff12aabd0c248578d78df3eb6b67efdc537a46d455c7f31"
SUITE_ARCHIVE = "lua-5.5.1-tests.tar.gz"
SUITE_ROOT = "lua-5.5.1-tests"
MAX_ARCHIVE_BYTES = 4 * 1024 * 1024
MAX_EXTRACTED_BYTES = 16 * 1024 * 1024
MAX_MEMBERS = 10_000


class SuiteError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_archive(path: Path, *, expected: str = SUITE_SHA256) -> None:
    try:
        size = path.stat().st_size
    except FileNotFoundError as error:
        raise SuiteError(f"suite archive not found: {path}") from error
    if size > MAX_ARCHIVE_BYTES:
        raise SuiteError(f"suite archive is unexpectedly large: {size} bytes")
    actual = sha256_file(path)
    if actual != expected:
        raise SuiteError(
            f"Lua 5.5.1 test-suite checksum mismatch: expected {expected}, got {actual}"
        )


def download_archive(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(SUITE_URL, headers={"User-Agent": "LuaPyre-conformance/0.10"})
    tmp = destination.with_name(destination.name + ".tmp")
    try:
        with urlopen(request, timeout=30) as response, tmp.open("wb") as out:
            total = 0
            while True:
                block = response.read(64 * 1024)
                if not block:
                    break
                total += len(block)
                if total > MAX_ARCHIVE_BYTES:
                    raise SuiteError("downloaded test-suite archive exceeds size limit")
                out.write(block)
        verify_archive(tmp)
        os.replace(tmp, destination)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return destination


def ensure_archive(path: Path, *, allow_download: bool) -> Path:
    if path.exists():
        verify_archive(path)
        return path
    if not allow_download:
        raise SuiteError(f"suite archive not found: {path}")
    return download_archive(path)


def _safe_member_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise SuiteError(f"unsafe path in test-suite archive: {name!r}")
    parts = tuple(part for part in path.parts if part not in ("", "."))
    if not parts:
        raise SuiteError("empty path in test-suite archive")
    return PurePosixPath(*parts)


def safe_extract(
    archive: Path,
    destination: Path,
    *,
    max_members: int = MAX_MEMBERS,
    max_extracted_bytes: int = MAX_EXTRACTED_BYTES,
) -> Path:
    """Extract the pinned suite without trusting tar metadata or tar.extractall."""
    verify_archive(archive)
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()

    with tarfile.open(archive, mode="r:gz") as tf:
        members = tf.getmembers()
        if len(members) > max_members:
            raise SuiteError("test-suite archive contains too many members")

        total = 0
        checked: list[tuple[tarfile.TarInfo, PurePosixPath]] = []
        for member in members:
            relative = _safe_member_path(member.name)
            if not (member.isdir() or member.isfile()):
                raise SuiteError(
                    f"unsupported archive member type for {member.name!r}; links and devices are rejected"
                )
            if member.isfile():
                if member.size < 0:
                    raise SuiteError(f"negative member size for {member.name!r}")
                total += member.size
                if total > max_extracted_bytes:
                    raise SuiteError("test-suite archive exceeds extracted-size limit")
            checked.append((member, relative))

        for member, relative in checked:
            target = (root / Path(*relative.parts)).resolve()
            if target != root and root not in target.parents:
                raise SuiteError(f"archive path escapes destination: {member.name!r}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            source = tf.extractfile(member)
            if source is None:
                raise SuiteError(f"cannot read archive member: {member.name!r}")
            with source, target.open("wb") as out:
                shutil.copyfileobj(source, out, length=64 * 1024)

    suite_root = root / SUITE_ROOT
    if not suite_root.is_dir():
        raise SuiteError(f"archive does not contain expected root directory {SUITE_ROOT!r}")
    return suite_root


def prepare_suite(archive: Path, extract_to: Path, *, allow_download: bool) -> Path:
    archive = ensure_archive(archive, allow_download=allow_download)
    expected_root = extract_to / SUITE_ROOT
    if expected_root.is_dir():
        # Always re-verify the archive; reuse the already extracted tree only as
        # a convenience for manual runs. A clean CI checkout extracts afresh.
        verify_archive(archive)
        return expected_root.resolve()
    return safe_extract(archive, extract_to)


def list_lua_files(suite_root: Path) -> list[str]:
    return sorted(
        path.relative_to(suite_root).as_posix()
        for path in suite_root.rglob("*.lua")
        if path.is_file()
    )


def resolve_suite_file(suite_root: Path, name: str) -> Path:
    relative = _safe_member_path(name)
    root = suite_root.resolve()
    path = (root / Path(*relative.parts)).resolve()
    if path != root and root not in path.parents:
        raise SuiteError(f"test path escapes suite root: {name!r}")
    if not path.is_file():
        raise SuiteError(f"test file not found: {name}")
    if path.suffix != ".lua":
        raise SuiteError(f"not a Lua test file: {name}")
    return path


def run_selected(suite_root: Path, names: list[str], *, unrestricted: bool = True) -> bool:
    """Run explicitly selected upstream Lua files in isolated LuaPyre runtimes.

    `_U=true` matches Lua's documented basic-suite mode. This helper does not
    add ambient filesystem, OS, package, C-module, or debug-library access; a
    selected upstream file that requires those facilities fails explicitly.
    """
    from luapyre import LuaRuntime

    success = True
    for name in names:
        path = resolve_suite_file(suite_root, name)
        lua = LuaRuntime()
        if unrestricted:
            lua.set("_U", True)
        source = path.read_bytes()
        try:
            text = source.decode("utf-8")
        except UnicodeDecodeError as error:
            print(f"FAIL {name}: not UTF-8: {error}", file=sys.stderr)
            success = False
            continue
        try:
            lua.execute(text, chunkname="@" + name)
        except Exception as error:  # developer harness: report exact failing boundary
            print(f"FAIL {name}: {type(error).__name__}: {error}", file=sys.stderr)
            success = False
        else:
            print(f"PASS {name}")
    return success


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify and inspect the exact official Lua 5.5.1 test suite."
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(".official-lua-551"),
        help="cache/extraction directory (default: .official-lua-551)",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="use this pre-downloaded lua-5.5.1-tests.tar.gz instead of the work-dir cache",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="fail rather than downloading the pinned archive when missing",
    )
    parser.add_argument("--list", action="store_true", help="list all .lua files in the suite")
    parser.add_argument(
        "--files",
        nargs="*",
        default=[],
        metavar="FILE",
        help="explicit suite-relative .lua files to execute in fresh LuaPyre runtimes",
    )
    parser.add_argument(
        "--full-mode",
        action="store_true",
        help="do not set _U=true for selected tests (does not grant extra host capabilities)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    work_dir: Path = args.work_dir
    archive = args.archive or (work_dir / SUITE_ARCHIVE)
    extract_to = work_dir / "extracted"
    try:
        suite_root = prepare_suite(
            archive,
            extract_to,
            allow_download=not args.no_download,
        )
        files = list_lua_files(suite_root)
        print(f"Lua 5.5.1 official suite: {suite_root}")
        print(f"archive sha256: {SUITE_SHA256}")
        print(f"Lua files: {len(files)}")
        if args.list:
            for name in files:
                print(name)
        if args.files and not run_selected(
            suite_root,
            args.files,
            unrestricted=not args.full_mode,
        ):
            return 1
        return 0
    except SuiteError as error:
        print(f"official-suite error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
