from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
from urllib.request import Request, urlopen


SUITE_URL = "https://www.lua.org/tests/lua-5.5.1-tests.tar.gz"
SUITE_SHA256 = "da07b543872dc0bb2ff12aabd0c248578d78df3eb6b67efdc537a46d455c7f31"
SUITE_ARCHIVE = "lua-5.5.1-tests.tar.gz"
SUITE_ROOT = "lua-5.5.1-tests"
MAX_ARCHIVE_BYTES = 4 * 1024 * 1024
MAX_EXTRACTED_BYTES = 16 * 1024 * 1024
MAX_SUITE_FILE_BYTES = 16 * 1024 * 1024
MAX_MEMBERS = 10_000
DEFAULT_FILE_FUEL = 20_000_000

# Provisional probe set: all files classified as sandbox-safe or requiring
# only the read-only suite module loader. The release baseline is reduced to
# the subset that passes unchanged against the pinned Lua 5.5.1 archive.
BASELINE_FILES = (
    "bwcoercion.lua",
    "pm.lua",
    "tpack.lua",
    "vararg.lua",
    "bitwise.lua",
    "math.lua",
    "utf8.lua",
)

_STRESS_FILES = {
    "big.lua",
    "cstack.lua",
    "gc.lua",
    "gengc.lua",
    "memerr.lua",
    "verybig.lua",
}

_DEPENDENCY_RULES = (
    ("internal-test-api", re.compile(rb"\bT\s*[\.\[]")),
    ("debug-library", re.compile(rb"(?:\brequire\s*\(?\s*['\"]debug['\"]|\bdebug\s*[\.:])")),
    ("io-library", re.compile(rb"\bio\s*[\.:]")),
    ("os-library", re.compile(rb"\bos\s*[\.:]")),
    ("package-library", re.compile(rb"\bpackage\s*[\.:]")),
    ("module-loader", re.compile(rb"\brequire\b")),
    ("file-loader", re.compile(rb"\b(?:loadfile|dofile)\b")),
)


class SuiteError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FileClassification:
    name: str
    category: str
    hints: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RunResult:
    name: str
    status: str
    error_type: str | None = None
    error: str | None = None


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
    request = Request(SUITE_URL, headers={"User-Agent": "LuaPyre-conformance/0.11"})
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
        verify_archive(archive)
        return expected_root.resolve()
    return safe_extract(archive, extract_to)


def list_lua_files(suite_root: Path) -> list[str]:
    return sorted(
        path.relative_to(suite_root).as_posix()
        for path in suite_root.rglob("*.lua")
        if path.is_file()
    )


def _resolve_suite_path(suite_root: Path, name: str, *, lua_only: bool) -> Path:
    relative = _safe_member_path(name)
    root = suite_root.resolve()
    path = (root / Path(*relative.parts)).resolve()
    if path != root and root not in path.parents:
        raise SuiteError(f"test path escapes suite root: {name!r}")
    if not path.is_file():
        raise SuiteError(f"test file not found: {name}")
    if lua_only and path.suffix != ".lua":
        raise SuiteError(f"not a Lua test file: {name}")
    if path.stat().st_size > MAX_SUITE_FILE_BYTES:
        raise SuiteError(f"suite file exceeds size limit: {name}")
    return path


def resolve_suite_file(suite_root: Path, name: str) -> Path:
    return _resolve_suite_path(suite_root, name, lua_only=True)


def resolve_suite_asset(suite_root: Path, name: str) -> Path:
    """Resolve a read-only asset below the suite root for the test harness."""
    return _resolve_suite_path(suite_root, name, lua_only=False)


def install_suite_capabilities(lua, suite_root: Path, *, echo: bool = False) -> None:
    """Install only host capabilities; Lua library semantics stay production-real."""
    root = suite_root.resolve()

    def suite_loader(name: str):
        try:
            return resolve_suite_asset(root, name).read_bytes()
        except (OSError, SuiteError):
            return None

    def output_sink(data: bytes):
        if echo:
            print(data.decode("utf-8", "replace"), end="")

    lua.set_file_loader(suite_loader)
    lua.set_output_sink(output_sink)
    lua.set("arg", [])


def make_suite_runtime(
    suite_root: Path,
    *,
    unrestricted: bool = True,
    echo: bool = False,
    fuel: int = DEFAULT_FILE_FUEL,
):
    from luapyre import LuaRuntime

    lua = LuaRuntime(fuel=fuel)
    install_suite_capabilities(lua, suite_root, echo=echo)
    if unrestricted:
        lua.set("_U", True)
    return lua


def classify_file(suite_root: Path, name: str) -> FileClassification:
    path = resolve_suite_file(suite_root, name)
    source = path.read_bytes()
    hints = [label for label, pattern in _DEPENDENCY_RULES if pattern.search(source)]
    if name in _STRESS_FILES:
        hints.append("stress/resource-sensitive")

    unique = tuple(dict.fromkeys(hints))
    if "internal-test-api" in unique:
        category = "internal-c"
    elif any(item in unique for item in ("debug-library", "io-library", "os-library")):
        category = "sandbox-host"
    elif "stress/resource-sensitive" in unique:
        category = "stress"
    elif any(item in unique for item in ("package-library", "module-loader", "file-loader")):
        category = "harness-assisted"
    else:
        category = "sandbox-safe"
    return FileClassification(name, category, unique)


def classify_suite(suite_root: Path) -> list[FileClassification]:
    return [classify_file(suite_root, name) for name in list_lua_files(suite_root)]


def run_files(
    suite_root: Path,
    names: list[str] | tuple[str, ...],
    *,
    unrestricted: bool = True,
    echo: bool = False,
    fuel: int = DEFAULT_FILE_FUEL,
) -> list[RunResult]:
    """Run selected upstream Lua files in isolated, suite-rooted runtimes."""
    results: list[RunResult] = []
    for name in names:
        path = resolve_suite_file(suite_root, name)
        lua = make_suite_runtime(
            suite_root,
            unrestricted=unrestricted,
            echo=echo,
            fuel=fuel,
        )
        source = path.read_bytes()
        try:
            text = source.decode("utf-8")
        except UnicodeDecodeError as error:
            result = RunResult(name, "fail", type(error).__name__, str(error))
            results.append(result)
            print(f"FAIL {name}: not UTF-8: {error}", file=sys.stderr)
            continue
        try:
            lua.execute(text, chunkname="@" + name)
        except Exception as error:
            result = RunResult(name, "fail", type(error).__name__, str(error))
            results.append(result)
            print(f"FAIL {name}: {type(error).__name__}: {error}", file=sys.stderr)
        else:
            results.append(RunResult(name, "pass"))
            print(f"PASS {name}")
    return results


def run_selected(
    suite_root: Path,
    names: list[str],
    *,
    unrestricted: bool = True,
    echo: bool = False,
    fuel: int = DEFAULT_FILE_FUEL,
) -> bool:
    return all(
        result.status == "pass"
        for result in run_files(
            suite_root,
            names,
            unrestricted=unrestricted,
            echo=echo,
            fuel=fuel,
        )
    )


def build_report(
    suite_root: Path,
    *,
    classifications: list[FileClassification] | None = None,
    runs: list[RunResult] | None = None,
) -> dict[str, object]:
    classifications = classifications or classify_suite(suite_root)
    counts = Counter(item.category for item in classifications)
    runs = runs or []
    return {
        "target": "Lua 5.5.1",
        "suite_sha256": SUITE_SHA256,
        "lua_files": len(classifications),
        "classification_counts": dict(sorted(counts.items())),
        "files": [asdict(item) for item in classifications],
        "runs": [asdict(item) for item in runs],
        "run_summary": {
            "total": len(runs),
            "passed": sum(item.status == "pass" for item in runs),
            "failed": sum(item.status != "pass" for item in runs),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify, classify, and run the exact official Lua 5.5.1 test suite."
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
        "--classify",
        action="store_true",
        help="print the static dependency category for every .lua file",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="run the committed sandbox-compatible 5.5.1 baseline",
    )
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
    parser.add_argument(
        "--echo",
        action="store_true",
        help="forward the upstream suite's print calls to stdout",
    )
    parser.add_argument(
        "--fuel",
        type=int,
        default=DEFAULT_FILE_FUEL,
        help=f"instruction fuel per selected file (default: {DEFAULT_FILE_FUEL})",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        help="write a machine-readable classification/run report",
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
        classifications = classify_suite(suite_root)
        print(f"Lua 5.5.1 official suite: {suite_root}")
        print(f"archive sha256: {SUITE_SHA256}")
        print(f"Lua files: {len(files)}")
        if args.list:
            for name in files:
                print(name)
        if args.classify:
            for item in classifications:
                detail = ", ".join(item.hints) if item.hints else "no host dependency hints"
                print(f"{item.category:16} {item.name:20} {detail}")
            counts = Counter(item.category for item in classifications)
            print("classification summary:")
            for category, count in sorted(counts.items()):
                print(f"  {category}: {count}")

        selected: list[str] = []
        if args.baseline:
            selected.extend(BASELINE_FILES)
        selected.extend(args.files)
        selected = list(dict.fromkeys(selected))

        runs: list[RunResult] = []
        if selected:
            runs = run_files(
                suite_root,
                selected,
                unrestricted=not args.full_mode,
                echo=args.echo,
                fuel=args.fuel,
            )

        if args.report_json:
            args.report_json.parent.mkdir(parents=True, exist_ok=True)
            report = build_report(
                suite_root,
                classifications=classifications,
                runs=runs,
            )
            args.report_json.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        if any(result.status != "pass" for result in runs):
            return 1
        return 0
    except SuiteError as error:
        print(f"official-suite error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
