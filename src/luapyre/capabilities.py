from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import warnings


OutputSink = Callable[[bytes], object]
WarningSink = Callable[[bytes], object]
FileLoader = Callable[[str], bytes | str | None]


def default_output_sink(data: bytes) -> None:
    """Write Lua-formatted bytes to Python stdout."""
    print(data.decode("utf-8", "replace"), end="")


def default_warning_sink(data: bytes) -> None:
    """Route Lua warnings through Python's warnings subsystem."""
    warnings.warn(data.decode("utf-8", "replace"), RuntimeWarning, stacklevel=3)


@dataclass(slots=True)
class RuntimeCapabilities:
    """Mutable host capabilities shared by the installed standard library.

    Output and warning sinks receive already Lua-formatted bytes. The default
    output sink simply delegates to Python ``print``. Filesystem access is
    absent unless ``file_loader`` is explicitly supplied by the embedder.
    """

    output_sink: OutputSink = default_output_sink
    warning_sink: WarningSink = default_warning_sink
    file_loader: FileLoader | None = None

    def set_output_sink(self, sink: OutputSink | None) -> None:
        if sink is not None and not callable(sink):
            raise TypeError("output sink must be callable or None")
        self.output_sink = default_output_sink if sink is None else sink

    def set_warning_sink(self, sink: WarningSink | None) -> None:
        if sink is not None and not callable(sink):
            raise TypeError("warning sink must be callable or None")
        self.warning_sink = default_warning_sink if sink is None else sink

    def set_file_loader(self, loader: FileLoader | None) -> None:
        if loader is not None and not callable(loader):
            raise TypeError("file loader must be callable or None")
        self.file_loader = loader

    def read_file(self, name: bytes | str) -> tuple[bytes | None, bytes | None]:
        loader = self.file_loader
        if loader is None:
            return None, b"file loading is disabled"
        if isinstance(name, bytes):
            try:
                text = name.decode("utf-8")
            except UnicodeDecodeError:
                return None, b"filename is not valid UTF-8"
        elif isinstance(name, str):
            text = name
        else:
            return None, b"filename must be a string"
        try:
            result = loader(text)
        except OSError as error:
            return None, str(error).encode("utf-8", "replace")
        if result is None:
            return None, f"cannot open {text}".encode("utf-8", "replace")
        if isinstance(result, str):
            result = result.encode("utf-8")
        if not isinstance(result, bytes):
            raise TypeError("file loader must return bytes, str, or None")
        return result, None
