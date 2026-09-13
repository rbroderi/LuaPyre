from __future__ import annotations

import datetime as _datetime
import time as _time

from .capabilities import RuntimeCapabilities
from .errors import LuaRuntimeError
from .stdlib_support import lua_c_function, need_bytes, need_number
from .table import LuaTable
from .values import MultiValue


def install_os_library(globals_table: LuaTable, capabilities: RuntimeCapabilities) -> LuaTable:
    """Install OS helpers that never invoke a process or access the host filesystem."""
    library = LuaTable()
    temporary_counter = 0

    def put(name, fn):
        library.rawset(name.encode("ascii"), lua_c_function(fn, f"os.{name}"))

    def clock():
        return _time.process_time()

    put("clock", clock)
    put("difftime", lambda first, second: need_number(first, 1, "difftime") - need_number(second, 2, "difftime"))

    def time_fn(value=None):
        if value is None:
            return int(_time.time())
        if not isinstance(value, LuaTable):
            raise LuaRuntimeError("bad argument #1 to 'time' (table expected)")
        for required in (b"year", b"month", b"day"):
            if value.rawget(required) is None:
                raise LuaRuntimeError(f"field '{required.decode()}' missing in date table")
        now = _datetime.datetime.now()
        fields = {
            "year": now.year,
            "month": now.month,
            "day": now.day,
            "hour": 12,
            "min": 0,
            "sec": 0,
        }
        for name in fields:
            item = value.rawget(name.encode("ascii"))
            if item is not None:
                if type(item) is not int and not (
                    type(item) is float and item.is_integer()
                ):
                    raise LuaRuntimeError(f"field '{name}' is not an integer")
                fields[name] = int(item)
        try:
            normalized_year = fields["year"] + (fields["month"] - 1) // 12
            normalized_month = (fields["month"] - 1) % 12 + 1
            date = _datetime.datetime(normalized_year, normalized_month, 1) + _datetime.timedelta(
                days=fields["day"] - 1,
                hours=fields["hour"], minutes=fields["min"], seconds=fields["sec"],
            )
        except (ValueError, OverflowError) as error:
            if not 1 <= fields["year"] <= 9999:
                raise LuaRuntimeError("field 'year' is out-of-bound") from None
            raise LuaRuntimeError(str(error)) from None
        for name, item in (
            (b"year", date.year), (b"month", date.month), (b"day", date.day),
            (b"hour", date.hour), (b"min", date.minute), (b"sec", date.second),
            (b"wday", (date.weekday() + 1) % 7 + 1),
            (b"yday", date.timetuple().tm_yday),
            (b"isdst", bool(date.dst())),
        ):
            value.rawset(name, item)
        return int(date.timestamp())

    put("time", time_fn)

    def date_fn(fmt=b"%c", timestamp=None):
        fmt = need_bytes(fmt, 1, "date")
        timestamp = _time.time() if timestamp is None else need_number(timestamp, 2, "date")
        utc = fmt.startswith(b"!")
        if utc:
            fmt = fmt[1:]
        index = 0
        conversions = b"aAbBcdDFHhIjklMmnpPrRSTtUuVwWxXyYzZ%"
        while index < len(fmt):
            if fmt[index:index + 1] != b"%":
                index += 1
                continue
            if index + 1 >= len(fmt) or fmt[index + 1] not in conversions:
                raise LuaRuntimeError("invalid conversion specifier")
            index += 2
        value = _datetime.datetime.fromtimestamp(timestamp, _datetime.UTC if utc else None)
        if fmt == b"*t":
            result = LuaTable()
            for name, item in (
                (b"year", value.year), (b"month", value.month), (b"day", value.day),
                (b"hour", value.hour), (b"min", value.minute), (b"sec", value.second),
                (b"wday", (value.weekday() + 1) % 7 + 1), (b"yday", value.timetuple().tm_yday),
                (b"isdst", bool(value.dst())),
            ):
                result.rawset(name, item)
            return result
        try:
            return value.strftime(fmt.decode("ascii")).encode("utf-8")
        except (UnicodeDecodeError, ValueError) as error:
            raise LuaRuntimeError(str(error)) from None

    put("date", date_fn)
    def getenv(name):
        return capabilities.environment.get(need_bytes(name, 1, "getenv"))

    put("getenv", getenv)

    def setlocale(locale=None, category=b"all"):
        if locale is not None:
            locale = need_bytes(locale, 1, "setlocale")
        need_bytes(category, 2, "setlocale")
        return b"C" if locale in (None, b"C", b"POSIX") else None

    put("setlocale", setlocale)

    def tmpname():
        nonlocal temporary_counter
        temporary_counter += 1
        name = f"@luapyre-tmp/{temporary_counter}"
        capabilities.write_virtual_file(name, b"")
        return name.encode("utf-8")

    put("tmpname", tmpname)

    def remove(name):
        name = need_bytes(name, 1, "remove").decode("utf-8", "surrogateescape")
        if name not in capabilities.virtual_files:
            return MultiValue((None, b"file is outside the in-memory sandbox"))
        del capabilities.virtual_files[name]
        return True

    def rename(old, new):
        old = need_bytes(old, 1, "rename").decode("utf-8", "surrogateescape")
        new = need_bytes(new, 2, "rename").decode("utf-8", "surrogateescape")
        if old not in capabilities.virtual_files:
            return MultiValue((None, b"file is outside the in-memory sandbox"))
        capabilities.virtual_files[new] = capabilities.virtual_files.pop(old)
        return True

    put("remove", remove)
    put("rename", rename)
    globals_table.rawset(b"os", library)
    return library
