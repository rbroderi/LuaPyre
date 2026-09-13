from __future__ import annotations

from .bytecode import Cell, Closure
from .capabilities import RuntimeCapabilities
from .diagnostics import chunk_id, format_traceback, frame_line
from .errors import LuaRuntimeError
from .stdlib_support import lua_c_function, need_integer
from .table import LuaTable
from .values import MultiValue
from .vm import HostFunction


def install_debug_library(
    globals_table: LuaTable,
    vm,
    capabilities: RuntimeCapabilities,
) -> LuaTable:
    """Install Lua-only introspection without exposing Python or host state."""
    library = LuaTable()
    registry = LuaTable()

    def put(name, fn):
        library.rawset(name.encode("ascii"), lua_c_function(fn, f"debug.{name}"))

    def resolve_target(target):
        if isinstance(target, Closure):
            return target, None
        level = need_integer(target, 1, "getinfo")
        frames = vm._active_frames or ()
        frame = frames[-level] if 0 < level <= len(frames) else None
        return (frame.closure, frame) if frame is not None else (None, None)

    def getinfo(target=1, options=b"flnSrtu"):
        if not isinstance(options, bytes) or any(
            option not in b"flnSrtuL" for option in options
        ):
            raise LuaRuntimeError("invalid option")
        if isinstance(target, HostFunction):
            result = LuaTable()
            result.rawset(b"source", b"=[C]")
            result.rawset(b"short_src", b"[C]")
            result.rawset(b"linedefined", -1)
            result.rawset(b"lastlinedefined", -1)
            result.rawset(b"what", b"C")
            result.rawset(b"currentline", -1)
            result.rawset(b"nups", 0)
            result.rawset(b"nparams", 0)
            result.rawset(b"isvararg", True)
            result.rawset(b"func", target)
            return result
        if type(target) is int and target == 2:
            # Protected calls are VM continuations. Preserve Lua's observable
            # C boundary without exposing the Python stack or adding a real
            # host callback frame.
            protected = next(
                (fn for fn in reversed(getattr(vm, "_host_call_stack", ()))
                 if fn.name in ("pcall", "xpcall")),
                None,
            )
            if protected is None:
                protected_frame = next(
                    (
                        frame
                        for frame in reversed(vm._active_frames or ())
                        if frame.protected_name in ("pcall", "xpcall")
                    ),
                    None,
                )
                if protected_frame is not None:
                    protected = HostFunction(
                        lambda: None,
                        protected_frame.protected_name,
                    )
            if protected is not None:
                result = getinfo(protected, options)
                result.rawset(b"name", protected.name.encode())
                result.rawset(b"namewhat", b"global")
                return result
        closure, frame = resolve_target(target)
        if closure is None:
            return None
        proto = closure.proto
        source = proto.source
        if isinstance(source, str):
            source = source.encode("utf-8", "surrogateescape")
        if source is None:
            source = b"=?"
        result = LuaTable()
        fields = {
            b"source": source,
            b"short_src": chunk_id(source),
            b"linedefined": proto.linedefined,
            b"lastlinedefined": proto.lastlinedefined,
            b"what": b"main" if proto.name == "<chunk>" else b"Lua",
            b"currentline": frame_line(frame) if frame is not None else -1,
            b"name": (
                None
                if frame is None or proto.name in ("<chunk>", "<anonymous>")
                else (frame.call_name or proto.name).encode()
            ),
            b"namewhat": (
                (frame.call_namewhat or proto.debug_namewhat).encode()
                if frame is not None else b""
            ),
            b"nups": len(proto.upvalues) + int(proto.env_reg >= 0),
            b"nparams": proto.param_count,
            b"isvararg": proto.is_vararg,
            b"istailcall": False,
            b"ftransfer": 0,
            b"ntransfer": 0,
            b"extraargs": len(frame.varargs) if frame is not None else 0,
            b"func": closure,
        }
        for key, value in fields.items():
            if value is not None:
                result.rawset(key, value)
        if b"L" in options:
            active_lines = LuaTable()
            for line in set(proto.lineinfo):
                if proto.linedefined < line <= proto.lastlinedefined:
                    active_lines.rawset(line, True)
            result.rawset(b"activelines", active_lines)
        return result

    put("getinfo", getinfo)

    def getlocal(target, index):
        index = need_integer(index, 2, "getlocal")
        if isinstance(target, Closure):
            active = target.proto.debug_locals
            if 1 <= index <= len(active):
                return active[index - 1][0].encode()
            return None
        level = need_integer(target, 1, "getlocal")
        frames = vm._active_frames or ()
        frame = frames[-level] if 0 < level <= len(frames) else None
        if frame is None:
            return None
        pc = max(0, frame.pc - 1)
        active = [item for item in frame.proto.debug_locals if item[2] <= pc <= item[3]]
        if not 1 <= index <= len(active):
            return None
        name, register, _start, _end = active[index - 1]
        cell = frame.cells.get(register)
        value = frame.regs[register] if cell is None else cell.value
        return MultiValue((name.encode(), value))

    put("getlocal", getlocal)

    def getupvalue(fn, index):
        if not isinstance(fn, Closure):
            return None
        index = need_integer(index, 2, "getupvalue")
        if 1 <= index <= len(fn.proto.upvalues):
            return MultiValue((fn.proto.upvalues[index - 1].name.encode(), fn.upvalues[index - 1].value))
        if index == len(fn.proto.upvalues) + 1 and fn.proto.env_reg >= 0:
            return MultiValue((b"_ENV", fn.env))
        return None

    def setupvalue(fn, index, value):
        if not isinstance(fn, Closure):
            raise LuaRuntimeError("bad argument #1 to 'setupvalue' (function expected)")
        index = need_integer(index, 2, "setupvalue")
        if 1 <= index <= len(fn.proto.upvalues):
            fn.upvalues[index - 1].value = value
            vm.gc.write_barrier(fn, value)
            return fn.proto.upvalues[index - 1].name.encode()
        if index == len(fn.proto.upvalues) + 1 and fn.proto.env_reg >= 0:
            fn.env = value
            vm.gc.write_barrier(fn, value)
            return b"_ENV"
        return None

    put("getupvalue", getupvalue)
    put("setupvalue", setupvalue)

    def getuservalue(value, index=1):
        index = need_integer(index, 2, "getuservalue")
        return MultiValue((None, False))

    def setuservalue(value, new_value, index=1):
        index = need_integer(index, 3, "setuservalue")
        if isinstance(value, Cell):
            raise LuaRuntimeError(
                "bad argument #1 to 'setuservalue' "
                "(userdata expected, got light userdata)"
            )
        return None

    put("getuservalue", getuservalue)
    put("setuservalue", setuservalue)

    def upvalueid(fn, index):
        if not isinstance(fn, (Closure, HostFunction)):
            raise LuaRuntimeError("bad argument #1 to 'upvalueid' (function expected)")
        index = need_integer(index, 2, "upvalueid")
        if isinstance(fn, HostFunction):
            return fn if index == 1 else None
        if not 1 <= index <= len(fn.upvalues):
            return None
        return fn.upvalues[index - 1]

    def upvaluejoin(first, first_index, second, second_index):
        if not isinstance(first, Closure) or not isinstance(second, Closure):
            raise LuaRuntimeError("function expected")
        first_index = need_integer(first_index, 2, "upvaluejoin")
        second_index = need_integer(second_index, 4, "upvaluejoin")
        if not 1 <= first_index <= len(first.upvalues) or not 1 <= second_index <= len(second.upvalues):
            raise LuaRuntimeError("invalid upvalue index")
        first.upvalues[first_index - 1] = second.upvalues[second_index - 1]

    put("upvalueid", upvalueid)
    put("upvaluejoin", upvaluejoin)

    def getmetatable(value):
        return vm.metatable_for(value)

    def setmetatable(value, metatable):
        if metatable is not None and not isinstance(metatable, LuaTable):
            raise LuaRuntimeError("bad argument #2 to 'setmetatable' (nil or table expected)")
        if isinstance(value, LuaTable):
            value.metatable = metatable
            value.version += 1
            vm.gc.adopt(metatable)
            vm.gc.write_barrier(value, metatable)
            vm.gc.observe_weak_table(value)
            vm.gc.mark_finalizable(value, metatable)
        else:
            if isinstance(value, bytes):
                kind = b"string"
            elif type(value) in (int, float):
                kind = b"number"
            elif type(value) is bool:
                kind = b"boolean"
            elif value is None:
                kind = b"nil"
            elif isinstance(value, (Closure, HostFunction)):
                kind = b"function"
            else:
                raise LuaRuntimeError("unsupported value for debug.setmetatable")
            if metatable is None:
                vm.type_metatables.pop(kind, None)
            else:
                vm.type_metatables[kind] = metatable
                vm.gc.adopt(metatable)
        return value

    put("getmetatable", getmetatable)
    put("setmetatable", setmetatable)

    def traceback(message=None, level=1):
        level = need_integer(level, 2, "traceback")
        protected_error = getattr(vm, "_active_protected_error", None)
        if protected_error is None:
            protected_error = next(
                (
                    frame.protected_error
                    for frame in reversed(vm._active_frames or ())
                    if frame.protected_error is not None
                ),
                None,
            )
        if protected_error is not None:
            return format_traceback(protected_error, include_message=True)
        prefix = b"" if message is None else (
            message if isinstance(message, bytes) else str(message).encode("utf-8", "replace")
        )
        lines = [prefix] if prefix else []
        lines.append(b"stack traceback:")
        frames = vm._active_frames or ()
        for frame in reversed(frames[: max(0, len(frames) - level + 1)]):
            source = frame.proto.source
            lines.append(b"\t" + chunk_id(source) + b":" + str(frame_line(frame)).encode() + b": in function")
        return b"\n".join(lines)

    put("traceback", traceback)
    put("getregistry", lambda: registry)
    put("gethook", lambda: MultiValue((None, b"", 0)))
    globals_table.rawset(b"debug", library)
    registry.rawset(b"_G", globals_table)
    registry.rawset(b"_MAINTHREAD", vm.main_thread)
    return library
