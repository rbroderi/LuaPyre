from __future__ import annotations

from .bytecode import Cell, Closure, Op
from .capabilities import RuntimeCapabilities
from .diagnostics import chunk_id, format_traceback, frame_line
from .errors import LuaRuntimeError
from .stdlib_support import lua_c_function, need_integer
from .table import LuaTable
from .values import MultiValue
from .vm import HostFunction
from .threadvm import LuaThread


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

    def resolve_target(target, thread=None):
        if isinstance(target, Closure):
            return target, None
        level = need_integer(target, 1, "getinfo")
        frames = thread.frames if thread is not None else (vm._active_frames or ())
        frame = frames[-level] if 0 < level <= len(frames) else None
        return (frame.closure, frame) if frame is not None else (None, None)

    def getinfo(*args):
        values = list(args)
        target_thread = None
        if values and isinstance(values[0], LuaThread):
            target_thread = values.pop(0)
        target = values[0] if values else 1
        options = values[1] if len(values) > 1 else b"flnSrtu"
        if not isinstance(options, bytes) or any(
            option not in b"flnSrtuL" for option in options
        ):
            raise LuaRuntimeError("invalid option")
        if (
            type(target) is int
            and target == 2
            and vm._hook_thread().hook_running
            and isinstance(getattr(vm, "_active_hook_subject", None), HostFunction)
        ):
            subject = vm._active_hook_subject
            result = getinfo(subject, options)
            result.rawset(b"name", subject.name.rsplit(".", 1)[-1].encode())
            result.rawset(b"namewhat", b"global")
            result.rawset(b"ftransfer", vm._active_hook_transfer_base)
            result.rawset(b"ntransfer", len(vm._active_hook_transfer))
            return result
        if isinstance(target, HostFunction):
            result = LuaTable()
            result.rawset(b"source", b"=[C]")
            result.rawset(b"short_src", b"[C]")
            result.rawset(b"linedefined", -1)
            result.rawset(b"lastlinedefined", -1)
            result.rawset(b"what", b"C")
            result.rawset(b"currentline", -1)
            result.rawset(b"nups", len(target._gc_refs))
            result.rawset(b"nparams", 0)
            result.rawset(b"isvararg", True)
            result.rawset(b"istailcall", False)
            result.rawset(b"extraargs", 0)
            result.rawset(b"ftransfer", 0)
            result.rawset(b"ntransfer", 0)
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
        closure, frame = resolve_target(target, target_thread)
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
                if frame is None or (
                    proto.name in ("<chunk>", "<anonymous>")
                    and frame.call_name is None
                    and frame.trace_name != "__close"
                )
                else (
                    frame.call_name
                    or ("close" if frame.trace_name == "__close" else proto.name)
                ).encode()
            ),
            b"namewhat": (
                (
                    frame.call_namewhat
                    or ("metamethod" if frame.trace_name == "__close" else proto.debug_namewhat)
                ).encode()
                if frame is not None else b""
            ),
            b"nups": len(proto.upvalues) + int(proto.env_reg >= 0),
            b"nparams": proto.param_count,
            b"isvararg": proto.is_vararg,
            b"istailcall": frame.is_tailcall if frame is not None else False,
            b"ftransfer": 0,
            b"ntransfer": 0,
            b"extraargs": len(frame.varargs) if frame is not None else 0,
            b"func": closure,
        }
        for key, value in fields.items():
            if value is not None:
                result.rawset(key, value)
        if type(target) is int and target == 2 and vm._hook_thread().hook_running:
            result.rawset(b"ftransfer", vm._active_hook_transfer_base)
            result.rawset(b"ntransfer", len(vm._active_hook_transfer))
        if b"L" in options:
            active_lines = LuaTable()
            for line in set(proto.lineinfo):
                if (
                    proto.linedefined < line <= proto.lastlinedefined
                    or (
                        proto.linedefined == proto.lastlinedefined
                        and line == proto.linedefined
                    )
                ):
                    active_lines.rawset(line, True)
            result.rawset(b"activelines", active_lines)
        return result

    put("getinfo", getinfo)

    def getlocal(*args):
        values = list(args)
        thread = None
        if values and isinstance(values[0], LuaThread):
            thread = values.pop(0)
        if len(values) < 2:
            raise LuaRuntimeError("bad argument to 'getlocal' (value expected)")
        target, index = values[:2]
        index = need_integer(index, 2, "getlocal")
        if isinstance(target, Closure):
            active = target.proto.debug_locals[:target.proto.param_count]
            if 1 <= index <= len(active):
                return active[index - 1][0].encode()
            return None
        if isinstance(target, HostFunction):
            return None
        level = need_integer(target, 1, "getlocal")
        if level == 0:
            if index == 1:
                return MultiValue((b"(C temporary)", target))
            if index == 2:
                return MultiValue((b"(C temporary)", index))
            return None
        if level == 2 and vm._hook_thread().hook_running:
            base = vm._active_hook_transfer_base
            offset = index - base
            if 0 <= offset < len(vm._active_hook_transfer):
                return MultiValue(
                    (b"(temporary)", vm._active_hook_transfer[offset])
                )
        frames = thread.frames if thread is not None else (vm._active_frames or ())
        frame = frames[-level] if 0 < level <= len(frames) else None
        if frame is None:
            raise LuaRuntimeError("level out of range")
        if index < 0:
            offset = -index
            if 1 <= offset <= len(frame.varargs):
                return MultiValue((b"(vararg)", frame.varargs[offset - 1]))
            return None
        pc = max(0, frame.pc - 1)
        active = [item for item in frame.proto.debug_locals if item[2] <= pc <= item[3]]
        if frame.proto.code[pc].op in (Op.RETURN, Op.RETURNV, Op.HALT):
            active = [item for item in active if item[0] != "(vararg table)"]
        if index > len(active):
            named_registers = {item[1] for item in active}
            live_pc = (
                pc
                if frame.proto.code[pc].op is Op.LOCAL
                else min(frame.pc, len(frame.proto.code))
            )
            temporaries = [
                register
                for register in sorted(vm.gc._live_sets(frame.proto)[live_pc])
                if register not in named_registers
                and 0 <= register < len(frame.regs)
                and frame.regs[register] is not None
            ]
            offset = index - len(active) - 1
            if 0 <= offset < len(temporaries):
                return MultiValue((b"(temporary)", frame.regs[temporaries[offset]]))
            return None
        if index <= 0:
            return None
        name, register, _start, _end = active[index - 1]
        if register == -1 and name == "(vararg table)":
            value = LuaTable.from_sequence(frame.varargs)
            value.rawset(b"n", len(frame.varargs))
            vm.gc.adopt(value)
            return MultiValue((name.encode(), value))
        cell = frame.cells.get(register)
        value = frame.regs[register] if cell is None else cell.value
        return MultiValue((name.encode(), value))

    put("getlocal", getlocal)

    def setlocal(*args):
        values = list(args)
        thread = None
        if values and isinstance(values[0], LuaThread):
            thread = values.pop(0)
        if len(values) < 3:
            raise LuaRuntimeError("bad argument to 'setlocal' (value expected)")
        level = need_integer(values[0], 1, "setlocal")
        index = need_integer(values[1], 2, "setlocal")
        new_value = values[2]
        frames = thread.frames if thread is not None else (vm._active_frames or ())
        frame = frames[-level] if 0 < level <= len(frames) else None
        if frame is None:
            raise LuaRuntimeError("level out of range")
        if index < 0:
            offset = -index
            if not 1 <= offset <= len(frame.varargs):
                return None
            items = list(frame.varargs)
            items[offset - 1] = new_value
            frame.varargs = tuple(items)
            return b"(vararg)"
        pc = max(0, frame.pc - 1)
        active = [item for item in frame.proto.debug_locals if item[2] <= pc <= item[3]]
        if frame.proto.code[pc].op in (Op.RETURN, Op.RETURNV, Op.HALT):
            active = [item for item in active if item[0] != "(vararg table)"]
        if index > len(active):
            named_registers = {item[1] for item in active}
            live_pc = (
                pc
                if frame.proto.code[pc].op is Op.LOCAL
                else min(frame.pc, len(frame.proto.code))
            )
            temporaries = [
                register
                for register in sorted(vm.gc._live_sets(frame.proto)[live_pc])
                if register not in named_registers
                and 0 <= register < len(frame.regs)
                and frame.regs[register] is not None
            ]
            offset = index - len(active) - 1
            if 0 <= offset < len(temporaries):
                frame.regs[temporaries[offset]] = new_value
                return b"(temporary)"
            return None
        if index <= 0:
            return None
        name, register, _start, _end = active[index - 1]
        if register == -1:
            return None
        cell = frame.cells.get(register)
        if cell is None:
            frame.regs[register] = new_value
        else:
            cell.value = new_value
            frame.regs[register] = new_value
            vm.gc.write_barrier(cell, new_value)
        return name.encode()

    # Stack mutation belongs to the explicit full-debug profile, alongside
    # hooks. The default sandbox keeps read-only Lua-state inspection only.
    if vm.debug_hooks_enabled:
        put("setlocal", setlocal)

    def getupvalue(fn, index):
        if isinstance(fn, HostFunction):
            index = need_integer(index, 2, "getupvalue")
            return MultiValue((b"", fn)) if index == 1 else None
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

    def traceback(*args):
        values = list(args)
        target_thread = None
        if values and isinstance(values[0], LuaThread):
            target_thread = values.pop(0)
        message = values[0] if values else None
        level = values[1] if len(values) > 1 else 1
        include_yield = target_thread is not None and level is None
        if level is None:
            level = 1
        level = need_integer(level, 2, "traceback")
        if message is not None and not isinstance(message, bytes):
            return message
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
        if getattr(vm, "_direct_protected_host", False):
            lines.append(b"\t[C]: in function 'pcall'")
        if level == 0:
            lines.append(b"\t[C]: in function 'traceback'")
        if include_yield and target_thread.status == "suspended":
            lines.append(b"\t[C]: in function 'yield'")
        elif (
            include_yield
            and target_thread.status == "dead"
            and target_thread.error is not None
        ):
            lines.append(b"\t[C]: in function 'error'")
        frames = (
            target_thread.frames
            if target_thread is not None
            else (vm._active_frames or ())
        )
        trace_frames = list(
            reversed(frames[: max(0, len(frames) - level + 1)])
        )
        # Lua keeps tracebacks useful without letting recursive programs
        # produce unbounded diagnostic strings: ten innermost frames, a
        # skipped-level marker, and eleven outermost frames.
        if len(trace_frames) > 21:
            skipped = len(trace_frames) - 21
            trace_items = (
                trace_frames[:10]
                + [skipped]
                + trace_frames[-11:]
            )
        else:
            trace_items = trace_frames
        for frame in trace_items:
            if isinstance(frame, int):
                lines.append(
                    b"\t...\t(skipping " + str(frame).encode() + b" levels)"
                )
                continue
            source = frame.proto.source
            if frame.call_namewhat == "hook" or frame.trace_name == "hook":
                context = b"in hook"
            elif frame.proto.name == "<anonymous>":
                context = (
                    b"in function <" + chunk_id(source) + b":"
                    + str(frame.proto.linedefined).encode() + b">"
                )
            elif frame.proto.name == "<chunk>":
                context = b"in function"
            else:
                name = (frame.call_name or frame.proto.name).encode()
                context = b"in function '" + name + b"'"
            lines.append(
                b"\t" + chunk_id(source) + b":"
                + str(frame_line(frame)).encode() + b": " + context
            )
        return b"\n".join(lines)

    put("traceback", traceback)
    put("getregistry", lambda: registry)

    if vm.debug_hooks_enabled:
        def sethook(*args):
            values = list(args)
            thread = None
            if values and isinstance(values[0], LuaThread):
                thread = values.pop(0)
            hook = values[0] if values else None
            mask = values[1] if len(values) > 1 else b""
            count = values[2] if len(values) > 2 else 0
            return vm.set_hook(thread, hook, mask, count)

        def gethook(thread=None):
            return vm.get_hook(thread)

        put("sethook", sethook)
        put("gethook", gethook)

        hook_keys = LuaTable()
        hook_keys.metatable = LuaTable()
        hook_keys.metatable.rawset(b"__mode", b"k")
        registry.rawset(b"_HOOKKEY", hook_keys)
    else:
        put("gethook", lambda: MultiValue((None, b"", 0)))
    globals_table.rawset(b"debug", library)
    registry.rawset(b"_G", globals_table)
    registry.rawset(b"_MAINTHREAD", vm.main_thread)
    return library
