from __future__ import annotations

import ast
from dataclasses import dataclass, field
from types import FunctionType

from .ast_backend import JumpListLayout, inline_expression_helper, inline_local_jump_list
from .bytecode import Closure, Ins, Op, Proto
from .errors import LuaRuntimeError
from .opdispatch import _float_divide, _float_modulo
from .range_analysis import analyze_integer_ranges
from .table import LuaTable, _ABSENT, _hash_key
from .typed_ir import TypedIRCompiler, _reads, _writes
from .values import lua_equal, static_value_type, type_matches


_FUNC_RETURN = -3
_FUNC_SUSPEND = -2

_FUNCTION_CONTROL = frozenset(
    (
        Op.JMP,
        Op.JMPIF,
        Op.JMPIFNOT,
        Op.JMPIFNIL,
        Op.FORPREP,
        Op.FORLOOP,
        Op.JFORLOOP,
    )
)

_FUNCTION_OPS = frozenset(
    {
        Op.LOADK,
        Op.MOVE,
        Op.LOCAL,
        Op.GETUPVAL,
        Op.SETUPVAL,
        Op.NEWTABLE,
        Op.GETTABLE,
        Op.SETTABLE,
        Op.ADD,
        Op.ADD_I,
        Op.ADD_F,
        Op.SUB,
        Op.SUB_I,
        Op.SUB_F,
        Op.MUL,
        Op.MUL_I,
        Op.MUL_F,
        Op.DIV,
        Op.MOD,
        Op.EQ,
        Op.LT,
        Op.LE,
        Op.NOT,
        Op.TOBOOL,
        Op.GUARD,
        Op.JMP,
        Op.JMPIF,
        Op.JMPIFNOT,
        Op.JMPIFNIL,
        Op.FORPREP,
        Op.FORLOOP,
        Op.JFORLOOP,
        Op.CALL,
        Op.RETURN,
        Op.HALT,
    }
)


@dataclass(frozen=True, slots=True)
class CompiledAstFunction:
    proto: Proto
    runner: FunctionType
    frame_pool: list[object] = field(default_factory=list, compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class CompiledCallEntry:
    """Final execution target for one compiled Proto and argument shape."""

    proto: Proto
    runner: FunctionType
    virtual: bool
    arg_count: int | None
    trusted_args: bool


def _materialized_call_runner(compiled: CompiledAstFunction, *, trusted_args: bool):
    """Build the stable real-Frame entry used when virtualization is unsafe."""

    if compiled.proto.param_count == 1:
        return _one_argument_materialized_call_runner(
            compiled, trusted_args=trusted_args
        )

    def run(vm, frames, closure, args, dest, want, budget, meter):
        if len(frames) >= vm.max_frames:
            raise LuaRuntimeError("stack overflow")
        child = vm._acquire_compiled_frame(
            compiled,
            closure,
            args,
            dest,
            want,
            validate_args=not trusted_args,
        )
        frames.append(child)
        status, values = compiled.runner(vm, frames, child, budget, meter)
        if status == _FUNC_RETURN:
            if not frames or frames[-1] is not child:
                raise RuntimeError("compiled function stack mismatch")
            frames.pop()
            vm.jit.function_executions += 1
            vm._release_compiled_frame(compiled, child)
            return _FUNC_RETURN, values
        vm.jit.function_suspends += 1
        return _FUNC_SUSPEND, None

    return run


def _one_argument_materialized_call_runner(
    compiled: CompiledAstFunction, *, trusted_args: bool
):
    """Fuse frame recycling into the common one-argument call entry."""

    expected = compiled.proto.param_types[0].name

    def run(vm, frames, closure, args, dest, want, budget, meter):
        if len(frames) >= vm.max_frames:
            raise LuaRuntimeError("stack overflow")
        pool = compiled.frame_pool
        if pool:
            child = pool.pop()
            value = args[0] if args else None
            if not trusted_args and not type_matches(expected, value):
                raise LuaRuntimeError(
                    f"argument 1: expected {expected}, "
                    f"got {static_value_type(value).name}"
                )
            child.regs[0] = value
            if compiled.proto.env_reg >= 0:
                child.regs[compiled.proto.env_reg] = closure.env
            child.closure = closure
            child.pc = 0
            child.return_reg = dest
            child.return_want = want
            if vm.debug_hooks_enabled:
                child.hook_call_values = tuple(args[:1])
        else:
            child = vm._acquire_compiled_frame(
                compiled,
                closure,
                args,
                dest,
                want,
                validate_args=not trusted_args,
            )
        frames.append(child)
        status, values = compiled.runner(vm, frames, child, budget, meter)
        if status == _FUNC_RETURN:
            if not frames or frames[-1] is not child:
                raise RuntimeError("compiled function stack mismatch")
            frames.pop()
            vm.jit.function_executions += 1
            if len(pool) < min(128, vm.max_frames):
                pool.append(child)
            return _FUNC_RETURN, values
        vm.jit.function_suspends += 1
        return _FUNC_SUSPEND, None

    return run


@dataclass(frozen=True, slots=True)
class _FunctionBlock:
    start: int
    end: int
    instructions: tuple[tuple[int, Ins], ...]


class TypedFunctionJITMixin:
    """Whole-function AST compilation for certified fully typed closures.

    The VM keeps real ``Frame`` objects on its stack while generated Python runs.
    A guard miss therefore does not restart a function: promoted locals are
    spilled, ``frame.pc`` is set to the exact instruction, and execution simply
    resumes in Tier 0. Nested compiled calls use the same rule, which makes
    direct recursion safe and keeps traceback/unwind state available.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._function_hot: dict[int, tuple[Proto, int]] = {}
        self._function_cache: dict[int, tuple[Proto, CompiledAstFunction | None]] = {}
        self._call_entry_cache: dict[
            tuple[int, int | None, bool], tuple[Proto, CompiledCallEntry]
        ] = {}
        self._compiling_closures: dict[int, Closure] = {}
        self.function_compiles = 0
        self.function_executions = 0
        self.function_suspends = 0

    @staticmethod
    def _function_target(ins: Ins, fallthrough: int) -> tuple[int, ...]:
        if ins.op is Op.JMP:
            return (ins.a,)
        if ins.op in (Op.JMPIF, Op.JMPIFNOT, Op.JMPIFNIL):
            return (ins.a, fallthrough)
        if ins.op is Op.FORPREP:
            return (ins.d, fallthrough)
        if ins.op in (Op.FORLOOP, Op.JFORLOOP):
            return (ins.d, fallthrough)
        return (fallthrough,)

    @staticmethod
    def _function_writes_register(ins: Ins, reg: int) -> bool:
        if ins.op is Op.CALL:
            return ins.e > 0 and ins.a <= reg < ins.a + ins.e
        if ins.op in (Op.SETUPVAL, Op.SETTABLE):
            return False
        return ins.op not in _FUNCTION_CONTROL and ins.op not in (
            Op.RETURN, Op.HALT
        ) and ins.a == reg

    def _function_blocks(self, proto: Proto) -> tuple[_FunctionBlock, ...] | None:
        if not proto.code or any(ins.op not in _FUNCTION_OPS for ins in proto.code):
            return None
        leaders = {0, len(proto.code)}
        for pc, ins in enumerate(proto.code):
            if ins.op is Op.CALL:
                leaders.add(pc + 1)
                continue
            if ins.op in _FUNCTION_CONTROL:
                for target in self._function_target(ins, pc + 1):
                    if target < 0 or target > len(proto.code):
                        return None
                    leaders.add(target)
                leaders.add(pc + 1)
            elif ins.op in (Op.RETURN, Op.HALT):
                leaders.add(pc + 1)

        ordered = sorted(leaders)
        blocks: list[_FunctionBlock] = []
        for index, start in enumerate(ordered[:-1]):
            end = ordered[index + 1]
            instructions = tuple((pc, proto.code[pc]) for pc in range(start, end))
            if not instructions:
                continue
            for _pc, ins in instructions[:-1]:
                if ins.op in _FUNCTION_CONTROL or ins.op in (Op.CALL, Op.RETURN, Op.HALT):
                    return None
            blocks.append(_FunctionBlock(start, end, instructions))

        starts = {block.start for block in blocks}
        starts.add(len(proto.code))
        for block in blocks:
            terminal = block.instructions[-1][1]
            if terminal.op in _FUNCTION_CONTROL:
                for target in self._function_target(terminal, block.end):
                    if target not in starts:
                        return None
        return tuple(blocks)

    def get_compiled_function(self, closure: Closure) -> CompiledAstFunction | None:
        proto = closure.proto
        if not proto.jit_fully_typed or proto.is_vararg:
            return None
        ident = id(proto)
        cached = self._function_cache.get(ident)
        if cached is not None and cached[0] is proto:
            return cached[1]
        self._compiling_closures[ident] = closure
        try:
            compiled = self._compile_ast_function(proto)
        finally:
            self._compiling_closures.pop(ident, None)
        self._function_cache[ident] = (proto, compiled)
        if compiled is not None:
            self.function_compiles += 1
        return compiled

    def maybe_function(self, closure: Closure) -> CompiledAstFunction | None:
        proto = closure.proto
        if not proto.jit_fully_typed or proto.is_vararg:
            return None
        ident = id(proto)
        cached = self._function_cache.get(ident)
        if cached is not None and cached[0] is proto:
            return cached[1]
        hot_entry = self._function_hot.get(ident)
        hot = hot_entry[1] if hot_entry is not None and hot_entry[0] is proto else 0
        hot += 1
        self._function_hot[ident] = (proto, hot)
        if hot < self.threshold:
            return None
        return self.get_compiled_function(closure)

    def run_compiled_child(
        self,
        vm,
        frames,
        parent,
        closure: Closure,
        args: tuple,
        dest: int,
        want: int,
        budget: int,
        meter: list[int],
        compiled: CompiledAstFunction,
    ):
        entry = self.get_call_entry(compiled, arg_count=len(args))
        return entry.runner(vm, frames, closure, args, dest, want, budget, meter)

    def get_call_entry(
        self,
        compiled: CompiledAstFunction,
        *,
        arg_count: int | None = None,
        trusted_args: bool = False,
    ) -> CompiledCallEntry:
        """Resolve virtualization and adapters once for a stable call shape."""

        proto = compiled.proto
        key = (id(proto), arg_count, trusted_args)
        cached = self._call_entry_cache.get(key)
        if cached is not None and cached[0] is proto:
            return cached[1]
        from .virtual_frame import compile_virtual_frame

        runner = compile_virtual_frame(
            proto,
            arg_count=arg_count,
            trusted_args=trusted_args,
        )
        entry = CompiledCallEntry(
            proto,
            runner if runner is not None else _materialized_call_runner(
                compiled, trusted_args=trusted_args
            ),
            runner is not None,
            arg_count,
            trusted_args,
        )
        self._call_entry_cache[key] = (proto, entry)
        return entry

    def _compile_ast_function(self, proto: Proto) -> CompiledAstFunction | None:
        blocks = self._function_blocks(proto)
        if blocks is None:
            return None
        # Functions that create child closures need exact open-cell identity.
        # Those stay in Tier 0 until the closure/cell lowering tranche.
        if proto.children:
            return None

        registers = tuple(range(max(1, proto.register_count)))
        ranges = analyze_integer_ranges(proto)
        block_index = {block.start: index for index, block in enumerate(blocks)}
        typed_plan = TypedIRCompiler(proto).compile(
            tuple(block.instructions for block in blocks)
        )
        compiling_closure = self._compiling_closures.get(id(proto))

        # Continuation liveness is used only at successful compiled-child
        # boundaries.  All side exits retain the full diagnostic spill.
        live_in = [set() for _ in range(len(proto.code) + 1)]
        changed = True
        while changed:
            changed = False
            for pc in range(len(proto.code) - 1, -1, -1):
                ins = proto.code[pc]
                if ins.op in (Op.RETURN, Op.HALT):
                    successors = ()
                elif ins.op is Op.JMP:
                    successors = (ins.a,)
                elif ins.op in (Op.JMPIF, Op.JMPIFNOT, Op.JMPIFNIL):
                    successors = (ins.a, pc + 1)
                elif ins.op in (Op.FORPREP, Op.FORLOOP, Op.JFORLOOP):
                    successors = (ins.d, pc + 1)
                else:
                    successors = (pc + 1,)
                out: set[int] = set()
                for successor in successors:
                    if 0 <= successor < len(live_in):
                        out.update(live_in[successor])
                value = set(_reads(ins)) | (out - set(_writes(ins)))
                if value != live_in[pc]:
                    live_in[pc] = value
                    changed = True

        diagnostic_live: list[set[int]] = [set() for _ in live_in]
        for _name, reg, start, end in proto.debug_locals:
            for pc in range(max(0, start), min(len(diagnostic_live), end + 1)):
                diagnostic_live[pc].add(reg)

        def state_for(pc: int) -> int:
            if pc == len(proto.code):
                return _FUNC_RETURN
            return block_index[pc]

        def spill(indent: str, subset=None) -> list[str]:
            selected = registers if subset is None else sorted(subset)
            return [f"{indent}regs[{reg}] = _r{reg}" for reg in selected]

        def suspend(pc: int, indent: str) -> list[str]:
            return [
                *spill(indent),
                f"{indent}frame.pc = {pc}",
                f"{indent}return _FUNC_SUSPEND",
            ]

        lines = [
            "def _jit_ast_function(vm, frames, frame, budget, meter):",
            "    regs = frame.regs",
            "    consts = frame.proto.constants",
            "    upvalues = frame.closure.upvalues",
            "    used = 0",
            "    _result = ()",
        ]
        for reg in registers:
            lines.append(f"    _r{reg} = regs[{reg}]")
        lines.append("    _state = 0")

        block_names = tuple(f"_jump_{index}" for index in range(len(blocks)))

        def i64(
            dest: str, expression: str, tag: str, indent: str, *, overflow_free=False
        ) -> list[str]:
            if overflow_free:
                return [f"{indent}{dest} = {expression}"]
            tmp = f"_i64_{tag}"
            return [
                f"{indent}{tmp} = ({expression}) & _MASK64",
                f"{indent}{dest} = {tmp} - _TWO64 if {tmp} & _SIGN64 else {tmp}",
            ]

        def deopt(out: list[str], condition: str, pc: int, indent: str) -> None:
            out.append(f"{indent}if {condition}:")
            out.extend(f"{indent}    {line.strip()}" for line in suspend(pc, ""))

        def emit_forprep(out: list[str], ins: Ins, true_state: int, false_state: int, indent: str):
            a, b, c = f"_r{ins.a}", f"_r{ins.b}", f"_r{ins.c}"
            out.extend(
                [
                    f"{indent}if not (type({a}) in _NUM_TYPES and type({b}) in _NUM_TYPES and type({c}) in _NUM_TYPES):",
                    f"{indent}    raise _LuaRuntimeError(\"'for' limit must be a number\")",
                    f"{indent}if {c} == 0:",
                    f"{indent}    raise _LuaRuntimeError(\"'for' step is zero\")",
                    f"{indent}if type({a}) is float or type({b}) is float or type({c}) is float:",
                    f"{indent}    {a} = float({a})",
                    f"{indent}    {b} = float({b})",
                    f"{indent}    {c} = float({c})",
                    f"{indent}return {true_state} if ({a} <= {b} if {c} > 0 else {a} >= {b}) else {false_state}",
                ]
            )

        def emit_forloop(out: list[str], ins: Ins, loop_state: int, exit_state: int, pc: int, indent: str):
            a, b, c = f"_r{ins.a}", f"_r{ins.b}", f"_r{ins.c}"
            tag = f"f_{pc}_{ins.a}"
            if all(ranges.range_at(pc, reg) is not None for reg in (ins.a, ins.b, ins.c)):
                out.extend(
                    [
                        f"{indent}_next_{tag} = {a} + {c}",
                        f"{indent}if _next_{tag} < _INT_MIN or _next_{tag} > _INT_MAX or ({c} > 0 and _next_{tag} > {b}) or ({c} < 0 and _next_{tag} < {b}):",
                        f"{indent}    return {exit_state}",
                        f"{indent}{a} = _next_{tag}",
                        f"{indent}return {loop_state}",
                    ]
                )
                return
            out.extend(
                [
                    f"{indent}if type({a}) is int and type({b}) is int and type({c}) is int:",
                    f"{indent}    _next_{tag} = {a} + {c}",
                    f"{indent}    if _next_{tag} < _INT_MIN or _next_{tag} > _INT_MAX or ({c} > 0 and _next_{tag} > {b}) or ({c} < 0 and _next_{tag} < {b}):",
                    f"{indent}        return {exit_state}",
                    f"{indent}    {a} = _next_{tag}",
                    f"{indent}    return {loop_state}",
                    f"{indent}_next_{tag} = float({a}) + float({c})",
                    f"{indent}if ({c} > 0 and _next_{tag} > {b}) or ({c} < 0 and _next_{tag} < {b}):",
                    f"{indent}    return {exit_state}",
                    f"{indent}{a} = _next_{tag}",
                    f"{indent}return {loop_state}",
                ]
            )

        constant_tokens: dict[str, object] = {}
        call_target_caches: dict[str, list[object | None]] = {}

        def leaf_sequence(closure: Closure):
            child = closure.proto
            if (
                not child.jit_fully_typed
                or child.is_vararg
                or child.upvalues
                or child.children
            ):
                return None
            allowed = {
                Op.LOADK, Op.MOVE, Op.LOCAL,
                Op.ADD, Op.ADD_I, Op.ADD_F,
                Op.SUB, Op.SUB_I, Op.SUB_F,
                Op.MUL, Op.MUL_I, Op.MUL_F,
                Op.DIV, Op.MOD, Op.NOT, Op.TOBOOL,
                Op.RETURN,
            }
            sequence = []
            for child_pc, child_ins in enumerate(child.code):
                if child_ins.op not in allowed:
                    return None
                sequence.append((child_pc, child_ins))
                if child_ins.op is Op.RETURN:
                    break
            if not sequence or sequence[-1][1].op is not Op.RETURN:
                return None
            return tuple(sequence) if len(sequence) <= 24 else None

        def emit_inline_leaf(
            out: list[str], pc: int, ins: Ins, closure: Closure, indent: str
        ) -> bool:
            sequence = leaf_sequence(closure)
            if sequence is None:
                return False
            child = closure.proto
            expected_proto = f"_inline_proto_{pc}"
            child_consts = f"_inline_consts_{pc}"
            constant_tokens[expected_proto] = child
            constant_tokens[child_consts] = child.constants
            fn = f"_r{ins.b}"
            out.append(
                f"{indent}if not isinstance({fn}, _Closure) or {fn}.proto is not {expected_proto}:"
            )
            out.extend(f"{indent}    {line.strip()}" for line in suspend(pc, ""))
            total_cost = 1 + len(sequence)
            out.append(f"{indent}if budget - meter[0] - used < {total_cost}:")
            out.extend(f"{indent}    {line.strip()}" for line in suspend(pc, ""))
            out.append(f"{indent}used += 1")
            out.append(f"{indent}if len(frames) >= vm.max_frames:")
            out.append(f"{indent}    raise _LuaRuntimeError('stack overflow')")
            prefix = f"_inl_{pc}_r"
            for index in range(child.param_count):
                source = f"_r{ins.c + index}" if index < ins.d else "None"
                expected = child.param_types[index].name
                out.append(f"{indent}if not _type_matches({expected!r}, {source}):")
                out.append(
                    f"{indent}    raise _LuaRuntimeError(f'argument {index + 1}: expected {expected}, got {{_static_value_type({source}).name}}')"
                )
                out.append(f"{indent}{prefix}{index} = {source}")
            child_ranges = analyze_integer_ranges(child)
            child_constants: dict[int, object] = {}
            returned = None
            for child_pc, child_ins in sequence:
                op = child_ins.op
                out.append(f"{indent}used += 1")
                a = f"{prefix}{child_ins.a}"
                b = f"{prefix}{child_ins.b}"
                c = f"{prefix}{child_ins.c}"
                if op is Op.LOADK:
                    out.append(f"{indent}{a} = {child_consts}[{child_ins.b}]")
                    child_constants[child_ins.a] = child.constants[child_ins.b]
                elif op in (Op.MOVE, Op.LOCAL):
                    out.append(f"{indent}{a} = {b}")
                    if child_ins.b in child_constants:
                        child_constants[child_ins.a] = child_constants[child_ins.b]
                    else:
                        child_constants.pop(child_ins.a, None)
                elif op in (Op.ADD_I, Op.SUB_I, Op.MUL_I):
                    symbol = {Op.ADD_I: "+", Op.SUB_I: "-", Op.MUL_I: "*"}[op]
                    out.extend(i64(
                        a, f"{b} {symbol} {c}", f"inl_{pc}_{child_pc}", indent,
                        overflow_free=child_ranges.overflow_free(child_pc),
                    ))
                elif op in (Op.ADD_F, Op.SUB_F, Op.MUL_F):
                    symbol = {Op.ADD_F: "+", Op.SUB_F: "-", Op.MUL_F: "*"}[op]
                    out.append(f"{indent}{a} = float({b} {symbol} {c})")
                elif op in (Op.ADD, Op.SUB, Op.MUL):
                    symbol = {Op.ADD: "+", Op.SUB: "-", Op.MUL: "*"}[op]
                    tmp = f"_inl_{pc}_v{child_pc}"
                    out.append(f"{indent}{tmp} = {b} {symbol} {c}")
                    out.append(f"{indent}if type({b}) is int and type({c}) is int:")
                    out.extend(i64(a, tmp, f"inlg_{pc}_{child_pc}", indent + "    "))
                    out.append(f"{indent}else:")
                    out.append(f"{indent}    {a} = {tmp}")
                elif op is Op.DIV:
                    divisor = child_constants.get(child_ins.c, _ABSENT)
                    if type(divisor) in (int, float) and divisor != 0:
                        out.append(f"{indent}{a} = float({b}) / {float(divisor)!r}")
                    else:
                        out.append(f"{indent}{a} = _float_divide({b}, {c})")
                elif op is Op.MOD:
                    out.append(f"{indent}if type({b}) is int and type({c}) is int:")
                    out.append(f"{indent}    if {c} == 0:")
                    out.append(f"{indent}        raise _LuaRuntimeError(\"attempt to perform 'n%0'\")")
                    out.append(f"{indent}    {a} = {b} % {c}")
                    out.append(f"{indent}else:")
                    out.append(f"{indent}    {a} = _float_modulo({b}, {c})")
                elif op is Op.NOT:
                    out.append(f"{indent}{a} = not _truthy({b})")
                elif op is Op.TOBOOL:
                    out.append(f"{indent}{a} = _truthy({b})")
                elif op is Op.EQ:
                    out.append(f"{indent}{a} = _lua_equal({b}, {c})")
                elif op in (Op.LT, Op.LE):
                    symbol = "<" if op is Op.LT else "<="
                    out.append(f"{indent}{a} = {b} {symbol} {c}")
                elif op is Op.GUARD:
                    expected = child.constants[child_ins.b]
                    out.append(f"{indent}if not _type_matches({expected!r}, {a}):")
                    out.append(
                        f"{indent}    raise _LuaRuntimeError(f'expected {expected!s}, got {{_static_value_type({a}).name}}')"
                    )
                elif op is Op.RETURN:
                    returned = [f"{prefix}{child_ins.a + i}" for i in range(child_ins.b)]
                    break
                if op not in (Op.LOADK, Op.MOVE, Op.LOCAL):
                    child_constants.pop(child_ins.a, None)
            if returned is None:
                return False
            if ins.e > 0:
                for index in range(ins.e):
                    value = returned[index] if index < len(returned) else "None"
                    out.append(f"{indent}_r{ins.a + index} = {value}")
            return True

        for block_no, block in enumerate(blocks):
            lines.append(f"    def {block_names[block_no]}():")
            indent = "        "
            known_constants: dict[int, object] = {}
            known_closures: dict[int, Closure] = {}
            cost = len(block.instructions)
            lines.append(f"{indent}if budget - meter[0] - used < {cost}:")
            lines.extend(f"{indent}    {line.strip()}" for line in suspend(block.start, ""))

            terminal_pc, terminal_ins = block.instructions[-1]
            terminal_control = terminal_ins.op in _FUNCTION_CONTROL or terminal_ins.op in (
                Op.RETURN,
                Op.HALT,
            )
            ordinary = block.instructions[:-1] if terminal_control else block.instructions

            batchable = {
                Op.LOADK, Op.MOVE, Op.LOCAL, Op.GETUPVAL,
                Op.ADD_I, Op.SUB_I, Op.MUL_I,
                Op.ADD_F, Op.SUB_F, Op.MUL_F,
                Op.NOT, Op.TOBOOL,
            }
            pending_cost = 0

            for pc, ins in ordinary:
                op = ins.op
                if op not in batchable and pending_cost:
                    lines.append(f"{indent}used += {pending_cost}")
                    pending_cost = 0
                a, b, c = f"_r{ins.a}", f"_r{ins.b}", f"_r{ins.c}"
                if op is Op.LOADK:
                    lines.append(f"{indent}{a} = consts[{ins.b}]")
                    known_constants[ins.a] = proto.constants[ins.b]
                elif op in (Op.MOVE, Op.LOCAL):
                    lines.append(f"{indent}{a} = {b}")
                    if ins.b in known_constants:
                        known_constants[ins.a] = known_constants[ins.b]
                    else:
                        known_constants.pop(ins.a, None)
                    if ins.b in known_closures:
                        known_closures[ins.a] = known_closures[ins.b]
                    else:
                        known_closures.pop(ins.a, None)
                elif op is Op.GETUPVAL:
                    lines.append(f"{indent}{a} = upvalues[{ins.b}].value")
                    if (
                        compiling_closure is not None
                        and ins.b < len(compiling_closure.upvalues)
                        and isinstance(compiling_closure.upvalues[ins.b].value, Closure)
                    ):
                        known_closures[ins.a] = compiling_closure.upvalues[ins.b].value
                    else:
                        known_closures.pop(ins.a, None)
                elif op is Op.SETUPVAL:
                    lines.extend([
                        f"{indent}used += 1",
                        f"{indent}vm.gc.write_barrier(upvalues[{ins.a}], {b})",
                        f"{indent}upvalues[{ins.a}].value = {b}",
                    ])
                elif op is Op.NEWTABLE:
                    lines.append(f"{indent}used += 1")
                    lines.extend(spill(indent))
                    lines.append(f"{indent}{a} = vm._new_table()")
                elif op is Op.GETTABLE:
                    deopt(lines, f"not isinstance({b}, _LuaTable) or {b}.metatable is not None", pc, indent)
                    key = known_constants.get(ins.c, _ABSENT)
                    if type(key) is float and key.is_integer():
                        key = int(key)
                    if key is _ABSENT:
                        lines.extend([f"{indent}used += 1", f"{indent}{a} = {b}.rawget({c})"])
                    else:
                        token_name = f"_key_token_{pc}"
                        constant_tokens[token_name] = _hash_key(key)
                        lines.append(f"{indent}used += 1")
                        if type(key) is int and key >= 1:
                            lines.extend([
                                f"{indent}if {key} <= len({b}.array):",
                                f"{indent}    {a} = {b}.array[{key - 1}]",
                                f"{indent}else:",
                                f"{indent}    _item_{pc} = {b}.hash.get({token_name}, _ABSENT)",
                                f"{indent}    {a} = None if _item_{pc} is _ABSENT else _item_{pc}[1]",
                            ])
                        else:
                            lines.extend([
                                f"{indent}_item_{pc} = {b}.hash.get({token_name}, _ABSENT)",
                                f"{indent}{a} = None if _item_{pc} is _ABSENT else _item_{pc}[1]",
                            ])
                    known_constants.pop(ins.a, None)
                elif op is Op.SETTABLE:
                    deopt(lines, f"not isinstance({a}, _LuaTable) or {a}.metatable is not None", pc, indent)
                    deopt(lines, f"{b} is None or (type({b}) is float and _isnan({b}))", pc, indent)
                    lines.extend([f"{indent}used += 1", f"{indent}{a}.rawset({b}, {c})"])
                elif op in (Op.ADD_I, Op.SUB_I, Op.MUL_I):
                    symbol = {Op.ADD_I: "+", Op.SUB_I: "-", Op.MUL_I: "*"}[op]
                    lines.extend(
                        i64(
                            a, f"{b} {symbol} {c}", str(pc), indent,
                            overflow_free=ranges.overflow_free(pc),
                        )
                    )
                elif op in (Op.ADD_F, Op.SUB_F, Op.MUL_F):
                    symbol = {Op.ADD_F: "+", Op.SUB_F: "-", Op.MUL_F: "*"}[op]
                    lines.append(f"{indent}{a} = float({b} {symbol} {c})")
                elif op in (Op.ADD, Op.SUB, Op.MUL):
                    symbol = {Op.ADD: "+", Op.SUB: "-", Op.MUL: "*"}[op]
                    deopt(lines, f"type({b}) not in _NUM_TYPES or type({c}) not in _NUM_TYPES", pc, indent)
                    tmp = f"_arith_{pc}"
                    lines.extend([f"{indent}used += 1", f"{indent}{tmp} = {b} {symbol} {c}"])
                    lines.append(f"{indent}if type({b}) is int and type({c}) is int:")
                    lines.extend(i64(a, tmp, f"g_{pc}", indent + "    "))
                    lines.append(f"{indent}else:")
                    lines.append(f"{indent}    {a} = {tmp}")
                elif op is Op.DIV:
                    deopt(lines, f"type({b}) not in _NUM_TYPES or type({c}) not in _NUM_TYPES", pc, indent)
                    lines.extend([f"{indent}used += 1", f"{indent}{a} = _float_divide({b}, {c})"])
                elif op is Op.MOD:
                    deopt(lines, f"type({b}) not in _NUM_TYPES or type({c}) not in _NUM_TYPES", pc, indent)
                    lines.append(f"{indent}used += 1")
                    lines.append(f"{indent}if type({b}) is int and type({c}) is int:")
                    lines.append(f"{indent}    if {c} == 0:")
                    lines.append(f"{indent}        raise _LuaRuntimeError(\"attempt to perform 'n%0'\")")
                    lines.extend(i64(a, f"{b} % {c}", f"m_{pc}", indent + "    "))
                    lines.append(f"{indent}else:")
                    lines.append(f"{indent}    {a} = _float_modulo({b}, {c})")
                elif op is Op.NOT:
                    lines.append(f"{indent}{a} = not _truthy({b})")
                elif op is Op.TOBOOL:
                    lines.append(f"{indent}{a} = _truthy({b})")
                elif op is Op.EQ:
                    deopt(
                        lines,
                        f"isinstance({b}, _LuaTable) and isinstance({c}, _LuaTable) and not _lua_equal({b}, {c}) and ({b}.metatable is not None or {c}.metatable is not None)",
                        pc,
                        indent,
                    )
                    lines.extend([f"{indent}used += 1", f"{indent}{a} = _lua_equal({b}, {c})"])
                elif op in (Op.LT, Op.LE):
                    symbol = "<" if op is Op.LT else "<="
                    deopt(
                        lines,
                        f"not ((type({b}) in _NUM_TYPES and type({c}) in _NUM_TYPES) or (isinstance({b}, bytes) and isinstance({c}, bytes)))",
                        pc,
                        indent,
                    )
                    lines.extend([f"{indent}used += 1", f"{indent}{a} = {b} {symbol} {c}"])
                elif op is Op.GUARD:
                    lines.append(f"{indent}used += 1")
                    lines.append(f"{indent}if not _type_matches({proto.constants[ins.b]!r}, {a}):")
                    lines.append(
                        f"{indent}    raise _LuaRuntimeError(f\"expected {{consts[{ins.b}]!s}}, got {{_static_value_type({a}).name}}\")"
                    )
                elif op is Op.CALL:
                    inline_target = known_closures.get(ins.b)
                    if inline_target is not None and emit_inline_leaf(
                        lines, pc, ins, inline_target, indent
                    ):
                        known_constants.pop(ins.a, None)
                        known_closures.pop(ins.a, None)
                        continue
                    # Do not execute a dynamic/host call inside a partially
                    # compiled function. Suspend before the CALL so Tier 0 can
                    # perform every Lua metamethod/callability check exactly.
                    lines.append(f"{indent}_fn_{pc} = {b}")
                    cache_name = f"_call_target_{pc}"
                    call_target_caches.setdefault(cache_name, [None, None])
                    call_site = typed_plan.instruction(pc)
                    actual_types = tuple(
                        call_site.value_for(ins.c + i).type_name
                        for i in range(ins.d)
                    )
                    accepted_names = {
                        "integer": ("Any", "integer", "integer_lua", "number"),
                        "integer_lua": ("Any", "integer", "integer_lua", "number"),
                        "float": ("Any", "float", "number"),
                        "number": ("Any", "number"),
                    }
                    trust_terms = []
                    for index, actual in enumerate(actual_types):
                        accepted = accepted_names.get(actual, ("Any", actual))
                        trust_terms.append(
                            f"_compiled_{pc}.proto.param_types[{index}].name in {accepted!r}"
                        )
                    trusted_expr = " and ".join(
                        [f"_compiled_{pc}.proto.param_count == {ins.d}", *trust_terms]
                    )
                    lines.extend([
                        f"{indent}_target_cache_{pc} = {cache_name}",
                        f"{indent}if _fn_{pc} is _target_cache_{pc}[0]:",
                        f"{indent}    _entry_{pc} = _target_cache_{pc}[1]",
                        f"{indent}else:",
                        f"{indent}    _compiled_{pc} = vm.jit.get_compiled_function(_fn_{pc}) if isinstance(_fn_{pc}, _Closure) else None",
                        f"{indent}    _trusted_{pc} = ({trusted_expr}) if _compiled_{pc} is not None else False",
                        f"{indent}    _entry_{pc} = vm.jit.get_call_entry(_compiled_{pc}, arg_count={ins.d}, trusted_args=_trusted_{pc}) if _compiled_{pc} is not None else None",
                        f"{indent}    _target_cache_{pc}[0] = _fn_{pc}",
                        f"{indent}    _target_cache_{pc}[1] = _entry_{pc}",
                    ])
                    lines.append(f"{indent}if _entry_{pc} is None:")
                    lines.extend(f"{indent}    {line.strip()}" for line in suspend(pc, ""))
                    lines.append(f"{indent}used += 1")
                    # Flush the parent's exact instruction count before the
                    # child runs. The child's own finally block uses the same
                    # meter, so exceptions and recursive suspension preserve
                    # fuel exactly without a per-op mutable-list increment.
                    lines.append(f"{indent}meter[0] += used")
                    lines.append(f"{indent}used = 0")
                    outputs = set(range(ins.a, ins.a + max(0, ins.e))) if ins.e > 0 else set()
                    call_spill = (live_in[pc + 1] | diagnostic_live[pc + 1]) - outputs
                    lines.extend(spill(indent, call_spill))
                    for output_reg in sorted(
                        outputs & (live_in[pc + 1] | diagnostic_live[pc + 1])
                    ):
                        lines.append(f"{indent}regs[{output_reg}] = None")
                    lines.append(f"{indent}frame.pc = {pc + 1}")
                    args = ", ".join(f"_r{ins.c + i}" for i in range(ins.d))
                    if ins.d == 1:
                        args += ","
                    lines.append(
                        f"{indent}_status_{pc}, _values_{pc} = _entry_{pc}.runner(vm, frames, _fn_{pc}, ({args}), {ins.a}, {ins.e}, budget, meter)"
                    )
                    lines.append(f"{indent}if _status_{pc} == _FUNC_SUSPEND:")
                    lines.append(f"{indent}    return _FUNC_SUSPEND")
                    if ins.e > 0:
                        for value_index in range(ins.e):
                            if ins.e == 1:
                                lines.append(
                                    f"{indent}_r{ins.a} = _values_{pc}[0] if _values_{pc} else None"
                                )
                            else:
                                lines.append(
                                    f"{indent}_r{ins.a + value_index} = _values_{pc}[{value_index}] if {value_index} < len(_values_{pc}) else None"
                                )
                else:
                    return None

                if op in batchable:
                    pending_cost += 1

                if op not in (Op.LOADK, Op.MOVE, Op.LOCAL, Op.GETUPVAL, Op.CALL):
                    known_closures.pop(ins.a, None)
                elif op is Op.LOADK:
                    known_closures.pop(ins.a, None)

                if op is Op.CALL and ins.e > 0:
                    for result_reg in range(ins.a, ins.a + ins.e):
                        known_constants.pop(result_reg, None)
                elif op not in (Op.LOADK, Op.MOVE, Op.LOCAL, Op.GETTABLE) and self._function_writes_register(ins, ins.a):
                    known_constants.pop(ins.a, None)

            if pending_cost:
                lines.append(f"{indent}used += {pending_cost}")

            if terminal_control:
                ins = terminal_ins
                pc = terminal_pc
                lines.append(f"{indent}used += 1")
                if ins.op is Op.JMP:
                    lines.append(f"{indent}return {state_for(ins.a)}")
                elif ins.op in (Op.JMPIF, Op.JMPIFNOT):
                    condition = f"_truthy(_r{ins.b})"
                    if ins.op is Op.JMPIFNOT:
                        condition = f"not ({condition})"
                    lines.append(
                        f"{indent}return {state_for(ins.a)} if {condition} else {state_for(block.end)}"
                    )
                elif ins.op is Op.JMPIFNIL:
                    lines.append(
                        f"{indent}return {state_for(ins.a)} if _r{ins.b} is None else {state_for(block.end)}"
                    )
                elif ins.op is Op.FORPREP:
                    emit_forprep(lines, ins, state_for(block.end), state_for(ins.d), indent)
                elif ins.op in (Op.FORLOOP, Op.JFORLOOP):
                    emit_forloop(lines, ins, state_for(ins.d), state_for(block.end), pc, indent)
                elif ins.op is Op.RETURN:
                    values = ", ".join(f"_r{ins.a + i}" for i in range(ins.b))
                    if ins.b == 1:
                        values += ","
                    lines.append(f"{indent}_result = ({values})")
                    lines.append(f"{indent}return _FUNC_RETURN")
                elif ins.op is Op.HALT:
                    lines.append(f"{indent}_result = ()")
                    lines.append(f"{indent}return _FUNC_RETURN")
                else:
                    return None
            else:
                lines.append(f"{indent}return {state_for(block.end)}")

        lines.append("    _jump_list = (" + ", ".join(block_names) + ",)")
        lines.extend(
            [
                "    try:",
                "        _state = 0",
                "        while _state >= 0:",
                "            _state = _jump_list[_state]()",
                "        if _state == _FUNC_RETURN:",
                "            return _FUNC_RETURN, _result",
                "        return _FUNC_SUSPEND, None",
                "    finally:",
                "        meter[0] += used",
            ]
        )

        tree = ast.parse("\n".join(lines))
        tree = inline_expression_helper(
            tree,
            "def _truthy(value):\n    return not (value is None or value is False)\n",
        )
        tree = inline_local_jump_list(
            tree,
            JumpListLayout("_jump_list", "_state", block_names),
        )
        ast.fix_missing_locations(tree)
        namespace = {
            "_FUNC_RETURN": _FUNC_RETURN,
            "_FUNC_SUSPEND": _FUNC_SUSPEND,
            "_INT_MIN": -(1 << 63),
            "_INT_MAX": (1 << 63) - 1,
            "_MASK64": (1 << 64) - 1,
            "_SIGN64": 1 << 63,
            "_TWO64": 1 << 64,
            "_NUM_TYPES": (int, float),
            "_Closure": Closure,
            "_LuaRuntimeError": LuaRuntimeError,
            "_LuaTable": LuaTable,
            "_float_divide": _float_divide,
            "_float_modulo": _float_modulo,
            "_isnan": __import__("math").isnan,
            "_lua_equal": lua_equal,
            "_static_value_type": static_value_type,
            "_type_matches": type_matches,
            "_ABSENT": _ABSENT,
            **constant_tokens,
            **call_target_caches,
        }
        exec(compile(tree, "<luapyre-ast-function>", "exec"), namespace)
        return CompiledAstFunction(proto, namespace["_jit_ast_function"])
