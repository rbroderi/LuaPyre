from __future__ import annotations

from dataclasses import dataclass

from .bytecode import Cell, Ins, Op
from .jit import CompiledLoop, IRBlock, IRInstruction, IRLoop, PythonJIT
from .jit_policy import JIT_LOOP_BODY_OPS
from .table import LuaTable
from .values import i64, lua_equal, type_matches


_BRANCH_OPS = frozenset((Op.JMP, Op.JMPIF, Op.JMPIFNOT))
_REGION_ONLY_OPS = _BRANCH_OPS | frozenset((Op.GUARD,))


@dataclass(frozen=True, slots=True)
class _Block:
    start: int
    end: int
    instructions: tuple[IRInstruction, ...]


class RegionPythonJIT(PythonJIT):
    """0.14 JIT extension for acyclic control flow inside numeric loops.

    Straight-line loops continue to use the 0.13 backend unchanged. Loops that
    contain forward branches or type guards are lowered into basic blocks and
    emitted as one generated-Python region. The generated region dispatches at
    block boundaries, not at every Lua opcode.

    Exact fuel behavior is preserved by entering a compiled iteration only when
    the remaining budget can cover the *longest* path. Near a quota boundary we
    simply return to the interpreter, which can consume the shorter path one
    instruction at a time.
    """

    def _compile_loop(self, frame, start_pc: int, backedge_pc: int):
        body = frame.proto.code[start_pc:backedge_pc]
        if not any(ins.op in _REGION_ONLY_OPS for ins in body):
            return super()._compile_loop(frame, start_pc, backedge_pc)
        return self._compile_region_loop(frame, start_pc, backedge_pc)

    def _lower_region(
        self, frame, start_pc: int, backedge_pc: int
    ) -> tuple[IRLoop, tuple[_Block, ...], int] | None:
        proto = frame.proto
        body = proto.code[start_pc:backedge_pc]
        if not body or any(ins.op not in JIT_LOOP_BODY_OPS for ins in body):
            return None

        loop_ins = proto.code[backedge_pc]
        if loop_ins.op not in (Op.FORLOOP, Op.JFORLOOP) or loop_ins.d != start_pc:
            return None

        lowered: list[IRInstruction] = []
        leaders = {start_pc, backedge_pc}
        for offset, ins in enumerate(body):
            pc = start_pc + offset
            specialization = None
            if ins.op in (Op.ADD, Op.SUB, Op.MUL, Op.MOD, Op.EQ, Op.LT, Op.LE):
                specialization = self._profile_binary(frame.regs, ins)
            lowered.append(IRInstruction(pc, ins, specialization))

            if ins.op in _BRANCH_OPS:
                target = ins.a
                # Internal region control flow is deliberately acyclic in this
                # tranche. Nested loops therefore deopt rather than creating a
                # second interpreter inside generated Python.
                if target < start_pc or target > backedge_pc or target <= pc:
                    return None
                leaders.add(target)
                if pc + 1 <= backedge_pc:
                    leaders.add(pc + 1)

        ordered = sorted(leaders)
        by_pc = {item.pc: item for item in lowered}
        blocks: list[_Block] = []
        for index, leader in enumerate(ordered[:-1]):
            end = ordered[index + 1]
            instructions = tuple(by_pc[pc] for pc in range(leader, end))
            if not instructions:
                continue
            # Any branch must terminate its block; leader construction should
            # guarantee this, but fail closed if malformed bytecode violates it.
            if any(item.ins.op in _BRANCH_OPS for item in instructions[:-1]):
                return None
            blocks.append(_Block(leader, end, instructions))

        starts = {block.start for block in blocks}
        starts.add(backedge_pc)
        for block in blocks:
            terminal = block.instructions[-1].ins
            if terminal.op in _BRANCH_OPS and terminal.a not in starts:
                return None

        block_map = {block.start: block for block in blocks}
        visiting: set[int] = set()
        memo: dict[int, int] = {backedge_pc: 0}

        def longest(pc: int) -> int:
            cached = memo.get(pc)
            if cached is not None:
                return cached
            if pc in visiting:
                raise ValueError("cyclic region")
            block = block_map.get(pc)
            if block is None:
                raise ValueError("missing region block")
            visiting.add(pc)
            terminal = block.instructions[-1].ins
            own = len(block.instructions)
            if terminal.op is Op.JMP:
                tail = longest(terminal.a)
            elif terminal.op in (Op.JMPIF, Op.JMPIFNOT):
                tail = max(longest(terminal.a), longest(block.end))
            else:
                tail = longest(block.end)
            visiting.remove(pc)
            memo[pc] = own + tail
            return memo[pc]

        try:
            max_body_cost = longest(start_pc)
        except ValueError:
            return None

        ir = IRLoop(
            start_pc,
            backedge_pc,
            backedge_pc + 1,
            IRBlock(start_pc, backedge_pc, tuple(lowered)),
            loop_ins,
        )
        return ir, tuple(blocks), max_body_cost + 1

    @staticmethod
    def _guard(
        lines: list[str], condition: str, pc: int, offset: int, indent: str
    ) -> None:
        lines.append(f"{indent}if not ({condition}):")
        lines.append(f"{indent}    frame.pc = {pc}")
        lines.append(f"{indent}    return used + {offset}, False")

    def _emit_instruction(
        self,
        lines: list[str],
        item: IRInstruction,
        *,
        offset: int,
        indent: str,
        trusted: bool,
    ) -> bool:
        ins, pc, op = item.ins, item.pc, item.ins.op
        if op is Op.LOADK:
            lines.append(f"{indent}regs[{ins.a}] = consts[{ins.b}]")
        elif op is Op.MOVE:
            lines.append(f"{indent}regs[{ins.a}] = regs[{ins.b}]")
        elif op is Op.LOCAL:
            lines.append(f"{indent}_v = regs[{ins.b}]")
            lines.append(f"{indent}regs[{ins.a}] = _v")
            lines.append(f"{indent}if {ins.a} in cells:")
            lines.append(f"{indent}    cells[{ins.a}] = vm._new_cell(_v)")
        elif op is Op.NEWTABLE:
            lines.append(f"{indent}regs[{ins.a}] = vm._new_table()")
        elif op is Op.GETTABLE:
            self._guard(
                lines,
                f"isinstance(regs[{ins.b}], _LuaTable) and regs[{ins.b}].metatable is None",
                pc,
                offset,
                indent,
            )
            lines.append(f"{indent}regs[{ins.a}] = regs[{ins.b}].rawget(regs[{ins.c}])")
        elif op is Op.SETTABLE:
            self._guard(
                lines,
                f"isinstance(regs[{ins.a}], _LuaTable) and regs[{ins.a}].metatable is None",
                pc,
                offset,
                indent,
            )
            lines.append(f"{indent}_key = regs[{ins.b}]")
            self._guard(
                lines,
                "_key is not None and not (type(_key) is float and _isnan(_key))",
                pc,
                offset,
                indent,
            )
            lines.append(f"{indent}regs[{ins.a}].rawset(_key, regs[{ins.c}])")
        elif op in (Op.ADD_I, Op.SUB_I, Op.MUL_I):
            symbol = {Op.ADD_I: "+", Op.SUB_I: "-", Op.MUL_I: "*"}[op]
            if not trusted:
                self._guard(
                    lines,
                    f"type(regs[{ins.b}]) is int and type(regs[{ins.c}]) is int",
                    pc,
                    offset,
                    indent,
                )
            lines.append(
                f"{indent}regs[{ins.a}] = _i64(regs[{ins.b}] {symbol} regs[{ins.c}])"
            )
        elif op in (Op.ADD_F, Op.SUB_F, Op.MUL_F):
            symbol = {Op.ADD_F: "+", Op.SUB_F: "-", Op.MUL_F: "*"}[op]
            if not trusted:
                self._guard(
                    lines,
                    f"type(regs[{ins.b}]) in _NUM_TYPES and type(regs[{ins.c}]) in _NUM_TYPES",
                    pc,
                    offset,
                    indent,
                )
            lines.append(
                f"{indent}regs[{ins.a}] = float(regs[{ins.b}] {symbol} regs[{ins.c}])"
            )
        elif op in (Op.ADD, Op.SUB, Op.MUL):
            symbol = {Op.ADD: "+", Op.SUB: "-", Op.MUL: "*"}[op]
            specialization = item.specialization
            if specialization == "int":
                self._guard(
                    lines,
                    f"type(regs[{ins.b}]) is int and type(regs[{ins.c}]) is int",
                    pc,
                    offset,
                    indent,
                )
                lines.append(
                    f"{indent}regs[{ins.a}] = _i64(regs[{ins.b}] {symbol} regs[{ins.c}])"
                )
            elif specialization == "number":
                self._guard(
                    lines,
                    f"type(regs[{ins.b}]) in _NUM_TYPES and type(regs[{ins.c}]) in _NUM_TYPES",
                    pc,
                    offset,
                    indent,
                )
                lines.append(f"{indent}_a = regs[{ins.b}]; _b = regs[{ins.c}]")
                lines.append(f"{indent}_v = _a {symbol} _b")
                lines.append(
                    f"{indent}regs[{ins.a}] = _i64(_v) if type(_a) is int and type(_b) is int else _v"
                )
            else:
                return False
        elif op is Op.MOD:
            if item.specialization != "int":
                return False
            self._guard(
                lines,
                f"type(regs[{ins.b}]) is int and type(regs[{ins.c}]) is int and regs[{ins.c}] != 0",
                pc,
                offset,
                indent,
            )
            lines.append(f"{indent}regs[{ins.a}] = _i64(regs[{ins.b}] % regs[{ins.c}])")
        elif op is Op.EQ:
            specialization = item.specialization
            if specialization == "int":
                guard = f"type(regs[{ins.b}]) is int and type(regs[{ins.c}]) is int"
            elif specialization == "number":
                guard = f"type(regs[{ins.b}]) in _NUM_TYPES and type(regs[{ins.c}]) in _NUM_TYPES"
            elif specialization == "bytes":
                guard = f"isinstance(regs[{ins.b}], bytes) and isinstance(regs[{ins.c}], bytes)"
            else:
                return False
            self._guard(lines, guard, pc, offset, indent)
            lines.append(f"{indent}regs[{ins.a}] = _lua_equal(regs[{ins.b}], regs[{ins.c}])")
        elif op in (Op.LT, Op.LE):
            specialization = item.specialization
            if specialization == "int":
                guard = f"type(regs[{ins.b}]) is int and type(regs[{ins.c}]) is int"
            elif specialization == "number":
                guard = f"type(regs[{ins.b}]) in _NUM_TYPES and type(regs[{ins.c}]) in _NUM_TYPES"
            elif specialization == "bytes":
                guard = f"isinstance(regs[{ins.b}], bytes) and isinstance(regs[{ins.c}], bytes)"
            else:
                return False
            self._guard(lines, guard, pc, offset, indent)
            symbol = "<" if op is Op.LT else "<="
            lines.append(f"{indent}regs[{ins.a}] = regs[{ins.b}] {symbol} regs[{ins.c}]")
        elif op is Op.NOT:
            lines.append(f"{indent}_v = regs[{ins.b}]")
            lines.append(f"{indent}regs[{ins.a}] = (_v is None or _v is False)")
        elif op is Op.TOBOOL:
            lines.append(f"{indent}_v = regs[{ins.b}]")
            lines.append(f"{indent}regs[{ins.a}] = not (_v is None or _v is False)")
        elif op is Op.GUARD:
            self._guard(
                lines,
                f"_type_matches(consts[{ins.b}], regs[{ins.a}])",
                pc,
                offset,
                indent,
            )
        else:
            return False
        return True

    def _compile_region_loop(self, frame, start_pc: int, backedge_pc: int):
        lowered = self._lower_region(frame, start_pc, backedge_pc)
        if lowered is None:
            return None
        ir, blocks, max_cost = lowered
        trusted = bool(frame.proto.jit_trust_types)

        lines = [
            "def _jit_region(vm, frame, budget):",
            "    regs = frame.regs",
            "    consts = frame.proto.constants",
            "    cells = frame.cells",
            "    used = 0",
            f"    while budget - used >= {max_cost}:",
            f"        _pc = {start_pc}",
            f"        while _pc != {backedge_pc}:",
        ]

        for block_index, block in enumerate(blocks):
            keyword = "if" if block_index == 0 else "elif"
            lines.append(f"            {keyword} _pc == {block.start}:")
            indent = "                "
            terminal = block.instructions[-1]
            has_branch = terminal.ins.op in _BRANCH_OPS
            ordinary = block.instructions[:-1] if has_branch else block.instructions

            for offset, item in enumerate(ordinary):
                if not self._emit_instruction(
                    lines,
                    item,
                    offset=offset,
                    indent=indent,
                    trusted=trusted,
                ):
                    return None
            if ordinary:
                lines.append(f"{indent}used += {len(ordinary)}")

            if has_branch:
                ins = terminal.ins
                if ins.op is Op.JMP:
                    lines.append(f"{indent}used += 1")
                    lines.append(f"{indent}_pc = {ins.a}")
                else:
                    lines.append(f"{indent}_v = regs[{ins.b}]")
                    condition = "not (_v is None or _v is False)"
                    if ins.op is Op.JMPIFNOT:
                        condition = f"not ({condition})"
                    lines.append(f"{indent}used += 1")
                    lines.append(f"{indent}if {condition}:")
                    lines.append(f"{indent}    _pc = {ins.a}")
                    lines.append(f"{indent}else:")
                    lines.append(f"{indent}    _pc = {block.end}")
            else:
                lines.append(f"{indent}_pc = {block.end}")
            lines.append(f"{indent}continue")

        lines.extend([
            "            else:",
            "                frame.pc = _pc",
            "                return used, False",
            "        if _forloop(regs, _loop_ins):",
            "            used += 1",
            "            continue",
            "        used += 1",
            f"        frame.pc = {ir.exit_pc}",
            "        return used, True",
            f"    frame.pc = {ir.start_pc}",
            "    return used, used > 0",
        ])

        compiled_ns = {
            "_Cell": Cell,
            "_LuaTable": LuaTable,
            "_NUM_TYPES": (int, float),
            "_i64": i64,
            "_lua_equal": lua_equal,
            "_type_matches": type_matches,
            "_isnan": __import__("math").isnan,
            "_forloop": None,
            "_loop_ins": ir.loop_ins,
        }
        source = "\n".join(lines)
        exec(compile(source, "<luapyre-jit-region>", "exec"), compiled_ns)
        raw_runner = compiled_ns["_jit_region"]

        def runner(vm, frame, budget, _raw=raw_runner, _ins=ir.loop_ins):
            compiled_ns["_forloop"] = vm._forloop
            compiled_ns["_loop_ins"] = _ins
            return _raw(vm, frame, budget)

        # cost_per_iteration is used for statistics only for branch regions;
        # exact fuel consumption comes from the dynamic `used` count returned
        # by the generated runner.
        return CompiledLoop(ir, max_cost, runner)
