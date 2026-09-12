from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from enum import Enum
import math
import struct

from .bytecode import Op, Proto
from .call_ir import StaticCallSite, StaticClosureRef
from .typed_ir import TypedIRCompiler, TypedIRPlan
from .typesys import accepts, parse_simple_type


_MASK64 = (1 << 64) - 1
_SIGN64 = 1 << 63
_TWO64 = 1 << 64


class ValueKind(str, Enum):
    """Kinds of immutable SSA-like values understood by the 0.17 optimizer."""

    ARGUMENT = "argument"
    CONSTANT = "constant"
    LITERAL = "literal"
    EXPRESSION = "expression"


@dataclass(frozen=True, slots=True)
class ValueNode:
    id: int
    kind: ValueKind
    type_name: str
    op: str | None = None
    args: tuple[int, ...] = ()
    payload: object = None
    def_pc: int = -1


@dataclass(frozen=True, slots=True)
class ValueIRPlan:
    """Backend-neutral value graph layered on the 0.16 typed IR.

    LOADK/MOVE/LOCAL are aliases, typed pure operations produce immutable SSA
    values, identical expressions share a value number, scalar constant
    subgraphs fold to LITERAL nodes, and eligible static CALLs splice the
    callee's value graph into the caller. Only live EXPRESSION nodes need runtime
    assignments. ``instruction_count`` retains the exact Lua cost, including
    inlined callee bytecode, even when generated Python operations disappear.
    """

    typed_plan: TypedIRPlan
    nodes: tuple[ValueNode, ...]
    register_values: tuple[tuple[int, int], ...]
    return_values: tuple[int, ...]
    return_pc: int
    instruction_count: int
    live_nodes: frozenset[int]
    definition_pcs: tuple[tuple[int, int], ...]
    folded_pcs: tuple[int, ...]
    cse_pcs: tuple[int, ...]
    eliminated_pcs: tuple[int, ...]
    call_sites: tuple[StaticCallSite, ...] = ()

    def node(self, node_id: int) -> ValueNode:
        return self.nodes[node_id]

    @property
    def has_optimization(self) -> bool:
        return bool(
            self.folded_pcs
            or self.cse_pcs
            or self.eliminated_pcs
            or self.call_sites
        )


_INT_BINOPS = {
    Op.ADD_I: "add_i",
    Op.SUB_I: "sub_i",
    Op.MUL_I: "mul_i",
}
_FLOAT_BINOPS = {
    Op.ADD_F: "add_f",
    Op.SUB_F: "sub_f",
    Op.MUL_F: "mul_f",
}
_BOOL_UNARY = {
    Op.NOT: "not",
    Op.TOBOOL: "tobool",
}
_COMPARE_OPS = {
    Op.EQ: "eq",
    Op.LT: "lt",
    Op.LE: "le",
}
_PURE_VALUE_OPS = frozenset(
    {
        Op.LOADK,
        Op.MOVE,
        Op.LOCAL,
        *_INT_BINOPS,
        *_FLOAT_BINOPS,
        *_BOOL_UNARY,
        *_COMPARE_OPS,
    }
)
_SUPPORTED = _PURE_VALUE_OPS | frozenset({Op.CLOSURE, Op.CALL, Op.RETURN, Op.HALT})


def _i64(value: int) -> int:
    value &= _MASK64
    return value - _TWO64 if value & _SIGN64 else value


def _lua_truthy(value: object) -> bool:
    return not (value is None or value is False)


def _literal_key(value: object) -> tuple[object, ...]:
    if type(value) is float:
        return (float, struct.pack(">d", value))
    return (type(value), value)


def _literal_type(value: object) -> str:
    if value is None:
        return "nil"
    if type(value) is bool:
        return "boolean"
    if type(value) is int:
        return "integer"
    if type(value) is float:
        return "float"
    if isinstance(value, bytes):
        return "string"
    return "Any"


def _safe_scalar_type(type_name: str) -> bool:
    return type_name in {"nil", "boolean", "integer", "float", "number", "string"}


def _safe_literal(value: object) -> bool:
    return value is None or type(value) in (bool, int, float) or isinstance(value, bytes)


def _reachable_prefix(proto: Proto) -> Proto | None:
    terminal = next(
        (pc for pc, ins in enumerate(proto.code) if ins.op in (Op.RETURN, Op.HALT)),
        None,
    )
    if terminal is None:
        return None
    result = copy(proto)
    result.code = list(proto.code[: terminal + 1])
    return result


class ValueIRCompiler:
    """Build an SSA-ish value graph on top of ``TypedIRPlan``.

    The admitted domain is deliberately exact rather than broad. Pure typed
    scalar operations are value-numbered. A lexical CALL is admitted only when
    its function register is statically traced to a child CLOSURE, the child is a
    fully typed non-vararg leaf with no upvalues/children, every argument type is
    statically compatible, and the child's reachable body recursively compiles
    to this pure value IR. Any escaping/dynamic function use fails closed.
    """

    def __init__(self, proto: Proto, typed_plan: TypedIRPlan):
        self.proto = proto
        self.typed_plan = typed_plan
        self._nodes: list[ValueNode] = []
        self._intern: dict[tuple[object, ...], int] = {}
        self._definition_pcs: dict[int, int] = {}
        self._folded_pcs: set[int] = set()
        self._cse_pcs: set[int] = set()
        self._call_sites: list[StaticCallSite] = []
        self._extra_instruction_cost = 0

    def _new_node(
        self,
        kind: ValueKind,
        type_name: str,
        *,
        op: str | None = None,
        args: tuple[int, ...] = (),
        payload: object = None,
        def_pc: int = -1,
        key: tuple[object, ...] | None = None,
    ) -> int:
        if key is not None:
            existing = self._intern.get(key)
            if existing is not None:
                return existing
        node_id = len(self._nodes)
        self._nodes.append(ValueNode(node_id, kind, type_name, op, args, payload, def_pc))
        if key is not None:
            self._intern[key] = node_id
        return node_id

    def _argument(self, index: int, type_name: str) -> int:
        return self._new_node(
            ValueKind.ARGUMENT,
            type_name,
            payload=index,
            key=(ValueKind.ARGUMENT, index, type_name),
        )

    def _constant(self, index: int) -> int:
        value = self.proto.constants[index]
        if not _safe_literal(value):
            raise ValueError("non-scalar constant in value IR")
        return self._new_node(
            ValueKind.CONSTANT,
            _literal_type(value),
            payload=index,
            key=(ValueKind.CONSTANT, index),
        )

    def _literal(self, value: object) -> int:
        if not _safe_literal(value):
            raise ValueError("non-scalar literal in value IR")
        return self._new_node(
            ValueKind.LITERAL,
            _literal_type(value),
            payload=value,
            key=(ValueKind.LITERAL, *_literal_key(value)),
        )

    def _known_value(self, node_id: int) -> tuple[bool, object]:
        node = self._nodes[node_id]
        if node.kind is ValueKind.LITERAL:
            return True, node.payload
        if node.kind is ValueKind.CONSTANT:
            value = self.proto.constants[int(node.payload)]
            if _safe_literal(value):
                return True, value
        return False, None

    def _fold_binary(self, op: str, left: object, right: object) -> tuple[bool, object]:
        try:
            if op == "add_i" and type(left) is int and type(right) is int:
                return True, _i64(left + right)
            if op == "sub_i" and type(left) is int and type(right) is int:
                return True, _i64(left - right)
            if op == "mul_i" and type(left) is int and type(right) is int:
                return True, _i64(left * right)
            if op == "add_f" and type(left) in (int, float) and type(right) in (int, float):
                return True, float(left + right)
            if op == "sub_f" and type(left) in (int, float) and type(right) in (int, float):
                return True, float(left - right)
            if op == "mul_f" and type(left) in (int, float) and type(right) in (int, float):
                return True, float(left * right)
            if op == "eq":
                if type(left) in (int, float) and type(right) in (int, float):
                    return True, left == right
                if type(left) is type(right) and (
                    left is None or type(left) is bool or isinstance(left, bytes)
                ):
                    return True, left == right
                if type(left) is not type(right):
                    return True, False
            if op in ("lt", "le"):
                numeric = type(left) in (int, float) and type(right) in (int, float)
                strings = isinstance(left, bytes) and isinstance(right, bytes)
                if numeric or strings:
                    return True, left < right if op == "lt" else left <= right
        except (ArithmeticError, OverflowError, ValueError):
            return False, None
        return False, None

    def _binary(self, pc: int, op: str, left_id: int, right_id: int, type_name: str) -> int:
        left_known, left = self._known_value(left_id)
        right_known, right = self._known_value(right_id)
        if left_known and right_known:
            folded, value = self._fold_binary(op, left, right)
            if folded and not (type(value) is float and math.isnan(value)):
                self._folded_pcs.add(pc)
                return self._literal(value)
        key = (ValueKind.EXPRESSION, op, left_id, right_id, type_name)
        existing = self._intern.get(key)
        if existing is not None:
            self._cse_pcs.add(pc)
            return existing
        node_id = self._new_node(
            ValueKind.EXPRESSION,
            type_name,
            op=op,
            args=(left_id, right_id),
            def_pc=pc,
            key=key,
        )
        self._definition_pcs[node_id] = pc
        return node_id

    def _unary(self, pc: int, op: str, value_id: int, type_name: str) -> int:
        known, value = self._known_value(value_id)
        if known and op in ("not", "tobool"):
            folded = (not _lua_truthy(value)) if op == "not" else _lua_truthy(value)
            self._folded_pcs.add(pc)
            return self._literal(folded)
        key = (ValueKind.EXPRESSION, op, value_id, type_name)
        existing = self._intern.get(key)
        if existing is not None:
            self._cse_pcs.add(pc)
            return existing
        node_id = self._new_node(
            ValueKind.EXPRESSION,
            type_name,
            op=op,
            args=(value_id,),
            def_pc=pc,
            key=key,
        )
        self._definition_pcs[node_id] = pc
        return node_id

    def _mark_live(self, roots: tuple[int, ...]) -> frozenset[int]:
        live: set[int] = set()
        stack = list(roots)
        while stack:
            node_id = stack.pop()
            if node_id in live:
                continue
            live.add(node_id)
            stack.extend(self._nodes[node_id].args)
        return frozenset(live)

    @staticmethod
    def _node_ref(value: int | StaticClosureRef) -> int | None:
        return value if type(value) is int else None

    def _child_plan(self, child: Proto) -> tuple[Proto, ValueIRPlan] | None:
        if (
            not child.jit_fully_typed
            or child.is_vararg
            or child.upvalues
            or child.children
        ):
            return None
        analysis_child = _reachable_prefix(child)
        if analysis_child is None:
            return None
        block = (tuple(enumerate(analysis_child.code)),)
        typed = TypedIRCompiler(analysis_child).compile(block)
        plan = ValueIRCompiler(analysis_child, typed).compile()
        if plan is None or plan.call_sites:
            return None
        return analysis_child, plan

    def _clone_child_value(
        self,
        pc: int,
        child: Proto,
        child_plan: ValueIRPlan,
        args: tuple[int, ...],
    ) -> tuple[int, ...] | None:
        mapping: dict[int, int] = {}
        for node in child_plan.nodes:
            if node.kind is ValueKind.ARGUMENT:
                index = int(node.payload)
                if index >= len(args):
                    return None
                mapping[node.id] = args[index]
            elif node.kind is ValueKind.CONSTANT:
                value = child.constants[int(node.payload)]
                if not _safe_literal(value):
                    return None
                mapping[node.id] = self._literal(value)
            elif node.kind is ValueKind.LITERAL:
                mapping[node.id] = self._literal(node.payload)
            elif node.kind is ValueKind.EXPRESSION:
                mapped = tuple(mapping[arg] for arg in node.args)
                if len(mapped) == 1:
                    mapping[node.id] = self._unary(pc, node.op or "", mapped[0], node.type_name)
                elif len(mapped) == 2:
                    mapping[node.id] = self._binary(
                        pc, node.op or "", mapped[0], mapped[1], node.type_name
                    )
                else:
                    return None
            else:
                return None
        return tuple(mapping[node_id] for node_id in child_plan.return_values)

    def compile(self) -> ValueIRPlan | None:
        code = self.proto.code
        if not code or code[-1].op not in (Op.RETURN, Op.HALT):
            return None
        if any(ins.op not in _SUPPORTED for ins in code):
            return None
        if any(ins.op in (Op.RETURN, Op.HALT) for ins in code[:-1]):
            return None

        try:
            regs: dict[int, int | StaticClosureRef] = {
                index: self._literal(None) for index in range(self.proto.register_count)
            }
        except ValueError:
            return None
        for index in range(min(self.proto.param_count, self.proto.register_count)):
            typ = (
                self.proto.param_types[index].name
                if index < len(self.proto.param_types)
                else "Any"
            )
            if typ == "Any":
                return None
            regs[index] = self._argument(index, typ)

        inlined_call_pcs: set[int] = set()
        closure_pcs: set[int] = set()

        for pc, ins in enumerate(code[:-1]):
            op = ins.op
            if op is Op.LOADK:
                try:
                    regs[ins.a] = self._constant(ins.b)
                except ValueError:
                    return None
                continue
            if op in (Op.MOVE, Op.LOCAL):
                regs[ins.a] = regs[ins.b]
                continue
            if op is Op.CLOSURE:
                if ins.b < 0 or ins.b >= len(self.proto.children):
                    return None
                child = self.proto.children[ins.b]
                # Capturing lexical cells is observable state and must stay on
                # the existing exact closure/cell compiler for now.
                if child.upvalues or child.is_vararg:
                    return None
                regs[ins.a] = StaticClosureRef(ins.b)
                closure_pcs.add(pc)
                continue
            if op in _INT_BINOPS:
                left = self._node_ref(regs[ins.b])
                right = self._node_ref(regs[ins.c])
                if left is None or right is None:
                    return None
                if self._nodes[left].type_name != "integer" or self._nodes[right].type_name != "integer":
                    return None
                regs[ins.a] = self._binary(pc, _INT_BINOPS[op], left, right, "integer")
                continue
            if op in _FLOAT_BINOPS:
                left = self._node_ref(regs[ins.b])
                right = self._node_ref(regs[ins.c])
                if left is None or right is None:
                    return None
                numeric = {"integer", "float", "number"}
                if self._nodes[left].type_name not in numeric or self._nodes[right].type_name not in numeric:
                    return None
                regs[ins.a] = self._binary(pc, _FLOAT_BINOPS[op], left, right, "float")
                continue
            if op in _BOOL_UNARY:
                source = self._node_ref(regs[ins.b])
                if source is None:
                    return None
                regs[ins.a] = self._unary(pc, _BOOL_UNARY[op], source, "boolean")
                continue
            if op in _COMPARE_OPS:
                left = self._node_ref(regs[ins.b])
                right = self._node_ref(regs[ins.c])
                if left is None or right is None:
                    return None
                left_type = self._nodes[left].type_name
                right_type = self._nodes[right].type_name
                if not (_safe_scalar_type(left_type) and _safe_scalar_type(right_type)):
                    return None
                if op in (Op.LT, Op.LE):
                    numeric = {"integer", "float", "number"}
                    if not (
                        (left_type in numeric and right_type in numeric)
                        or (left_type == right_type == "string")
                    ):
                        return None
                regs[ins.a] = self._binary(pc, _COMPARE_OPS[op], left, right, "boolean")
                continue
            if op is Op.CALL:
                function_value = regs[ins.b]
                if not isinstance(function_value, StaticClosureRef):
                    return None
                if ins.e < 0:
                    return None
                child_index = function_value.child_index
                child = self.proto.children[child_index]
                child_pair = self._child_plan(child)
                if child_pair is None or ins.d != child.param_count:
                    return None
                analysis_child, child_plan = child_pair
                arg_values: list[int] = []
                for arg_index in range(ins.d):
                    node_id = self._node_ref(regs[ins.c + arg_index])
                    if node_id is None:
                        return None
                    actual = parse_simple_type(self._nodes[node_id].type_name)
                    expected = child.param_types[arg_index]
                    if not accepts(expected, actual):
                        return None
                    arg_values.append(node_id)
                returned = self._clone_child_value(
                    pc, analysis_child, child_plan, tuple(arg_values)
                )
                if returned is None:
                    return None
                nil_value = self._literal(None)
                for result_index in range(ins.e):
                    regs[ins.a + result_index] = (
                        returned[result_index]
                        if result_index < len(returned)
                        else nil_value
                    )
                self._extra_instruction_cost += child_plan.instruction_count
                self._call_sites.append(
                    StaticCallSite(
                        pc,
                        child_index,
                        ins.d,
                        ins.e,
                        child_plan.instruction_count,
                    )
                )
                inlined_call_pcs.add(pc)
                continue
            return None

        terminal = code[-1]
        return_values: tuple[int, ...]
        if terminal.op is Op.HALT:
            return_values = ()
        else:
            values: list[int] = []
            for index in range(terminal.a, terminal.a + max(0, terminal.b)):
                node_id = self._node_ref(regs[index])
                # Returning a function value would make closure allocation and
                # identity observable, so that shape must stay in the old tier.
                if node_id is None:
                    return None
                values.append(node_id)
            return_values = tuple(values)

        live = self._mark_live(return_values)
        definition_pcs = tuple(sorted((node_id, pc) for node_id, pc in self._definition_pcs.items()))
        live_definition_pcs = {pc for node_id, pc in definition_pcs if node_id in live}
        eliminated = tuple(
            pc
            for pc, ins in enumerate(code[:-1])
            if (
                (ins.op in _PURE_VALUE_OPS and pc not in live_definition_pcs)
                or pc in closure_pcs
                or pc in inlined_call_pcs
            )
        )
        return ValueIRPlan(
            self.typed_plan,
            tuple(self._nodes),
            tuple(
                sorted(
                    (reg, node_id)
                    for reg, value in regs.items()
                    if (node_id := self._node_ref(value)) is not None
                )
            ),
            return_values,
            len(code) - 1,
            len(code) + self._extra_instruction_cost,
            live,
            definition_pcs,
            tuple(sorted(self._folded_pcs)),
            tuple(sorted(self._cse_pcs)),
            eliminated,
            tuple(self._call_sites),
        )
