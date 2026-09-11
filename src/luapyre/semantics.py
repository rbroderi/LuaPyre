from __future__ import annotations

from dataclasses import dataclass

from . import astnodes as A
from .errors import LuaSyntaxError


@dataclass(slots=True)
class _LabelInfo:
    stmt: A.LabelStmt
    block_id: int
    active_locals: frozenset[int]
    close_depth: int


@dataclass(slots=True)
class _GotoInfo:
    stmt: A.GotoStmt
    block_id: int
    active_locals: frozenset[int]


class ControlFlowAnalyzer:
    """Validate Lua goto visibility/scope-entry rules and annotate close depths.

    Labels are visible throughout their block (but never across function
    boundaries). A goto may leave local scopes, but it may not enter the scope
    of a local declaration. The analyzer records each target label's active
    to-be-closed depth so the bytecode compiler can unwind before the jump.
    """

    def __init__(self):
        self._next_block = 0
        self._next_local = 0
        self._next_label = 0
        self._parents: dict[int, int | None] = {}
        self._labels: list[_LabelInfo] = []
        self._gotos: list[_GotoInfo] = []
        self._local_names: dict[int, str] = {}

    def analyze(self, body: list[A.Stmt]) -> None:
        root = self._new_block(None)
        self._walk_block(body, root, frozenset(), 0)
        self._resolve_gotos()

    def _new_block(self, parent: int | None) -> int:
        block = self._next_block
        self._next_block += 1
        self._parents[block] = parent
        return block

    def _new_local(self, name: str) -> int:
        local_id = self._next_local
        self._next_local += 1
        self._local_names[local_id] = name
        return local_id

    def _is_ancestor(self, ancestor: int, block: int) -> bool:
        current: int | None = block
        while current is not None:
            if current == ancestor:
                return True
            current = self._parents[current]
        return False

    def _distance(self, ancestor: int, block: int) -> int:
        distance = 0
        current: int | None = block
        while current is not None and current != ancestor:
            current = self._parents[current]
            distance += 1
        return distance

    def _walk_child(self, body, parent, active, close_depth, synthetic=()):
        block = self._new_block(parent)
        child_active = set(active)
        for name in synthetic:
            child_active.add(self._new_local(name))
        self._walk_block(body, block, frozenset(child_active), close_depth)

    def _walk_block(self, body, block_id, initial_active, initial_close_depth):
        active = set(initial_active)
        close_depth = initial_close_depth

        for stmt in body:
            if isinstance(stmt, A.LabelStmt):
                for existing in self._labels:
                    if existing.stmt.name == stmt.name and self._is_ancestor(existing.block_id, block_id):
                        raise LuaSyntaxError(
                            f"line {stmt.line}: label '{stmt.name}' already defined and visible"
                        )
                stmt.label_id = self._next_label
                self._next_label += 1
                stmt.close_depth = close_depth
                self._labels.append(
                    _LabelInfo(stmt, block_id, frozenset(active), close_depth)
                )
                continue

            if isinstance(stmt, A.GotoStmt):
                self._gotos.append(
                    _GotoInfo(stmt, block_id, frozenset(active))
                )
                continue

            if isinstance(stmt, A.LocalDecl):
                for declared in stmt.names:
                    active.add(self._new_local(declared.name))
                    if declared.attribute == "close":
                        close_depth += 1
                continue

            if isinstance(stmt, A.FunctionDef) and stmt.local:
                active.add(self._new_local(stmt.name))
                continue

            if isinstance(stmt, A.DoStmt):
                self._walk_child(stmt.body, block_id, active, close_depth)
                continue

            if isinstance(stmt, A.WhileStmt):
                self._walk_child(stmt.body, block_id, active, close_depth)
                continue

            if isinstance(stmt, A.RepeatStmt):
                self._walk_child(stmt.body, block_id, active, close_depth)
                continue

            if isinstance(stmt, A.NumericForStmt):
                self._walk_child(
                    stmt.body, block_id, active, close_depth, (stmt.name,)
                )
                continue

            if isinstance(stmt, A.GenericForStmt):
                self._walk_child(
                    stmt.body, block_id, active, close_depth, tuple(stmt.names)
                )
                continue

            if isinstance(stmt, A.IfStmt):
                for _, clause in stmt.clauses:
                    self._walk_child(clause, block_id, active, close_depth)
                if stmt.else_body:
                    self._walk_child(stmt.else_body, block_id, active, close_depth)

    def _resolve_gotos(self):
        for goto in self._gotos:
            candidates = [
                label for label in self._labels
                if label.stmt.name == goto.stmt.name
                and self._is_ancestor(label.block_id, goto.block_id)
            ]
            if not candidates:
                raise LuaSyntaxError(
                    f"line {goto.stmt.line}: no visible label '{goto.stmt.name}' for goto"
                )
            target = min(
                candidates,
                key=lambda label: self._distance(label.block_id, goto.block_id),
            )
            entered = target.active_locals.difference(goto.active_locals)
            if entered:
                local_id = min(entered)
                name = self._local_names.get(local_id, "?")
                raise LuaSyntaxError(
                    f"line {goto.stmt.line}: goto '{goto.stmt.name}' jumps into the scope of local '{name}'"
                )
            goto.stmt.target_id = target.stmt.label_id
            goto.stmt.close_depth = target.close_depth


def analyze_control_flow(body: list[A.Stmt]) -> None:
    ControlFlowAnalyzer().analyze(body)
