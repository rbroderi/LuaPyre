from __future__ import annotations

import ast
import copy
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class JumpListLayout:
    """Description of a generated local jump list before AST inlining."""

    list_name: str
    state_name: str
    block_names: tuple[str, ...]


class _NameReplacer(ast.NodeTransformer):
    def __init__(self, mapping: dict[str, ast.expr]):
        self.mapping = mapping

    def visit_Name(self, node: ast.Name):
        replacement = self.mapping.get(node.id)
        if replacement is not None and isinstance(node.ctx, ast.Load):
            return ast.copy_location(copy.deepcopy(replacement), node)
        return node


class ExpressionFunctionInliner(ast.NodeTransformer):
    """Inline side-effect-free single-expression helper calls.

    The JIT only applies this to generated helper calls whose arguments are
    already Python locals/constants.  That restriction avoids the classic
    ``f(expensive()) -> expensive() + expensive()`` duplication bug in naive
    AST inliners.
    """

    def __init__(self, function_def: ast.FunctionDef):
        super().__init__()
        if len(function_def.body) != 1 or not isinstance(
            function_def.body[0], ast.Return
        ):
            raise ValueError("inline helper must contain exactly one return")
        if function_def.args.vararg or function_def.args.kwarg:
            raise ValueError("inline helper cannot be variadic")
        if function_def.args.kwonlyargs:
            raise ValueError("inline helper cannot have keyword-only arguments")
        self.name = function_def.name
        self.params = [arg.arg for arg in function_def.args.args]
        self.expression = function_def.body[0].value
        if self.expression is None:
            raise ValueError("inline helper must return an expression")

    @staticmethod
    def _cheap_argument(node: ast.expr) -> bool:
        return isinstance(node, (ast.Name, ast.Constant))

    def visit_Call(self, node: ast.Call):
        node = self.generic_visit(node)
        if not isinstance(node.func, ast.Name) or node.func.id != self.name:
            return node
        if node.keywords or len(node.args) != len(self.params):
            return node
        if not all(self._cheap_argument(arg) for arg in node.args):
            return node
        mapping = dict(zip(self.params, node.args))
        replacement = _NameReplacer(mapping).visit(copy.deepcopy(self.expression))
        return ast.copy_location(replacement, node)


class _ReturnToJump(ast.NodeTransformer):
    def __init__(self, state_name: str):
        self.state_name = state_name

    def visit_Return(self, node: ast.Return):
        value = node.value if node.value is not None else ast.Constant(-1)
        return [
            ast.Assign(
                targets=[ast.Name(self.state_name, ast.Store())],
                value=value,
            ),
            ast.Continue(),
        ]


class LocalJumpListInliner(ast.NodeTransformer):
    """Inline generated local jump-list block functions into a match dispatcher.

    The backend intentionally builds a simple threaded-code shape first::

        def _jump_0(): ...; return 1
        def _jump_1(): ...; return -1
        _jump_list = (_jump_0, _jump_1)
        while _state >= 0:
            _state = _jump_list[_state]()

    Keeping this intermediate representation makes CFG lowering simple and is a
    useful non-optimized reference form.  Before CPython compilation this pass
    removes the Python function calls and the jump-list lookup by splicing each
    block body directly into a ``match _state`` case.  In other words, the
    convenient local jump list is the compiler representation; the hot machine
    receives inlined Python bytecode rather than a function-dispatch loop.
    """

    def __init__(self, layout: JumpListLayout):
        super().__init__()
        self.layout = layout
        self.blocks: dict[str, ast.FunctionDef] = {}
        self._inside_target_function = False

    def _is_jump_list_assignment(self, stmt: ast.stmt) -> bool:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            return False
        target = stmt.targets[0]
        return isinstance(target, ast.Name) and target.id == self.layout.list_name

    def _is_dispatch_loop(self, node: ast.While) -> bool:
        if len(node.body) != 1 or not isinstance(node.body[0], ast.Assign):
            return False
        assign = node.body[0]
        if len(assign.targets) != 1:
            return False
        target = assign.targets[0]
        if not isinstance(target, ast.Name) or target.id != self.layout.state_name:
            return False
        call = assign.value
        if not isinstance(call, ast.Call) or call.args or call.keywords:
            return False
        subscript = call.func
        if not isinstance(subscript, ast.Subscript):
            return False
        if not isinstance(subscript.value, ast.Name):
            return False
        if subscript.value.id != self.layout.list_name:
            return False
        index = subscript.slice
        return isinstance(index, ast.Name) and index.id == self.layout.state_name

    def _case_for(self, index: int, block_name: str) -> ast.match_case:
        block = self.blocks[block_name]
        body = copy.deepcopy(block.body)
        rewriter = _ReturnToJump(self.layout.state_name)
        rewritten: list[ast.stmt] = []
        for stmt in body:
            result = rewriter.visit(stmt)
            if result is None:
                continue
            if isinstance(result, list):
                rewritten.extend(result)
            else:
                rewritten.append(result)
        if not rewritten or not isinstance(rewritten[-1], ast.Continue):
            rewritten.extend(
                [
                    ast.Assign(
                        targets=[ast.Name(self.layout.state_name, ast.Store())],
                        value=ast.Constant(-1),
                    ),
                    ast.Continue(),
                ]
            )
        return ast.match_case(
            pattern=ast.MatchValue(ast.Constant(index)),
            guard=None,
            body=rewritten,
        )

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if self._inside_target_function:
            return node

        block_names = set(self.layout.block_names)
        found = {
            stmt.name: stmt
            for stmt in node.body
            if isinstance(stmt, ast.FunctionDef) and stmt.name in block_names
        }
        if set(found) != block_names:
            return self.generic_visit(node)

        self.blocks = found
        self._inside_target_function = True
        try:
            new_body: list[ast.stmt] = []
            for stmt in node.body:
                if isinstance(stmt, ast.FunctionDef) and stmt.name in block_names:
                    continue
                if self._is_jump_list_assignment(stmt):
                    continue
                new_body.append(self.visit(stmt))
            node.body = new_body
        finally:
            self._inside_target_function = False
        return node

    def visit_While(self, node: ast.While):
        node = self.generic_visit(node)
        if not self._is_dispatch_loop(node):
            return node
        cases = [
            self._case_for(index, name)
            for index, name in enumerate(self.layout.block_names)
        ]
        cases.append(
            ast.match_case(
                pattern=ast.MatchAs(name=None),
                guard=None,
                body=[
                    ast.Assign(
                        targets=[ast.Name(self.layout.state_name, ast.Store())],
                        value=ast.Constant(-2),
                    ),
                    ast.Continue(),
                ],
            )
        )
        node.body = [
            ast.Match(
                subject=ast.Name(self.layout.state_name, ast.Load()),
                cases=cases,
            )
        ]
        return node


def inline_expression_helper(tree: ast.AST, source: str) -> ast.AST:
    helper = ast.parse(source).body[0]
    if not isinstance(helper, ast.FunctionDef):
        raise ValueError("helper source must define a function")
    tree = ExpressionFunctionInliner(helper).visit(tree)
    ast.fix_missing_locations(tree)
    return tree


def inline_local_jump_list(tree: ast.AST, layout: JumpListLayout) -> ast.AST:
    tree = LocalJumpListInliner(layout).visit(tree)
    ast.fix_missing_locations(tree)
    return tree
