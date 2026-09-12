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


class SemanticFastPathOptimizer(ast.NodeTransformer):
    """Inline tiny Lua semantic helpers in generated Python AST.

    Expression helpers are expanded only for locals/constants, so repeated uses
    cannot duplicate observable argument work. Dense table paths are guarded by
    an empty hash-part check: sparse/mixed tables immediately stay on the exact
    ``LuaTable.rawget/rawset`` implementation, while array-like tables can avoid
    method/key-tagging overhead for positive integer accesses.
    """

    @staticmethod
    def _cheap(node: ast.expr) -> bool:
        return isinstance(node, (ast.Name, ast.Constant))

    @staticmethod
    def _type_is(value: ast.expr, type_name: str) -> ast.expr:
        return ast.Compare(
            left=ast.Call(ast.Name("type", ast.Load()), [copy.deepcopy(value)], []),
            ops=[ast.Is()],
            comparators=[ast.Name(type_name, ast.Load())],
        )

    @classmethod
    def _lua_equal_expr(cls, left: ast.expr, right: ast.expr) -> ast.expr:
        left_bool = cls._type_is(left, "bool")
        right_bool = cls._type_is(right, "bool")
        bool_case = ast.BoolOp(ast.Or(), [copy.deepcopy(left_bool), copy.deepcopy(right_bool)])
        bool_value = ast.BoolOp(
            ast.And(),
            [
                left_bool,
                right_bool,
                ast.Compare(copy.deepcopy(left), [ast.Is()], [copy.deepcopy(right)]),
            ],
        )
        left_number = ast.Compare(
            ast.Call(ast.Name("type", ast.Load()), [copy.deepcopy(left)], []),
            [ast.In()],
            [ast.Name("_NUM_TYPES", ast.Load())],
        )
        right_number = ast.Compare(
            ast.Call(ast.Name("type", ast.Load()), [copy.deepcopy(right)], []),
            [ast.In()],
            [ast.Name("_NUM_TYPES", ast.Load())],
        )
        number_case = ast.BoolOp(ast.And(), [left_number, right_number])
        number_value = ast.Compare(copy.deepcopy(left), [ast.Eq()], [copy.deepcopy(right)])
        bytes_or_nil = ast.BoolOp(
            ast.Or(),
            [
                ast.Call(ast.Name("isinstance", ast.Load()), [copy.deepcopy(left), ast.Name("bytes", ast.Load())], []),
                ast.Compare(copy.deepcopy(left), [ast.Is()], [ast.Constant(None)]),
                ast.Call(ast.Name("isinstance", ast.Load()), [copy.deepcopy(right), ast.Name("bytes", ast.Load())], []),
                ast.Compare(copy.deepcopy(right), [ast.Is()], [ast.Constant(None)]),
            ],
        )
        same_type = ast.Compare(
            ast.Call(ast.Name("type", ast.Load()), [copy.deepcopy(left)], []),
            [ast.Is()],
            [ast.Call(ast.Name("type", ast.Load()), [copy.deepcopy(right)], [])],
        )
        same_value = ast.Compare(copy.deepcopy(left), [ast.Eq()], [copy.deepcopy(right)])
        bytes_nil_value = ast.BoolOp(ast.And(), [same_type, same_value])
        identity_value = ast.Compare(copy.deepcopy(left), [ast.Is()], [copy.deepcopy(right)])
        return ast.IfExp(
            bool_case,
            bool_value,
            ast.IfExp(number_case, number_value, ast.IfExp(bytes_or_nil, bytes_nil_value, identity_value)),
        )

    @classmethod
    def _type_matches_expr(cls, type_name: str, value: ast.expr) -> ast.expr | None:
        if type_name == "Any":
            return ast.Constant(True)
        if " | " in type_name:
            parts = [cls._type_matches_expr(part, value) for part in type_name.split(" | ")]
            if any(part is None for part in parts):
                return None
            return ast.BoolOp(ast.Or(), [part for part in parts if part is not None])
        if type_name == "number":
            return ast.Compare(
                ast.Call(ast.Name("type", ast.Load()), [copy.deepcopy(value)], []),
                [ast.In()],
                [ast.Name("_NUM_TYPES", ast.Load())],
            )
        if type_name == "integer":
            return cls._type_is(value, "int")
        if type_name == "float":
            return cls._type_is(value, "float")
        if type_name == "boolean":
            return cls._type_is(value, "bool")
        if type_name == "string":
            return ast.Call(ast.Name("isinstance", ast.Load()), [copy.deepcopy(value), ast.Name("bytes", ast.Load())], [])
        if type_name == "table":
            return ast.Call(ast.Name("isinstance", ast.Load()), [copy.deepcopy(value), ast.Name("_LuaTable", ast.Load())], [])
        if type_name == "nil":
            return ast.Compare(copy.deepcopy(value), [ast.Is()], [ast.Constant(None)])
        return None

    @staticmethod
    def _raw_method(call: ast.Call, name: str, arity: int) -> tuple[ast.Name, list[ast.expr]] | None:
        if call.keywords or len(call.args) != arity:
            return None
        func = call.func
        if not isinstance(func, ast.Attribute) or func.attr != name:
            return None
        if not isinstance(func.value, ast.Name):
            return None
        if not all(SemanticFastPathOptimizer._cheap(arg) for arg in call.args):
            return None
        return func.value, call.args

    def visit_Call(self, node: ast.Call):
        node = self.generic_visit(node)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "_lua_equal"
            and not node.keywords
            and len(node.args) == 2
            and all(self._cheap(arg) for arg in node.args)
        ):
            return ast.copy_location(self._lua_equal_expr(node.args[0], node.args[1]), node)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "_type_matches"
            and not node.keywords
            and len(node.args) == 2
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and self._cheap(node.args[1])
        ):
            replacement = self._type_matches_expr(node.args[0].value, node.args[1])
            if replacement is not None:
                return ast.copy_location(replacement, node)
        rawlen = self._raw_method(node, "rawlen", 0)
        if rawlen is not None:
            table, _ = rawlen
            return ast.copy_location(
                ast.Call(ast.Name("len", ast.Load()), [ast.Attribute(copy.deepcopy(table), "array", ast.Load())], []),
                node,
            )
        return node

    def visit_Assign(self, node: ast.Assign):
        node = self.generic_visit(node)
        if len(node.targets) != 1 or not isinstance(node.value, ast.Call):
            return node
        rawget = self._raw_method(node.value, "rawget", 1)
        if rawget is None:
            return node
        table, args = rawget
        key = args[0]
        target = node.targets[0]
        array = ast.Attribute(copy.deepcopy(table), "array", ast.Load())
        table_hash = ast.Attribute(copy.deepcopy(table), "hash", ast.Load())
        fallback = ast.Call(
            ast.Attribute(copy.deepcopy(table), "rawget", ast.Load()),
            [copy.deepcopy(key)],
            [],
        )
        dense_key = ast.BoolOp(
            ast.And(),
            [
                self._type_is(key, "int"),
                ast.Compare(copy.deepcopy(key), [ast.GtE()], [ast.Constant(1)]),
                ast.Compare(
                    copy.deepcopy(key),
                    [ast.LtE()],
                    [ast.Call(ast.Name("len", ast.Load()), [copy.deepcopy(array)], [])],
                ),
            ],
        )
        direct_value = ast.Subscript(
            copy.deepcopy(array),
            ast.BinOp(copy.deepcopy(key), ast.Sub(), ast.Constant(1)),
            ast.Load(),
        )
        dense_or_fallback = ast.If(
            test=dense_key,
            body=[ast.Assign([copy.deepcopy(target)], direct_value)],
            orelse=[ast.Assign([copy.deepcopy(target)], copy.deepcopy(fallback))],
        )
        return ast.copy_location(
            ast.If(
                test=copy.deepcopy(table_hash),
                body=[ast.Assign([copy.deepcopy(target)], copy.deepcopy(fallback))],
                orelse=[dense_or_fallback],
            ),
            node,
        )

    def visit_Expr(self, node: ast.Expr):
        node = self.generic_visit(node)
        if not isinstance(node.value, ast.Call):
            return node
        rawset = self._raw_method(node.value, "rawset", 2)
        if rawset is None:
            return node
        table, args = rawset
        key, value = args
        array = ast.Attribute(copy.deepcopy(table), "array", ast.Load())
        table_hash = ast.Attribute(copy.deepcopy(table), "hash", ast.Load())
        version = ast.Attribute(copy.deepcopy(table), "version", ast.Store())
        fallback = ast.Expr(
            ast.Call(
                ast.Attribute(copy.deepcopy(table), "rawset", ast.Load()),
                [copy.deepcopy(key), copy.deepcopy(value)],
                [],
            )
        )
        dense_key_value = ast.BoolOp(
            ast.And(),
            [
                self._type_is(key, "int"),
                ast.Compare(copy.deepcopy(key), [ast.GtE()], [ast.Constant(1)]),
                ast.Compare(copy.deepcopy(value), [ast.IsNot()], [ast.Constant(None)]),
            ],
        )
        existing = ast.Compare(
            copy.deepcopy(key),
            [ast.LtE()],
            [ast.Call(ast.Name("len", ast.Load()), [copy.deepcopy(array)], [])],
        )
        appendable = ast.Compare(
            copy.deepcopy(key),
            [ast.Eq()],
            [
                ast.BinOp(
                    ast.Call(ast.Name("len", ast.Load()), [copy.deepcopy(array)], []),
                    ast.Add(),
                    ast.Constant(1),
                )
            ],
        )
        bump_version = ast.AugAssign(version, ast.Add(), ast.Constant(1))
        store_existing = ast.Assign(
            [
                ast.Subscript(
                    copy.deepcopy(array),
                    ast.BinOp(copy.deepcopy(key), ast.Sub(), ast.Constant(1)),
                    ast.Store(),
                )
            ],
            copy.deepcopy(value),
        )
        append_value = ast.Expr(
            ast.Call(ast.Attribute(copy.deepcopy(array), "append", ast.Load()), [copy.deepcopy(value)], [])
        )
        dense_body = ast.If(
            test=dense_key_value,
            body=[
                ast.If(
                    test=existing,
                    body=[copy.deepcopy(bump_version), store_existing],
                    orelse=[
                        ast.If(
                            test=appendable,
                            body=[copy.deepcopy(bump_version), append_value],
                            orelse=[copy.deepcopy(fallback)],
                        )
                    ],
                )
            ],
            orelse=[copy.deepcopy(fallback)],
        )
        return ast.copy_location(
            ast.If(
                test=copy.deepcopy(table_hash),
                body=[copy.deepcopy(fallback)],
                orelse=[dense_body],
            ),
            node,
        )


class _ReturnToJump(ast.NodeTransformer):
    def __init__(self, state_name: str):
        self.state_name = state_name

    def visit_Return(self, node: ast.Return):
        value = node.value if node.value is not None else ast.Constant(-1)
        return [
            ast.Assign(targets=[ast.Name(self.state_name, ast.Store())], value=value),
            ast.Continue(),
        ]


class LocalJumpListInliner(ast.NodeTransformer):
    """Inline generated local jump-list block functions into a match dispatcher."""

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
        if not isinstance(subscript.value, ast.Name) or subscript.value.id != self.layout.list_name:
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
                    ast.Assign(targets=[ast.Name(self.layout.state_name, ast.Store())], value=ast.Constant(-1)),
                    ast.Continue(),
                ]
            )
        return ast.match_case(pattern=ast.MatchValue(ast.Constant(index)), guard=None, body=rewritten)

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
        cases = [self._case_for(index, name) for index, name in enumerate(self.layout.block_names)]
        cases.append(
            ast.match_case(
                pattern=ast.MatchAs(name=None),
                guard=None,
                body=[
                    ast.Assign(targets=[ast.Name(self.layout.state_name, ast.Store())], value=ast.Constant(-2)),
                    ast.Continue(),
                ],
            )
        )
        node.body = [ast.Match(subject=ast.Name(self.layout.state_name, ast.Load()), cases=cases)]
        return node


def inline_expression_helper(tree: ast.AST, source: str) -> ast.AST:
    helper = ast.parse(source).body[0]
    if not isinstance(helper, ast.FunctionDef):
        raise ValueError("helper source must define a function")
    tree = ExpressionFunctionInliner(helper).visit(tree)
    ast.fix_missing_locations(tree)
    return tree


def optimize_semantic_helpers(tree: ast.AST) -> ast.AST:
    tree = SemanticFastPathOptimizer().visit(tree)
    ast.fix_missing_locations(tree)
    return tree


def inline_local_jump_list(tree: ast.AST, layout: JumpListLayout) -> ast.AST:
    tree = LocalJumpListInliner(layout).visit(tree)
    tree = optimize_semantic_helpers(tree)
    ast.fix_missing_locations(tree)
    return tree
