from __future__ import annotations

import ast
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "src/luapyre/binary_chunks.py"
WORKFLOW = ROOT / ".github/workflows/_dispatch-refactor-once.yml"
SELF = Path(__file__)

source = TARGET.read_text(encoding="utf-8")
module = ast.parse(source)
lines = source.splitlines()

translator = next(
    node for node in module.body
    if isinstance(node, ast.ClassDef) and node.name == "_Translator"
)
translate = next(
    node for node in translator.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "translate"
)
while_node = next(node for node in translate.body if isinstance(node, ast.While))
if_node = next(node for node in while_node.body if isinstance(node, ast.If))


def segment(node: ast.AST) -> str:
    # ast.unparse deliberately normalizes indentation. get_source_segment keeps
    # original continuation indentation after removing the first-line column,
    # which makes moved compound statements invalid when re-indented.
    return ast.unparse(node)


def handler_name(test: ast.expr) -> str:
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
        raise RuntimeError(f"unsupported translator condition: {ast.unparse(test)}")
    if not isinstance(test.left, ast.Name) or test.left.id != "opcode":
        raise RuntimeError(f"unsupported translator discriminator: {ast.unparse(test)}")
    right = test.comparators[0]
    if isinstance(test.ops[0], ast.Eq) and isinstance(right, ast.Name):
        name = right.id
        return name[2:].lower() if name.startswith("P_") else name.lower()
    if isinstance(test.ops[0], ast.In) and isinstance(right, ast.Name):
        return right.id.strip("_").lower()
    if isinstance(test.ops[0], ast.In) and isinstance(right, ast.Tuple):
        names = []
        for item in right.elts:
            if not isinstance(item, ast.Name):
                raise RuntimeError(f"unsupported translator tuple: {ast.unparse(test)}")
            names.append(item.id[2:].lower() if item.id.startswith("P_") else item.id.lower())
        known = {
            ("mmbin", "mmbini", "mmbink"): "orphan_mm",
            ("unm", "bnot", "not", "len"): "unary",
            ("eq", "lt", "le"): "compare_reg",
            ("eqi", "lti", "lei", "gti", "gei"): "compare_imm",
        }
        return known.get(tuple(names), "_".join(names))
    raise RuntimeError(f"unsupported translator condition: {ast.unparse(test)}")


def table_lines(test: ast.expr, name: str) -> list[str]:
    compare = test
    right = compare.comparators[0]
    handler = f"_Translator._translate_{name}"
    if isinstance(compare.ops[0], ast.Eq):
        return [f"    {right.id}: {handler},"]
    if isinstance(right, ast.Name):
        return [
            f"for _puc_opcode in {right.id}:",
            f"    _PUC_TRANSLATORS[_puc_opcode] = {handler}",
        ]
    return [f"    {item.id}: {handler}," for item in right.elts]

branches: list[tuple[ast.expr, list[ast.stmt]]] = []
node = if_node
final_else: list[ast.stmt] = []
while True:
    branches.append((node.test, node.body))
    if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
        node = node.orelse[0]
        continue
    final_else = node.orelse
    break

if len(branches) < 40:
    raise RuntimeError(f"translator branch extraction unexpectedly found only {len(branches)} branches")
if not final_else or "unsupported PUC-Lua opcode" not in "\n".join(segment(stmt) for stmt in final_else):
    raise RuntimeError("translator terminal unsupported-opcode branch changed")

handlers: list[str] = []
table_dict_entries: list[str] = []
table_post: list[str] = []
seen_names: set[str] = set()
for test, body in branches:
    name = handler_name(test)
    if name in seen_names:
        raise RuntimeError(f"duplicate generated handler name {name}")
    seen_names.add(name)
    body_text = "\n".join(segment(stmt) for stmt in body)
    indented_body = textwrap.indent(body_text, "        ")
    handlers.append(
        f"    def _translate_{name}(self, pc: int, word: int) -> int:\n"
        "        code = self.source.code\n"
        "        opcode = _op(word)\n"
        "        a, b, c, k = _a(word), _b(word), _c(word), _k(word)\n"
        f"{indented_body}\n"
        "        return pc"
    )
    generated = table_lines(test, name)
    if generated and generated[0].startswith("for "):
        table_post.extend(generated)
    else:
        table_dict_entries.extend(generated)

# Preserve all finalization logic after the translation loop semantically.
after_while = translate.body[translate.body.index(while_node) + 1:]
after_text = "\n".join(segment(stmt) for stmt in after_while)
after_text = textwrap.indent(after_text, "        ")

new_translate = (
    "    def translate(self) -> Proto:\n"
    "        code = self.source.code\n"
    "        pc = 0\n"
    "        handlers = _PUC_TRANSLATORS\n"
    "        while pc < len(code):\n"
    "            self.pcmap[pc] = len(self.proto.code)\n"
    "            word = code[pc]\n"
    "            opcode = _op(word)\n"
    "            handler = handlers.get(opcode)\n"
    "            if handler is None:\n"
    "                raise BinaryChunkError(f\"unsupported PUC-Lua opcode {opcode}\")\n"
    "            pc = handler(self, pc, word)\n\n"
    f"{after_text}"
)

replacement = "\n\n".join(handlers + [new_translate])
start = translate.lineno - 1
end = translate.end_lineno
updated_lines = lines[:start] + replacement.splitlines() + lines[end:]
updated = "\n".join(updated_lines) + ("\n" if source.endswith("\n") else "")

marker = "\n\ndef load_puc55_chunk(data: bytes) -> Proto:\n"
if marker not in updated:
    raise RuntimeError("load_puc55_chunk insertion marker not found")

table = "\n\n_PUC_TRANSLATORS = {\n" + "\n".join(table_dict_entries) + "\n}\n"
if table_post:
    table += "\n" + "\n".join(table_post) + "\n"
table += (
    "\n_missing_puc_translators = set(range(85)).difference(_PUC_TRANSLATORS)\n"
    "_extra_puc_translators = set(_PUC_TRANSLATORS).difference(range(85))\n"
    "if _missing_puc_translators or _extra_puc_translators:\n"
    "    raise RuntimeError(\n"
    "        \"PUC translator handler table mismatch: \"\n"
    "        f\"missing={sorted(_missing_puc_translators)} \"\n"
    "        f\"extra={sorted(_extra_puc_translators)}\"\n"
    "    )\n"
)
updated = updated.replace(marker, table + marker, 1)

# Syntax-check the generated module before touching the working tree.
ast.parse(updated)
TARGET.write_text(updated, encoding="utf-8")

# Architecture regression: closed opcode set must stay table-dispatched.
test_path = ROOT / "tests/test_dispatch_architecture.py"
test_path.write_text(
    "from __future__ import annotations\n\n"
    "import inspect\n\n"
    "from luapyre.binary_chunks import _PUC_TRANSLATORS, _Translator\n\n\n"
    "def test_puc_translator_dispatch_table_covers_all_opcodes():\n"
    "    assert set(_PUC_TRANSLATORS) == set(range(85))\n\n\n"
    "def test_puc_translate_loop_uses_table_dispatch():\n"
    "    source = inspect.getsource(_Translator.translate)\n"
    "    assert \"_PUC_TRANSLATORS\" in source\n"
    "    assert \"elif opcode\" not in source\n",
    encoding="utf-8",
)

# One-shot scaffolding must not land in the branch result.
SELF.unlink()
if WORKFLOW.exists():
    WORKFLOW.unlink()
