"""Measure corrected 0.40 general IR paths and former headline controls.

Run unchanged against baseline and candidate trees. Release acceptance requires
all three differently shaped cases in a generality group to improve; admission
counts are stored with every result. Tree and matrix cases are regression
controls and no longer use whole-algorithm compilers.
"""
from __future__ import annotations

import sys

import speed_039_ab as _base
from speed_039_ab import *  # noqa: F401,F403
from python_headroom import binary_trees, spectral_norm
from vm_programs import WORKLOADS


def tree_build_only():
    def make(depth):
        if depth == 0:
            return {"left": None, "right": None}
        return {"left": make(depth - 1), "right": make(depth - 1)}
    node = make(8)
    return 1 if node["left"] is not None else 0


def tree_traversal_only():
    def make(depth):
        if depth == 0:
            return {"left": None, "right": None}
        return {"left": make(depth - 1), "right": make(depth - 1)}

    def check(node):
        if node["left"] is None:
            return 1
        return 1 + check(node["left"]) + check(node["right"])

    node = make(8)

    def run():
        total = 0
        for _ in range(30):
            total += check(node)
        return total
    return run


def matrix_checksum(transpose=False):
    values = [None, *[1.0 for _ in range(30)]]
    result = [None]
    for i in range(1, 31):
        total = 0.0
        for j in range(1, 31):
            left, right = (j - 1, i - 1) if transpose else (i - 1, j - 1)
            diagonal = left + right
            weight = 1.0 / (
                diagonal * (diagonal + 1) / 2.0 + left + 1.0
            )
            total += weight * values[j]
        result.append(total)
    return sum(result[1:])


MATRIX_FUNCTIONS = """
local function A(i: integer, j: integer): float
    return 1.0 / (((i+j)*(i+j+1)/2.0)+i+1.0)
end
local function multiplyAv(x: table, n: integer): table
    local out: table = {}
    for i = 1, n do
        local s: float = 0.0
        for j = 1, n do
            local xj: float = x[j]
            s = s + A(i-1, j-1) * xj
        end
        out[i] = s
    end
    return out
end
local function multiplyAtv(x: table, n: integer): table
    local out: table = {}
    for i = 1, n do
        local s: float = 0.0
        for j = 1, n do
            local xj: float = x[j]
            s = s + A(j-1, i-1) * xj
        end
        out[i] = s
    end
    return out
end
"""


def scalar_dag_reference(kind):
    total = 0
    for value in range(1, 20001):
        if kind == 0:
            total += (value + 2) * (value - 1)
        elif kind == 1:
            total += value * value + 3 * value + 1
        else:
            left, right = value + 1, value - 1
            total += left * right + value
    return total


def nested_reduction_reference(kind):
    total = 1 if kind == 1 else 0
    for outer in range(1, 201):
        row = 0
        for inner in range(1, 101):
            if kind == 0:
                total += outer * inner
            elif kind == 1:
                row += outer + inner
            else:
                total += (outer * 3 + inner * 5) % 997
        if kind == 1:
            total += row
    return total


def sparse_reference(kind):
    marked = set()
    total = 0
    for value in range(1, 5001):
        key = value * (3 if kind == 0 else 5 if kind == 1 else 7)
        if kind == 1 and key in marked:
            continue
        marked.add(key)
        total += 1 if kind < 2 else value
    return total


def recursive_reference(kind, value):
    def run(n):
        if kind == 0:
            return 0 if n <= 0 else n + run(n - 1)
        if kind == 1:
            return n if n < 2 else run(n - 1) + run(n - 2)
        return 1 if n <= 0 else run(n - 1) + run(n - 1) + run(n - 1)
    return run(value)


def record_reference(arity):
    total = 0
    names = ("a", "b", "c", "d", "e")[:arity]
    for value in range(1, 5001):
        record = {name: value + index for index, name in enumerate(names)}
        total += record["a"]
    return total


CASES_040 = {
    "binary_trees_headroom": Case(
        WORKLOADS["binary_trees"].source_for("luapyre").split("\n", 1)[1],
        binary_trees, 15330, "Complete Binary Trees lower-bound workload",
    ),
    "spectral_norm_headroom": Case(
        WORKLOADS["spectral_norm"].source_for("luapyre").split("\n", 1)[1],
        spectral_norm, 1.274097380427096,
        "Complete Spectral Norm lower-bound workload",
    ),
    "tree_build_only": Case("""
local function make(depth: integer): table
    if depth == 0 then return {left=nil, right=nil} end
    return {left=make(depth-1), right=make(depth-1)}
end
local node: table = make(8)
if node.left ~= nil then return 1 end
return 0
""", tree_build_only, 1, "Recursive construction with only a shallow result check"),
    "tree_traversal_only": Case("""
local function make(depth: integer): table
    if depth == 0 then return {left=nil, right=nil} end
    return {left=make(depth-1), right=make(depth-1)}
end
local function check(node: table): integer
    if node.left == nil then return 1 end
    return 1 + check(node.left) + check(node.right)
end
local node: table = make(8)
return function(): integer
    local total: integer = 0
    for round = 1, 30 do total = total + check(node) end
    return total
end
""", tree_traversal_only(), 15330,
        "Traversal of one prebuilt tree", ((),)),
    "matrix_av": Case(MATRIX_FUNCTIONS + """
local values: table = {}
for i = 1, 30 do values[i] = 1.0 end
local result: table = multiplyAv(values, 30)
local total: float = 0.0
for i = 1, 30 do total = total + result[i] end
return total
""", lambda: matrix_checksum(False), matrix_checksum(False),
        "One dense Av matrix product and checked reduction"),
    "matrix_atv": Case(MATRIX_FUNCTIONS + """
local values: table = {}
for i = 1, 30 do values[i] = 1.0 end
local result: table = multiplyAtv(values, 30)
local total: float = 0.0
for i = 1, 30 do total = total + result[i] end
return total
""", lambda: matrix_checksum(True), matrix_checksum(True),
        "One dense Atv matrix product and checked reduction"),
    "scalar_dag_product": Case("""
local function f(x: integer): integer return (x+2)*(x-1) end
local total: integer=0 for i=1,20000 do total=total+f(i) end return total
""", lambda: scalar_dag_reference(0), scalar_dag_reference(0),
        "Pure scalar product DAG"),
    "scalar_dag_polynomial": Case("""
local function f(x: integer): integer local square: integer=x*x return square+3*x+1 end
local total: integer=0 for i=1,20000 do total=total+f(i) end return total
""", lambda: scalar_dag_reference(1), scalar_dag_reference(1),
        "Pure scalar polynomial with a temporary"),
    "scalar_dag_split": Case("""
local function f(x: integer): integer local a: integer=x+1 local b: integer=x-1 return a*b+x end
local total: integer=0 for i=1,20000 do total=total+f(i) end return total
""", lambda: scalar_dag_reference(2), scalar_dag_reference(2),
        "Pure scalar DAG with independent temporaries"),
    "nested_reduction_product": Case("""
local total: integer=0 for i=1,200 do for j=1,100 do total=total+i*j end end return total
""", lambda: nested_reduction_reference(0), nested_reduction_reference(0),
        "Nested product reduction"),
    "nested_reduction_rows": Case("""
local total: integer=1 for i=1,200 do local row: integer=0 for j=1,100 do row=row+i+j end total=total+row end return total
""", lambda: nested_reduction_reference(1), nested_reduction_reference(1),
        "Nested reduction through a row temporary"),
    "nested_reduction_modulo": Case("""
local total: integer=0 for i=1,200 do for j=1,100 do total=total+((i*3+j*5)%997) end end return total
""", lambda: nested_reduction_reference(2), nested_reduction_reference(2),
        "Nested modulo reduction with another expression tree"),
    "sparse_slots": Case("""
local slots: table={} local total: integer=0 for i=1,5000 do local k: integer=i*3 slots[k]=true if slots[k] then total=total+1 end end return total
""", lambda: sparse_reference(0), sparse_reference(0), "Sparse slot map"),
    "sparse_visited": Case("""
local seen: table={} local total: integer=0 for i=1,5000 do local k: integer=i*5 if not seen[k] then seen[k]=true total=total+1 end end return total
""", lambda: sparse_reference(1), sparse_reference(1), "Sparse visited set"),
    "sparse_occupancy": Case("""
local occupied: table={} local total: integer=0 for i=1,5000 do local k: integer=i*7 occupied[k]=true if occupied[k] then total=total+i end end return total
""", lambda: sparse_reference(2), sparse_reference(2), "Sparse occupancy map"),
    "recursive_linear_general": Case("""
local function f(n: integer): integer if n <= 0 then return 0 end return n+f(n-1) end
return f(200)
""", lambda: recursive_reference(0, 200), recursive_reference(0, 200),
        "Linear recursion with a scalar base"),
    "recursive_fibonacci_general": Case("""
local function f(n: integer): integer if n < 2 then return n end return f(n-1)+f(n-2) end
return f(20)
""", lambda: recursive_reference(1, 20), recursive_reference(1, 20),
        "Two-branch scalar recursion"),
    "recursive_ternary_general": Case("""
local function f(n: integer): integer if 0 >= n then return 1 end return f(n-1)+f(n-1)+f(n-1) end
return f(8)
""", lambda: recursive_reference(2, 8), recursive_reference(2, 8),
        "Three-branch recursion with reversed comparison"),
    "record_layout_one": Case("""
local function make(v: integer): table return {a=v} end
local total: integer=0 for i=1,5000 do local r: table=make(i) total=total+r.a end return total
""", lambda: record_reference(1), record_reference(1), "One-field record layout"),
    "record_layout_three": Case("""
local function make(v: integer): table return {a=v,b=v+1,c=v+2} end
local total: integer=0 for i=1,5000 do local r: table=make(i) total=total+r.a end return total
""", lambda: record_reference(3), record_reference(3), "Three-field record layout"),
    "record_layout_five": Case("""
local function make(v: integer): table return {e=v+4,c=v+2,a=v,b=v+1,d=v+3} end
local total: integer=0 for i=1,5000 do local r: table=make(i) total=total+r.a end return total
""", lambda: record_reference(5), record_reference(5),
        "Five-field reordered record layout"),
}

_base.CASES.update(CASES_040)
CASES = _base.CASES

GENERALITY_GROUPS = {
    "pure_scalar_expression_dag": (
        "scalar_dag_product", "scalar_dag_polynomial", "scalar_dag_split",
    ),
    "nested_numeric_region": (
        "nested_reduction_product", "nested_reduction_rows", "nested_reduction_modulo",
    ),
    "sparse_integer_boolean_region": (
        "sparse_slots", "sparse_visited", "sparse_occupancy",
    ),
    "scalar_call_entry": (
        "recursive_linear_general", "recursive_fibonacci_general",
        "recursive_ternary_general",
    ),
    "fresh_record_layout": (
        "record_layout_one", "record_layout_three", "record_layout_five",
    ),
}
assert all(len(set(cases)) >= 3 for cases in GENERALITY_GROUPS.values())
FOCUSED_CASES = tuple(case for cases in GENERALITY_GROUPS.values() for case in cases)


if __name__ == "__main__":
    if "--case" not in sys.argv:
        for name in FOCUSED_CASES:
            sys.argv.extend(("--case", name))
    main()
