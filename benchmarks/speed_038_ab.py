"""Measure the Python/AST-only 0.38 performance tranche.

The inherited 0.36/0.37 probes remain controls.  The focused default cases
cover pure recursive bases, scalar materialized calls, nested dense regions,
and record construction. Run unchanged against baseline and candidate trees.
"""
from __future__ import annotations

import sys

import speed_037_ab as _base
from speed_037_ab import *  # noqa: F401,F403


def scalar_internal_binary():
    total = 0
    for value in range(1, 2001):
        scratch = [value]
        total += scratch[0] + value + 1
    return total


def table_mix_nested():
    values = [None]
    for value in range(1, 6001):
        values.append((value * 17) % 1009)
    total = 0
    for round_ in range(1, 5):
        for index in range(1, 6001):
            value = (values[index] * 33 + index + round_) % 10007
            values[index] = value
            total = (total + value) % 1000000007
    return total


CASES_038 = {
    "scalar_internal_binary": Case("""
local function combine(left: integer, right: integer): integer
    local scratch: table = {left}
    return scratch[1] + right
end
local total: integer = 0
for i = 1, 2000 do total = total + combine(i, i + 1) end
return total
""", scalar_internal_binary, 4004000,
        "Two-scalar internal calls through a materialized child"),
    "table_mix_nested": Case("""
local n: integer = 6000
local values: table = {}
for i = 1, n do values[i] = (i * 17) % 1009 end
local total: integer = 0
for round = 1, 4 do
    for i = 1, n do
        local value: integer = values[i]
        value = (value * 33 + i + round) % 10007
        values[i] = value
        total = (total + value) % 1000000007
    end
end
return total
""", table_mix_nested, 119882313,
        "Nested-loop dense extent and primitive-write proof"),
}

_base.CASES.update(CASES_038)
CASES = _base.CASES


FOCUSED_CASES = (
    "recursive_linear",
    "recursive_balanced",
    "scalar_internal_binary",
    "record_continuations",
    "dense_read",
    "dense_alias_write",
    "table_mix_nested",
)


if __name__ == "__main__":
    if "--case" not in sys.argv:
        for name in FOCUSED_CASES:
            sys.argv.extend(("--case", name))
    main()
