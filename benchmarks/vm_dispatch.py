from __future__ import annotations

import gc
import random
import sys
from statistics import median
from time import perf_counter_ns

from luapyre.bytecode import Ins, Op

EARLY = [
    Ins(Op.LOADK), Ins(Op.MOVE), Ins(Op.GETUPVAL),
    Ins(Op.ADD_I), Ins(Op.NOT), Ins(Op.JMPIFNOT),
] * 3

LATE = [
    Ins(Op.VARARG), Ins(Op.TBC), Ins(Op.CLOSE), Ins(Op.CHECKNIL),
    Ins(Op.RETURN), Ins(Op.GUARD), Ins(Op.HALT),
] * 3

UNIFORM = [Ins(op) for op in Op]

WEIGHTED = (
    [Ins(Op.LOADK)] * 15
    + [Ins(Op.MOVE)] * 20
    + [Ins(Op.LOCAL)] * 5
    + [Ins(Op.GETUPVAL)] * 5
    + [Ins(Op.GETTABLE)] * 8
    + [Ins(Op.SETTABLE)] * 5
    + [Ins(Op.ADD_I)] * 15
    + [Ins(Op.NOT)] * 3
    + [Ins(Op.EQ)] * 4
    + [Ins(Op.JMPIFNOT)] * 8
    + [Ins(Op.FORLOOP)] * 5
    + [Ins(Op.CALL)] * 4
    + [Ins(Op.RETURN)] * 2
    + [Ins(Op.HALT)]
)


def _build_if_enum():
    lines = [
        "def run(code, repeats):",
        "    score = 0",
        "    for _ in range(repeats):",
        "        for ins in code:",
        "            op = ins.op",
    ]
    for index, op in enumerate(Op, 1):
        keyword = "if" if index == 1 else "elif"
        lines.append(f"            {keyword} op is Op.{op.name}: score += {index}")
    lines.append("    return score")
    namespace = {"Op": Op}
    exec("\n".join(lines), namespace)
    return namespace["run"]


def _build_if_literal():
    lines = [
        "def run(code, repeats):",
        "    score = 0",
        "    for _ in range(repeats):",
        "        for ins in code:",
        "            op = ins.op",
    ]
    for index, _op in enumerate(Op, 1):
        keyword = "if" if index == 1 else "elif"
        lines.append(f"            {keyword} op == {index}: score += {index}")
    lines.append("    return score")
    namespace = {}
    exec("\n".join(lines), namespace)
    return namespace["run"]


def _build_match_enum():
    lines = [
        "def run(code, repeats):",
        "    score = 0",
        "    for _ in range(repeats):",
        "        for ins in code:",
        "            op = ins.op",
        "            match op:",
    ]
    for index, op in enumerate(Op, 1):
        lines.append(f"                case Op.{op.name}: score += {index}")
    lines.append("    return score")
    namespace = {"Op": Op}
    exec("\n".join(lines), namespace)
    return namespace["run"]


def _build_match_literal():
    lines = [
        "def run(code, repeats):",
        "    score = 0",
        "    for _ in range(repeats):",
        "        for ins in code:",
        "            op = ins.op",
        "            match op:",
    ]
    for index, _op in enumerate(Op, 1):
        lines.append(f"                case {index}: score += {index}")
    lines.append("    return score")
    namespace = {}
    exec("\n".join(lines), namespace)
    return namespace["run"]


def _handler(value):
    def handle(state, _ins):
        state[0] += value
    return handle


DICT_HANDLERS = {op: _handler(int(op)) for op in Op}
LIST_HANDLERS = [None] * (max(Op) + 1)
for _op, _fn in DICT_HANDLERS.items():
    LIST_HANDLERS[_op] = _fn


def _run_dict(code, repeats):
    state = [0]
    handlers = DICT_HANDLERS
    for _ in range(repeats):
        for ins in code:
            handlers[ins.op](state, ins)
    return state[0]


def _run_list(code, repeats):
    state = [0]
    handlers = LIST_HANDLERS
    for _ in range(repeats):
        for ins in code:
            handlers[ins.op](state, ins)
    return state[0]


RUNNERS = [
    ("if enum is", _build_if_enum()),
    ("if literal ==", _build_if_literal()),
    ("match enum", _build_match_enum()),
    ("match literal", _build_match_literal()),
    ("dict handler", _run_dict),
    ("list handler", _run_list),
]


def bench(label, code, *, target_dispatches=400_000, rounds=7):
    repeats = max(1, target_dispatches // len(code))
    dispatches = repeats * len(code)

    expected = {name: fn(code, 2) for name, fn in RUNNERS}
    assert len(set(expected.values())) == 1, expected

    timings = {name: [] for name, _ in RUNNERS}
    for round_index in range(rounds):
        order = RUNNERS[:]
        random.Random(20260911 + round_index).shuffle(order)
        for name, fn in order:
            gc.collect()
            start = perf_counter_ns()
            fn(code, repeats)
            elapsed = perf_counter_ns() - start
            timings[name].append(elapsed / dispatches)

    medians = {name: median(values) for name, values in timings.items()}
    baseline = medians["if enum is"]
    print(f"\n{label}: {dispatches:,} dispatches/round")
    for name, _ in RUNNERS:
        value = medians[name]
        print(f"{name:14s} {value:9.1f} ns/dispatch  {value / baseline:5.2f}x current")


if __name__ == "__main__":
    print(sys.version.replace("\n", " "))
    bench("early/common", EARLY)
    bench("late-op-heavy", LATE)
    bench("uniform", UNIFORM)
    bench("weighted", WEIGHTED)
