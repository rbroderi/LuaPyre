from __future__ import annotations

from dataclasses import dataclass
import gc
import math
import statistics
import sys
import time

from luapyre import LuaRuntime


@dataclass(frozen=True, slots=True)
class Workload:
    source: str
    expected: object
    group: str = "micro"
    luapyre_source: str | None = None
    abs_tolerance: float | None = None

    def source_for(self, dialect: str) -> str:
        if dialect == "luapyre" and self.luapyre_source is not None:
            return self.luapyre_source
        return self.source

    def validate(self, result: object) -> bool:
        if self.abs_tolerance is None:
            return result == self.expected
        if type(result) not in (int, float) or type(self.expected) not in (int, float):
            return False
        return math.isclose(
            float(result),
            float(self.expected),
            rel_tol=0.0,
            abs_tol=self.abs_tolerance,
        )


_ARITHMETIC = """
local s = 0
for i = 1, 30000 do
    s = s + i
end
return s
"""

_TYPED_ARITHMETIC = """-- luapyre: typed
local s: integer = 0
for i = 1, 30000 do
    s = s + i
end
return s
"""

_CALLS = """
local function bump(x)
    return x + 1
end
local s = 0
for i = 1, 8000 do
    s = bump(s)
end
return s
"""

_TYPED_CALLS = """-- luapyre: typed
local function bump(x: integer): integer
    return x + 1
end
local s: integer = 0
for i = 1, 8000 do
    s = bump(s)
end
return s
"""

_BRANCHES = """
local s = 0
for i = 1, 30000 do
    if i % 2 == 0 then
        s = s + i
    else
        s = s - 1
    end
end
return s
"""

_TYPED_BRANCHES = """-- luapyre: typed
local s: integer = 0
for i = 1, 30000 do
    if i % 2 == 0 then
        s = s + i
    else
        s = s - 1
    end
end
return s
"""

_GLOBAL_READ = """
g = 7
local s = 0
for i = 1, 30000 do
    local v = g
    s = s + v
end
return s
"""

_TYPED_GLOBAL_READ = """-- luapyre: typed
global g: integer
g = 7
local s: integer = 0
for i = 1, 30000 do
    local v: integer = g
    s = s + v
end
return s
"""

_CONST_FIELD = """
local t = {value = 7}
local s = 0
for i = 1, 30000 do
    local v = t.value
    s = s + v
end
return s
"""

_TYPED_CONST_FIELD = """-- luapyre: typed
local t = {value = 7}
local s: integer = 0
for i = 1, 30000 do
    local v: integer = t.value
    s = s + v
end
return s
"""

_FIB = """
local function fib(n)
    if n < 2 then
        return n
    end
    return fib(n - 1) + fib(n - 2)
end
return fib(20)
"""

_TYPED_FIB = """-- luapyre: typed
local function fib(n: integer): integer
    if n < 2 then
        return n
    end
    return fib(n - 1) + fib(n - 2)
end
return fib(20)
"""

_SIEVE = """
local n = 5000
local composite = {}
local count = 0
for p = 2, n do
    if not composite[p] then
        count = count + 1
        local multiple = p * p
        while multiple <= n do
            composite[multiple] = true
            multiple = multiple + p
        end
    end
end
return count
"""

_TYPED_SIEVE = """-- luapyre: typed
local n = 5000
local composite = {}
local count = 0
for p = 2, n do
    if not composite[p] then
        count = count + 1
        local multiple = p * p
        while multiple <= n do
            composite[multiple] = true
            multiple = multiple + p
        end
    end
end
return count
"""

_BINARY_TREES = """
local function make(depth)
    if depth == 0 then
        return {left = nil, right = nil}
    end
    return {left = make(depth - 1), right = make(depth - 1)}
end

local function check(node)
    if node.left == nil then
        return 1
    end
    return 1 + check(node.left) + check(node.right)
end

local total = 0
for i = 1, 30 do
    total = total + check(make(8))
end
return total
"""

_TYPED_BINARY_TREES = """-- luapyre: typed
local function make(depth: integer): table
    if depth == 0 then
        return {left = nil, right = nil}
    end
    return {left = make(depth - 1), right = make(depth - 1)}
end

local function check(node: table): integer
    if node.left == nil then
        return 1
    end
    return 1 + check(node.left) + check(node.right)
end

local total = 0
for i = 1, 30 do
    total = total + check(make(8))
end
return total
"""

_TABLE_MIX = """
local n = 6000
local t = {}
for i = 1, n do
    t[i] = (i * 17) % 1009
end
local s = 0
for round = 1, 4 do
    for i = 1, n do
        local v = t[i]
        v = (v * 33 + i + round) % 10007
        t[i] = v
        s = (s + v) % 1000000007
    end
end
return s
"""

_TYPED_TABLE_MIX = """-- luapyre: typed
local n = 6000
local t = {}
for i = 1, n do
    t[i] = (i * 17) % 1009
end
local s = 0
for round = 1, 4 do
    for i = 1, n do
        local v: integer = t[i]
        v = (v * 33 + i + round) % 10007
        t[i] = v
        s = (s + v) % 1000000007
    end
end
return s
"""

_STRING_BUILD = """
local s = ""
for i = 1, 3000 do
    if i % 2 == 0 then
        s = s .. "ab"
    else
        s = s .. "xyz"
    end
end
return #s
"""

_TYPED_STRING_BUILD = """-- luapyre: typed
local s = ""
for i = 1, 3000 do
    if i % 2 == 0 then
        s = s .. "ab"
    else
        s = s .. "xyz"
    end
end
return #s
"""

_SPECTRAL_NORM = """
local function A(i, j)
    return 1.0 / (((i + j) * (i + j + 1) / 2.0) + i + 1.0)
end

local function multiplyAv(x, n)
    local out = {}
    for i = 1, n do
        local s = 0.0
        for j = 1, n do
            s = s + A(i - 1, j - 1) * x[j]
        end
        out[i] = s
    end
    return out
end

local function multiplyAtv(x, n)
    local out = {}
    for i = 1, n do
        local s = 0.0
        for j = 1, n do
            s = s + A(j - 1, i - 1) * x[j]
        end
        out[i] = s
    end
    return out
end

local function multiplyAtAv(x, n)
    return multiplyAtv(multiplyAv(x, n), n)
end

local n = 30
local u = {}
for i = 1, n do
    u[i] = 1.0
end
local v = {}
for i = 1, 5 do
    v = multiplyAtAv(u, n)
    u = multiplyAtAv(v, n)
end
local vBv = 0.0
local vv = 0.0
for i = 1, n do
    vBv = vBv + u[i] * v[i]
    vv = vv + v[i] * v[i]
end
return math.sqrt(vBv / vv)
"""

_TYPED_SPECTRAL_NORM = """-- luapyre: typed
global math: table
local function A(i: integer, j: integer): float
    return 1.0 / (((i + j) * (i + j + 1) / 2.0) + i + 1.0)
end

local function multiplyAv(x: table, n: integer): table
    local out = {}
    for i = 1, n do
        local s = 0.0
        for j = 1, n do
            local xj: float = x[j]
            s = s + A(i - 1, j - 1) * xj
        end
        out[i] = s
    end
    return out
end

local function multiplyAtv(x: table, n: integer): table
    local out = {}
    for i = 1, n do
        local s = 0.0
        for j = 1, n do
            local xj: float = x[j]
            s = s + A(j - 1, i - 1) * xj
        end
        out[i] = s
    end
    return out
end

local function multiplyAtAv(x: table, n: integer): table
    return multiplyAtv(multiplyAv(x, n), n)
end

local n = 30
local u = {}
for i = 1, n do
    u[i] = 1.0
end
local v = {}
for i = 1, 5 do
    v = multiplyAtAv(u, n)
    u = multiplyAtAv(v, n)
end
local vBv = 0.0
local vv = 0.0
for i = 1, n do
    local ui: float = u[i]
    local vi: float = v[i]
    vBv = vBv + ui * vi
    vv = vv + vi * vi
end
return math.sqrt(vBv / vv)
"""


WORKLOADS: dict[str, Workload] = {
    "arithmetic": Workload(_ARITHMETIC, 450015000),
    "typed_arith": Workload(
        _ARITHMETIC,
        450015000,
        group="typed",
        luapyre_source=_TYPED_ARITHMETIC,
    ),
    "tables": Workload(
        """
        local t = {}
        for i = 1, 8000 do
            t[i] = i * 3
        end
        local s = 0
        for i = 1, 8000 do
            s = s + t[i]
        end
        return s
        """,
        96012000,
    ),
    "calls": Workload(_CALLS, 8000),
    "typed_calls": Workload(
        _CALLS,
        8000,
        group="typed",
        luapyre_source=_TYPED_CALLS,
    ),
    "branches": Workload(_BRANCHES, 225000000),
    "typed_branch": Workload(
        _BRANCHES,
        225000000,
        group="typed",
        luapyre_source=_TYPED_BRANCHES,
    ),
    "typed_global_read": Workload(
        _GLOBAL_READ,
        210000,
        group="typed",
        luapyre_source=_TYPED_GLOBAL_READ,
    ),
    "typed_const_field": Workload(
        _CONST_FIELD,
        210000,
        group="typed",
        luapyre_source=_TYPED_CONST_FIELD,
    ),
    "coroutines": Workload(
        """
        local function worker(n)
            local s = 0
            for i = 1, n do
                s = s + i
                if i % 10 == 0 then
                    coroutine.yield(s)
                end
            end
            return s
        end

        local co = coroutine.create(worker)
        local ok, value = coroutine.resume(co, 2000)
        while coroutine.status(co) ~= "dead" do
            ok, value = coroutine.resume(co)
        end
        return value
        """,
        2001000,
    ),
    "fib_recursive": Workload(
        _FIB,
        6765,
        group="algorithm",
        luapyre_source=_TYPED_FIB,
    ),
    "sieve": Workload(
        _SIEVE,
        669,
        group="algorithm",
        luapyre_source=_TYPED_SIEVE,
    ),
    "binary_trees": Workload(
        _BINARY_TREES,
        15330,
        group="algorithm",
        luapyre_source=_TYPED_BINARY_TREES,
    ),
    "table_mix": Workload(
        _TABLE_MIX,
        119882313,
        group="algorithm",
        luapyre_source=_TYPED_TABLE_MIX,
    ),
    "string_build": Workload(
        _STRING_BUILD,
        7500,
        group="algorithm",
        luapyre_source=_TYPED_STRING_BUILD,
    ),
    "spectral_norm": Workload(
        _SPECTRAL_NORM,
        1.274097380427096,
        group="algorithm",
        luapyre_source=_TYPED_SPECTRAL_NORM,
        abs_tolerance=1e-12,
    ),
}


def bench(workload: Workload, repeats: int = 7) -> tuple[float, float]:
    runtime = LuaRuntime(fuel=20_000_000)
    proto = runtime.compile(workload.source_for("luapyre"))

    for _ in range(2):
        result = runtime.vm.run(proto, fuel=20_000_000)
        if not workload.validate(result):
            raise AssertionError(f"expected {workload.expected!r}, got {result!r}")

    samples = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter_ns()
            result = runtime.vm.run(proto, fuel=20_000_000)
            elapsed = time.perf_counter_ns() - start
            if not workload.validate(result):
                raise AssertionError(f"expected {workload.expected!r}, got {result!r}")
            samples.append(elapsed / 1_000_000)
    finally:
        if was_enabled:
            gc.enable()

    return statistics.median(samples), min(samples)


def main() -> None:
    print(sys.version)
    for name, workload in WORKLOADS.items():
        median_ms, best_ms = bench(workload)
        print(
            f"{name:16s} group={workload.group:9s} "
            f"median={median_ms:9.3f} ms  best={best_ms:9.3f} ms"
        )


if __name__ == "__main__":
    main()
