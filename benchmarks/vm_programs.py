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

_N_BODY = """
local PI = 3.141592653589793
local SOLAR_MASS = 4.0 * PI * PI
local DAYS_PER_YEAR = 365.24
local bodies = {
    {x = 0.0, y = 0.0, z = 0.0, vx = 0.0, vy = 0.0, vz = 0.0, mass = SOLAR_MASS},
    {x = 4.841431442464721, y = -1.1603200440274284, z = -0.10362204447112311,
     vx = 0.001660076642744037 * DAYS_PER_YEAR,
     vy = 0.007699011184197404 * DAYS_PER_YEAR,
     vz = -0.0000690460016972063 * DAYS_PER_YEAR,
     mass = 0.0009547919384243266 * SOLAR_MASS},
    {x = 8.34336671824458, y = 4.124798564124305, z = -0.4035234171143214,
     vx = -0.002767425107268624 * DAYS_PER_YEAR,
     vy = 0.004998528012349172 * DAYS_PER_YEAR,
     vz = 0.00002304172975737639 * DAYS_PER_YEAR,
     mass = 0.0002858859806661308 * SOLAR_MASS},
    {x = 12.894369562139131, y = -15.111151401698631, z = -0.22330757889265573,
     vx = 0.002964601375647616 * DAYS_PER_YEAR,
     vy = 0.0023784717395948095 * DAYS_PER_YEAR,
     vz = -0.000029658956854023756 * DAYS_PER_YEAR,
     mass = 0.00004366244043351563 * SOLAR_MASS},
    {x = 15.379697114850917, y = -25.919314609987964, z = 0.17925877295037118,
     vx = 0.0026806777249038932 * DAYS_PER_YEAR,
     vy = 0.001628241700382423 * DAYS_PER_YEAR,
     vz = -0.00009515922545197159 * DAYS_PER_YEAR,
     mass = 0.000051513890204661145 * SOLAR_MASS},
}

local px, py, pz = 0.0, 0.0, 0.0
for i = 1, #bodies do
    local b = bodies[i]
    px = px + b.vx * b.mass
    py = py + b.vy * b.mass
    pz = pz + b.vz * b.mass
end
bodies[1].vx = -px / SOLAR_MASS
bodies[1].vy = -py / SOLAR_MASS
bodies[1].vz = -pz / SOLAR_MASS

for step = 1, 1000 do
    for i = 1, #bodies - 1 do
        local bi = bodies[i]
        for j = i + 1, #bodies do
            local bj = bodies[j]
            local dx = bi.x - bj.x
            local dy = bi.y - bj.y
            local dz = bi.z - bj.z
            local distance2 = dx * dx + dy * dy + dz * dz
            local magnitude = 0.01 / (distance2 * math.sqrt(distance2))
            bi.vx = bi.vx - dx * bj.mass * magnitude
            bi.vy = bi.vy - dy * bj.mass * magnitude
            bi.vz = bi.vz - dz * bj.mass * magnitude
            bj.vx = bj.vx + dx * bi.mass * magnitude
            bj.vy = bj.vy + dy * bi.mass * magnitude
            bj.vz = bj.vz + dz * bi.mass * magnitude
        end
    end
    for i = 1, #bodies do
        local b = bodies[i]
        b.x = b.x + 0.01 * b.vx
        b.y = b.y + 0.01 * b.vy
        b.z = b.z + 0.01 * b.vz
    end
end

local energy = 0.0
for i = 1, #bodies do
    local bi = bodies[i]
    energy = energy + 0.5 * bi.mass *
        (bi.vx * bi.vx + bi.vy * bi.vy + bi.vz * bi.vz)
    for j = i + 1, #bodies do
        local bj = bodies[j]
        local dx = bi.x - bj.x
        local dy = bi.y - bj.y
        local dz = bi.z - bj.z
        energy = energy - bi.mass * bj.mass / math.sqrt(dx * dx + dy * dy + dz * dz)
    end
end
return energy
"""

_MANDELBROT = """
local size = 40
local checksum = 0
for y = 0, size - 1 do
    local ci = 2.0 * y / size - 1.0
    local byte = 0
    local bits = 0
    for x = 0, size - 1 do
        local cr = 2.0 * x / size - 1.5
        local zr, zi, tr, ti = 0.0, 0.0, 0.0, 0.0
        local iterations = 0
        while iterations < 50 and tr + ti <= 4.0 do
            zi = 2.0 * zr * zi + ci
            zr = tr - ti + cr
            tr = zr * zr
            ti = zi * zi
            iterations = iterations + 1
        end
        byte = byte * 2 + ((tr + ti <= 4.0) and 1 or 0)
        bits = bits + 1
        if bits == 8 then
            checksum = (checksum * 131 + byte) % 2147483647
            byte, bits = 0, 0
        end
    end
end
return checksum
"""

_FANNKUCH_REDUX = """
local n = 7
local perm1, count = {}, {}
for i = 1, n do perm1[i] = i - 1 end
local max_flips, checksum, sign, r = 0, 0, 1, n
while true do
    while r ~= 1 do
        count[r] = r
        r = r - 1
    end
    local perm = {}
    for i = 1, n do perm[i] = perm1[i] end
    local flips = 0
    local k = perm[1]
    while k ~= 0 do
        local left, right = 1, k + 1
        while left < right do
            perm[left], perm[right] = perm[right], perm[left]
            left, right = left + 1, right - 1
        end
        flips = flips + 1
        k = perm[1]
    end
    checksum = checksum + sign * flips
    if flips > max_flips then max_flips = flips end

    while true do
        if r == n then return checksum * 100 + max_flips end
        local first = perm1[1]
        for i = 1, r do perm1[i] = perm1[i + 1] end
        perm1[r + 1] = first
        count[r + 1] = count[r + 1] - 1
        if count[r + 1] > 0 then
            sign = -sign
            break
        end
        r = r + 1
    end
end
"""

_FASTA = """
local alu = "GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGGGAGGCCGAGGCGGGCGGATCACCTGAGGTCAGGAGTTCGAGACCAGCCTGGCCAACATGGTGAAACCCCGTCTCTACTAAAAATACAAAAATTAGCCGGGCGTGGTGGCGCGCGCCTGTAATCCCAGCTACTCGGGAGGCTGAGGCAGGAGAATCGCTTGAACCCGGGAGGCGGAGGTTGCAGTGAGCCGAGATCGCGCCACTGCACTCCAGCCTGGGCGACAGAGCGAGACTCCGTCTCAAAAA"
local iub = {
    {char = "a", code = 97, p = 0.27}, {char = "c", code = 99, p = 0.12},
    {char = "g", code = 103, p = 0.12}, {char = "t", code = 116, p = 0.27},
    {char = "B", code = 66, p = 0.02}, {char = "D", code = 68, p = 0.02},
    {char = "H", code = 72, p = 0.02}, {char = "K", code = 75, p = 0.02},
    {char = "M", code = 77, p = 0.02}, {char = "N", code = 78, p = 0.02},
    {char = "R", code = 82, p = 0.02}, {char = "S", code = 83, p = 0.02},
    {char = "V", code = 86, p = 0.02}, {char = "W", code = 87, p = 0.02},
    {char = "Y", code = 89, p = 0.02},
}
local homo = {
    {char = "a", code = 97, p = 0.3029549426680},
    {char = "c", code = 99, p = 0.1979883004921},
    {char = "g", code = 103, p = 0.1975473066391},
    {char = "t", code = 116, p = 0.3015094502008},
}
local function cumulative(dist)
    local total = 0.0
    for i = 1, #dist do total = total + dist[i].p; dist[i].p = total end
end
cumulative(iub); cumulative(homo)
local seed = 42
local function pick(dist)
    seed = (seed * 3877 + 29573) % 139968
    local value = seed / 139968
    for i = 1, #dist do
        if value < dist[i].p then return dist[i] end
    end
    return dist[#dist]
end
local chunks, checksum = {}, 0
for i = 1, 1000 do
    local at = ((i - 1) % #alu) + 1
    local ch = string.sub(alu, at, at)
    chunks[#chunks + 1] = ch
    checksum = checksum + string.byte(ch)
end
for i = 1, 1500 do
    local item = pick(iub); chunks[#chunks + 1] = item.char; checksum = checksum + item.code
end
for i = 1, 2500 do
    local item = pick(homo); chunks[#chunks + 1] = item.char; checksum = checksum + item.code
end
local output = table.concat(chunks)
return #output * 1000000 + checksum
"""

_K_NUCLEOTIDE = """
local motif = "GGTATTTTAATTTATAGTGGTAAGATATTAAGATAATATTTGGTGGTAGTTTTAATGTGTAA"
local sequence = string.rep(motif, 20)
local sizes = {1, 2, 3, 4, 6, 12, 18}
local checksum = 0
for s = 1, #sizes do
    local k = sizes[s]
    local counts = {}
    for i = 1, #sequence - k + 1 do
        local key = string.sub(sequence, i, i + k - 1)
        counts[key] = (counts[key] or 0) + 1
    end
    checksum = (checksum * 131 + (counts[string.sub(sequence, 1, k)] or 0)) % 2147483647
    checksum = (checksum * 131 + (counts[string.sub(sequence, 7, 6 + k)] or 0)) % 2147483647
end
return checksum
"""

_REVERSE_COMPLEMENT = """
local motif = "ACGTUMRWSYKVHDBN"
local sequence = string.rep(motif, 250)
local complement = {
    A = "T", C = "G", G = "C", T = "A", U = "A", M = "K", R = "Y",
    W = "W", S = "S", Y = "R", K = "M", V = "B", H = "D", D = "H",
    B = "V", N = "N",
}
local output = {}
for i = #sequence, 1, -1 do
    output[#output + 1] = complement[string.sub(sequence, i, i)]
end
local reversed = table.concat(output)
local checksum = 0
for i = 1, #reversed do
    checksum = (checksum + i * string.byte(reversed, i)) % 2147483647
end
return checksum
"""


LANGUAGE_BENCHMARKS = (
    "n_body",
    "mandelbrot",
    "spectral_norm",
    "fannkuch_redux",
    "binary_trees",
    "fasta",
    "k_nucleotide",
    "reverse_complement",
)


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
        group="language",
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
        group="language",
        luapyre_source=_TYPED_SPECTRAL_NORM,
        abs_tolerance=1e-12,
    ),
    "n_body": Workload(
        _N_BODY,
        -0.16908760523460614,
        group="language",
        abs_tolerance=1e-12,
    ),
    "mandelbrot": Workload(_MANDELBROT, 759728559, group="language"),
    "fannkuch_redux": Workload(_FANNKUCH_REDUX, 22816, group="language"),
    "fasta": Workload(_FASTA, 5000481113, group="language"),
    "k_nucleotide": Workload(_K_NUCLEOTIDE, 1510947559, group="language"),
    "reverse_complement": Workload(
        _REVERSE_COMPLEMENT,
        607629500,
        group="language",
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
