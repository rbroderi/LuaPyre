from __future__ import annotations

import math
import time

from .errors import LuaRuntimeError
from .table import LuaTable
from .values import MultiValue, i64
from .vm import HostFunction
from .stdlib_support import INT_MAX, INT_MIN, UINT_MASK, need_integer, need_number, to_integer


_MASK = (1 << 64) - 1


def _rotl(value: int, count: int) -> int:
    value &= _MASK
    return ((value << count) | (value >> (64 - count))) & _MASK


class _LuaRandom:
    """Lua 5.5's xoshiro256** generator for reproducible explicit seeds."""

    __slots__ = ("state",)

    def __init__(self):
        self.state = [0, 0xff, 0, 0]
        self.seed(time.time_ns(), id(self))

    def next_u64(self) -> int:
        s0, s1, s2, s3 = self.state
        result = (_rotl((s1 * 5) & _MASK, 7) * 9) & _MASK
        # Lua's implementation computes this from the original state[1],
        # before state[1] is XORed with the updated state[2].
        t = (s1 << 17) & _MASK
        s2 ^= s0
        s3 ^= s1
        s1 ^= s2
        s0 ^= s3
        s2 ^= t
        s3 = _rotl(s3, 45)
        self.state[:] = (s0 & _MASK, s1 & _MASK, s2 & _MASK, s3 & _MASK)
        return result

    def seed(self, x: int, y: int) -> tuple[int, int]:
        x &= _MASK
        y &= _MASK
        self.state[:] = (x, 0xff, y, 0)
        for _ in range(16):
            self.next_u64()
        return i64(x), i64(y)

    def project(self, random_value: int, maximum: int) -> int:
        limit = maximum
        shift = 1
        while (limit & (limit + 1)) != 0:
            limit |= limit >> shift
            shift *= 2
        value = random_value & limit
        while value > maximum:
            value = self.next_u64() & limit
        return value


def _number_result(value: float):
    if math.isfinite(value) and value.is_integer() and INT_MIN <= value <= INT_MAX:
        return int(value)
    return float(value)


def _libm_unary(function, value):
    value = float(need_number(value))
    try:
        return float(function(value))
    except ValueError:
        return math.nan
    except OverflowError:
        return math.copysign(math.inf, value)


def install_math_library(globals_table: LuaTable, vm) -> LuaTable:
    lib = LuaTable()
    rng = _LuaRandom()

    def register(name, fn):
        lib.rawset(name.encode("ascii"), HostFunction(fn, f"math.{name}"))

    def abs_fn(x):
        x = need_number(x, 1, "abs")
        if type(x) is int:
            return i64(-x) if x < 0 else x
        return abs(x)

    register("abs", abs_fn)
    register("acos", lambda x: _libm_unary(math.acos, x))
    register("asin", lambda x: _libm_unary(math.asin, x))

    def atan(y, x=1):
        return math.atan2(float(need_number(y, 1, "atan")), float(need_number(x, 2, "atan")))

    register("atan", atan)

    def ceil(x):
        x = need_number(x, 1, "ceil")
        if type(x) is int:
            return x
        if not math.isfinite(x):
            return x
        return _number_result(float(math.ceil(x)))

    def floor(x):
        x = need_number(x, 1, "floor")
        if type(x) is int:
            return x
        if not math.isfinite(x):
            return x
        return _number_result(float(math.floor(x)))

    register("ceil", ceil)
    register("cos", lambda x: _libm_unary(math.cos, x))
    register("deg", lambda x: float(need_number(x, 1, "deg")) * (180.0 / math.pi))

    def exp(x):
        x = float(need_number(x, 1, "exp"))
        try:
            return math.exp(x)
        except OverflowError:
            return math.inf

    register("exp", exp)
    register("floor", floor)

    def fmod(x, y):
        x = need_number(x, 1, "fmod")
        y = need_number(y, 2, "fmod")
        if type(x) is int and type(y) is int:
            if y == 0:
                raise LuaRuntimeError("bad argument #2 to 'fmod' (zero)")
            if y == -1:
                return 0
            quotient = abs(x) // abs(y)
            if (x < 0) != (y < 0):
                quotient = -quotient
            return i64(x - quotient * y)
        try:
            return math.fmod(float(x), float(y))
        except ValueError:
            return math.nan

    register("fmod", fmod)

    def frexp(x):
        mantissa, exponent = math.frexp(float(need_number(x, 1, "frexp")))
        return MultiValue((mantissa, exponent))

    register("frexp", frexp)

    def ldexp(m, e):
        m = float(need_number(m, 1, "ldexp"))
        e = need_integer(e, 2, "ldexp")
        try:
            return math.ldexp(m, e)
        except OverflowError:
            return math.copysign(math.inf, m)

    register("ldexp", ldexp)

    def log(x, base=None):
        x = float(need_number(x, 1, "log"))
        if x == 0.0:
            result = -math.inf
        elif x < 0.0:
            result = math.nan
        else:
            result = math.log(x)
        if base is None:
            return result
        base = float(need_number(base, 2, "log"))
        if base <= 0.0 or base == 1.0:
            try:
                return result / math.log(base)
            except (ValueError, ZeroDivisionError):
                return math.nan
        return result / math.log(base)

    register("log", log)

    def maximum(*values):
        if not values:
            raise LuaRuntimeError("bad argument #1 to 'max' (value expected)")
        best = values[0]
        for value in values[1:]:
            if vm.less_than_sync(best, value):
                best = value
        return best

    def minimum(*values):
        if not values:
            raise LuaRuntimeError("bad argument #1 to 'min' (value expected)")
        best = values[0]
        for value in values[1:]:
            if vm.less_than_sync(value, best):
                best = value
        return best

    register("max", maximum)
    register("min", minimum)

    def modf(x):
        x = need_number(x, 1, "modf")
        if type(x) is int:
            return MultiValue((x, 0.0))
        if math.isinf(x):
            return MultiValue((x, 0.0))
        if math.isnan(x):
            return MultiValue((math.nan, math.nan))
        integer = math.ceil(x) if x < 0 else math.floor(x)
        return MultiValue((_number_result(float(integer)), float(x - integer)))

    register("modf", modf)
    register("rad", lambda x: float(need_number(x, 1, "rad")) * (math.pi / 180.0))

    def random_fn(*args):
        random_value = rng.next_u64()
        if not args:
            return (random_value >> 11) * (2.0 ** -53)
        if len(args) == 1:
            upper = need_integer(args[0], 1, "random")
            if upper == 0:
                return i64(random_value)
            low = 1
        elif len(args) == 2:
            low = need_integer(args[0], 1, "random")
            upper = need_integer(args[1], 2, "random")
        else:
            raise LuaRuntimeError("wrong number of arguments to 'random'")
        if low > upper:
            raise LuaRuntimeError("bad argument #1 to 'random' (interval is empty)")
        width = ((upper & UINT_MASK) - (low & UINT_MASK)) & UINT_MASK
        projected = rng.project(random_value, width)
        return i64(projected + (low & UINT_MASK))

    def randomseed(*args):
        if not args:
            x = time.time_ns() & _MASK
            y = (time.perf_counter_ns() ^ id(rng)) & _MASK
        elif len(args) <= 2:
            x = need_integer(args[0], 1, "randomseed") & _MASK
            y = (need_integer(args[1], 2, "randomseed") if len(args) == 2 else 0) & _MASK
        else:
            raise LuaRuntimeError("wrong number of arguments to 'randomseed'")
        a, b = rng.seed(x, y)
        return MultiValue((a, b))

    register("random", random_fn)
    register("randomseed", randomseed)
    register("sin", lambda x: _libm_unary(math.sin, x))
    register("sqrt", lambda x: _libm_unary(math.sqrt, x))
    register("tan", lambda x: _libm_unary(math.tan, x))

    def tointeger(x):
        return to_integer(x)

    def type_fn(x):
        if type(x) is int:
            return b"integer"
        if type(x) is float:
            return b"float"
        return None

    def ult(m, n):
        return (need_integer(m, 1, "ult") & UINT_MASK) < (need_integer(n, 2, "ult") & UINT_MASK)

    register("tointeger", tointeger)
    register("type", type_fn)
    register("ult", ult)

    lib.rawset(b"pi", math.pi)
    lib.rawset(b"huge", math.inf)
    lib.rawset(b"maxinteger", INT_MAX)
    lib.rawset(b"mininteger", INT_MIN)
    globals_table.rawset(b"math", lib)
    return lib
