from __future__ import annotations

from luapyre import LuaRuntime
from luapyre.bytecode import Ins, Op, Proto, UpvalueDesc
from luapyre.typed_ir import IRValueKind, TypedIRCompiler


def test_ir_recognizes_root_environment_constant_key_and_dses_key_load():
    proto = Proto(
        "root",
        code=[
            Ins(Op.LOADK, 1, 0),
            Ins(Op.GETTABLE, 2, 0, 1),
        ],
        constants=[b"answer"],
        register_count=3,
        env_reg=0,
        jit_fully_typed=True,
    )
    plan = TypedIRCompiler(proto).compile(
        (((0, proto.code[0]), (1, proto.code[1])),)
    )

    key_load = plan.instruction(0)
    table_get = plan.instruction(1)
    assert key_load is not None and key_load.dead_definition
    assert table_get is not None
    assert table_get.specialization == "global_get"
    assert table_get.value_for(1).kind is IRValueKind.CONSTANT
    assert plan.invariant_sites == (1,)
    assert plan.cache_sites == ()


def test_ir_virtualizes_env_upvalue_and_constant_key_together():
    proto = Proto(
        "child",
        code=[
            Ins(Op.GETUPVAL, 0, 0),
            Ins(Op.LOADK, 1, 0),
            Ins(Op.GETTABLE, 2, 0, 1),
        ],
        constants=[b"value"],
        upvalues=[UpvalueDesc("upvalue", 0, "_ENV")],
        register_count=3,
        jit_fully_typed=True,
    )
    plan = TypedIRCompiler(proto).compile(
        (((0, proto.code[0]), (1, proto.code[1]), (2, proto.code[2])),)
    )

    env_load = plan.instruction(0)
    key_load = plan.instruction(1)
    table_get = plan.instruction(2)
    assert env_load is not None and env_load.dead_definition
    assert key_load is not None and key_load.dead_definition
    assert table_get is not None and table_get.specialization == "global_get"
    assert table_get.value_for(0).kind is IRValueKind.UPVALUE
    assert table_get.value_for(0).is_environment
    assert plan.invariant_sites == (2,)


def test_global_load_is_not_hoisted_across_possible_table_mutation():
    proto = Proto(
        "root",
        code=[
            Ins(Op.LOADK, 1, 0),
            Ins(Op.GETTABLE, 2, 0, 1),
            Ins(Op.SETTABLE, 3, 4, 5),
        ],
        constants=[b"answer"],
        register_count=6,
        env_reg=0,
        jit_fully_typed=True,
    )
    plan = TypedIRCompiler(proto).compile(
        (((0, proto.code[0]), (1, proto.code[1]), (2, proto.code[2])),)
    )
    assert plan.invariant_sites == ()
    assert plan.cache_sites == (1,)


def test_ir_does_not_virtualize_constants_for_unconverted_arithmetic():
    proto = Proto(
        "arith",
        code=[
            Ins(Op.LOADK, 0, 0),
            Ins(Op.ADD_I, 2, 0, 1),
        ],
        constants=[7],
        register_count=3,
        jit_fully_typed=True,
    )
    plan = TypedIRCompiler(proto).compile(
        (((0, proto.code[0]), (1, proto.code[1])),)
    )
    assert not plan.instruction(0).dead_definition
    assert plan.instruction(1).value_for(0).kind is IRValueKind.REGISTER
    assert plan.instruction(1).result_type == "integer"


def test_typed_global_read_loop_runs_through_jit_ir_cache():
    runtime = LuaRuntime(jit_threshold=1, fuel=2_000_000)
    source = """-- luapyre: typed
global g: integer = 7
local total = 0
for i = 1, 5000 do
    local value: integer = g
    total = total + value
end
return total
"""
    proto = runtime.compile(source)
    assert Op.JFORLOOP in [ins.op for ins in proto.code]
    assert runtime.vm.run(proto) == 35000
    assert runtime.jit_stats.loop_compiles >= 1


def test_typed_constant_field_cache_observes_mutation_version():
    runtime = LuaRuntime(jit_threshold=1, fuel=2_000_000)
    source = """-- luapyre: typed
local t = {value = 1}
local total = 0
for i = 1, 200 do
    local value: integer = t.value
    total = total + value
    if i == 100 then
        t.value = 3
    end
end
return total
"""
    # 100 iterations see 1; iterations 101..200 see 3.
    assert runtime.execute(source) == 400
    assert runtime.jit_stats.loop_compiles >= 1


def _compiled_function_filenames(runtime: LuaRuntime) -> set[str]:
    filenames: set[str] = set()
    for _proto, compiled in runtime.vm.jit._function_cache.values():
        if compiled is not None:
            filenames.add(compiled.runner.__code__.co_filename)
    return filenames


def test_whole_function_ir_hoists_invariant_global_read():
    runtime = LuaRuntime(jit_threshold=1, fuel=2_000_000)
    source = """-- luapyre: typed
global g: integer = 5
local function sum_global(n: integer): integer
    local total = 0
    for i = 1, n do
        local value: integer = g
        total = total + value
    end
    return total
end
local result: integer = sum_global(2000)
return result
"""
    assert runtime.execute(source) == 10000
    assert "<luapyre-ir-function>" in _compiled_function_filenames(runtime)


def test_whole_function_ir_cache_observes_table_version_mutation():
    runtime = LuaRuntime(jit_threshold=1, fuel=2_000_000)
    source = """-- luapyre: typed
local function sum_field(t: table): integer
    local total = 0
    for i = 1, 200 do
        local value: integer = t.value
        total = total + value
        if i == 100 then
            t.value = 3
        end
    end
    return total
end
local t = {value = 1}
local result: integer = sum_field(t)
return result
"""
    assert runtime.execute(source) == 400
    assert "<luapyre-ir-function>" in _compiled_function_filenames(runtime)
