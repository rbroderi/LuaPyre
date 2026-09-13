from __future__ import annotations

from dataclasses import dataclass

import pytest

from luapyre import LuaFunction, LuaRuntime


Scalar = int | float | str


@dataclass
class Point:
    x: int
    y: int
    label: str


@dataclass
class Payload:
    values: list[Scalar]
    tags: set[str]
    counts: dict[str, int]


def test_python_collections_are_available_as_lua_tables():
    lua = LuaRuntime()
    lua.set("numbers", [10, 20, 12])
    lua.set("allowed", {"fox", "rabbit"})
    lua.set("weights", {"nick": 20, "judy": 22})

    assert lua.execute(
        "return numbers[1] + numbers[2] + numbers[3], "
        "allowed.fox, allowed.wolf, weights.nick + weights.judy"
    ) == (42, True, None, 42)


def test_python_facing_results_support_typed_containers():
    lua = LuaRuntime()

    assert lua.execute_python("return {1, 2.5, 'three'}") == [1, 2.5, "three"]
    assert lua.execute_python(
        "return {one = 1, two = 2}",
        return_type=dict[str, int],
    ) == {"one": 1, "two": 2}
    assert lua.execute_python(
        "return {fox = true, rabbit = true}",
        return_type=set[str],
    ) == {"fox", "rabbit"}
    assert lua.execute_python(
        "return {1, 2.5, 'three'}",
        return_type=list[Scalar],
    ) == [1, 2.5, "three"]
    assert lua.execute_python(
        "return {fox = 1, [2] = 'two'}",
        return_type=dict[Scalar],
    ) == {"fox": 1, 2: "two"}

    lua.execute("answer = {value = 42}")
    assert lua.get_python("answer", return_type=dict[str, int]) == {"value": 42}


def test_dataclasses_round_trip_through_lua_tables():
    lua = LuaRuntime()
    lua.set("point", Point(20, 22, "start"))

    result = lua.execute_python(
        "return {x = point.x + 1, y = point.y + 2, label = point.label .. '!'}",
        return_type=Point,
    )

    assert result == Point(21, 24, "start!")


def test_dataclass_container_fields_and_numeric_sets_round_trip():
    lua = LuaRuntime()
    lua.set(
        "payload",
        Payload([1, 2.5, "three"], {"ready"}, {"items": 3}),
    )
    lua.set("numbers", {1, 2, 4})

    payload, numbers = lua.execute_python(
        "payload.tags.done = true; payload.counts.items = 4; "
        "return payload, numbers",
        return_type=tuple[Payload, set[int]],
    )

    assert payload == Payload(
        [1, 2.5, "three"],
        {"ready", "done"},
        {"items": 4},
    )
    assert numbers == {1, 2, 4}


@pytest.mark.parametrize("jit", [False, True])
def test_lua_functions_are_callable_from_python(jit):
    lua = LuaRuntime(jit=jit)
    add = lua.execute_python("return function(a, b) return a + b end")

    assert isinstance(add, LuaFunction)
    assert add(20, 22, return_type=int) == 42

    lua.execute("function join(a, b) return a .. ':' .. b end")
    assert lua.function("join")("Nick", "Judy", return_type=str) == "Nick:Judy"
    assert lua.call("join", "Finn", "Bell", return_type=str) == "Finn:Bell"


def test_explicit_python_functions_receive_converted_values():
    lua = LuaRuntime()

    def move(point: Point, delta: list[int]) -> Point:
        return Point(
            point.x + delta[0],
            point.y + delta[1],
            point.label,
        )

    lua.set("move", move)
    result = lua.execute_python(
        "return move({x = 20, y = 20, label = 'office'}, {1, 2})",
        return_type=Point,
    )

    assert result == Point(21, 22, "office")


def test_multiple_returns_convert_independently():
    lua = LuaRuntime()
    result = lua.execute_python(
        "return 42, 'answer', {1, 2}",
        return_type=tuple[int, str, list[int]],
    )
    assert result == (42, "answer", [1, 2])


def test_functions_and_cycles_cannot_cross_runtime_boundaries():
    first = LuaRuntime()
    second = LuaRuntime()
    function = first.execute_python("return function() return 1 end")

    with pytest.raises(ValueError, match="cross between runtimes"):
        second.set("foreign", function)

    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="cyclic Python"):
        first.set("cyclic", cyclic)
