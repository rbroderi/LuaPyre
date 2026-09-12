from __future__ import annotations

from luapyre import LuaRuntime


def test_host_userdata_identity_roundtrip_survives_temporary_tables_and_gc():
    """Regression guard for the failure mode reported in Lupa GH-294.

    Opaque host objects must round-trip through Lua by identity even while
    temporary Lua objects are being created and collection cycles run.  A host
    reference must never be silently rebound to another live Python object.
    """

    lua = LuaRuntime()
    entities = [object(), object(), object()]

    def entity_for(index):
        return entities[index % len(entities)]

    def verify_entity(index, actual):
        expected = entities[index % len(entities)]
        assert actual is expected, (index, id(expected), id(actual))
        return True

    lua.expose("entity_for", entity_for)
    lua.expose("verify_entity", verify_entity)

    result = lua.execute(
        '''
local function echo(entity, event)
  return entity
end

for i = 0, 19999 do
  local expected = entity_for(i)
  local actual = echo(expected, {name = "update"})
  if not verify_entity(i, actual) then
    return false
  end
  if i % 97 == 0 then
    collectgarbage("collect")
  end
end

return true
'''
    )

    assert result is True


def test_distinct_host_userdata_remain_distinct_as_table_keys_across_gc():
    """Identity-keyed Lua table entries must not alias after collection."""

    lua = LuaRuntime()
    entities = [object(), object(), object()]

    def entity_for(index):
        return entities[index % len(entities)]

    lua.expose("entity_for", entity_for)

    result = lua.execute(
        '''
local keys = {entity_for(0), entity_for(1), entity_for(2)}
local seen = {}
for i = 1, 3 do
  seen[keys[i]] = i
end

for n = 1, 250 do
  local temporary = {n = n}
  if n % 17 == 0 then
    collectgarbage("collect")
  end
end

return seen[keys[1]], seen[keys[2]], seen[keys[3]],
       keys[1] ~= keys[2] and keys[2] ~= keys[3] and keys[1] ~= keys[3]
'''
    )

    assert result == (1, 2, 3, True)
