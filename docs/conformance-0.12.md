# LuaPyre 0.12 exact-conformance tranche

LuaPyre 0.12 expands the checksum-pinned Lua 5.5.1 release gate from the original `bwcoercion.lua` baseline to seven unchanged upstream files:

- `bwcoercion.lua`
- `pm.lua`
- `tpack.lua`
- `vararg.lua`
- `bitwise.lua`
- `math.lua`
- `utf8.lua`

The authoritative suite remains `lua-5.5.1-tests.tar.gz` with SHA-256:

```text
da07b543872dc0bb2ff12aabd0c248578d78df3eb6b67efdc537a46d455c7f31
```

`tools/official_551.py --baseline` is the permanent release gate. Candidate files are not copied, patched, skipped, or marked as expected failures; promotion means the upstream file runs unchanged in an isolated LuaPyre runtime using the production file-loader/output capability path.

## Semantics covered by this tranche

The 0.12 work closes a set of exact Lua 5.5.1 differences exposed while promoting the six new files. Focused regressions cover:

- Lua numeral-string coercion for arithmetic and bitwise operators, including wide hexadecimal and hexadecimal-float edge cases
- extreme logical shifts and signed 64-bit integer boundaries
- named-vararg mutability, `n` handling, const behavior, capture behavior, and return-slot semantics
- Lua-compatible `string.gsub` capture/replacement diagnostics
- `string.pack`, `packsize`, and `unpack` option, alignment, integral-size, and overflow diagnostics
- integer floor-division and power boundary behavior
- UTF-8 offset behavior for incomplete sequences, bounds checks, empty ranges, and iterator controls
- stable observable `collectgarbage("count")` across the non-allocating named-vararg path used by the official suite
- Lua-compatible infinity-to-integer diagnostics for bitwise operations, including the `math.huge` field attribution required by `math.lua`

The last 0.12 blocker was `math.huge << 1`: a duplicated bitwise fast path bypassed the generic integer-conversion diagnostic and therefore omitted Lua's `field 'huge'` context. The dispatch path now preserves that context while retaining the ordinary `number has no integer representation` diagnostic for other non-integral floats.

## Validation

Normal CI runs the complete pytest/differential suite on Python 3.13 and 3.14 and verifies that the Lupa oracle is actually Lua 5.5.

The official conformance workflow separately downloads and verifies the pinned Lua 5.5.1 archive, classifies the 34 upstream Lua files, and runs every file in `BASELINE_FILES`. A release candidate is acceptable only when both the Python-version matrix and the exact official baseline are green.

The complete upstream suite is still intentionally larger than the current gate. Many remaining files rely on Lua's internal C test API, `debug`, `io`, `os`, native-module behavior, allocator details, or stress/resource assumptions that are not part of LuaPyre's default sandbox. Other non-gated files remain useful sources of genuine semantic work for later tranches.
