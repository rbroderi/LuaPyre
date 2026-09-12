from __future__ import annotations

from copy import deepcopy

from .binary_chunks import (
    BinaryChunkError,
    P_EXTRAARG,
    P_NEWTABLE,
    P_RETURN,
    P_TAILCALL,
    _Translator,
    _b,
    _decode_puc_chunk,
    _op,
)


_POS_K = 15
_POS_B = 16
_MASK_B = 0xFF << _POS_B


def _normalise_compiler_pairs(proto) -> None:
    """Normalize PUC compiler pairs before lowering to LuaPyre IR.

    Lua 5.5 always emits EXTRAARG after NEWTABLE; LuaPyre ignores allocation
    hints, so setting NEWTABLE's k bit lets the core translator consume the
    mandatory companion without changing observable table semantics.

    A source-level tail return is emitted as TAILCALL followed by an
    unreachable RETURN B=0.  The LuaPyre TAILCALL already terminates the frame;
    rewrite only that immediately-following dead RETURN to a zero-result return
    so it remains a valid (but unreachable) translated instruction. Reachable
    open RETURN instructions keep their strict multi-result validation.
    """
    code = proto.code
    for pc, word in enumerate(code):
        opcode = _op(word)
        if opcode == P_NEWTABLE:
            if pc + 1 >= len(code) or _op(code[pc + 1]) != P_EXTRAARG:
                raise BinaryChunkError("NEWTABLE without mandatory EXTRAARG")
            code[pc] = word | (1 << _POS_K)
        elif opcode == P_TAILCALL and pc + 1 < len(code):
            following = code[pc + 1]
            if _op(following) == P_RETURN and _b(following) == 0:
                code[pc + 1] = (following & ~_MASK_B) | (1 << _POS_B)
    for child in proto.children:
        _normalise_compiler_pairs(child)


def load_puc55_chunk(data: bytes):
    proto = _decode_puc_chunk(data)
    _normalise_compiler_pairs(proto)
    return _Translator(proto).translate()
