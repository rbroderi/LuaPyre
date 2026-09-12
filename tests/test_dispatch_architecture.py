from __future__ import annotations

import inspect

from luapyre.binary_chunks import _PUC_TRANSLATORS, _Translator


def test_puc_translator_dispatch_table_covers_all_opcodes():
    assert set(_PUC_TRANSLATORS) == set(range(85))


def test_puc_translate_loop_uses_table_dispatch():
    source = inspect.getsource(_Translator.translate)
    assert "_PUC_TRANSLATORS" in source
    assert "elif opcode" not in source
