"""Offline tests for the structural injector set (proposal 4.3: zero-width + bidi).

No models/network: these assert the exact Unicode code points the structural family
injects, and that a narrower injected charset (declared ablation) is honoured and logged.
"""
from __future__ import annotations

import asyncio

from smuggle.engine import (
    _StructuralInjectionConverter,
    _STRUCTURAL_INJECTORS,
    _ZW_INJECTORS,
)

_ZW = {0x200B, 0x200C, 0x200D}
_BIDI_EMB_OVR = set(range(0x202A, 0x202F))   # U+202A-U+202E
_BIDI_ISO = set(range(0x2066, 0x206A))       # U+2066-U+2069


def _injected_codepoints(original: str, mutated: str) -> set[int]:
    return {ord(ch) for ch in mutated if ch not in original}


def test_default_structural_set_covers_zw_and_bidi() -> None:
    cps = {ord(ch) for ch in _STRUCTURAL_INJECTORS}
    assert _ZW <= cps, "zero-width controls missing"
    assert _BIDI_EMB_OVR <= cps, "bidi embedding/override U+202A-U+202E missing"
    assert _BIDI_ISO <= cps, "bidi isolate U+2066-U+2069 missing"
    # exactly these three groups, nothing stray
    assert cps == _ZW | _BIDI_EMB_OVR | _BIDI_ISO


def test_converter_injects_from_the_structural_set() -> None:
    conv = _StructuralInjectionConverter(rng_seed=0)
    out = asyncio.run(conv.convert_async(prompt="abcdef")).output_text
    injected = _injected_codepoints("abcdef", out)
    assert injected, "expected structural controls to be inserted between characters"
    assert injected <= (_ZW | _BIDI_EMB_OVR | _BIDI_ISO)


def test_zero_width_only_ablation_is_honoured_and_excludes_bidi() -> None:
    conv = _StructuralInjectionConverter(rng_seed=1, charset=_ZW_INJECTORS)
    out = asyncio.run(conv.convert_async(prompt="abcdef")).output_text
    injected = _injected_codepoints("abcdef", out)
    assert injected <= _ZW
    assert not (injected & (_BIDI_EMB_OVR | _BIDI_ISO)), "ablation must not inject bidi controls"


def test_injected_charset_is_stored_exactly() -> None:
    conv = _StructuralInjectionConverter(rng_seed=0, charset=["\u200d", "\u202e"])
    assert conv._charset == ["\u200d", "\u202e"]
    # and it only ever injects from that set
    out = asyncio.run(conv.convert_async(prompt="abcdef")).output_text
    assert _injected_codepoints("abcdef", out) <= {0x200D, 0x202E}


def test_empty_charset_is_rejected() -> None:
    try:
        _StructuralInjectionConverter(rng_seed=0, charset=[])
    except ValueError:
        return
    raise AssertionError("empty structural charset should raise ValueError")
