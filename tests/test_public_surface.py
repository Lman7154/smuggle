"""Offline property tests for the ``smuggle`` package public surface.

No models/network: these assert import-time behaviour of the package namespace
only. Task 1.1 deleted nine shipped-CipherChat symbols (five from
``context_templates``, four from ``experiments``) and scrubbed them from
``smuggle.__init__``; this module proves they are no longer reachable from the
package surface.
"""
from __future__ import annotations

import pytest

import smuggle

# The nine CipherChat symbols removed from the package in task 1.1. None of them
# may be reachable from the ``smuggle`` package surface any more.
REMOVED_NAMES = (
    # removed from smuggle.context_templates
    "load_cipherchat_prepend",
    "load_cipherchat_category_bank",
    "DEFAULT_CIPHERCHAT_PREPEND_PATH",
    "DEFAULT_CIPHERCHAT_CATEGORY_PATH",
    "LLAMA_GUARD_TEXT_CATEGORIES",
    # removed from smuggle.experiments
    "make_cipherchat_prepend",
    "make_cipherchat_prepend_async",
    "make_category_cipherchat_prepend",
    "make_category_cipherchat_prepend_async",
)


@pytest.mark.parametrize("name", REMOVED_NAMES)
def test_removed_symbol_is_unimportable(name: str) -> None:
    """**Feature: crescendo-unicode-arm, Property 1: Removed symbols are unimportable**

    A removed CipherChat name must not be reachable on the package: attribute
    access raises ``AttributeError`` and a ``from smuggle import <name>``
    statement raises ``ImportError``.

    ``exec`` is used because it faithfully reproduces a real
    ``from smuggle import <name>`` statement, whose ``IMPORT_FROM`` opcode raises
    ``ImportError`` for a missing name. (``__import__("smuggle", fromlist=[name])``
    does NOT raise for a missing non-submodule name -- verified in the installed
    runtime -- so it would silently pass; ``exec`` is the correct check.)

    **Validates: Requirements 1.1, 1.2**
    """
    with pytest.raises(AttributeError):
        getattr(smuggle, name)

    with pytest.raises(ImportError):
        exec(f"from smuggle import {name}", {})


# ---------------------------------------------------------------------------
# Property 2: the public export list fully resolves.
#
# Task 1.1 removed the nine CipherChat names from the package surface. Every name
# still advertised in
# ``smuggle.__all__`` must resolve on the package to a present, non-``None`` object
# (Requirement 1.3), and a plain ``import smuggle`` must succeed without any removed
# CipherChat name reappearing on the surface. The count is intentionally NOT pinned:
# the invariant is resolution, not a magic number, so legitimate future exports do
# not break this test.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", smuggle.__all__)
def test_exported_name_resolves(name: str) -> None:
    """**Feature: crescendo-unicode-arm, Property 2: The public export list fully resolves**

    Every name advertised in ``smuggle.__all__`` must be a present, non-``None``
    attribute of the package. A listed name that is missing (``hasattr`` is False)
    or bound to ``None`` is a real export defect in ``smuggle/__init__.py``, not a
    test artefact -- the failure message names the offending symbol.

    **Validates: Requirements 1.3**
    """
    assert hasattr(smuggle, name), f"{name!r} is listed in __all__ but is not an attribute of the package"
    assert getattr(smuggle, name) is not None, f"{name!r} is listed in __all__ but resolves to None"


def test_public_surface_import_is_clean() -> None:
    """**Feature: crescendo-unicode-arm, Property 2: The public export list fully resolves**

    A fresh ``import smuggle`` succeeds and none of the nine removed CipherChat
    names reappear in ``smuggle.__all__``. This is a simple, deterministic proxy
    for "the package initialises with no CipherChat path access" (Requirement 1.4):
    the package imports cleanly and advertises no removed harmful-bank symbol. A
    heavier filesystem-access assertion is intentionally avoided as unnecessary and
    flaky.

    **Validates: Requirements 1.3**
    """
    import importlib

    mod = importlib.import_module("smuggle")
    assert mod is smuggle  # import succeeds and yields the package object

    leaked = set(mod.__all__).intersection(REMOVED_NAMES)
    assert not leaked, f"removed CipherChat names leaked back into __all__: {sorted(leaked)}"
