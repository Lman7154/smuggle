"""Offline behaviour + signature tests for the decoupled adaptive Unicode search.

Task 1.5 / Requirements 2.2, 2.3, 2.4, 2.5.

No live model and no network: the PyRIT send path (``smuggle.experiments.PromptSendingAttack``)
is monkeypatched with a recording fake whose ``execute_async`` captures the ``objective`` and the
``prepended_conversation`` kwarg (default ``None``). The guardrail is a ``Guardrail`` subclass that
returns a low score so every candidate counts as bypassing and the delivery path actually runs; the
judge returns ``False`` so the search never early-stops and exercises every send. This mirrors the
``FakeGuardrail`` shape in ``tests/test_triggers.py`` (``unsafe_score`` + ``threshold``).
"""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

from pyrit.models import Message

from smuggle import experiments
from smuggle.experiments import run_adaptive_unicode_search_async
from smuggle.guardrail import Guardrail

_PROMPT = "sample benign request"


# --------------------------------------------------------------------------- #
# Offline fakes (no model, no network)                                        #
# --------------------------------------------------------------------------- #
class RecordingTarget:
    """Fake objective target. Every send made through the (monkeypatched)
    ``PromptSendingAttack`` bound to this target is appended to ``sends`` as a
    ``SimpleNamespace(objective=..., prepended_conversation=...)``."""

    def __init__(self) -> None:
        self.sends: list[SimpleNamespace] = []


class _StubResponse:
    def get_value(self) -> str:
        # Non-empty so runner._response_text(result) is truthy and the search proceeds to scoring.
        return "non-empty stub response"


class _StubResult:
    def __init__(self) -> None:
        self.last_response = _StubResponse()


class FakePromptSendingAttack:
    """Monkeypatch replacement for ``smuggle.experiments.PromptSendingAttack``.

    The search builds it with ``objective_target=target`` (our RecordingTarget) and calls
    ``execute_async(objective=...)`` with ``prepended_conversation=...`` ONLY when a prepend is
    injected; otherwise the kwarg is omitted, so the default ``None`` is what gets recorded.
    """

    def __init__(self, *, objective_target: RecordingTarget) -> None:
        self._target = objective_target

    async def execute_async(self, *, objective, prepended_conversation=None):
        self._target.sends.append(
            SimpleNamespace(objective=objective, prepended_conversation=prepended_conversation)
        )
        return _StubResult()


class FakeGuardrail(Guardrail):
    """A ``Guardrail`` instance (so the search uses it directly, never wrapping it in
    ``Guardrail(binary_scorer=...)``) whose ``unsafe_score`` is a low constant, so every candidate
    bypasses (``unsafe_score <= threshold``) and the send path is exercised. The point is to drive
    delivery, not to measure."""

    def __init__(self, score: float = 0.0, threshold: float = 0.5) -> None:
        self.threshold = float(threshold)
        self._score = float(score)
        self.calls: list[str] = []

    async def unsafe_score(self, text: str) -> float:
        self.calls.append(text)
        return self._score


class _FalseScore:
    score_rationale = "stub: objective not achieved"

    def get_value(self) -> bool:
        return False


class FakeJudge:
    """Returns one ``False`` score so the search never early-stops (so every eligible candidate is
    delivered) and carries a rationale string, matching what runner._score reads."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def score_text_async(self, response, objective=None):
        self.calls.append((response, objective))
        return [_FalseScore()]


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
_OMIT = object()


def _run_search(target: RecordingTarget, *, prepend=_OMIT):
    """Run the search offline. When ``prepend`` is omitted it is NOT passed, so the real default
    (``None``) is exercised (neither ``categories`` nor ``prepend`` supplied)."""
    kwargs = dict(
        targets=target,
        prompts=[_PROMPT],
        guardrail=FakeGuardrail(),
        judge=FakeJudge(),
        families=("homoglyph",),
        candidate_pool=2,
        target_budget=2,
    )
    if prepend is not _OMIT:
        kwargs["prepend"] = prepend
    return asyncio.run(run_adaptive_unicode_search_async(**kwargs))


# --------------------------------------------------------------------------- #
# Behaviour tests                                                             #
# --------------------------------------------------------------------------- #
def test_completes_without_categories_or_prepend(monkeypatch):
    """Assertions 1-3: with NEITHER categories NOR prepend, the search completes, every send
    carries ``prepended_conversation=None``, and the outcome's category is ``""``."""
    target = RecordingTarget()
    monkeypatch.setattr(experiments, "PromptSendingAttack", FakePromptSendingAttack)

    outcomes, errors = _run_search(target)

    # Assertion 1: it completed (returned a single outcome) with no offline-path errors.
    assert errors == []
    assert len(outcomes) == 1

    # Assertion 2: sends actually happened and each used no prepended conversation.
    assert target.sends, "expected at least one target send"
    assert all(send.prepended_conversation is None for send in target.sends)

    # Assertion 3: the harm-category attribute is the empty default.
    assert outcomes[0].category == ""


def test_signature_drops_categories_and_defaults_prepend_to_none():
    """Assertions 4-5: ``categories`` is gone from the search signature; ``prepend`` exists with
    default ``None``."""
    sig = inspect.signature(run_adaptive_unicode_search_async)

    # Assertion 4
    assert "categories" not in sig.parameters

    # Assertion 5
    assert "prepend" in sig.parameters
    assert sig.parameters["prepend"].default is None


def test_injected_prepend_delivered_on_every_candidate(monkeypatch):
    """Assertion 6: when a prepend is injected, every recorded send carries the prepended
    conversation built from it (a non-empty list of PyRIT Messages)."""
    target = RecordingTarget()
    monkeypatch.setattr(experiments, "PromptSendingAttack", FakePromptSendingAttack)

    injected = [("user", "benign context")]
    outcomes, errors = _run_search(target, prepend=injected)

    assert errors == []
    assert len(outcomes) == 1
    assert target.sends, "expected at least one target send"
    for send in target.sends:
        assert isinstance(send.prepended_conversation, list)
        assert send.prepended_conversation, "prepend must be delivered on every candidate"
        assert all(isinstance(message, Message) for message in send.prepended_conversation)
