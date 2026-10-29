"""Offline unit tests for proposal-defined trigger-word detection."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from smuggle.triggers import TriggerWordDetector


class FakeGuardrail:
    """Deterministic async guardrail with no model or network dependency."""

    threshold = 0.5

    def __init__(self, score: Callable[[str], float]) -> None:
        self._score = score
        self.calls: list[str] = []

    async def unsafe_score(self, text: str) -> float:
        self.calls.append(text)
        return self._score(text)


def words_present(text: str) -> set[str]:
    return set(text.lower().split())


def test_detects_single_word_flip() -> None:
    guardrail = FakeGuardrail(lambda text: 0.0 if "alpha" not in words_present(text) else 1.0)

    result = asyncio.run(TriggerWordDetector(guardrail).detect("single", "alpha bravo charlie"))

    assert result.trigger_words == ("alpha",)
    assert result.removal_depth == 1
    assert not result.used_fallback


def test_detects_non_prefix_pair_flip() -> None:
    def score(text: str) -> float:
        words = words_present(text)
        if "bravo" not in words and "delta" not in words:
            return 0.0
        return {"alpha": 0.6, "bravo": 0.7, "charlie": 0.8, "delta": 0.9}.get(
            next(iter({"alpha", "bravo", "charlie", "delta"} - words), ""), 1.0
        )

    result = asyncio.run(
        TriggerWordDetector(FakeGuardrail(score)).detect("pair", "alpha bravo charlie delta")
    )

    assert result.ranked_words == ("alpha", "bravo", "charlie", "delta")
    assert result.trigger_words == ("bravo", "delta")
    assert result.removal_depth == 2
    assert not result.used_fallback


def test_detects_triple_flip() -> None:
    def score(text: str) -> float:
        words = words_present(text)
        if {"alpha", "charlie", "delta"}.isdisjoint(words):
            return 0.0
        missing = {"alpha", "bravo", "charlie", "delta"} - words
        return 1.0 - 0.1 * len(missing)

    result = asyncio.run(
        TriggerWordDetector(FakeGuardrail(score)).detect("triple", "alpha bravo charlie delta")
    )

    assert result.trigger_words == ("alpha", "charlie", "delta")
    assert result.removal_depth == 3
    assert not result.used_fallback


def test_falls_back_to_all_content_words_when_no_subset_flips() -> None:
    result = asyncio.run(
        TriggerWordDetector(FakeGuardrail(lambda _text: 1.0)).detect(
            "fallback", "alpha bravo charlie delta"
        )
    )

    assert result.trigger_words == ("alpha", "bravo", "charlie", "delta")
    assert result.removal_depth == 0
    assert result.used_fallback


def test_safe_input_uses_fallback_without_removal_scores() -> None:
    guardrail = FakeGuardrail(lambda _text: 0.0)

    result = asyncio.run(TriggerWordDetector(guardrail).detect("safe", "ordinary benign discussion"))

    assert result.trigger_words == ("ordinary", "benign", "discussion")
    assert result.removal_depth == 0
    assert result.used_fallback
    assert guardrail.calls == ["ordinary benign discussion"]


def test_cache_key_includes_text_when_seed_id_is_reused() -> None:
    def score(text: str) -> float:
        words = words_present(text)
        return 0.0 if "alpha" not in words and "bravo" not in words else 1.0

    guardrail = FakeGuardrail(score)
    detector = TriggerWordDetector(guardrail)

    first = asyncio.run(detector.detect("reused-id", "alpha task"))
    second = asyncio.run(detector.detect("reused-id", "bravo task"))

    assert first.trigger_words == ("alpha",)
    assert second.trigger_words == ("bravo",)
    assert first is not second
    assert guardrail.calls.count("alpha task") == 1
    assert guardrail.calls.count("bravo task") == 1
