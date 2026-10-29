"""Proposal §4.3.1 trigger-word detection.

For every harmful seed prompt, content words are ranked by their Llama Guard
P(unsafe) saliency.  The proposal's *hierarchical removal* decision is then
applied exactly: test every single word, then every pair, then every triple in
that ranked order.  The first subset that flips the guardrail to safe is the
trigger set.  If no subset of size at most three flips, all content words are
used as an explicitly recorded fallback.

Saliency is used only to make the exhaustive, size-first search deterministic;
it is not substituted for the proposal's pair/triple tests.  Prompts are scored
as ``user`` input by :class:`smuggle.guardrail.Guardrail`.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Sequence

from .guardrail import Guardrail

try:  # Use PyRIT's content-word stoplist; do not maintain a second list.
    from pyrit.converter.text_selection_strategy import DEFAULT_CONTENT_STOPWORDS as _STOP
except Exception:  # pragma: no cover
    _STOP = frozenset()

_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _remove(text: str, words: Sequence[str]) -> str:
    """Remove every case-insensitive occurrence of each selected content word."""
    out = text
    for word in words:
        out = re.sub(r"\b" + re.escape(word) + r"\b", " ", out, flags=re.IGNORECASE)
    return _norm_ws(out)


def content_words(text: str, *, min_len: int = 3, stopwords=_STOP) -> list[str]:
    """Return distinct candidate content words in first-occurrence order."""
    stop = {word.lower() for word in stopwords}
    out: list[str] = []
    seen: set[str] = set()
    for word in _WORD_RE.findall(text):
        lowered = word.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        letters = re.sub(r"[^A-Za-z]", "", word)
        if len(letters) >= min_len and lowered not in stop:
            out.append(word)
    return out


@dataclass(frozen=True)
class TriggerResult:
    seed_id: str
    trigger_words: tuple[str, ...]
    ranked_words: tuple[str, ...]
    removal_depth: int                 # 1, 2, 3; 0 means all-content fallback
    used_fallback: bool
    base_score: float                  # P(unsafe) for the original user prompt


class TriggerWordDetector:
    """Guardrail-saliency ranking plus proposal-defined hierarchical removal.

    ``max_words`` exists only for an explicitly declared pilot/ablation.  The
    proposal-default is three, because it defines single, pair, and triple
    removal.  A lower value is rejected to avoid silently changing that method.
    """

    def __init__(self, guardrail: Guardrail, *, max_words: int = 3,
                 min_word_length: int = 3, min_saliency: float = 1e-3) -> None:
        if max_words != 3:
            raise ValueError("The proposal-defined trigger procedure requires max_words=3.")
        self._g = guardrail
        self._min_len = int(min_word_length)
        self._min_sal = float(min_saliency)
        # seed ids are not globally unique in ad-hoc callers; cache actual text too.
        self._cache: dict[tuple[str, str], TriggerResult] = {}

    async def detect(self, seed_id: str, text: str) -> TriggerResult:
        key = (seed_id, text)
        if key in self._cache:
            return self._cache[key]

        base = await self._g.unsafe_score(text)
        candidates = content_words(text, min_len=self._min_len)
        if base <= self._g.threshold or not candidates:
            result = TriggerResult(seed_id, tuple(candidates), tuple(candidates), 0, True, base)
            self._cache[key] = result
            return result

        # First rank individual candidates by graded probability drop.
        saliency: list[tuple[str, float]] = []
        for word in candidates:
            score = await self._g.unsafe_score(_remove(text, (word,)))
            saliency.append((word, base - score))
        saliency.sort(key=lambda item: item[1], reverse=True)
        ranked = tuple(word for word, score in saliency if score > self._min_sal)
        # Do not discard zero/negative-saliency words: interactions can still flip.
        ranked = ranked + tuple(word for word, _ in saliency if word not in ranked)

        # Exhaustive within the proposal's single/pair/triple hierarchy.  The
        # saliency order supplies deterministic ordering within each same-size set.
        for size in (1, 2, 3):
            if len(ranked) < size:
                continue
            for subset in itertools.combinations(ranked, size):
                if await self._g.unsafe_score(_remove(text, subset)) <= self._g.threshold:
                    result = TriggerResult(seed_id, subset, ranked, size, False, base)
                    self._cache[key] = result
                    return result

        result = TriggerResult(seed_id, tuple(candidates), ranked, 0, True, base)
        self._cache[key] = result
        return result
