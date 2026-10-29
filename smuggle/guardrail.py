"""Guardrail used to its full potential (Meta Llama Guard model card).

Instead of a bare safe/unsafe label, we read the FIRST-TOKEN probability of the
"unsafe" class - the exact method Meta documents for turning Llama Guard into a
classifier score. That graded signal is what lets trigger detection RANK words by
how much each contributes to the unsafe verdict (TextFooler / "Not All Tokens Are
Created Equal"), rather than flipping a binary trip-wire.

The classifier (model + endpoint) is injected - nothing hardcoded. Swap the model
to swap guardrails; any OpenAI-compatible classifier that emits a safe/unsafe
first token works, and a plain PyRIT TrueFalseScorer can be used as a binary-only
fallback.

A prompt is judged in the USER role (it is user input), which is the correct
frame for trigger detection - not the AGENT role PyRIT's LlamaGuardScorer defaults
to (that frames the text as an assistant reply and skews the verdict).
"""
from __future__ import annotations

import asyncio
import math
from typing import Any


class Guardrail:
    def __init__(self, *, client: Any = None, model: str | None = None,
                 role: str = "user", threshold: float = 0.5,
                 binary_scorer: Any = None, unsafe_is_true: bool = True,
                 unsafe_token: str = "unsafe", safe_token: str = "safe") -> None:
        self._client = client
        self._model = model
        self._role = role
        self.threshold = float(threshold)
        self._binary = binary_scorer
        self._unsafe_is_true = unsafe_is_true
        # The verdict tokens the classifier emits. Llama Guard writes "safe"/"unsafe";
        # ShieldGemma and Granite Guardian write "No"/"Yes" (Yes == unsafe). Swapping the
        # guardrail is then just swapping the model name and these two tokens - nothing is
        # hardcoded to Llama Guard's vocabulary.
        self._unsafe_token = unsafe_token
        self._safe_token = safe_token

    async def unsafe_score(self, text: str) -> float:
        """Graded P(unsafe) in [0, 1] (Meta's first-token-probability method)."""
        if not text or not text.strip():
            return 0.0
        if self._client is not None and self._model is not None:
            return await asyncio.to_thread(self._score_logprob, text)
        if self._binary is not None:
            scores = await self._binary.score_text_async(text)
            if not scores:
                return 0.0
            raw = scores[0].get_value()
            # A float-scale scorer already returns a graded value in [0, 1]; keep it. Only a
            # genuine bool is mapped to 0/1. (Previously bool(raw) turned 0.3 into unsafe=1.0.)
            if isinstance(raw, bool):
                val = 1.0 if raw else 0.0
            else:
                val = float(raw)
            return val if self._unsafe_is_true else (1.0 - val)
        raise ValueError("Guardrail needs either (client+model) or a binary_scorer.")

    async def is_unsafe(self, text: str) -> bool:
        return (await self.unsafe_score(text)) > self.threshold

    def _score_logprob(self, text: str) -> float:
        r = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": self._role, "content": text}],
            max_tokens=1, temperature=0, logprobs=True, top_logprobs=20,
        )
        content = r.choices[0].logprobs.content if r.choices[0].logprobs else None
        u_tok, s_tok = self._unsafe_token.strip().lower(), self._safe_token.strip().lower()
        if not content:
            txt = (r.choices[0].message.content or "").strip().lower()
            return 1.0 if txt.startswith(u_tok) else 0.0
        probs: dict[str, float] = {}
        for t in content[0].top_logprobs:
            key = t.token.strip().lower()
            if key in (u_tok, s_tok) and key not in probs:
                probs[key] = math.exp(t.logprob)
        u, s = probs.get(u_tok, 0.0), probs.get(s_tok, 0.0)
        if u + s == 0.0:
            tok = content[0].token.strip().lower()
            return 1.0 if tok.startswith(u_tok) else 0.0
        return u / (u + s)


GuardrailAdapter = Guardrail  # backwards-compatible alias
