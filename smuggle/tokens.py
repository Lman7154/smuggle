"""Token-space fragmentation metric (analysis layer, net-new).

Quantifies how far an encoding shifts the *token surface* a model or guardrail sees.
A byte-level BPE tokenizer turns the clean prompt into a few word / sub-word tokens, but
the encoded prompt into many byte-fallback tokens that barely overlap the originals. That
shift is the mechanism behind external-filter bypass (the safety-bearing token IDs the
classifier keys on are simply no longer present). This module reports it as a number so it
can sit in the result tables beside imperceptibility, guardrail bypass, and ASR.

IMPORTANT — this is a *reference* measurement, not the target's own tokenizer. Hosted and
Ollama models rarely expose their tokenizer, so we use a fixed, well-known byte-level BPE
tokenizer (default: tiktoken ``o200k_base``, the GPT-4o family vocab). The absolute token
IDs differ per model, but the comparative shift - inflation and near-zero overlap - is
robust across byte-level BPE tokenizers (verified: o200k and GPT-2 agree on the direction
and magnitude), so it is a valid *relative* measure across families and variants. The
tokenizer name is recorded in every record. Override with the ``SMUGGLE_TOKENIZER`` env var
(any tiktoken encoding name) or by passing ``encoder=`` explicitly.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Optional


@lru_cache(maxsize=4)
def _load_named_encoder(name: str):
    """Load and cache a tiktoken encoding by name. Raises if tiktoken/the vocab is absent."""
    import tiktoken
    return tiktoken.get_encoding(name)


def _default_encoder() -> tuple[Any, str]:
    """Return ``(encoder, name)`` for the configured default, or ``(None, "unavailable")``
    when tiktoken (or its vocab download) is not available - so measurement degrades to a
    recorded ``None`` instead of crashing a run."""
    name = os.environ.get("SMUGGLE_TOKENIZER", "o200k_base")
    try:
        return _load_named_encoder(name), name
    except Exception:
        return None, "unavailable"


def token_fragmentation(
    clean: str, encoded: str, *, encoder: Any = None, tokenizer_name: str = ""
) -> Optional[dict]:
    """Measure the token-space shift from ``clean`` to its ``encoded`` form.

    Returns a dict (or ``None`` if no tokenizer is available) with:

    - ``tokenizer``      : the reference tokenizer used (for provenance in the record)
    - ``clean_tokens``   : token count of the clean prompt
    - ``encoded_tokens`` : token count of the encoded prompt
    - ``shared_ids``     : number of distinct token IDs present in BOTH
    - ``retained``       : shared_ids / distinct clean IDs  (1.0 = unchanged, 0.0 = fully shifted)
    - ``inflation``      : encoded_tokens / clean_tokens     (how many x the sequence grew)

    A big ``inflation`` with ``retained`` near 0 is the signature of a whole-prompt invisible
    encoding: the guardrail's safety-bearing tokens are gone and replaced by byte fragments.
    """
    if encoder is None:
        encoder, tokenizer_name = _default_encoder()
    if encoder is None:
        return None
    try:
        clean_ids = encoder.encode(clean)
        encoded_ids = encoder.encode(encoded)
    except Exception:
        return None
    clean_set, encoded_set = set(clean_ids), set(encoded_ids)
    shared = len(clean_set & encoded_set)
    clean_n = len(clean_ids)
    return {
        "tokenizer": tokenizer_name or getattr(encoder, "name", "custom"),
        "clean_tokens": clean_n,
        "encoded_tokens": len(encoded_ids),
        "shared_ids": shared,
        "retained": (shared / len(clean_set)) if clean_set else 0.0,
        "inflation": (len(encoded_ids) / clean_n) if clean_n else 0.0,
    }
