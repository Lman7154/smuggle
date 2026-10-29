"""Lossless result export.

The console transcript (``smuggle.printing``) is a human PREVIEW and may truncate long
responses. For the record of a real run, write the complete, untruncated outcomes to JSONL
here: every field is preserved and invisible/encoding code points are escaped with
``ensure_ascii=True`` so the exact smuggled bytes are reproducible from the file. One JSON
object per line; works for single-turn ``VariantOutcome`` and multi-turn ``MultiTurnOutcome``.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Iterable


def _rows(outcomes: Any) -> list[Any]:
    if hasattr(outcomes, "outcomes"):          # SmuggleResult
        return list(outcomes.outcomes)
    return list(outcomes)


def to_record(o: Any) -> dict:
    """Convert one outcome (dataclass) to a JSON-ready dict with all fields intact."""
    if dataclasses.is_dataclass(o):
        d = dataclasses.asdict(o)
    else:
        d = dict(o)
    # tuples -> lists for clean JSON; values are preserved exactly.
    return json.loads(json.dumps(d, ensure_ascii=True, default=str))


def save_jsonl(outcomes: Iterable[Any], path: str | Path) -> int:
    """Write outcomes to ``path`` as JSONL (one record per line), losslessly. Returns the
    number of records written. Invisible/encoding characters are \\uXXXX-escaped so the exact
    bytes are recoverable; nothing is truncated."""
    rows = _rows(outcomes)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for o in rows:
            fh.write(json.dumps(to_record(o), ensure_ascii=True) + "\n")
    return len(rows)


def load_jsonl(path: str | Path) -> list[dict]:
    """Read back a JSONL export as a list of dicts (for analysis/reporting)."""
    p = Path(path)
    out: list[dict] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
