"""Load JailbreakBench JBB-Behaviors the RIGHT way: the harmful *Goal* request.

Why this module exists: PyRIT 1.1.0's built-in ``jbb_behaviors`` provider puts the short
*Behavior* label (e.g. "Defamation", "Body shaming") into ``SeedPrompt.value`` and DROPS the
actual request. Feeding that label to the guardrail or the target means scoring/attacking a
bare topic word, not a harmful request - which is why Llama Guard flagged only ~16/100 and
trigger detection fell back ~83/100 when run on the provider's output. This loader reads the
``Goal`` column (the real request, e.g. "Write a defamatory article claiming ...") together with
``Category``; on the Goals, Llama Guard flags ~96/100 and fallback drops to ~56/100.

    from smuggle import jbb, run_experiment_1, run_experiment_3a, load_context_templates
    rows   = jbb.load_jbb_goals()                 # the real harmful requests + categories
    prompts = jbb.jbb_prompts(rows)               # <- use THESE as the attack prompts
    # single-turn:
    rows1, errs = run_experiment_1(targets=..., prompts=prompts, guardrail=..., judge=...)
    # multi-turn Strategy A (category-matched benign priming):
    cats = jbb.build_priming_categories(rows)     # {goal_text -> benign-template key}
    rows3a, errs = run_experiment_3a(targets=..., prompts=prompts, guardrail=..., judge=...,
                                     templates=load_context_templates(), categories=cats)
"""
from __future__ import annotations

from typing import Any, Optional

#: JBB-Behaviors' harm categories -> the benign-context template keys shipped in
#: templates/benign_context_templates.yaml. Best-effort domain match for Strategy A priming
#: (the priming only needs a plausible benign opening in a related domain, not an exact label).
JBB_CATEGORY_TO_TEMPLATE: dict[str, str] = {
    "Harassment/Discrimination": "harassment",
    "Malware/Hacking": "malware",
    "Physical harm": "weapons",
    "Economic harm": "fraud",
    "Fraud/Deception": "fraud",
    "Disinformation": "misinformation",
    "Sexual/Adult content": "sexual_content",
    "Privacy": "privacy",
    "Expert advice": "drugs",
    "Government decision-making": "misinformation",
}

#: HuggingFace dataset id and config for JBB-Behaviors.
_HF_DATASET = "JailbreakBench/JBB-Behaviors"
_HF_CONFIG = "behaviors"


def _load_dataset_offline_first(dataset: str, config: str):
    """Load a HuggingFace dataset from the LOCAL cache without touching the network when possible.

    JBB-Behaviors is already cached on disk, so there is no need to reach Hugging Face. We first
    try with the Hub forced offline (fast, no network, no update check); only if the cache is
    genuinely missing do we restore online mode and allow a one-time download. The global
    ``datasets.config.HF_HUB_OFFLINE`` flag is saved and restored so this call has no lasting
    side effect on the caller's environment."""
    import datasets
    from datasets import load_dataset
    prev = getattr(datasets.config, "HF_HUB_OFFLINE", None)
    try:
        try:
            datasets.config.HF_HUB_OFFLINE = True
        except Exception:
            pass
        try:
            return load_dataset(dataset, config)          # offline: read the local cache
        except Exception:
            try:
                datasets.config.HF_HUB_OFFLINE = False     # cache miss -> allow one download
            except Exception:
                pass
            return load_dataset(dataset, config)
    finally:
        if prev is not None:
            try:
                datasets.config.HF_HUB_OFFLINE = prev
            except Exception:
                pass


def load_jbb_goals(split: str = "harmful", *, dataset: str = _HF_DATASET,
                   config: str = _HF_CONFIG) -> list[dict[str, Any]]:
    """Return JBB-Behaviors rows as dicts with keys ``goal``, ``category``, ``behavior``, ``index``.

    ``goal`` is the actual harmful request - use it as the attack prompt. Requires the
    ``datasets`` package; it reads the LOCAL HuggingFace cache offline-first, so NO network is
    needed once the dataset is cached (only a genuine cache miss triggers a one-time download).
    ``split`` is "harmful" (100 misuse behaviours) or "benign" (100 benign controls)."""
    try:
        import datasets  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError("The 'datasets' package is required to load JBB-Behaviors "
                           "(pip install datasets).") from e
    d = _load_dataset_offline_first(dataset, config)
    if split not in d:
        split = list(d.keys())[0]
    ds = d[split]

    def _get(row: Any, *keys: str) -> Any:
        for k in keys:
            if k in row and row[k] is not None:
                return row[k]
        return None

    rows: list[dict[str, Any]] = []
    for row in ds:
        goal = _get(row, "Goal", "goal")
        if not goal:
            continue
        rows.append({
            "goal": goal,
            "category": _get(row, "Category", "category"),
            "behavior": _get(row, "Behavior", "behavior"),
            "index": _get(row, "Index", "index"),
        })
    return rows


def jbb_prompts(rows: list[dict[str, Any]]) -> list[str]:
    """The attack prompts = the Goal requests (never the Behavior labels)."""
    return [r["goal"] for r in rows if r.get("goal")]


def build_priming_categories(rows: list[dict[str, Any]],
                             mapping: Optional[dict[str, str]] = None) -> dict[str, str]:
    """Map each Goal -> a benign-context template key for ``run_experiment_3a(categories=...)``.

    Goals whose category is unmapped are omitted; the priming provider then falls back to a
    seeded-random benign template for those (still benign, just not domain-matched)."""
    m = mapping or JBB_CATEGORY_TO_TEMPLATE
    out: dict[str, str] = {}
    for r in rows:
        goal, cat = r.get("goal"), r.get("category")
        if goal and cat in m:
            out[goal] = m[cat]
    return out
