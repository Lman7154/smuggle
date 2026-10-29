"""Benign context templates for multi-turn Strategy A (context priming), wired to prompts
by harm category (H2).

The template CONTENT lives in ``templates/benign_context_templates.yaml`` (10 harm
categories x 4 hand-crafted benign exchanges). This module only loads that injected content
and turns it into a ``priming`` provider the experiment driver can pass straight to
``run_experiment_3a`` / ``run_multiturn_a``. Nothing is hardcoded here: the categories, the
per-prompt category mapping, and the YAML path are all supplied by the caller.

    from smuggle import context_templates as T
    from smuggle import run_experiment_3a
    tmpl = T.load_context_templates()                         # injected benign content
    cats = {objective_text: category for ...}                 # from JBB-Behaviors metadata
    rows, errs = run_experiment_3a(targets=..., prompts=..., guardrail=..., judge=...,
                                   templates=tmpl, categories=cats)
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Optional, Union

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

#: Default location of the injected benign-context content shipped with the package.
DEFAULT_TEMPLATES_PATH = Path(__file__).parent / "templates" / "benign_context_templates.yaml"

Context = list  # list[tuple[role, content]]


def load_context_templates(path: Union[str, Path] = DEFAULT_TEMPLATES_PATH) -> dict[str, list[Context]]:
    """Load the YAML into ``{category: [template, ...]}`` where each template is a list of
    ``(role, content)`` turns. The path is injected; swap it to supply different content."""
    if yaml is None:
        raise RuntimeError("PyYAML is required to load context templates (pip install pyyaml).")
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    out: dict[str, list[Context]] = {}
    for category, templates in (raw or {}).items():
        norm: list[Context] = []
        for t in templates:
            norm.append([(pair[0], pair[1]) for pair in t])
        out[category] = norm
    return out


def make_priming_provider(
    templates: dict[str, list[Context]],
    *,
    categories: Optional[Union[dict, Callable[[str], Optional[str]]]] = None,
    seed: int = 0,
    default_category: Optional[str] = None,
) -> Callable[[str], Context]:
    """Build a ``priming`` provider: ``provider(objective) -> [(role, content), ...]``.

    ``categories`` maps a prompt to its harm category - a dict (keyed by the objective text)
    or a callable ``objective -> category``; when a prompt's category is unknown the provider
    falls back to ``default_category`` or a deterministic pseudo-random category. Selection is
    seeded by ``(seed, objective)`` so it is reproducible and spreads prompts across the four
    templates of a category. The returned context is a copy of injected YAML content."""
    cats = list(templates.keys())
    if not cats:
        raise ValueError("No templates provided.")

    def category_of(objective: str) -> Optional[str]:
        if callable(categories):
            return categories(objective)
        if isinstance(categories, dict):
            return categories.get(objective)
        return None

    def provider(objective: str) -> Context:
        rng = random.Random(f"{seed}:{objective}")
        cat = category_of(objective) or default_category
        if cat not in templates:
            cat = rng.choice(cats)
        return list(rng.choice(templates[cat]))

    return provider
