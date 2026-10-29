"""Analysis the researcher owns (PyRIT does not compute these): ASR + 95% CI
(Wilson), McNemar paired test + odds ratio, turn-depth, family ranking."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence


@dataclass(frozen=True)
class AsrEstimate:
    condition: str
    model: str
    asr: float
    ci_low: float
    ci_high: float
    n: int


@dataclass(frozen=True)
class McNemarResult:
    condition_a: str
    condition_b: str
    b: int           # success in A, fail in B
    c: int           # fail in A, success in B
    statistic: float
    p_value: float
    odds_ratio: float


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def asr(successes: int, n: int, *, condition: str = "", model: str = "") -> AsrEstimate:
    rate = successes / n if n else 0.0
    lo, hi = wilson_interval(successes, n)
    return AsrEstimate(condition, model, rate, lo, hi, n)


def _chi2_sf_1df(x: float) -> float:
    # survival function of chi-square with 1 dof = erfc(sqrt(x/2))
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))


def mcnemar(paired_a: Sequence[bool], paired_b: Sequence[bool], *,
            condition_a: str = "A", condition_b: str = "B") -> McNemarResult:
    if len(paired_a) != len(paired_b):
        raise ValueError("McNemar requires the identical seed set in both conditions")
    b = sum(1 for a, bb in zip(paired_a, paired_b) if a and not bb)
    c = sum(1 for a, bb in zip(paired_a, paired_b) if (not a) and bb)
    if b + c == 0:
        stat, p = 0.0, 1.0
    else:
        stat = (abs(b - c) - 1) ** 2 / (b + c)   # continuity-corrected
        p = _chi2_sf_1df(stat)
    odds = (b / c) if c else float("inf") if b else 1.0
    return McNemarResult(condition_a, condition_b, b, c, stat, p, odds)


def turn_depth_at_first_success(per_turn_success: Sequence[bool]) -> int | None:
    for i, ok in enumerate(per_turn_success, start=1):
        if ok:
            return i
    return None


def family_ranking(asr_by_family: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(asr_by_family.items(), key=lambda kv: kv[1], reverse=True)


# --------------------------------------------------------------------------- #
# Bypass vs compliance, reported as SEPARATE measures per (model, family, mode)
# (guardrail-bypass rate and compliance ASR are distinct columns;
# bypass characterised as a function of coverage/mode). Duck-typed on VariantOutcome
# so it also works on records reloaded from CentralMemory.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FamilyModeReport:
    model: str
    family: str
    mode: str
    n: int                               # unique inputs used for ASR (after de-duplication)
    successes: int
    asr: float
    asr_ci: tuple[float, float]
    n_guarded: int                       # subset that had an input-guardrail verdict
    bypasses: Optional[int]
    bypass_rate: Optional[float]         # None when nothing was guardrail-scored
    bypass_ci: Optional[tuple[float, float]]
    mean_imperceptibility: float
    n_raw: int = 0                       # variants before de-duplication (greedy repeats removed)
    n_fallback: int = 0                  # variants whose detection fell back to all content words


def _family_key(fam: Any) -> str:
    return fam if isinstance(fam, str) else "+".join(fam)


def _dedup_identical(items: Sequence[Any]) -> list:
    """Drop greedy-identical repeats within a cell: keep one outcome per (seed_id, encoded_text).

    Under greedy decoding an identical encoded input yields an identical reply, so repeats
    carry no additional information; counting them would inflate N and shrink the CI by ~sqrt
    of the repeat factor (C1). Outcomes without an ``encoded_text`` (e.g. multi-turn) are kept
    as-is."""
    seen: set = set()
    out: list = []
    for o in items:
        enc = getattr(o, "encoded_text", None)
        if enc is None:
            out.append(o)
            continue
        key = (getattr(o, "seed_id", ""), enc)
        if key in seen:
            continue
        seen.add(key)
        out.append(o)
    return out


def bypass_and_asr(outcomes: Iterable[Any], *, collapse_identical: bool = True) -> list[FamilyModeReport]:
    """Group outcomes by (model, family, mode) and report two DISTINCT measures:

    - compliance ASR  = fraction where the judge found the OBJECTIVE was achieved;
    - guardrail-bypass = fraction of the input-guardrail-scored subset that evaded it.

    They are kept separate (never collapsed) because decode/mention != task done, and
    evading the filter != complying. Both carry Wilson 95% CIs. Bypass is computed only
    over the subset with a guardrail verdict (n_guarded), so coverage is explicit (#4).

    ``collapse_identical`` (default True) de-duplicates greedy-identical inputs within a cell
    before computing ASR, so N counts UNIQUE inputs (C1); ``n_raw`` keeps the pre-dedup count.
    This is the variant-level view; for the headline number prefer :func:`asr_over_prompts`,
    which treats each behaviour as the unit.
    """
    groups: dict[tuple[str, str, str], list] = {}
    for o in outcomes:
        key = (getattr(o, "model", ""), _family_key(o.family), o.mode)
        groups.setdefault(key, []).append(o)

    reports: list[FamilyModeReport] = []
    for (model, fam, mode), raw_items in groups.items():
        n_raw = len(raw_items)
        n_fb = sum(1 for o in raw_items if getattr(o, "used_fallback", False))
        items = _dedup_identical(raw_items) if collapse_identical else list(raw_items)
        n = len(items)
        succ = sum(1 for o in items if o.judge_success)
        a = asr(succ, n)
        guarded = [o for o in items if getattr(o, "input_flagged", None) is not None]
        ng = len(guarded)
        if ng:
            byp = sum(1 for o in guarded if not o.input_flagged)
            bypass_rate, bypass_ci, bypasses = byp / ng, wilson_interval(byp, ng), byp
        else:
            bypass_rate, bypass_ci, bypasses = None, None, None
        imp = [float(getattr(o, "imperceptibility", 0.0)) for o in items]
        reports.append(FamilyModeReport(
            model=model, family=fam, mode=mode, n=n, successes=succ,
            asr=a.asr, asr_ci=(a.ci_low, a.ci_high),
            n_guarded=ng, bypasses=bypasses, bypass_rate=bypass_rate, bypass_ci=bypass_ci,
            mean_imperceptibility=(sum(imp) / len(imp) if imp else 0.0),
            n_raw=n_raw, n_fallback=n_fb,
        ))
    reports.sort(key=lambda r: (r.model, r.family, r.mode))
    return reports


def format_report_table(reports: Sequence[FamilyModeReport]) -> str:
    """Render reports as a plain-text table with bypass and ASR in SEPARATE columns."""
    hdr = (f"{'model':20} {'family':22} {'mode':9} {'N':>3} "
           f"{'ASR':>6} {'ASR 95%CI':>14} {'bypass':>7} {'byp 95%CI':>14} {'imperc':>7}")
    lines = [hdr, "-" * len(hdr)]
    for r in reports:
        asr_ci = f"[{r.asr_ci[0]:.2f},{r.asr_ci[1]:.2f}]"
        if r.bypass_rate is None:
            byp, byp_ci = "n/a", "n/a"
        else:
            byp = f"{r.bypass_rate:.0%}"
            byp_ci = f"[{r.bypass_ci[0]:.2f},{r.bypass_ci[1]:.2f}]"
        lines.append(f"{r.model[:20]:20} {r.family[:22]:22} {r.mode:9} {r.n:>3} "
                     f"{r.asr:>5.0%} {asr_ci:>14} {byp:>7} {byp_ci:>14} {r.mean_imperceptibility:>7.3f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Prompt-level (behaviour-level) ASR - the trustworthy headline (C1)
#
# Under greedy decoding the variants of one behaviour are not independent samples, so
# treating them as independent Bernoulli trials is pseudo-replication: it understates the
# CI. The honest unit is the BEHAVIOUR (one of the 100 seed prompts). We collapse a
# behaviour's variants to a single binary - "attack succeeds if ANY variant works"
# (best-of-K, the usual jailbreak convention), or "majority"/"all" - and compute ASR +
# Wilson CI over the behaviours. This is also what proposal 4.8's paired-prompt McNemar
# requires (one binary per prompt per condition).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PromptLevelAsr:
    model: str
    family: str
    mode: str
    aggregate: str
    asr: float
    ci_low: float
    ci_high: float
    n_prompts: int
    n_variants: int


_AGGREGATORS = {
    "any": any,
    "all": all,
    "majority": lambda vals: sum(bool(v) for v in vals) > len(vals) / 2,
}


def per_prompt_success(outcomes: Iterable[Any], *, aggregate: str = "any") -> dict[str, bool]:
    """Collapse each behaviour's variant outcomes to one binary via ``aggregate``
    ('any' = best-of-K, 'majority', or 'all'). Keyed by seed_id."""
    if aggregate not in _AGGREGATORS:
        raise ValueError(f"aggregate must be one of {sorted(_AGGREGATORS)}")
    agg = _AGGREGATORS[aggregate]
    by_prompt: dict[str, list[bool]] = {}
    for o in outcomes:
        by_prompt.setdefault(getattr(o, "seed_id", ""), []).append(bool(o.judge_success))
    return {pid: bool(agg(vals)) for pid, vals in by_prompt.items()}


def asr_over_prompts(outcomes: Iterable[Any], *, aggregate: str = "any") -> list[PromptLevelAsr]:
    """Behaviour-level ASR + Wilson CI per (model, family, mode). This is the headline
    number: N is the number of distinct behaviours, not the number of variants (C1)."""
    groups: dict[tuple[str, str, str], list] = {}
    for o in outcomes:
        groups.setdefault((getattr(o, "model", ""), _family_key(o.family), o.mode), []).append(o)
    out: list[PromptLevelAsr] = []
    for (model, fam, mode), items in groups.items():
        succ_map = per_prompt_success(items, aggregate=aggregate)
        n = len(succ_map)
        s = sum(1 for v in succ_map.values() if v)
        lo, hi = wilson_interval(s, n)
        out.append(PromptLevelAsr(model, fam, mode, aggregate, (s / n if n else 0.0),
                                  lo, hi, n, len(items)))
    out.sort(key=lambda r: (r.model, r.family, r.mode))
    return out


def _fallback_by_prompt(outcomes: Iterable[Any]) -> dict[str, bool]:
    """A behaviour counts as fallback if ANY of its variants fell back to all content words."""
    fb: dict[str, bool] = {}
    for o in outcomes:
        pid = getattr(o, "seed_id", "")
        fb[pid] = fb.get(pid, False) or bool(getattr(o, "used_fallback", False))
    return fb


def fallback_rate(outcomes: Iterable[Any]) -> tuple[float, int, int]:
    """Fraction of distinct behaviours whose trigger detection fell back to all content
    words (H3). Returns (rate, n_fallback, n_prompts). When fallback is common the
    saliency-ranked trigger set is mostly degenerate and the targeted-vs-random contrast
    is undefined on those behaviours (handled by :func:`mcnemar_over_prompts`)."""
    fb = _fallback_by_prompt(outcomes)
    n = len(fb)
    k = sum(1 for v in fb.values() if v)
    return ((k / n if n else 0.0), k, n)


def mcnemar_over_prompts(outcomes_a: Iterable[Any], outcomes_b: Iterable[Any], *,
                         aggregate: str = "any", exclude_fallback: bool = False,
                         condition_a: str = "A", condition_b: str = "B") -> McNemarResult:
    """Paired McNemar between two conditions at the BEHAVIOUR level (proposal 4.8).

    Collapses each condition's variants to one binary per behaviour, pairs on the behaviours
    present in BOTH, and (when ``exclude_fallback``) drops behaviours that fell back in either
    condition - which is required for the targeted-vs-random comparison (H3), since random ==
    targeted on fallback behaviours."""
    a_list, b_list = list(outcomes_a), list(outcomes_b)
    a = per_prompt_success(a_list, aggregate=aggregate)
    b = per_prompt_success(b_list, aggregate=aggregate)
    common = set(a) & set(b)
    if exclude_fallback:
        fb_a, fb_b = _fallback_by_prompt(a_list), _fallback_by_prompt(b_list)
        common -= {p for p in common if fb_a.get(p) or fb_b.get(p)}
    ordered = sorted(common)
    return mcnemar([a[p] for p in ordered], [b[p] for p in ordered],
                   condition_a=condition_a, condition_b=condition_b)


def targeted_vs_random(outcomes: Iterable[Any], *, aggregate: str = "any") -> McNemarResult:
    """Convenience: split outcomes by mode and run the fallback-excluding paired McNemar
    (targeted vs random), the Objective-1 baseline contrast (H3)."""
    items = list(outcomes)
    tg = [o for o in items if getattr(o, "mode", None) == "targeted"]
    rd = [o for o in items if getattr(o, "mode", None) == "random"]
    return mcnemar_over_prompts(tg, rd, aggregate=aggregate, exclude_fallback=True,
                                condition_a="targeted", condition_b="random")
