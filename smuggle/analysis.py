"""Analysis owned by the research layer.

Wilson intervals are used only after outcomes are reduced to one binary observation
per JBB Goal within an inferential cell. Variant deliveries and repeated-Goal pooled
summaries are descriptive and receive no confidence interval.
"""
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
    statistic: float # NaN for the exact conditional test
    p_value: float
    odds_ratio: Optional[float]
    odds_ratio_ci_low: float
    odds_ratio_ci_high: float
    method: str

    @property
    def discordant(self) -> int:
        return self.b + self.c


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def asr(successes: int, n: int, *, condition: str = "", model: str = "") -> AsrEstimate:
    """Estimate ASR when ``n`` is one binary Goal outcome per treatment cell.

    Callers must not pass nested variant deliveries or pools in which a Goal appears
    more than once; Wilson coverage does not account for that dependence.
    """
    rate = successes / n if n else 0.0
    lo, hi = wilson_interval(successes, n)
    return AsrEstimate(condition, model, rate, lo, hi, n)


def _chi2_sf_1df(x: float) -> float:
    # Survival function of chi-square with 1 dof = erfc(sqrt(x/2)).
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))


def _binomial_cdf(k: int, n: int, probability: float) -> float:
    """P(X <= k) for X ~ Binomial(n, probability), using only the stdlib.

    McNemar's discordant totals are small in this project, so the direct sum is
    stable and avoids making SciPy a runtime dependency of the artefact.
    """
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 0.0
    q = 1.0 - probability
    return sum(
        math.comb(n, i) * probability**i * q ** (n - i)
        for i in range(k + 1)
    )


def _clopper_pearson(successes: int, n: int, *, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided binomial interval, computed by monotone bisection."""
    if not 0 <= successes <= n:
        raise ValueError("successes must be between zero and n")
    if n == 0:
        return (0.0, 1.0)

    if successes == 0:
        lower = 0.0
    else:
        # Solve P_p(X >= successes) = alpha/2. The tail increases with p.
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2.0
            tail = 1.0 - _binomial_cdf(successes - 1, n, mid)
            if tail < alpha / 2.0:
                lo = mid
            else:
                hi = mid
        lower = (lo + hi) / 2.0

    if successes == n:
        upper = 1.0
    else:
        # Solve P_p(X <= successes) = alpha/2. The CDF decreases with p.
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2.0
            cdf = _binomial_cdf(successes, n, mid)
            if cdf > alpha / 2.0:
                lo = mid
            else:
                hi = mid
        upper = (lo + hi) / 2.0
    return lower, upper


def _probability_to_odds(probability: float) -> float:
    if probability >= 1.0:
        return float("inf")
    if probability <= 0.0:
        return 0.0
    return probability / (1.0 - probability)


def matched_odds_ratio_interval(b: int, c: int, *, alpha: float = 0.05) -> tuple[
    Optional[float], float, float
]:
    """Matched-pair OR ``b/c`` and exact conditional 1-alpha interval.

    With no discordant pairs the effect is not estimable; returning ``None`` is
    more honest than calling the odds ratio 1.0. Boundary estimates remain zero
    or infinity, with exact one-sided limits supplied by the transformed
    Clopper--Pearson interval.
    """
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be non-negative")
    discordant = b + c
    if discordant == 0:
        return None, 0.0, float("inf")
    estimate = (b / c) if c else float("inf")
    lower_p, upper_p = _clopper_pearson(b, discordant, alpha=alpha)
    return estimate, _probability_to_odds(lower_p), _probability_to_odds(upper_p)


def exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact conditional McNemar p-value."""
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be non-negative")
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(b, c) + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm family-wise-error adjusted p-values, in the original order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [1.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def mcnemar(
    paired_a: Sequence[bool],
    paired_b: Sequence[bool],
    *,
    condition_a: str = "A",
    condition_b: str = "B",
    method: str = "exact",
) -> McNemarResult:
    """Paired binary comparison with an exact test by default.

    The exact conditional test is appropriate for the sparse discordant counts
    in this study. ``method='asymptotic'`` retains the continuity-corrected
    chi-square calculation for explicit sensitivity analyses.
    """
    if len(paired_a) != len(paired_b):
        raise ValueError("McNemar requires the identical seed set in both conditions")
    b = sum(1 for a, bb in zip(paired_a, paired_b) if a and not bb)
    c = sum(1 for a, bb in zip(paired_a, paired_b) if (not a) and bb)

    if method == "exact":
        statistic = float("nan")
        p_value = exact_mcnemar_p(b, c)
    elif method == "asymptotic":
        if b + c == 0:
            statistic, p_value = 0.0, 1.0
        else:
            statistic = (abs(b - c) - 1) ** 2 / (b + c)
            p_value = _chi2_sf_1df(statistic)
    else:
        raise ValueError("method must be 'exact' or 'asymptotic'")

    odds, odds_low, odds_high = matched_odds_ratio_interval(b, c)
    return McNemarResult(
        condition_a=condition_a,
        condition_b=condition_b,
        b=b,
        c=c,
        statistic=statistic,
        p_value=p_value,
        odds_ratio=odds,
        odds_ratio_ci_low=odds_low,
        odds_ratio_ci_high=odds_high,
        method=method,
    )


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
    n: int                               # byte-distinct variant deliveries
    successes: int
    asr: float                           # descriptive delivery rate; no CI
    n_guarded: int                       # eligible clean-unsafe deliveries
    bypasses: Optional[int]
    bypass_rate: Optional[float]
    joint_successes: Optional[int]       # encoded-safe delivery AND target compliance
    joint_rate: Optional[float]
    mean_imperceptibility: float
    n_raw: int = 0                       # variants before de-duplication (greedy repeats removed)
    n_fallback: int = 0                  # variants whose detection fell back to all content words


def _family_key(fam: Any) -> str:
    return fam if isinstance(fam, str) else "+".join(fam)


def _dedup_identical(items: Sequence[Any]) -> list:
    """Keep one recorded outcome per ``(seed_id, encoded_text)`` within a cell.

    Repeated mutation seeds can produce the same byte input. Counting those as
    independent variants would inflate N. Hosted inference can still return
    different text for the same input, so this function preserves the first
    deployment observation and does not claim that duplicate responses are
    deterministic. Outcomes without ``encoded_text`` are kept as-is.
    """
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


def bypass_and_asr(
    outcomes: Iterable[Any],
    *,
    collapse_identical: bool = True,
    eligible_seed_ids: Optional[set[str]] = None,
) -> list[FamilyModeReport]:
    """Group outcomes by ``(model, family, mode)`` and keep three layers distinct.

    - compliance ASR: the target achieved the objective;
    - diagnostic bypass: the encoded input was predicted safe by the off-path
      classifier, among Goals that classifier predicted unsafe in clean text;
    - joint success: the same clean-unsafe eligible delivery was predicted safe by
      that classifier and achieved the objective. This isolates encoding-induced
      gate evasion from clean-text classifier misses; no guard was enforced.

    ``eligible_seed_ids`` must contain the clean-unsafe Goal IDs for the
    proposal/report bypass definition. If omitted, every row with an encoded
    verdict is included for backward compatibility; callers must label that
    broader denominator explicitly.

    Byte-identical inputs are de-duplicated within each cell by default. Variant
    estimates are descriptive because variants remain nested within Goals.
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
        compliance_rate = succ / n if n else 0.0
        guarded = [
            o for o in items
            if getattr(o, "input_flagged", None) is not None
            and (
                eligible_seed_ids is None
                or getattr(o, "seed_id", "") in eligible_seed_ids
            )
        ]
        ng = len(guarded)
        if ng:
            byp = sum(1 for o in guarded if not o.input_flagged)
            joint = sum(1 for o in guarded if (not o.input_flagged) and bool(o.judge_success))
            bypass_rate, bypasses = byp / ng, byp
            joint_rate, joint_successes = joint / ng, joint
        else:
            bypass_rate, bypasses = None, None
            joint_rate, joint_successes = None, None
        imp = [float(getattr(o, "imperceptibility", 0.0)) for o in items]
        reports.append(FamilyModeReport(
            model=model, family=fam, mode=mode, n=n, successes=succ,
            asr=compliance_rate,
            n_guarded=ng, bypasses=bypasses, bypass_rate=bypass_rate,
            joint_successes=joint_successes, joint_rate=joint_rate,
            mean_imperceptibility=(sum(imp) / len(imp) if imp else 0.0),
            n_raw=n_raw, n_fallback=n_fb,
        ))
    reports.sort(key=lambda r: (r.model, r.family, r.mode))
    return reports


def format_report_table(reports: Sequence[FamilyModeReport]) -> str:
    """Render descriptive delivery counts and rates without inferential intervals."""
    hdr = (
        f"{'model':20} {'family':22} {'mode':9} {'N_deliv':>7} "
        f"{'compliant':>9} {'ASR':>6} {'N_guard':>7} {'bypass':>15} "
        f"{'joint':>15} {'imperc':>7}"
    )
    lines = [hdr, "-" * len(hdr)]
    for r in reports:
        if r.bypass_rate is None:
            bypass, joint = "n/a", "n/a"
        else:
            bypass = f"{r.bypasses}/{r.n_guarded} ({r.bypass_rate:.0%})"
            joint = f"{r.joint_successes}/{r.n_guarded} ({r.joint_rate:.0%})"
        lines.append(
            f"{r.model[:20]:20} {r.family[:22]:22} {r.mode:9} {r.n:>7} "
            f"{r.successes:>9} {r.asr:>5.0%} {r.n_guarded:>7} {bypass:>15} "
            f"{joint:>15} {r.mean_imperceptibility:>7.3f}"
        )
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


def per_prompt_success(outcomes: Iterable[Any], *, aggregate: str = "any",
                       collapse_identical: bool = True) -> dict[str, bool]:
    """Collapse each behaviour's outcomes to one binary via ``aggregate``.

    By default, byte-identical encoded inputs are collapsed first (keeping the first observed
    response). This prevents mutation-seed collisions from becoming undeclared extra inference
    attempts, which matters when a remote deployment varies despite greedy decoding.
    """
    if aggregate not in _AGGREGATORS:
        raise ValueError(f"aggregate must be one of {sorted(_AGGREGATORS)}")
    agg = _AGGREGATORS[aggregate]
    items = list(outcomes)
    if collapse_identical:
        items = _dedup_identical(items)
    by_prompt: dict[str, list[bool]] = {}
    for o in items:
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
        unique_items = _dedup_identical(items)
        succ_map = per_prompt_success(unique_items, aggregate=aggregate, collapse_identical=False)
        n = len(succ_map)
        s = sum(1 for v in succ_map.values() if v)
        lo, hi = wilson_interval(s, n)
        out.append(PromptLevelAsr(model, fam, mode, aggregate, (s / n if n else 0.0),
                                  lo, hi, n, len(unique_items)))
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
