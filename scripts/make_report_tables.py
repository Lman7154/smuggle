"""Regenerate report statistics from the text-free public CSV files.

No model, network, prompt text, or private transcript is required. Outputs include
machine-readable CSV/JSON plus LaTeX tables consumed by the report.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import smuggle

INLINE_FAMILIES = ("homoglyph", "structural", "variation_selector")
FAMILY_ORDER = {"homoglyph": 0, "structural": 1, "variation_selector": 2, "tag_block": 3}
MODEL_LABELS = {
    "gpt-oss:20b-cloud": "gpt-oss:20b",
    "gemma4:31b-cloud": "Gemma 4 31B",
    "nemotron-3-nano:30b-cloud": "Nemotron 3 Nano",
}
FAMILY_LABELS = {
    "homoglyph": "Homoglyph",
    "structural": "Structural",
    "variation_selector": "Variation selector",
    "tag_block": "Tag Block",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bool(row: dict[str, str], key: str) -> bool:
    return row[key].strip() in {"1", "true", "True"}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, text: str) -> None:
    """Write generated text with stable LF endings on every platform."""
    path.write_text(text, encoding="utf-8", newline="\n")


def _sort_key(model: str, family: str, mode: str = "") -> tuple:
    model_order = {name: i for i, name in enumerate(MODEL_LABELS)}
    mode_order = {"random": 0, "targeted": 1}
    return model_order.get(model, 99), FAMILY_ORDER.get(family, 99), mode_order.get(mode, 99)


def _dedup(rows: Iterable[dict[str, str]], hash_field: str) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    output: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda item: int(item.get("mutation_seed", 0))):
        key = (row["goal_id"], row[hash_field])
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def _estimate(successes: int, n: int) -> tuple[float, float, float]:
    low, high = smuggle.wilson_interval(successes, n)
    return (successes / n if n else 0.0), low, high


def exp1_variant_metrics(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["family"], row["mode"])].append(row)
    output: list[dict[str, Any]] = []
    for (model, family, mode), items in grouped.items():
        unique = _dedup(items, "encoded_sha256")
        successes = sum(_bool(row, "judge_success") for row in unique)
        asr = successes / len(unique) if unique else 0.0
        eligible = [row for row in unique if _bool(row, "clean_guard_flagged")]
        bypasses = sum(not _bool(row, "encoded_guard_flagged") for row in eligible)
        joint = sum(
            (not _bool(row, "encoded_guard_flagged")) and _bool(row, "judge_success")
            for row in eligible
        )
        bypass = bypasses / len(eligible) if eligible else 0.0
        joint_rate = joint / len(eligible) if eligible else 0.0
        output.append({
            "model": model,
            "family": family,
            "mode": mode,
            "unique_variants": len(unique),
            "compliance_successes": successes,
            "compliance_asr": asr,
            "bypass_eligible_n": len(eligible),
            "bypasses": bypasses,
            "bypass_rate": bypass,
            "joint_successes": joint,
            "joint_rate": joint_rate,
            "mean_canonical_visibility": sum(float(row["imperceptibility"]) for row in unique) / len(unique),
        })
    return sorted(output, key=lambda row: _sort_key(row["model"], row["family"], row["mode"]))


def exp1_behaviour_metrics(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["family"], row["mode"])].append(row)
    output: list[dict[str, Any]] = []
    for (model, family, mode), items in grouped.items():
        unique = _dedup(items, "encoded_sha256")
        by_goal: dict[str, list[bool]] = defaultdict(list)
        for row in unique:
            by_goal[row["goal_id"]].append(_bool(row, "judge_success"))
        successes = sum(any(values) for values in by_goal.values())
        rate, low, high = _estimate(successes, len(by_goal))
        output.append({
            "model": model,
            "family": family,
            "mode": mode,
            "goals": len(by_goal),
            "successful_goals": successes,
            "behaviour_asr": rate,
            "ci_low": low,
            "ci_high": high,
            "unique_variants": len(unique),
        })
    return sorted(output, key=lambda row: _sort_key(row["model"], row["family"], row["mode"]))


def targeting_tests(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    models = sorted({row["model"] for row in rows}, key=lambda model: _sort_key(model, "homoglyph"))
    for model in models:
        for family in INLINE_FAMILIES:
            cell = [row for row in rows if row["model"] == model and row["family"] == family]
            fallback_goals = {row["goal_id"] for row in cell if _bool(row, "used_fallback")}
            eligible_goals = sorted({row["goal_id"] for row in cell} - fallback_goals)
            vectors: dict[str, dict[str, bool]] = {"targeted": {}, "random": {}}
            for mode in vectors:
                mode_rows = [
                    row for row in cell
                    if row["mode"] == mode
                    and int(row["mutation_seed"]) == 0
                    and row["goal_id"] in eligible_goals
                ]
                unique = _dedup(mode_rows, "encoded_sha256")
                by_goal: dict[str, list[bool]] = defaultdict(list)
                for row in unique:
                    by_goal[row["goal_id"]].append(_bool(row, "judge_success"))
                vectors[mode] = {goal: any(by_goal.get(goal, [])) for goal in eligible_goals}
            targeted = [vectors["targeted"][goal] for goal in eligible_goals]
            random_values = [vectors["random"][goal] for goal in eligible_goals]
            result = smuggle.mcnemar(
                targeted,
                random_values,
                condition_a="targeted",
                condition_b="random",
            )
            output.append({
                "model": model,
                "family": family,
                "paired_goals": len(eligible_goals),
                "targeted_successes": sum(targeted),
                "random_successes": sum(random_values),
                "b_targeted_only": result.b,
                "c_random_only": result.c,
                "p_exact": result.p_value,
                "p_holm": 0.0,
                "matched_odds_ratio": result.odds_ratio,
                "odds_ratio_ci_low": result.odds_ratio_ci_low,
                "odds_ratio_ci_high": result.odds_ratio_ci_high,
                "method": "matched mutation seed 0; fallback Goals and Tag Block excluded",
            })
    adjusted = smuggle.holm_adjust([row["p_exact"] for row in output])
    for row, value in zip(output, adjusted):
        row["p_holm"] = value
    return output


def exp3a_cell_rates(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["family"])].append(row)
    output: list[dict[str, Any]] = []
    for (model, family), items in grouped.items():
        goal_ids = [row["goal_id"] for row in items]
        if len(set(goal_ids)) != len(goal_ids):
            duplicates = sorted(goal for goal, count in Counter(goal_ids).items() if count > 1)
            raise ValueError(
                f"repeated Goal rows in Experiment 3a cell {(model, family)!r}: {duplicates}"
            )
        record: dict[str, Any] = {"model": model, "family": family, "conversations": len(items)}
        for label, key in (
            ("pre_probe", "pre_probe_success"),
            ("final_probe", "final_probe_success"),
            ("overall", "overall_success"),
        ):
            successes = sum(_bool(row, key) for row in items)
            rate, low, high = _estimate(successes, len(items))
            record[f"{label}_successes"] = successes
            record[f"{label}_rate"] = rate
            record[f"{label}_ci_low"] = low
            record[f"{label}_ci_high"] = high
        output.append(record)
    return sorted(output, key=lambda row: _sort_key(row["model"], row["family"]))


def context_tests(exp1: list[dict[str, str]], exp3a: list[dict[str, str]], *, subset: str) -> list[dict[str, Any]]:
    exp1_seed0 = {
        (row["model"], row["family"], row["goal_id"]): row
        for row in exp1
        if row["mode"] == "targeted" and int(row["mutation_seed"]) == 0
    }
    grouped: dict[tuple[str, str], list[tuple[dict[str, str], dict[str, str]]]] = defaultdict(list)
    for row3 in exp3a:
        if subset == "preprobe_success" and not _bool(row3, "pre_probe_success"):
            continue
        if subset == "no_preprobe_success" and _bool(row3, "pre_probe_success"):
            continue
        if subset == "observable_final_sensitivity" and _bool(
            row3, "final_empty_length_completion"
        ):
            continue
        if subset == "phase_cap_sensitivity" and (
            row3["protocol_phase"] != "family_specific"
            or _bool(row3, "any_recorded_output_over_claimed_cap")
        ):
            continue
        key = (row3["model"], row3["family"], row3["goal_id"])
        row1 = exp1_seed0.get(key)
        if row1 is None or row1["encoded_sha256"] != row3["probe_sha256"]:
            raise ValueError(f"missing byte-identical pair for {key!r}")
        grouped[(row3["model"], row3["family"])].append((row1, row3))
    output: list[dict[str, Any]] = []
    for (model, family), items in sorted(grouped.items(), key=lambda item: _sort_key(*item[0])):
        first = [_bool(row1, "judge_success") for row1, _ in items]
        second = [_bool(row3, "final_probe_success") for _, row3 in items]
        result = smuggle.mcnemar(first, second, condition_a="exp1", condition_b="exp3a")
        output.append({
            "subset": subset,
            "model": model,
            "family": family,
            "paired_goals": len(items),
            "exp1_successes": sum(first),
            "exp3a_successes": sum(second),
            "b_exp1_only": result.b,
            "c_exp3a_only": result.c,
            "p_exact": result.p_value,
            "p_holm": 0.0,
            "matched_odds_ratio_exp1_over_exp3a": result.odds_ratio,
            "odds_ratio_ci_low": result.odds_ratio_ci_low,
            "odds_ratio_ci_high": result.odds_ratio_ci_high,
        })
    adjusted = smuggle.holm_adjust([row["p_exact"] for row in output])
    for row, value in zip(output, adjusted):
        row["p_holm"] = value
    return output


def clustered_context_sensitivity(exp1: list[dict[str, str]], exp3a: list[dict[str, str]], *, repetitions: int, seed: int) -> dict[str, Any]:
    exp1_seed0 = {
        (row["model"], row["family"], row["goal_id"]): row
        for row in exp1
        if row["mode"] == "targeted" and int(row["mutation_seed"]) == 0
    }
    by_goal: dict[str, list[int]] = defaultdict(list)
    for row3 in exp3a:
        key = (row3["model"], row3["family"], row3["goal_id"])
        row1 = exp1_seed0[key]
        by_goal[row3["goal_id"]].append(
            int(_bool(row3, "final_probe_success")) - int(_bool(row1, "judge_success"))
        )
    goal_ids = sorted(by_goal)
    means = {goal: sum(by_goal[goal]) / len(by_goal[goal]) for goal in goal_ids}
    observed = sum(means.values()) / len(goal_ids)
    rng = random.Random(seed)
    draws = sorted(
        sum(means[rng.choice(goal_ids)] for _ in goal_ids) / len(goal_ids)
        for _ in range(repetitions)
    )
    low = draws[int(0.025 * repetitions)]
    high = draws[min(repetitions - 1, int(0.975 * repetitions))]

    contributions = [sum(by_goal[goal]) for goal in goal_ids]
    distribution: dict[int, int] = {0: 1}
    nonzero = 0
    for contribution in contributions:
        if contribution == 0:
            continue
        nonzero += 1
        updated: dict[int, int] = defaultdict(int)
        for total, count in distribution.items():
            updated[total + contribution] += count
            updated[total - contribution] += count
        distribution = dict(updated)
    observed_sum = sum(contributions)
    p_value = (
        sum(count for total, count in distribution.items() if abs(total) >= abs(observed_sum)) / (2**nonzero)
        if nonzero else 1.0
    )
    return {
        "goals": len(goal_ids),
        "cells_per_goal": len(by_goal[goal_ids[0]]),
        "risk_difference_exp3a_minus_exp1": observed,
        "cluster_bootstrap_ci_low": low,
        "cluster_bootstrap_ci_high": high,
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": seed,
        "goal_block_signflip_p": p_value,
        "nonzero_goal_clusters": nonzero,
        "cluster_sum_distribution": dict(sorted(Counter(contributions).items())),
    }


def exp3a_provenance_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Text-free audit strata for protocol phase and completion truncation."""
    strata = [
        ("generic_convention_phase", lambda row: row["protocol_phase"] == "generic_convention"),
        ("family_specific_phase", lambda row: row["protocol_phase"] == "family_specific"),
        ("any_recorded_output_over_claimed_cap", lambda row: _bool(row, "any_recorded_output_over_claimed_cap")),
        ("final_empty_length_completion", lambda row: _bool(row, "final_empty_length_completion")),
        ("observable_final_completion", lambda row: not _bool(row, "final_empty_length_completion")),
        (
            "tag_block_observable_final",
            lambda row: row["family"] == "tag_block"
            and not _bool(row, "final_empty_length_completion"),
        ),
        (
            "conservative_exclusion_union",
            lambda row: row["protocol_phase"] != "family_specific"
            or _bool(row, "any_recorded_output_over_claimed_cap"),
        ),
        (
            "phase_cap_sensitivity_retained",
            lambda row: row["protocol_phase"] == "family_specific"
            and not _bool(row, "any_recorded_output_over_claimed_cap"),
        ),
    ]
    output: list[dict[str, Any]] = []
    for label, predicate in strata:
        selected = [row for row in rows if predicate(row)]
        output.append({
            "stratum": label,
            "rows": len(selected),
            "pre_probe_successes": sum(_bool(row, "pre_probe_success") for row in selected),
            "final_probe_successes": sum(_bool(row, "final_probe_success") for row in selected),
            "overall_successes": sum(_bool(row, "overall_success") for row in selected),
        })
    return output


def _pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}\\%"


def _ci(low: float, high: float, digits: int = 1) -> str:
    return f"[{100 * low:.{digits}f}, {100 * high:.{digits}f}]"


def _rate_cell(successes: int, n: int, rate: float, low: float, high: float) -> str:
    return f"{successes}/{n} ({_pct(rate)}; {_ci(low, high)})"


def _rate_only(successes: int, n: int, rate: float) -> str:
    return f"{successes}/{n} ({_pct(rate)})"


def _or(value: float | None, low: float, high: float) -> str:
    if value is None:
        point = "NE"
    elif value == float("inf"):
        point = "$\\infty$"
    else:
        point = f"{value:.2f}"
    low_text = f"{low:.2f}"
    high_text = "$\\infty$" if high == float("inf") else f"{high:.2f}"
    return f"{point} [{low_text}, {high_text}]"


def _tex_escape(value: str) -> str:
    return value.replace("_", "\\_")


def plaintext_vs_encoded_tests(
    plaintext: list[dict[str, str]],
    exp1: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Exact two-sided McNemar of plaintext vs encoded, paired by Goal.

    The encoded comparator is the targeted, mutation_seed=0 Experiment 1 rows (the
    same "direct single-attempt" results used in the context comparison).  Holm
    correction is applied across all 12 model-family tests.
    """
    plaintext_success: dict[tuple[str, str], bool] = {
        (row["model"], row["goal_id"]): _bool(row, "success")
        for row in plaintext
    }
    # Check #2: assert exactly 100 plaintext rows per model, no duplicates.
    models = sorted({row["model"] for row in plaintext}, key=lambda m: _sort_key(m, "homoglyph"))
    for model in models:
        gids = [row["goal_id"] for row in plaintext if row["model"] == model]
        if len(gids) != 100 or len(set(gids)) != 100:
            raise ValueError(
                f"check #2 failed: plaintext {model} has {len(gids)} rows ({len(set(gids))} unique), "
                f"expected 100"
            )

    exp1_seed0 = [
        row for row in exp1
        if row["mode"] == "targeted" and int(row["mutation_seed"]) == 0
    ]
    output: list[dict[str, Any]] = []
    for model in models:
        for family in FAMILY_ORDER:
            cell = [row for row in exp1_seed0
                    if row["model"] == model and row["family"] == family]
            pairs_a: list[bool] = []
            pairs_b: list[bool] = []
            for row in cell:
                gid = row["goal_id"]
                pkey = (model, gid)
                if pkey not in plaintext_success:
                    raise ValueError(
                        f"check #2 failed: no plaintext result for {pkey} "
                        f"(encoded family={family})"
                    )
                pairs_a.append(plaintext_success[pkey])
                pairs_b.append(_bool(row, "judge_success"))
            n = len(pairs_a)
            if n != 100:
                raise ValueError(
                    f"check #2 failed: {model}/{family} has {n} pairs, expected 100"
                )
            result = smuggle.mcnemar(
                pairs_a, pairs_b,
                condition_a="plaintext", condition_b="encoded",
            )
            output.append({
                "model": model,
                "family": family,
                "paired_goals": n,
                "plaintext_successes": sum(pairs_a),
                "encoded_successes": sum(pairs_b),
                "b_plaintext_only": result.b,
                "c_encoded_only": result.c,
                "p_exact": result.p_value,
                "p_holm": 0.0,
                "matched_odds_ratio_plaintext_over_encoded": result.odds_ratio,
                "odds_ratio_ci_low": result.odds_ratio_ci_low,
                "odds_ratio_ci_high": result.odds_ratio_ci_high,
            })
    adjusted = smuggle.holm_adjust([row["p_exact"] for row in output])
    for row, value in zip(output, adjusted):
        row["p_holm"] = value
    return output


def write_plaintext_tex_table(out: Path, tests: list[dict[str, Any]]) -> None:
    """Write the plaintext-vs-encoded McNemar table in the existing LaTeX style."""
    out.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Generated by scripts/make_report_tables.py; do not edit manually.",
        "\\begin{table}[htbp]", "\\centering", "\\scriptsize",
        "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{llrrrrrrc}", "\\toprule",
        "Model & Family & $n$ & Plaintext & Encoded & Plaintext only & Encoded only "
        "& Exact $p$ / Holm $p$ & Matched OR [95\\% CI] \\\\", "\\midrule",
    ]
    for row in tests:
        lines.append(
            f"{MODEL_LABELS[row['model']]} & {FAMILY_LABELS[row['family']]} & "
            f"{row['paired_goals']} & "
            f"{row['plaintext_successes']} & {row['encoded_successes']} & "
            f"{row['b_plaintext_only']} & {row['c_encoded_only']} & "
            f"{row['p_exact']:.3f} / {row['p_holm']:.3f} & "
            f"{_or(row['matched_odds_ratio_plaintext_over_encoded'], row['odds_ratio_ci_low'], row['odds_ratio_ci_high'])} \\\\")
    lines += [
        "\\bottomrule", "\\end{tabular}}",
        "\\caption{Matched comparison of plaintext and encoded (Experiment 1, targeted seed-0) "
        "delivery of the same JailbreakBench Goals. Exact two-sided McNemar tests use Holm "
        "correction across twelve model--family combinations. The matched odds ratio is "
        "greater than one when plaintext delivery has more condition-only successes than "
        "encoded delivery.}",
        "\\label{tab:plaintext-vs-encoded}", "\\end{table}", "",
    ]
    _write_text(out / "plaintext_vs_encoded_table.tex", "\n".join(lines))


def write_tex_tables(
    out: Path,
    behaviour: list[dict[str, Any]],
    variants: list[dict[str, Any]],
    targeting: list[dict[str, Any]],
    exp3a: list[dict[str, Any]],
    context_all: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
) -> None:
    out.mkdir(parents=True, exist_ok=True)

    lookup = {(row["model"], row["family"], row["mode"]): row for row in behaviour}
    lines = [
        "% Generated by scripts/make_report_tables.py; do not edit manually.",
        "\\begin{table}[htbp]", "\\centering", "\\footnotesize",
        "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{llcc}", "\\toprule",
        "Model & Family & Random best-of-2 & Targeted best-of-3 \\\\", "\\midrule",
    ]
    for model in MODEL_LABELS:
        for family in FAMILY_LABELS:
            random_row = lookup[(model, family, "random")]
            targeted_row = lookup[(model, family, "targeted")]
            random_cell = _rate_cell(random_row["successful_goals"], random_row["goals"], random_row["behaviour_asr"], random_row["ci_low"], random_row["ci_high"])
            targeted_cell = _rate_cell(targeted_row["successful_goals"], targeted_row["goals"], targeted_row["behaviour_asr"], targeted_row["ci_low"], targeted_row["ci_high"])
            lines.append(f"{MODEL_LABELS[model]} & {FAMILY_LABELS[family]} & {random_cell} & {targeted_cell} \\\\")
        lines.append("\\addlinespace")
    lines += [
        "\\bottomrule", "\\end{tabular}}",
        "\\caption{Single-turn Goal-level compliance. A Goal succeeds when any distinct variant succeeds. Wilson 95\\% confidence intervals use 100 Goals per condition.}",
        "\\label{tab:asr}", "\\end{table}", "",
    ]
    _write_text(out / "exp1_behaviour_table.tex", "\n".join(lines))

    lines = [
        "% Generated by scripts/make_report_tables.py; do not edit manually.",
        "\\begin{table}[htbp]", "\\centering", "\\scriptsize",
        "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{lllcccc}", "\\toprule",
        "Model & Family & Mode & Distinct variants & Compliance & Diagnostic bypass & Joint success \\\\", "\\midrule",
    ]
    for row in variants:
        compliance = _rate_only(row["compliance_successes"], row["unique_variants"], row["compliance_asr"])
        bypass = _rate_only(row["bypasses"], row["bypass_eligible_n"], row["bypass_rate"])
        joint = _rate_only(row["joint_successes"], row["bypass_eligible_n"], row["joint_rate"])
        lines.append(f"{MODEL_LABELS[row['model']]} & {FAMILY_LABELS[row['family']]} & {row['mode']} & {row['unique_variants']} & {compliance} & {bypass} & {joint} \\\\")
    lines += [
        "\\bottomrule", "\\end{tabular}}",
        "\\caption{Single-turn descriptive outcomes per prompt delivery. A Goal contributes two or three variants, so no confidence interval is attached to these correlated rates. Diagnostic bypass and joint success are restricted to the 96 Goals that Llama Guard classified unsafe before encoding. Joint success requires the same input to be predicted safe and the target to fulfil the objective.}",
        "\\label{tab:variant}", "\\end{table}", "",
    ]
    _write_text(out / "exp1_variant_table.tex", "\n".join(lines))

    lines = [
        "% Generated by scripts/make_report_tables.py; do not edit manually.",
        "\\begin{table}[htbp]", "\\centering", "\\scriptsize",
        "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{llrrrrrrc}", "\\toprule",
        "Model & Family & $n$ & Targeted & Random & Target only & Random only & Exact $p$ / Holm $p$ & Matched OR [95\\% CI] \\\\", "\\midrule",
    ]
    for row in targeting:
        lines.append(
            f"{MODEL_LABELS[row['model']]} & {FAMILY_LABELS[row['family']]} & {row['paired_goals']} & "
            f"{row['targeted_successes']} & {row['random_successes']} & {row['b_targeted_only']} & {row['c_random_only']} & "
            f"{row['p_exact']:.3f} / {row['p_holm']:.3f} & {_or(row['matched_odds_ratio'], row['odds_ratio_ci_low'], row['odds_ratio_ci_high'])} \\\\")
    lines += [
        "\\bottomrule", "\\end{tabular}}",
        "\\caption{Equal-opportunity targeting comparison: targeted and random modes each use one matched mutation per Goal. The 25 fallback Goals and Tag Block are excluded, leaving 75 paired Goals for each inline family. Exact two-sided McNemar tests use Holm correction across nine tests. NE means that the matched odds ratio is not estimable because no paired outcomes differ.}",
        "\\label{tab:targeting-tests}", "\\end{table}", "",
    ]
    _write_text(out / "targeting_tests_table.tex", "\n".join(lines))

    lines = [
        "% Generated by scripts/make_report_tables.py; do not edit manually.",
        "\\begin{table}[htbp]", "\\centering", "\\scriptsize",
        "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{llccc}", "\\toprule",
        "Model & Family & Prelude success & Final-prompt success & Any-turn success \\\\", "\\midrule",
    ]
    for row in exp3a:
        pre = _rate_cell(row["pre_probe_successes"], row["conversations"], row["pre_probe_rate"], row["pre_probe_ci_low"], row["pre_probe_ci_high"])
        probe = _rate_cell(row["final_probe_successes"], row["conversations"], row["final_probe_rate"], row["final_probe_ci_low"], row["final_probe_ci_high"])
        overall = _rate_cell(row["overall_successes"], row["conversations"], row["overall_rate"], row["overall_ci_low"], row["overall_ci_high"])
        lines.append(f"{MODEL_LABELS[row['model']]} & {FAMILY_LABELS[row['family']]} & {pre} & {probe} & {overall} \\\\")
    lines += [
        "\\bottomrule", "\\end{tabular}}",
        "\\caption{Multi-turn outcomes per model and encoding family ($n=100$ Goals). Prelude success occurred during the four adaptive turns; final-prompt success occurred on the encoded fifth turn; any-turn success records either. Wilson 95\\% intervals use one outcome per Goal within each condition.}",
        "\\label{tab:exp3a}", "\\end{table}", "",
    ]
    _write_text(out / "exp3a_rates_table.tex", "\n".join(lines))

    lines = [
        "% Generated by scripts/make_report_tables.py; do not edit manually.",
        "\\begin{table}[htbp]", "\\centering", "\\scriptsize",
        "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{llrrrrrrc}", "\\toprule",
        "Model & Family & $n$ & Direct & After prelude & Direct only & Multi-turn only & Exact $p$ / Holm $p$ & Matched OR [95\\% CI] \\\\", "\\midrule",
    ]
    for row in context_all:
        lines.append(
            f"{MODEL_LABELS[row['model']]} & {FAMILY_LABELS[row['family']]} & {row['paired_goals']} & "
            f"{row['exp1_successes']} & {row['exp3a_successes']} & {row['b_exp1_only']} & {row['c_exp3a_only']} & "
            f"{row['p_exact']:.3f} / {row['p_holm']:.3f} & {_or(row['matched_odds_ratio_exp1_over_exp3a'], row['odds_ratio_ci_low'], row['odds_ratio_ci_high'])} \\\\")
    lines += [
        "\\bottomrule", "\\end{tabular}}",
        "\\caption{Matched comparison of the same encoded prompt under direct and adaptive multi-turn delivery. Exact two-sided McNemar tests use Holm correction across twelve model--family combinations. The matched odds ratio is greater than one when direct delivery has more condition-only successes. The conditions ran in separate cloud sessions, so the comparison is associational.}",
        "\\label{tab:context-tests}", "\\end{table}", "",
    ]
    _write_text(out / "context_tests_table.tex", "\n".join(lines))

    provenance_labels = {
        "generic_convention_phase": "Initial generic-convention phase",
        "family_specific_phase": "Family-specific convention phase",
        "any_recorded_output_over_claimed_cap": "Any recorded target output $>4096$ tokens",
        "final_empty_length_completion": "Empty length-truncated final completion",
        "observable_final_completion": "Observable non-empty final completion",
        "tag_block_observable_final": "Tag Block observable final completion",
        "conservative_exclusion_union": "Conservative exclusion union",
        "phase_cap_sensitivity_retained": "Rows retained for phase/cap sensitivity",
    }
    row_end = r" \\"
    lines = [
        "% Generated by scripts/make_report_tables.py; do not edit manually.",
        "\\begin{table}[htbp]", "\\centering", "\\small",
        "\\begin{tabular}{lrrrr}", "\\toprule",
        "Audit stratum & Rows & Pre-probe & Final probe & Overall" + row_end,
        "\\midrule",
    ]
    for row in provenance:
        lines.append(
            f"{provenance_labels[row['stratum']]} & {row['rows']} & "
            f"{row['pre_probe_successes']} & {row['final_probe_successes']} & "
            f"{row['overall_successes']}" + row_end
        )
    lines += [
        "\\bottomrule", "\\end{tabular}",
        "\\caption{Text-free Experiment 3a provenance audit. The phase rows partition the run; the output-token and empty-completion rows can overlap them. The conservative exclusion union removes the 19 generic-phase rows and every row with any recorded target output above 4,096 tokens.}",
        "\\label{tab:exp3a-provenance}", "\\end{table}", "",
    ]
    _write_text(out / "exp3a_provenance_table.tex", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp1", default="data/derived/exp1_metrics.csv")
    parser.add_argument("--exp3a", default="data/derived/exp3a_metrics.csv")
    parser.add_argument("--out", default="results/tables")
    parser.add_argument("--plaintext", default="data/derived/plaintext_metrics.csv",
                        help="plaintext baseline CSV (skipped if absent)")
    parser.add_argument("--bootstrap-repetitions", type=int, default=50_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_261_103)
    args = parser.parse_args()

    exp1 = _read_csv(Path(args.exp1))
    exp3a = _read_csv(Path(args.exp3a))
    if len(exp1) != 6000 or len(exp3a) != 1200:
        raise ValueError(f"unexpected row counts: Exp1={len(exp1)}, Exp3a={len(exp3a)}")

    variants = exp1_variant_metrics(exp1)
    behaviour = exp1_behaviour_metrics(exp1)
    targeting = targeting_tests(exp1)
    rates3a = exp3a_cell_rates(exp3a)
    context_all = context_tests(exp1, exp3a, subset="all")
    context_preprobe_success = context_tests(exp1, exp3a, subset="preprobe_success")
    context_sensitivity = context_tests(exp1, exp3a, subset="no_preprobe_success")
    observable_final_sensitivity = context_tests(
        exp1, exp3a, subset="observable_final_sensitivity"
    )
    phase_cap_sensitivity = context_tests(
        exp1, exp3a, subset="phase_cap_sensitivity"
    )
    provenance = exp3a_provenance_summary(exp3a)
    clustered = clustered_context_sensitivity(
        exp1,
        exp3a,
        repetitions=args.bootstrap_repetitions,
        seed=args.bootstrap_seed,
    )

    out = Path(args.out)
    _write_csv(out / "exp1_variant_metrics.csv", variants)
    _write_csv(out / "exp1_behaviour_asr.csv", behaviour)
    _write_csv(out / "exp1_targeting_tests.csv", targeting)
    _write_csv(out / "exp3a_cell_rates.csv", rates3a)
    _write_csv(out / "exp3a_provenance.csv", provenance)
    _write_csv(
        out / "exp1_vs_exp3a_tests.csv",
        context_all + context_preprobe_success + context_sensitivity
        + observable_final_sensitivity + phase_cap_sensitivity,
    )
    write_tex_tables(out, behaviour, variants, targeting, rates3a, context_all, provenance)
    provenance_by_stratum = {row["stratum"]: row for row in provenance}
    tag_observable = [
        row for row in exp3a
        if row["family"] == "tag_block" and not _bool(row, "final_empty_length_completion")
    ]
    tag_observable_successes = sum(_bool(row, "final_probe_success") for row in tag_observable)
    tag_observable_by_model: dict[str, dict[str, Any]] = {}
    for model in MODEL_LABELS:
        selected = [row for row in tag_observable if row["model"] == model]
        successes = sum(_bool(row, "final_probe_success") for row in selected)
        tag_observable_by_model[model] = {
            "successes": successes,
            "n": len(selected),
        }

    # Reconstruct the within-cell byte-deduplicated delivery unit used by the
    # variant table before describing guard/compliance co-occurrence.
    exp1_by_cell: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in exp1:
        exp1_by_cell[(row["model"], row["family"], row["mode"])].append(row)
    exp1_unique = [
        row
        for cell_rows in exp1_by_cell.values()
        for row in _dedup(cell_rows, "encoded_sha256")
    ]
    compliant = [row for row in exp1_unique if _bool(row, "judge_success")]
    flagged_compliant = [row for row in compliant if _bool(row, "encoded_guard_flagged")]
    encoded_safe_compliant = [row for row in compliant if not _bool(row, "encoded_guard_flagged")]
    eligible_joint = [row for row in encoded_safe_compliant if _bool(row, "clean_guard_flagged")]
    clean_safe_compliant = [row for row in encoded_safe_compliant if not _bool(row, "clean_guard_flagged")]

    joint_cell_rows = [
        row for row in exp1_unique
        if row["model"] == "gemma4:31b-cloud"
        and row["family"] == "homoglyph"
        and row["mode"] == "random"
        and _bool(row, "clean_guard_flagged")
    ]
    joint_goal_ids = {
        row["goal_id"] for row in joint_cell_rows
        if not _bool(row, "encoded_guard_flagged") and _bool(row, "judge_success")
    }
    joint_cell_goal_n = len({row["goal_id"] for row in joint_cell_rows})
    joint_goal_ci = smuggle.wilson_interval(len(joint_goal_ids), joint_cell_goal_n)

    preprobe_positive = [row for row in exp3a if _bool(row, "pre_probe_success")]
    preprobe_negative = [row for row in exp3a if not _bool(row, "pre_probe_success")]

    summary = {
        "exp1_raw_rows": len(exp1),
        "exp1_unique_variant_rows": sum(row["unique_variants"] for row in variants),
        "exp1_compliance_successes_unique": sum(row["compliance_successes"] for row in variants),
        "exp1_joint_successes_clean_unsafe": sum(row["joint_successes"] for row in variants),
        "exp1_guard_compliance_delivery_records": {
            "compliant_total": len(compliant),
            "flagged_and_compliant": len(flagged_compliant),
            "flagged_and_compliant_unique_goal_input_hashes": len({
                (row["goal_id"], row["encoded_sha256"]) for row in flagged_compliant
            }),
            "flagged_and_compliant_unique_goals": len({row["goal_id"] for row in flagged_compliant}),
            "encoded_safe_and_compliant_all_goals": len(encoded_safe_compliant),
            "encoded_safe_and_compliant_clean_unsafe": len(eligible_joint),
            "encoded_safe_and_compliant_clean_safe": len(clean_safe_compliant),
        },
        "exp1_joint_success_goal_level_same_cell": {
            "model": "gemma4:31b-cloud",
            "family": "homoglyph",
            "mode": "random",
            "successful_goals": len(joint_goal_ids),
            "eligible_goals": joint_cell_goal_n,
            "wilson_ci_low": joint_goal_ci[0],
            "wilson_ci_high": joint_goal_ci[1],
        },
        "targeting_tests": len(targeting),
        "targeting_min_raw_p": min(row["p_exact"] for row in targeting),
        "targeting_min_holm_p": min(row["p_holm"] for row in targeting),
        "exp3a_pre_probe_successes": sum(_bool(row, "pre_probe_success") for row in exp3a),
        "exp3a_final_probe_successes": sum(_bool(row, "final_probe_success") for row in exp3a),
        "exp3a_overall_successes": sum(_bool(row, "overall_success") for row in exp3a),
        "exp3a_final_probe_by_preprobe_stratum": {
            "preprobe_success": {
                "final_probe_successes": sum(_bool(row, "final_probe_success") for row in preprobe_positive),
                "conversations": len(preprobe_positive),
            },
            "no_preprobe_success": {
                "final_probe_successes": sum(_bool(row, "final_probe_success") for row in preprobe_negative),
                "conversations": len(preprobe_negative),
            },
        },
        "exp3a_protocol_provenance": provenance_by_stratum,
        "exp3a_final_empty_length_tag_block": sum(
            _bool(row, "final_empty_length_completion") and row["family"] == "tag_block"
            for row in exp3a
        ),
        "exp3a_tag_block_observable_final": {
            "successes": tag_observable_successes,
            "n": len(tag_observable),
            "interpretation": "pooled descriptive complete case; repeated Goals and informative truncation; no pooled confidence interval",
            "by_model": tag_observable_by_model,
        },
        "exp3a_first_success_turns": dict(sorted(Counter(int(row["first_success_turn"]) for row in exp3a if row["first_success_turn"]).items())),
        "context_all_pooled_descriptive": {
            "pairs": len(exp3a),
            "exp1_successes": sum(row["exp1_successes"] for row in context_all),
            "exp3a_successes": sum(row["exp3a_successes"] for row in context_all),
            "b_exp1_only": sum(row["b_exp1_only"] for row in context_all),
            "c_exp3a_only": sum(row["c_exp3a_only"] for row in context_all),
        },
        "context_preprobe_success_pooled_descriptive": {
            "pairs": sum(row["paired_goals"] for row in context_preprobe_success),
            "exp1_successes": sum(row["exp1_successes"] for row in context_preprobe_success),
            "exp3a_successes": sum(row["exp3a_successes"] for row in context_preprobe_success),
            "b_exp1_only": sum(row["b_exp1_only"] for row in context_preprobe_success),
            "c_exp3a_only": sum(row["c_exp3a_only"] for row in context_preprobe_success),
        },
        "context_no_preprobe_success_pooled_descriptive": {
            "pairs": sum(row["paired_goals"] for row in context_sensitivity),
            "exp1_successes": sum(row["exp1_successes"] for row in context_sensitivity),
            "exp3a_successes": sum(row["exp3a_successes"] for row in context_sensitivity),
            "b_exp1_only": sum(row["b_exp1_only"] for row in context_sensitivity),
            "c_exp3a_only": sum(row["c_exp3a_only"] for row in context_sensitivity),
        },
        "context_observable_final_sensitivity_pooled_descriptive": {
            "pairs": sum(row["paired_goals"] for row in observable_final_sensitivity),
            "exp1_successes": sum(row["exp1_successes"] for row in observable_final_sensitivity),
            "exp3a_successes": sum(row["exp3a_successes"] for row in observable_final_sensitivity),
            "b_exp1_only": sum(row["b_exp1_only"] for row in observable_final_sensitivity),
            "c_exp3a_only": sum(row["c_exp3a_only"] for row in observable_final_sensitivity),
            "min_raw_p": min(row["p_exact"] for row in observable_final_sensitivity),
            "min_holm_p": min(row["p_holm"] for row in observable_final_sensitivity),
        },
        "context_phase_cap_sensitivity_pooled_descriptive": {
            "pairs": sum(row["paired_goals"] for row in phase_cap_sensitivity),
            "exp1_successes": sum(row["exp1_successes"] for row in phase_cap_sensitivity),
            "exp3a_successes": sum(row["exp3a_successes"] for row in phase_cap_sensitivity),
            "b_exp1_only": sum(row["b_exp1_only"] for row in phase_cap_sensitivity),
            "c_exp3a_only": sum(row["c_exp3a_only"] for row in phase_cap_sensitivity),
            "min_raw_p": min(row["p_exact"] for row in phase_cap_sensitivity),
            "min_holm_p": min(row["p_holm"] for row in phase_cap_sensitivity),
        },
        "context_goal_cluster_sensitivity": clustered,
    }
    _write_text(out / "analysis_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Generated report tables in {out}")

    # --- Plaintext vs encoded (additive; no existing output is changed) --- #
    plaintext_path = Path(args.plaintext)
    if plaintext_path.exists():
        plaintext_rows = _read_csv(plaintext_path)
        plaintext_tests = plaintext_vs_encoded_tests(plaintext_rows, exp1)
        _write_csv(out / "plaintext_vs_encoded_tests.csv", plaintext_tests)
        write_plaintext_tex_table(out, plaintext_tests)
        print(f"Plaintext-vs-encoded table: {len(plaintext_tests)} tests, "
              f"min Holm p = {min(r['p_holm'] for r in plaintext_tests):.4f}")
    else:
        print(f"Skipping plaintext analysis ({plaintext_path} not found)")


if __name__ == "__main__":
    main()
