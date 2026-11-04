"""Offline, byte-exact Experiment 1 versus Experiment 3a comparison.

The primary comparison pairs each Experiment 3a final probe with the targeted,
mutation-seed-0 Experiment 1 input for the same model, family, and JBB Goal. Exact
byte equality is asserted. Per-cell inference uses exact two-sided McNemar tests,
matched-pair odds ratios with exact conditional 95% intervals, and Holm correction.

The same 100 Goals recur across model-family cells, so pooled counts are descriptive.
A Goal-cluster bootstrap risk-difference interval and Goal-block sign-flip test are
reported as dependence-aware sensitivity analyses. No model or network is called.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import smuggle


@dataclass(frozen=True)
class Pair:
    model: str
    family: str
    goal_id: str
    exp1_success: bool
    exp3a_success: bool
    pre_probe_success: bool


def _load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _family(row: dict) -> str:
    value = row["family"]
    return value[0] if isinstance(value, list) else value


def _load_seeded_exp1(progress_path: str | Path) -> dict[tuple[str, str, str, int, str], dict]:
    index: dict[tuple[str, str, str, int, str], dict] = {}
    for item in _load_jsonl(progress_path):
        key = tuple(item["key"])
        if len(key) != 5:
            raise ValueError(f"unexpected Experiment 1 progress key: {key!r}")
        normalized = (str(key[0]), str(key[1]), str(key[2]), int(key[3]), str(key[4]))
        if normalized in index:
            raise ValueError(f"duplicate Experiment 1 progress key: {normalized!r}")
        index[normalized] = item["outcome"]
    return index


def build_pairs(
    exp1_progress: str | Path,
    exp3a_rows: str | Path,
    *,
    expected_pairs: int | None = 1200,
) -> list[Pair]:
    """Return exact seed-0 pairs and fail on missing or byte-mismatched inputs."""
    exp1 = _load_seeded_exp1(exp1_progress)
    rows3a = _load_jsonl(exp3a_rows)
    pairs: list[Pair] = []
    seen: set[tuple[str, str, str]] = set()

    for row in rows3a:
        model, family, goal_id = str(row["model"]), _family(row), str(row["seed_id"])
        pair_key = (model, family, goal_id)
        if pair_key in seen:
            raise ValueError(f"duplicate Experiment 3a pair: {pair_key!r}")
        seen.add(pair_key)

        exp1_key = (model, family, "targeted", 0, goal_id)
        if exp1_key not in exp1:
            raise ValueError(f"missing targeted seed-0 Experiment 1 row: {exp1_key!r}")
        row1 = exp1[exp1_key]
        if row1.get("encoded_text") != row.get("final_probe_text"):
            raise ValueError(f"final probe is not byte-identical for {pair_key!r}")

        pairs.append(Pair(
            model=model,
            family=family,
            goal_id=goal_id,
            exp1_success=bool(row1.get("judge_success")),
            exp3a_success=bool(row.get("final_probe_success")),
            pre_probe_success=bool(row.get("pre_probe_success")),
        ))

    if expected_pairs is not None and len(pairs) != expected_pairs:
        raise ValueError(f"expected {expected_pairs} pairs, found {len(pairs)}")
    return pairs


def _or_text(value: float | None) -> str:
    if value is None:
        return "NE"
    if value == float("inf"):
        return "inf"
    return f"{value:.4f}"


def _limit_text(value: float) -> str:
    return "inf" if value == float("inf") else f"{value:.4f}"


def cell_results(pairs: Iterable[Pair], *, subset: str) -> list[dict]:
    grouped: dict[tuple[str, str], list[Pair]] = defaultdict(list)
    for pair in pairs:
        if subset == "no_preprobe_success" and pair.pre_probe_success:
            continue
        grouped[(pair.model, pair.family)].append(pair)

    rows: list[dict] = []
    for (model, family), items in sorted(grouped.items()):
        a = [item.exp1_success for item in items]
        b = [item.exp3a_success for item in items]
        result = smuggle.mcnemar(a, b, condition_a="exp1", condition_b="exp3a")
        rows.append({
            "subset": subset,
            "model": model,
            "family": family,
            "n": len(items),
            "exp1_successes": sum(a),
            "exp3a_successes": sum(b),
            "b_exp1_only": result.b,
            "c_exp3a_only": result.c,
            "p_exact": result.p_value,
            "odds_ratio_exp1_over_exp3a": result.odds_ratio,
            "odds_ratio_ci_low": result.odds_ratio_ci_low,
            "odds_ratio_ci_high": result.odds_ratio_ci_high,
        })

    adjusted = smuggle.holm_adjust([row["p_exact"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["p_holm"] = value
    return rows


def cluster_sensitivity(
    pairs: Iterable[Pair],
    *,
    bootstrap_repetitions: int = 50_000,
    seed: int = 20_261_103,
) -> dict:
    """Dependence-aware pooled sensitivity, retaining all cells within each Goal."""
    by_goal: dict[str, list[int]] = defaultdict(list)
    for pair in pairs:
        by_goal[pair.goal_id].append(int(pair.exp3a_success) - int(pair.exp1_success))
    goal_ids = sorted(by_goal)
    if not goal_ids:
        raise ValueError("cluster sensitivity needs at least one Goal")
    if len({len(by_goal[goal]) for goal in goal_ids}) != 1:
        raise ValueError("all Goal clusters must contain the same number of cells")

    cluster_means = {goal: sum(by_goal[goal]) / len(by_goal[goal]) for goal in goal_ids}
    risk_difference = sum(cluster_means.values()) / len(goal_ids)

    rng = random.Random(seed)
    bootstrap: list[float] = []
    for _ in range(bootstrap_repetitions):
        sampled = [rng.choice(goal_ids) for _ in goal_ids]
        bootstrap.append(sum(cluster_means[goal] for goal in sampled) / len(sampled))
    bootstrap.sort()
    lo_index = int(0.025 * bootstrap_repetitions)
    hi_index = min(bootstrap_repetitions - 1, int(0.975 * bootstrap_repetitions))

    cluster_sums = [sum(by_goal[goal]) for goal in goal_ids]
    distribution: dict[int, int] = {0: 1}
    nonzero_clusters = 0
    for contribution in cluster_sums:
        if contribution == 0:
            continue
        nonzero_clusters += 1
        updated: dict[int, int] = defaultdict(int)
        for total, count in distribution.items():
            updated[total + contribution] += count
            updated[total - contribution] += count
        distribution = dict(updated)
    observed_sum = sum(cluster_sums)
    signflip_p = (
        sum(count for total, count in distribution.items() if abs(total) >= abs(observed_sum))
        / (2**nonzero_clusters)
        if nonzero_clusters else 1.0
    )

    return {
        "goals": len(goal_ids),
        "cells_per_goal": len(by_goal[goal_ids[0]]),
        "risk_difference_exp3a_minus_exp1": risk_difference,
        "cluster_bootstrap_ci_low": bootstrap[lo_index],
        "cluster_bootstrap_ci_high": bootstrap[hi_index],
        "bootstrap_repetitions": bootstrap_repetitions,
        "bootstrap_seed": seed,
        "goal_block_signflip_p": signflip_p,
        "nonzero_goal_clusters": nonzero_clusters,
        "cluster_sum_distribution": dict(sorted(Counter(cluster_sums).items())),
    }


def pooled_descriptive(pairs: Iterable[Pair], *, subset: str) -> dict:
    items = [
        pair for pair in pairs
        if subset == "all" or not pair.pre_probe_success
    ]
    b = sum(pair.exp1_success and not pair.exp3a_success for pair in items)
    c = sum((not pair.exp1_success) and pair.exp3a_success for pair in items)
    return {
        "subset": subset,
        "pairs": len(items),
        "exp1_successes": sum(pair.exp1_success for pair in items),
        "exp3a_successes": sum(pair.exp3a_success for pair in items),
        "b_exp1_only": b,
        "c_exp3a_only": c,
        "note": "descriptive only; repeated Goals make a naive pooled McNemar test inappropriate",
    }


def _print_table(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    print(
        f"{'model':28} {'family':18} {'n':>3} {'E1':>3} {'3a':>3} "
        f"{'b':>2} {'c':>2} {'p exact':>8} {'p Holm':>8} {'OR [95% CI]':>28}"
    )
    print("-" * 112)
    for row in rows:
        interval = (
            f"{_or_text(row['odds_ratio_exp1_over_exp3a'])} "
            f"[{_limit_text(row['odds_ratio_ci_low'])},"
            f"{_limit_text(row['odds_ratio_ci_high'])}]"
        )
        print(
            f"{row['model'][:28]:28} {row['family']:18} {row['n']:3d} "
            f"{row['exp1_successes']:3d} {row['exp3a_successes']:3d} "
            f"{row['b_exp1_only']:2d} {row['c_exp3a_only']:2d} "
            f"{row['p_exact']:8.4f} {row['p_holm']:8.4f} {interval:>28}"
        )


def _write_outputs(out_dir: Path, pairs: list[Pair], rows: list[dict], summary: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "exp1_vs_exp3a_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(pairs[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(pair) for pair in pairs)
    with (out_dir / "exp1_vs_exp3a_tests.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "exp1_vs_exp3a_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp1-progress", default="runs/full/exp1_full_progress.jsonl")
    parser.add_argument("--exp3a", default="runs/exp3a_full/exp3a_full_rows.jsonl")
    parser.add_argument("--out", help="optional directory for text-free CSV/JSON outputs")
    parser.add_argument("--bootstrap-repetitions", type=int, default=50_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_261_103)
    args = parser.parse_args()

    pairs = build_pairs(args.exp1_progress, args.exp3a)
    all_rows = cell_results(pairs, subset="all")
    sensitivity_rows = cell_results(pairs, subset="no_preprobe_success")
    _print_table(all_rows, "All 1,200 byte-identical pairs (primary)")
    _print_table(sensitivity_rows, "Pairs with no pre-probe success (post-treatment sensitivity)")

    clustered = cluster_sensitivity(
        pairs,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.bootstrap_seed,
    )
    summary = {
        "pooled_descriptive": [
            pooled_descriptive(pairs, subset="all"),
            pooled_descriptive(pairs, subset="no_preprobe_success"),
        ],
        "clustered_all_pairs_sensitivity": clustered,
        "interpretation": (
            "Per-cell exact McNemar tests are primary. Pooled counts are descriptive; "
            "the cluster sensitivity retains all model-family cells for each repeated Goal."
        ),
    }
    print("\nPooled descriptive:")
    for row in summary["pooled_descriptive"]:
        print(row)
    print("\nGoal-cluster sensitivity:")
    print(json.dumps(clustered, indent=2, sort_keys=True))

    if args.out:
        combined = all_rows + sensitivity_rows
        _write_outputs(Path(args.out), pairs, combined, summary)
        print(f"\nWrote text-free analysis to {args.out}")


if __name__ == "__main__":
    main()
