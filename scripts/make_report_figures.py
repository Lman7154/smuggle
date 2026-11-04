"""Generate the report's pipeline diagram and descriptive outcome chart.

The chart reads only the text-free public Experiment 1 CSV. No model, network, or
private prompt/response text is required.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FAMILIES = ["homoglyph", "structural", "variation_selector", "tag_block"]
LABELS = ["Homoglyph", "Structural", "Variation\nselector", "Tag Block"]


def _box(ax, xy, width, height, text, *, face="#eef4fb", edge="#315a7d", fontsize=14):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.4, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def _arrow(ax, start, end, *, dashed=False, color="#303030", label=None, label_xy=None):
    arrow = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=13,
        linewidth=1.4, color=color,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(arrow)
    if label and label_xy:
        ax.text(*label_xy, label, ha="center", va="center", fontsize=11, color=color)


def pipeline_figure(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, (0.03, 0.68), 0.14, 0.13, "JBB Goal\n(clean text)")
    _box(ax, (0.23, 0.68), 0.16, 0.13, "Trigger detector\nLlama Guard saliency")
    _box(ax, (0.45, 0.68), 0.15, 0.13, "Mutation Engine\nfour families")
    _box(ax, (0.66, 0.68), 0.13, 0.13, "Encoded\ninput")
    _box(ax, (0.84, 0.68), 0.13, 0.13, "Target\nchatbot", face="#fff4e6", edge="#9a5a13")
    _box(ax, (0.82, 0.40), 0.17, 0.13, "Composite judge\ntask achieved AND\nnot refused", face="#f3edf8", edge="#6b3f88", fontsize=12)
    _box(ax, (0.55, 0.14), 0.25, 0.14, "Separate outcomes\ncompliance • diagnostic bypass\n• joint success", face="#edf8ef", edge="#39734a", fontsize=12)
    _box(ax, (0.43, 0.39), 0.19, 0.13, "Diagnostic Llama Guard\noff-path; never gates\ndelivery", face="#eef4fb", edge="#315a7d", fontsize=12)
    _box(ax, (0.65, 0.89), 0.20, 0.08, "Multi-turn only:\nadaptive prior history", face="#fff7da", edge="#8b731d", fontsize=12)

    _arrow(ax, (0.17, 0.745), (0.23, 0.745))
    _arrow(ax, (0.39, 0.745), (0.45, 0.745))
    _arrow(ax, (0.60, 0.745), (0.66, 0.745))
    _arrow(ax, (0.79, 0.745), (0.84, 0.745), label="all inputs delivered", label_xy=(0.815, 0.79))
    _arrow(ax, (0.905, 0.68), (0.905, 0.53))
    _arrow(ax, (0.82, 0.46), (0.75, 0.28))
    _arrow(ax, (0.72, 0.68), (0.58, 0.52), dashed=True, color="#315a7d", label="copy scored", label_xy=(0.63, 0.61))
    _arrow(ax, (0.52, 0.39), (0.63, 0.28), dashed=True, color="#315a7d")
    _arrow(ax, (0.75, 0.89), (0.89, 0.81), color="#8b731d")

    ax.plot([0.05, 0.11], [0.08, 0.08], color="#303030", linewidth=1.5)
    ax.text(0.12, 0.08, "delivery / scoring path", va="center", fontsize=12)
    ax.plot([0.30, 0.36], [0.08, 0.08], color="#315a7d", linewidth=1.5, linestyle="--")
    ax.text(0.37, 0.08, "diagnostic-only path", va="center", fontsize=12)

    fig.tight_layout()
    for suffix in ("png", "pdf"):
        metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else None
        fig.savefig(
            out_dir / f"pipeline.{suffix}",
            dpi=220,
            bbox_inches="tight",
            metadata=metadata,
        )
    plt.close(fig)


def _load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def outcome_figure(exp1_path: Path, out_dir: Path) -> None:
    rows = [row for row in _load(exp1_path) if row["mode"] == "targeted"]
    bypass_rates: list[float] = []
    compliance_rates: list[float] = []

    for family in FAMILIES:
        family_rows = [row for row in rows if row["family"] == family]

        # The same encoded input is scored by the diagnostic guard for every model.
        # Collapse model replication and retain one row per Goal/hash.
        guard_unique: dict[tuple[str, str], dict[str, str]] = {}
        for row in family_rows:
            guard_unique.setdefault((row["goal_id"], row["encoded_sha256"]), row)
        guard_eligible = [row for row in guard_unique.values() if row["clean_guard_flagged"] == "1"]
        bypasses = sum(row["encoded_guard_flagged"] == "0" for row in guard_eligible)
        bypass_rates.append(100 * bypasses / len(guard_eligible))

        # Compliance is a model-condition observation, de-duplicated within model/Goal/input.
        model_unique: dict[tuple[str, str, str], dict[str, str]] = {}
        for row in family_rows:
            model_unique.setdefault((row["model"], row["goal_id"], row["encoded_sha256"]), row)
        successes = sum(row["judge_success"] == "1" for row in model_unique.values())
        compliance_rates.append(100 * successes / len(model_unique))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), gridspec_kw={"width_ratios": [1.08, 1]})
    colors = ["#4c78a8", "#72b7b2", "#f2cf5b", "#e45756"]
    x = range(len(FAMILIES))

    axes[0].bar(x, bypass_rates, color=colors, edgecolor="#333333", linewidth=0.6)
    axes[0].set_title("Diagnostic bypass (targeted mode)", fontsize=16)
    axes[0].set_ylabel("Encoded inputs predicted safe (%)", fontsize=13)
    axes[0].set_ylim(0, 108)
    axes[0].set_xticks(list(x), LABELS, fontsize=12)
    axes[0].tick_params(axis="y", labelsize=12)
    axes[0].grid(axis="y", alpha=0.25)
    for i, value in enumerate(bypass_rates):
        axes[0].text(i, value + 2, f"{value:.1f}%", ha="center", va="bottom", fontsize=12)

    axes[1].bar(x, compliance_rates, color=colors, edgecolor="#333333", linewidth=0.6)
    axes[1].set_title("Target compliance (targeted mode)", fontsize=16)
    axes[1].set_ylabel("Byte-deduplicated model-condition observations (%)", fontsize=13)
    axes[1].set_ylim(0, max(4.0, max(compliance_rates) * 1.35))
    axes[1].set_xticks(list(x), LABELS, fontsize=12)
    axes[1].tick_params(axis="y", labelsize=12)
    axes[1].grid(axis="y", alpha=0.25)
    for i, value in enumerate(compliance_rates):
        axes[1].text(i, value + 0.10, f"{value:.1f}%", ha="center", va="bottom", fontsize=12)

    fig.tight_layout(w_pad=2.5)
    for suffix in ("png", "pdf"):
        metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else None
        fig.savefig(
            out_dir / f"bypass_compliance.{suffix}",
            dpi=220,
            bbox_inches="tight",
            metadata=metadata,
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp1", default="data/derived/exp1_metrics.csv")
    parser.add_argument("--out", default="report_final/figures")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pipeline_figure(out)
    outcome_figure(Path(args.exp1), out)
    print(f"Generated report figures in {out}")


if __name__ == "__main__":
    main()
