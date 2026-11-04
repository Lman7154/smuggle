"""Create a text-free manual-adjudication sheet for Experiment 3a.

The sheet includes every automated positive and a deterministic stratified sample of
automated failures. Reviewers consult the private transcripts separately and fill the
blank human-label columns; no response text is copied into the sheet.
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

HUMAN_FIELDS = [
    "human_pre_probe_success",
    "human_final_probe_success",
    "human_overall_success",
    "human_refusal_or_safe_completion",
    "human_decode_only",
    "reviewer",
    "review_date_utc",
    "notes",
]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="data/derived/exp3a_metrics.csv")
    parser.add_argument("--out", default="private/exp3a_judge_audit.csv")
    parser.add_argument("--failure-sample-per-cell", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261103)
    args = parser.parse_args()

    rows = load_rows(Path(args.rows))
    positives = [row for row in rows if row["overall_success"] == "1"]
    failures: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["overall_success"] == "0":
            failures[(row["model"], row["family"])].append(row)

    rng = random.Random(args.seed)
    selected: list[tuple[str, dict[str, str]]] = [("all_automated_positives", row) for row in positives]
    for cell in sorted(failures):
        candidates = sorted(failures[cell], key=lambda row: row["row_id"])
        count = min(args.failure_sample_per_cell, len(candidates))
        selected.extend(("stratified_failure_sample", row) for row in rng.sample(candidates, count))

    fields = [
        "review_stratum", "row_id", "model", "family", "goal_id",
        "automated_pre_probe_success", "automated_final_probe_success",
        "automated_overall_success", "protocol_phase", "final_empty_length_completion",
        *HUMAN_FIELDS,
    ]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for stratum, row in sorted(selected, key=lambda item: (item[0], item[1]["model"], item[1]["family"], item[1]["goal_id"])):
            writer.writerow({
                "review_stratum": stratum,
                "row_id": row["row_id"],
                "model": row["model"],
                "family": row["family"],
                "goal_id": row["goal_id"],
                "automated_pre_probe_success": row["pre_probe_success"],
                "automated_final_probe_success": row["final_probe_success"],
                "automated_overall_success": row["overall_success"],
                "protocol_phase": row["protocol_phase"],
                "final_empty_length_completion": row["final_empty_length_completion"],
                **{field: "" for field in HUMAN_FIELDS},
            })

    print(
        f"Wrote {len(selected)} review rows to {output}: "
        f"{len(positives)} automated positives + {len(selected) - len(positives)} sampled failures"
    )
    print("No human validation claim should be made until the blank fields are completed and checked.")


if __name__ == "__main__":
    main()
