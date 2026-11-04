"""Create a history-free, allowlisted public release directory.

The private repository and `runs/` archive are left untouched. The destination must
not already exist, preventing accidental overwrite or deletion.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_URL = "https://github.com/Lman7154/smuggle"

EXACT_FILES = [
    ".gitignore",
    ".gitattributes",
    "README.md",
    "REPRODUCING.md",
    "DATA_AVAILABILITY.md",
    "RESPONSIBLE_USE.md",
    "SECURITY.md",
    "CITATION.cff",
    "pyproject.toml",
    "run_experiment.py",
    "compare_exp1_exp3a.py",
    "configs/run_manifest.json",
    "scripts/export_public_results.py",
    "scripts/make_report_tables.py",
    "scripts/make_report_figures.py",
    "scripts/make_judge_audit_sheet.py",
    "scripts/prepare_public_release.py",
    "data/derived/README.md",
    "data/derived/manifest.json",
    "data/derived/exp1_metrics.csv",
    "data/derived/exp3a_metrics.csv",
    "results/tables/README.md",
    "results/observation_ledger.md",
    "results/tables/analysis_summary.json",
    "results/tables/exp1_behaviour_asr.csv",
    "results/tables/exp1_variant_metrics.csv",
    "results/tables/exp1_targeting_tests.csv",
    "results/tables/exp3a_cell_rates.csv",
    "results/tables/exp3a_provenance.csv",
    "results/tables/exp1_vs_exp3a_tests.csv",
    "results/tables/exp1_behaviour_table.tex",
    "results/tables/exp1_variant_table.tex",
    "results/tables/targeting_tests_table.tex",
    "results/tables/exp3a_rates_table.tex",
    "results/tables/exp3a_provenance_table.tex",
    "results/tables/context_tests_table.tex",
    "report_final/report.tex",
    "report_final/myBibFile.bib",
    "report_final/mdframed.sty",
    "report_final/needspace.sty",
    "report_final/up-logo.jpg",
    "report_final/figures/pipeline.png",
    "report_final/figures/pipeline.pdf",
    "report_final/figures/bypass_compliance.png",
    "report_final/figures/bypass_compliance.pdf",
    "report_final/docx_build/build_docx.py",
    "report_final/docx_build/ieee.csl",
    "report_final/docx_build/tex2docx.lua",
]

GLOBS = [
    "smuggle/templates/*.yaml",
    "tests/*.py",
]

PUBLIC_PACKAGE_FILES = [
    "smuggle/__init__.py",
    "smuggle/analysis.py",
    "smuggle/context_templates.py",
    "smuggle/engine.py",
    "smuggle/experiments.py",
    "smuggle/export.py",
    "smuggle/guardrail.py",
    "smuggle/jbb.py",
    "smuggle/printing.py",
    "smuggle/pyrit_wiring.py",
    "smuggle/runner.py",
    "smuggle/tokens.py",
    "smuggle/triggers.py",
]

FORBIDDEN_PARTS = {
    ".git", ".kiro", "runs", ".revert-backup-b-c", "__pycache__",
    ".pytest_cache", ".hypothesis", "research_notes", "transcript",
}
WINDOWS_USER_PATH = re.compile(
    r"\b[A-Za-z]:(?:\\{1,2}|/)Users(?:\\{1,2}|/)[^\\/\s\"']+",
    re.IGNORECASE,
)

SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "Windows user path": WINDOWS_USER_PATH,
}
TEXT_SUFFIXES = {".py", ".md", ".tex", ".bib", ".yaml", ".yml", ".json", ".csv", ".cff", ".toml", ".sty", ".lua"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _allowlist() -> list[Path]:
    files = {ROOT / relative for relative in EXACT_FILES + PUBLIC_PACKAGE_FILES}
    for pattern in GLOBS:
        files.update(path for path in ROOT.glob(pattern) if path.is_file())
    missing = [path for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError("missing release files: " + ", ".join(str(path) for path in missing))
    return sorted(files)


def _scan_text(path: Path) -> None:
    if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {".gitignore", ".gitattributes"}:
        return
    text = path.read_text(encoding="utf-8", errors="strict")
    for label, pattern in SECRET_PATTERNS.items():
        if pattern.search(text):
            raise ValueError(f"{label} pattern found in {path.relative_to(ROOT)}")


def _is_text(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.name in {".gitignore", ".gitattributes"}


def _copy_canonical(source: Path, target: Path) -> None:
    """Copy a release file, normalizing public text to Git's declared LF form."""
    if _is_text(source):
        text = source.read_text(encoding="utf-8", errors="strict")
        canonical = text.replace("\r\n", "\n").replace("\r", "\n")
        target.write_text(canonical, encoding="utf-8", newline="\n")
    else:
        shutil.copy2(source, target)


def _validate_derived_csv(path: Path) -> None:
    expected_schemas = {
        "exp1_metrics.csv": {
            "row_id", "model", "family", "mode", "mutation_seed", "goal_id",
            "encoded_sha256", "judge_success", "encoded_guard_flagged",
            "clean_guard_score", "clean_guard_flagged", "used_fallback",
            "imperceptibility", "reference_tokenizer", "clean_tokens", "encoded_tokens",
        },
        "exp3a_metrics.csv": {
            "row_id", "model", "family", "mode", "mutation_seed", "goal_id",
            "probe_sha256", "pre_probe_success", "final_probe_success", "overall_success",
            "encoded_guard_flagged", "clean_guard_score", "clean_guard_flagged",
            "used_fallback", "imperceptibility", "executed_turns", "first_success_turn",
            "per_turn_success_bits", "protocol_phase", "protocol_sha256",
            "provenance_database", "attack_total_retries", "attack_result_timestamp_utc",
            "final_message_timestamp_utc", "max_recorded_output_tokens",
            "final_recorded_output_tokens", "any_recorded_output_over_claimed_cap",
            "final_recorded_output_over_claimed_cap", "any_empty_length_completion",
            "final_empty_length_completion", "final_finish_reason",
        },
    }
    if path.name not in expected_schemas:
        return
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        headers = {header.lower() for header in (reader.fieldnames or [])}
        if headers != expected_schemas[path.name]:
            missing = expected_schemas[path.name] - headers
            extra = headers - expected_schemas[path.name]
            raise ValueError(f"schema mismatch in {path}: missing={sorted(missing)}, extra={sorted(extra)}")
        count = sum(1 for _ in reader)
    expected = 6000 if path.name == "exp1_metrics.csv" else 1200
    if count != expected:
        raise ValueError(f"expected {expected} rows in {path}, found {count}")


def _validate_derived_public_hashes(release_root: Path) -> None:
    """Verify the nested data manifest against the canonical files to be published."""
    manifest_path = release_root / "data" / "derived" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes = manifest.get("public_hashes")
    expected_names = {"exp1_metrics.csv", "exp3a_metrics.csv"}
    if not isinstance(hashes, dict) or set(hashes) != expected_names:
        raise ValueError(
            f"derived public_hashes must name exactly {sorted(expected_names)}"
        )
    for name in sorted(expected_names):
        actual = _sha256(manifest_path.parent / name)
        if hashes[name] != actual:
            raise ValueError(
                f"derived public hash mismatch for {name}: "
                f"manifest={hashes[name]} actual={actual}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="public_release")
    args = parser.parse_args()
    destination = (ROOT / args.out).resolve()
    if destination.exists():
        raise FileExistsError(
            f"destination already exists: {destination}. Choose a new directory; "
            "the release builder never deletes, overwrites, or retains stale files."
        )
    destination.mkdir(parents=True)

    manifest: list[dict[str, object]] = []
    for source in _allowlist():
        relative = source.relative_to(ROOT)
        if any(part in FORBIDDEN_PARTS for part in relative.parts):
            raise ValueError(f"forbidden release path: {relative}")
        _scan_text(source)
        _validate_derived_csv(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        _copy_canonical(source, target)
        manifest.append({
            "path": relative.as_posix(),
            "bytes": target.stat().st_size,
            "sha256": _sha256(target),
        })

    _validate_derived_public_hashes(destination)

    release_manifest = {
        "format": "history-free allowlisted snapshot",
        "repository_url": REPOSITORY_URL,
        "file_count": len(manifest),
        "files": manifest,
        "excluded": [
            "Git history and IDE configuration",
            "runs/ raw JSONL, SQLite, caches, and transcripts",
            "research notes, backup notebooks, and one-off replay scripts",
            "final report DOCX/PDF and LaTeX compilation artifacts",
        ],
        "license_status": "No LICENSE file; all rights reserved pending author/university approval.",
    }
    (destination / "RELEASE_MANIFEST.json").write_text(
        json.dumps(release_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Prepared {len(manifest)} allowlisted files in {destination}")
    print(f"Review this directory, then publish it at {REPOSITORY_URL}.")


if __name__ == "__main__":
    main()
