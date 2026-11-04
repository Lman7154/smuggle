# Reproducing the reported statistics

The preferred reproduction path is offline and uses only text-free derived outcomes.
It does not contact a model endpoint or expose the harmful prompts and responses in the
private evidence archive.

## Environment

Tested in the research workspace with Python 3.13 and PyRIT 1.1.0. The package declares
Python 3.10 or later.

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -e .
```

## Regenerate all numerical tables

From the repository root:

```bash
python scripts/make_report_tables.py
python scripts/make_report_figures.py
```

The commands validate the public rows, write statistical artifacts to
`results/tables/`, and regenerate the two report figures in `report_final/figures/`.

Expected headline checks in `results/tables/analysis_summary.json`:

| Check | Expected value |
|---|---:|
| Experiment 1 raw rows | 6,000 |
| Experiment 1 byte-unique rows | 5,811 |
| Experiment 1 unique-row compliance events | 69 |
| Eligible joint bypass-and-compliance events | 1 |
| Valid targeting tests | 9 |
| Smallest raw targeting p-value | 1.0 |
| Smallest Holm targeting p-value | 1.0 |
| Experiment 3a pre-probe successes | 101 |
| Experiment 3a final-probe successes | 16 |
| Final success after / without pre-probe success | 9/101 / 7/1,099 |
| Matched Exp. 1-only / Exp. 3a-only within those strata | 4/9 / 6/4 |
| Experiment 3a overall successes | 108 |
| Initial generic-convention conversations | 19 |
| Family-specific convention conversations | 1,181 |
| Conversations with any recorded output above 4,096 tokens | 187 |
| Empty length-truncated final completions | 138 |
| Conservative phase/cap sensitivity pairs retained | 1,002 |
| Minimum Holm p-value in that sensitivity | 1.0 |
| Observable-final sensitivity pairs retained | 1,062 |
| Observable Tag Block finals | Gemma 0/100; gpt-oss 0/41; Nemotron 0/37 |
| Nonretained adaptive attack-result entries | 914 (872 errors, 42 non-error) |
| Retained conversations with an internal retry | 12 |
| Byte-paired Exp. 1 successes | 13 |
| Byte-paired Exp. 3a final-probe successes | 16 |
| Exp. 1-only / Exp. 3a-only discordances | 10 / 13 |
| Clustered risk difference, Exp. 3a minus Exp. 1 | 0.0025 |
| Goal-block sign-flip p-value | 0.68359375 |

The cluster bootstrap uses 50,000 draws and seed `20261103`; its expected percentile
interval is `[-0.005, 0.010]`.

## Statistical choices

- The JBB Goal is the inferential unit for the primary compliance analysis.
- Wilson intervals are reported for binomial Goal-level rates because outcomes are
  sparse and close to zero. Variant-delivery rates are descriptive and receive no
  confidence interval in the report because each Goal contributes repeated variants.
- Exact two-sided McNemar tests compare paired binary outcomes using only discordant
  Goals.
- Matched odds ratios are `b/c`; exact conditional intervals come from a
  Clopper–Pearson interval on `b | (b+c)` transformed to odds.
- Holm adjustment controls family-wise error across nine targeting tests and,
  separately, twelve Experiment 1/3a model-family tests.
- The targeting analysis uses mutation seed 0 in each arm, excludes 25 fallback Goals,
  and excludes Tag Block because whole-request hiding has no word-position treatment.
- Pooled Experiment 1/3a counts are descriptive because the same 100 Goals recur in
  twelve cells. The bootstrap and sign-flip sensitivity retain all cells belonging to
  a Goal as one cluster.
- A protocol sensitivity excludes the union of 19 generic-template conversations and
  187 conversations with any recorded target output above 4,096 tokens. The sets
  overlap, leaving 1,002 byte-paired observations. The counts remain `b=10,c=13`
  because every excluded pair is a concordant failure; this verifies bookkeeping but
  provides little stress on the paired contrast.
- The 101 pre-probe-positive and 1,099 pre-probe-negative rows are post-treatment
  descriptive strata, not randomized context conditions.

## Rebuild the private byte-pair audit

Researchers with access to the ignored private archive can run:

```bash
python compare_exp1_exp3a.py \
  --exp1-progress runs/full/exp1_full_progress.jsonl \
  --exp3a runs/exp3a_full/exp3a_full_rows.jsonl \
  --out results/comparison
```

The command requires exactly 1,200 pairs and fails if the targeted seed-0 Experiment 1
bytes differ from an Experiment 3a final probe.

## Regenerate the sanitized public rows

This command reads private harmful material and should run only in the private research
checkout:

```bash
python scripts/export_public_results.py \
  --exp1-progress runs/full/exp1_full_progress.jsonl \
  --exp3a-progress runs/exp3a_full/exp3a_full_progress.jsonl \
  --guard-cache runs/guard_cache.json \
  --exp3a-databases runs/exp3a_full.archive1.db runs/exp3a_full.db \
  --out data/derived
```

Review `data/derived/manifest.json`, rerun the table generator, and inspect the Git diff
before publishing. The exporter records source hashes so an examiner can confirm which
private files produced the public rows.

## Optional human judge audit

```bash
python scripts/make_judge_audit_sheet.py
```

This creates the ignored `private/exp3a_judge_audit.csv` with all 108 automated
positive conversations and a deterministic, cell-stratified sample of 120 failures.
The worksheet contains identifiers and blank human-label fields, not model text. A
human reviewer must complete it against the private transcripts before the report can
claim Experiment 3a judge agreement or estimate false-negative frequency.

## Live experiment replay

Live replay is not required to verify the reported arithmetic. It may use paid cloud
quota and may not reproduce identical transcripts because hosted model state is not
pinned. Commands and caveats are documented in `README.md` and exact settings in
`configs/run_manifest.json`.

## Report build

The LaTeX source consumes generated files with paths such as:

```tex
\input{../results/tables/exp1_behaviour_table.tex}
```

Compile from `report_final/` after running the table generator. A TeX compiler was not
installed in the audited Windows workspace, so the native LaTeX/BibTeX PDF must still
be compiled and visually checked on a machine with a LaTeX distribution. As a layout
proxy, the editable path was executed successfully:

```bash
python report_final/docx_build/build_docx.py \
  --out report_review.docx --pdf report_final/report_review.pdf
```

The Word-produced review PDF has 42 A4 pages, text on every page, no unresolved references,
and legible sampled result tables. It does not replace the native TeX build.
