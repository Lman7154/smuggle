# Text-free derived outcomes

These CSV files are the public statistical record for the report. They contain no
prompt, encoded-input, response, adversarial-turn, trigger-word, or judge-rationale
text.

- `exp1_metrics.csv`: 6,000 Experiment 1 calls with mutation seed, input hash,
  compliance label, diagnostic guard labels, fallback marker, canonical-visibility
  value, and numeric reference-tokenizer counts.
- `exp3a_metrics.csv`: 1,200 conversations with final-probe hash, pre-probe,
  final-probe and overall success labels, per-turn success bits, first-success depth,
  protocol-phase hash/label, text-free output-token/truncation provenance, timestamps,
  and retained internal-retry counts.
- `manifest.json`: source/public hashes, row-count checks, guard threshold, dataset
  identity, and known provenance limits.

Run `python scripts/make_report_tables.py` from the repository root to regenerate all
CSV, JSON, and LaTeX result tables. See `DATA_AVAILABILITY.md` for the privacy boundary
and `REPRODUCING.md` for expected values.
