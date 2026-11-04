# Data availability and privacy boundary

## Public files

The designated repository is <https://github.com/Lman7154/smuggle>.

`data/derived/exp1_metrics.csv` and `data/derived/exp3a_metrics.csv` contain enough
text-free information to recompute the report's counts, confidence intervals, paired
tests, effect sizes, multiplicity correction, byte-pair checks, and turn-depth
summary.

Public fields include:

- model, family, mode, mutation seed, and stable Goal ID;
- SHA-256 of the encoded input/final probe for de-duplication and pairing;
- strict judge success booleans;
- encoded- and clean-input diagnostic guard outcomes;
- fallback and canonical-visibility measurements;
- numeric token counts;
- Experiment 3a per-turn success bits, first-success depth, protocol phase, text-free
  finish-reason/output-token/truncation fields, retained-message timestamps, and
  internal-retry counts.

The public manifest records schemas, row counts, source/public SHA-256 values, the
diagnostic guard threshold, dataset identifier, and known reproducibility limits.

## Withheld private evidence

The following remain in the ignored `runs/` archive and must not be pushed:

- JBB Goal text and reversible encoded prompts;
- target responses, including harmful successful outputs;
- adversarial conversation turns;
- judge rationales that may quote or summarize harmful material;
- trigger words and generated demonstrations;
- PyRIT SQLite stores, progress ledgers, raw JSONL, and full transcripts.

These records are retained for supervisor/examiner audit on request. Withholding them
prevents the public repository from becoming a corpus of directly reusable harmful
outputs while the text-free rows preserve statistical verifiability.

## Diagnostic guard terminology

Llama Guard was measured off-path and did not block requests. `diagnostic bypass`
therefore means that it predicted an encoded input safe. The bypass denominator is
restricted to the 96 Goals it predicted unsafe in clean text. `joint success` means
that the same eligible encoded input was predicted safe and the target completed the
Goal; it isolates encoding-induced gate evasion. A literal all-Goal gate also permits
clean-text classifier misses, so those operational misses are reported separately.
No guard was enforced during target delivery.

## Known provenance limits

- The original run did not record an immutable Hugging Face dataset revision.
- Ollama Cloud manifest pointers do not hash the served weights.
- Hosted responses can differ across sessions despite temperature 0 and a requested
  seed.
- Experiment 1 and Experiment 3a used different requested output-cap fields. The
  current adaptive launcher sends `max_tokens=4096`, but 187 archived conversations
  contain at least one recorded target output above 4,096 tokens; uniform enforcement
  is therefore unverified.
- The adaptive archive contains 19 initial generic-convention conversations and
  1,181 family-specific conversations. Public rows identify both phases.
- The final progress ledger contains 1,200 unique design cells. The two PyRIT stores
  also contain 914 nonretained attack-result entries (872 errors and 42 non-error
  attack-stage records); 12 retained conversations record one internal retry. These
  counts are provenance, not extra outcome observations.
- The Experiment 1 positive-only researcher check was not preserved as row-level
  adjudications, and its heuristic failure spot-check was not recorded as a fixed
  review set.
- No independent Experiment 3a human audit is included in the reported analysis. Its
  optional private 228-row worksheet contains 108 automated positives and 120 sampled
  failures, but its human label, reviewer, date, and note fields remain blank.

These omissions are declared in the report and manifest rather than reconstructed or
invented after the fact.
