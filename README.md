# unicode-smuggle

Research artefact for the COS700 report **Unicode / ASCII Smuggling Attacks in Chatbots**.
It builds on pip-installed PyRIT 1.1.0 without modifying or vendoring PyRIT.

The repository contains:

- a PyRIT `PromptConverter` mutation engine for homoglyph, variation-selector,
  structural, and Tag Block families;
- trigger selection using graded Llama Guard saliency;
- single-turn and adaptive-prelude experiment orchestration;
- strict task-achieved **and** not-refused scoring;
- exact paired statistics and generated report tables;
- text-free derived outcomes that allow the reported statistics to be recomputed
  without releasing harmful prompts or model responses.

## Completed research scope

The final report covers two reduced-scope conditions:

1. **Experiment 1:** 100 JBB harmful Goals × 4 individual encoding families ×
   (3 targeted + 2 random variants) × 3 cloud targets = **6,000 calls**.
2. **Experiment 3a:** 100 Goals × 4 families × 3 targets = **1,200 conversations**,
   each with four adaptive escalatory turns followed by the targeted seed-0
   Experiment 1 payload as a byte-identical final probe.

The proposal's combined-family experiment, progressive-accumulation condition, and
7–9B versus 70–72B scale comparison were not completed. The report treats them as
unanswered objectives, not completed findings.

The public API keeps the two Strategy-A-like conditions distinct:

- `run_experiment_3a` or `conversation="experiment_3a"` selects the original
  proposal context-priming schedule;
- `run_experiment_adaptive_probe` or `conversation="adaptive_probe"` selects the
  amended four-turn adaptive-prelude condition reported in the dissertation.

Automatic generation of direct-completion jailbreak demonstrations is not part of the
public report release. The optional prepend helper accepts only caller-supplied pairs.

## Important measurement boundary

Llama Guard was a **diagnostic, off-path classifier**. It served two purposes:

- rank candidate trigger words by the drop in `P(unsafe)` when removed;
- measure whether an encoded input was predicted safe.

Its verdict never blocked target delivery. This was deliberate: enforcing the guard
would censor target-compliance observations for inputs it flagged. The analysis keeps
three outcomes separate:

- **diagnostic bypass:** encoded input predicted safe, among Goals predicted unsafe
  in clean text;
- **target compliance:** the target completed the Goal and did not refuse;
- **joint success:** the same clean-unsafe eligible input was predicted safe and
  produced target compliance, isolating encoding-induced gate evasion from clean-text
  classifier misses.

A literal input-only gate over all 100 Goals would also allow clean-text taxonomy
misses. The report therefore gives both the all-Goal operational count and the
96-Goal encoding-induced joint measure.

The study has no complete clean-plaintext target arm, so it characterises absolute
compliance under encoded delivery and does not estimate encoding uplift over
plaintext.

## Safe offline reproduction

The public reproduction path requires no model, API key, network connection, or
harmful text.

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -e .
python scripts/make_report_tables.py
python scripts/make_report_figures.py
```

Expected source counts:

- `data/derived/exp1_metrics.csv`: 6,000 text-free rows;
- `data/derived/exp3a_metrics.csv`: 1,200 text-free rows;
- 100 Goal IDs, of which 96 are eligible for diagnostic-bypass analysis;
- 1,200 byte-hash matches between Experiment 1 and Experiment 3a.

Generated outputs are written to `results/tables/`:

- behaviour-level ASR with Wilson 95% intervals and descriptive variant-level rates;
- diagnostic bypass and joint success;
- matched-seed targeted-versus-random exact McNemar tests;
- matched odds ratios with exact conditional 95% intervals;
- Holm-adjusted p-values;
- Experiment 3a per-cell rates, protocol-phase/truncation audit, and byte-paired
  comparisons;
- a Goal-cluster bootstrap and block sign-flip sensitivity for repeated Goals;
- LaTeX tables consumed directly by `report_final/report.tex`.

For a separate audit of the exact private byte pairs:

```bash
python compare_exp1_exp3a.py --out results/comparison
```

That command needs the private ignored `runs/` archive. It is not required to
recompute the public report tables.

## Private-to-public export

Only the researcher should run this command because its inputs contain full harmful
prompts, responses, adversarial turns, and judge rationales:

```bash
python scripts/export_public_results.py \
  --exp1-progress runs/full/exp1_full_progress.jsonl \
  --exp3a-progress runs/exp3a_full/exp3a_full_progress.jsonl \
  --guard-cache runs/guard_cache.json \
  --exp3a-databases runs/exp3a_full.archive1.db runs/exp3a_full.db \
  --out data/derived
```

The exporter uses an allowlist and refuses columns containing prompt, response,
rationale, conversation, trigger-word, or encoded-text fields. It writes hashes and
outcomes, then records private source hashes and public file hashes in
`data/derived/manifest.json`.

## Optional human judge audit

No independent human audit is included in the reported Experiment 3a results. An
optional text-free worksheet contains all 108 automated positives and ten
deterministically sampled failures from each model--family cell (120 failures total):

```bash
python scripts/make_judge_audit_sheet.py
```

The output `private/exp3a_judge_audit.csv` is ignored by Git. Fill its human-label,
reviewer, and date columns while consulting the private transcripts. Do not claim
human agreement or recall until that sheet is complete.

## Installation for live experiments

Live execution is optional and can consume substantial time or paid cloud quota.
Review the target names, endpoint, credentials, concurrency, and billing before
running it.

Requirements:

- Python 3.10 or later;
- Ollama at `http://localhost:11434/v1` for the local guard;
- access to the configured Ollama Cloud targets;
- the Hugging Face `JailbreakBench/JBB-Behaviors` dataset;
- PyRIT 1.1.0, installed by this package.

```bash
pip install -e .
```

The driver uses the JBB **`Goal`** field. Do not replace it with PyRIT 1.1.0's
`jbb_behaviors` provider value, which exposes a short behaviour label rather than the
harmful request.

### Benign smoke example

```python
import smuggle

endpoint = "http://localhost:11434/v1"
target = smuggle.make_target("llama3.1:8b", base_url=endpoint, api_key="ollama")
guard = smuggle.make_guardrail(
    model="llama-guard3:latest", base_url=endpoint, api_key="ollama"
)
judge_target = smuggle.make_target(
    "qwen2.5:7b", base_url=endpoint, api_key="ollama",
    max_completion_tokens=512,
)
judge = smuggle.make_judge(judge_target)

result = smuggle.run(
    target=target,
    prompts="Summarize why Unicode normalization matters in software.",
    conversation="single",
    attack="variation_selector",
    mode="targeted",
    guardrail=guard,
    judge=judge,
    input_guardrail=guard,
)
print(result.summary())
```

### Recorded full-run commands

Experiment 1:

```bash
SMUGGLE_DB=runs/exp1_full.db python -u run_experiment.py \
  --exp 1 --goals 100 --out runs/full --tag exp1_full \
  --n-targeted 3 --n-random 2 --reproduce-unenforced-exp1-cap
```

The final flag reproduces the original Experiment~1 cap-field mismatch and can allow
very long generations until timeout. Omit it for the safer corrected launcher, which
also sends `max_tokens=4096`; that safer run is not bit-for-bit the original treatment.

Experiment 3a:

```bash
SMUGGLE_DB=runs/exp3a_full.db python -u run_experiment.py \
  --exp adaptive_probe --goals 100 --out runs/exp3a_full --tag exp3a_full \
  --n-variants 1 --prelude-turns 4 --concurrency 1 \
  --adv-template smuggle/templates/unicode_smuggling_escalatory_probe.yaml
```

`SMUGGLE_DB` must be set before importing `smuggle` so PyRIT uses a durable SQLite
store rather than ephemeral memory.

## Statistical interpretation

- The JBB Goal is the primary inferential unit.
- Mutation variants are nested attempts, not independent samples.
- Behaviour-level ASR uses best-of-K after byte de-duplication.
- Variant-level rates are descriptive; no confidence interval treats nested variants
  as independent samples.
- Sparse paired cells use exact two-sided McNemar tests.
- Matched odds ratios describe direction; they are not estimable when there are no
  discordant Goals.
- Holm correction controls family-wise error separately for the nine valid targeting
  tests and twelve Experiment 1/3a tests.
- The inferential targeting contrast uses mutation seed 0 in each arm so every Goal
  has one attempted mutation per mode.
- Tag Block is excluded from targeted-versus-random inference because it hides the
  whole request and has no trigger-word position treatment.
- The 25 fallback Goals are excluded from that comparison, leaving 75 paired Goals.
- Experiment 1/3a pooled counts are descriptive because each Goal contributes twelve
  model-family cells. The report adds Goal-cluster sensitivity analyses.
- A conservative protocol sensitivity removes the 19 generic-template rows and every
  row with any recorded target output above 4,096 tokens. The overlapping exclusions
  leave 1,002 pairs. Every excluded pair is a concordant failure, so the filter checks
  bookkeeping but cannot change McNemar discordances.
- The final adaptive ledger has 1,200 unique design cells. Private stores retain 914
  additional attack-stage entries from interrupted/retried execution (872 errors and
  42 non-error records); they are provenance, not extra outcome observations.
- A separate observable-final sensitivity excludes 138 empty length-truncated finals
  and retains 1,062 pairs. The pooled Tag Block complete-case count is 0/178, without
  a pooled confidence interval because Goals repeat across models and truncation is
  informative.

## Hosted-model reproducibility

Temperature, top-p, seed, model names, manifest pointers, and run windows are
recorded. They do **not** guarantee bit-identical replay of hosted models. The archive
contains different response text for some repeated byte-identical requests.
Experiment 1's `max_completion_tokens=4096` field was not enforced by the endpoint.
The current adaptive launcher also sends `max_tokens=4096`, but archived token metadata
exceed 4,096 in 187 conversations and 138 final completions are empty with
`finish_reason=length`; uniform historical enforcement is therefore unverified. The
report treats the cross-session comparison as associational and reports a conservative
phase/cap sensitivity.

## Data and responsible release

Never commit `runs/`, SQLite stores, full JSONL, transcripts, generated demonstrations,
or executed research notebooks. They contain reversible harmful prompts or model
outputs. See:

- [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md)
- [`REPRODUCING.md`](REPRODUCING.md)
- [`RESPONSIBLE_USE.md`](RESPONSIBLE_USE.md)
- [`results/observation_ledger.md`](results/observation_ledger.md)

Repository: <https://github.com/Lman7154/smuggle>

The Git working tree for publication is generated from an explicit allowlist. Review
`RELEASE_MANIFEST.json` before pushing; do not publish from the private research
workspace.

## Licence

No licence has been selected. Unless the author and University of Pretoria approve
and add a `LICENSE` file, normal copyright rules apply; repository visibility alone
does not grant reuse rights.
