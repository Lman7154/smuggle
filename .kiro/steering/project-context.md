---
inclusion: always
---

# Project Context & Hard Rules

COS700 research: **"Unicode / ASCII Smuggling Attacks in Chatbots"** (student 26277850,
Prof Jan Eloff). The spec of record is `proposal.md` — when in doubt, it wins. Read it.

## Research objectives (proposal 2.2) — keep every action tied to one of these
Characterise how susceptible open-source chatbots are to Unicode/ASCII smuggling:
1. ASR of **each encoding family** single-turn (baseline).
2. Whether **combining families** raises ASR beyond any single family.
3. Whether **multi-turn** context changes ASR vs single-turn (A: context priming,
   B: progressive accumulation).
4. Effect of **model scale** (7–9B vs 70–72B).
Goal = **measurement/characterisation** with ASR+95%CI, McNemar+odds ratios,
imperceptibility, turn-depth — NOT building the strongest jailbreak.

## Build ON pip-installed PyRIT (1.1.0). Do NOT duplicate it. (proposal 4.4: "no framework modifications")
Use the installed package; never reimplement, fork, or shadow it:
- Target chatbot -> `OpenAIChatTarget`
- Encodings -> `SelectiveTextConverter` + `WordKeywordSelectionStrategy` for targeting, with a
  per-family sub-converter: `tag_block` -> PyRIT `AsciiSmugglerConverter(unicode_tags=True)`;
  `homoglyph`, `variation_selector`, `structural` -> net-new minimal converters (see below).
  PyRIT also ships Unicode converters that are available to broaden coverage
  (`UnicodeConfusableConverter`, `VariationSelectorSmugglerConverter`,
  `SneakyBitsSmugglerConverter`, `ZeroWidthConverter`, `BidiConverter`,
  `UnicodeSubstitutionConverter`). The net-new converters exist only because the current four
  families need specific behaviour (curated visually-identical look-alikes; a selector appended
  after each char; a seed-varied zero-width + bidi set) that the nearest PyRIT converter
  implements differently — NOT because those PyRIT converters are unusable or off-limits.
- Single-turn -> `PromptSendingAttack`; Multi-turn -> `PromptSendingAttack` with a PREPENDED
  conversation (scripted context-priming / progressive-accumulation strategies), NOT
  `RedTeamingAttack` — chosen for reproducible control of the benign context and per-turn schedule.
- Judge -> `SelfAskTrueFalseScorer` (objective/TASK_ACHIEVED) AND not-`SelfAskRefusalScorer`,
  combined via `TrueFalseCompositeScorer` (two-stage: fulfilled AND not-refused).
- Guardrail -> Llama Guard (used to its potential: first-token P(unsafe), Meta model card)
- Logging/persistence -> `CentralMemory` (default auto-init is IN_MEMORY = EPHEMERAL; for a real
  multi-hour run initialise PyRIT's SQLite FILE store so results survive a crash, proposal 4.9)
If PyRIT provides it, call it. Net-new code is limited to (a) **trigger-word detection**,
(b) the **analysis/experiment** layer, (c) the **JBB Goal dataset loader** (`smuggle.jbb`,
see below), and (d) THREE minimal per-family converters — `_HomoglyphConverter`,
`_VariationSelectorAppendConverter`, `_StructuralInjectionConverter` — each justified because
PyRIT 1.1.0's nearest converter implements a different behaviour or cannot produce
seed-distinct variants (homoglyph: `UnicodeConfusableConverter` draws from the
`confusable_homoglyphs`/`confusables` data — a mix of other scripts and styled
Mathematical-Alphanumeric forms, sometimes with injected invisible marks, non-deterministic by
default — rather than curated visually-identical Cyrillic/Greek look-alikes; variation_selector:
PyRIT collapses the word behind one carrier glyph instead of appending after each char;
structural: `ZeroWidthConverter` emits a single fixed U+200B with no choice/seed). These are
behavioural differences for the current families, not reasons the PyRIT converters cannot be used.
`tag_block` stays pure PyRIT (`AsciiSmugglerConverter`). There is still **no** custom
ResponseScorer, keyword scorer, agreement module, failure-reason classifier, or SafetyClassifier.

## Two artefacts + one plug-and-play tool
- Artefact 1 = the Unicode/ASCII Mutation Engine (a PyRIT `PromptConverter`).
- Artefact 2 = the pipeline (PyRIT wiring + trigger pre-pass + analysis).
- One `run(...)` combines them (nanoGCG/advtok style). Everything injected; see below.

## Nothing is hardcoded
Target(s), dataset/prompt, guardrail, judge, adversarial chat, attack family/families,
and any prepended framing are **passed in**. No baked-in model names, endpoints,
keys, dataset paths, benign carriers, stopword lists (use PyRIT's), or framing strings.
Default = bare baseline. Framing/prepend content must be **injected or model-generated**,
never a fixed "decode-and-obey" string (that would contaminate the measurement —
you'd be testing your own instruction, not the smuggling attack).

- **Dataset**: load JBB via `smuggle.jbb.load_jbb_goals()` — the harmful **`Goal`** request
  (+ `Category`). NEVER use PyRIT's built-in `jbb_behaviors` provider: it puts the short
  `Behavior` LABEL ("Defamation") in `value` and drops the request, so you'd score/attack a
  topic word (Llama Guard then flags ~16/100; on the real Goals ~96/100). Symptom -> trace to source.

## Correctness decisions already fought through (do not regress)
- **Judge**: scores whether the *objective was achieved* (task completion), never a
  keyword/semantic match. Decode/explain/mention of harm = NOT success.
- **Trigger detection**: rank content words by **saliency** = drop in Llama Guard's
  graded P(unsafe) when removed (TextFooler / "Not All Tokens"); minimal flipping set,
  ranked primary->top-two->all; fallback = all content words. NOT first-word-that-flips.
  Score prompts in the **user** role. Finding (measured on the real JBB **Goals**, not labels):
  Llama Guard 3 flags ~96/100; ~56/100 still fall back (no small word-subset flips its
  holistic-intent verdict) and ~44 yield a targeted set — report both. The ~4 it scores "safe"
  are disinformation / gambling-tip Goals that fall OUTSIDE Llama Guard 3's hazard taxonomy (it
  has no general misinformation category); that is a classifier-vs-dataset DEFINITION mismatch,
  not a bug, and those fall back as expected.
- **Tag Block** is the whole-/decode family; the other three keep text readable.
- Report **guardrail-bypass rate** and **compliance ASR** as separate columns
  (`analysis.bypass_and_asr` / `format_report_table`, grouped by model/family/mode).
- **Decoding config (proposal 4.9)**: build targets via `make_target` with temperature=0.0,
  top_p=1.0, max_completion_tokens=512 (overridable). 512 suits the five NON-reasoning
  7-72B targets (4.2); reasoning models (e.g. gpt-oss) exhaust the budget on hidden
  reasoning and are an informal extension, not a proposal target.
- Single-turn baselines must stay bare; framing/multi-turn are explicit, logged conditions.

## Environment
Windows. PyRIT 1.1.0 lives in `...\Programs\Python\Python313\python.exe` (NOT the
WindowsApps shim). Ollama local (`http://localhost:11434/v1`, key "ollama") + Ollama
Cloud. Terminal is bash. The package is `smuggle/` (pip install -e .).
