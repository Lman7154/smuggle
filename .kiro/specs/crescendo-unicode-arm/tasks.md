# Implementation Plan: Crescendo-Unicode Arm

## Overview

Three workstreams, sequenced lowest-risk first so every step is independently verifiable and, wherever possible, needs no live model. Workstream A removes the shipped harmful CipherChat banks and decouples the adaptive search and jailbreak-prepend driver, which also yields a clean import surface. Workstream B adds the Crescendo-Unicode runner on the installed PyRIT `CrescendoAttack`, modelled directly on the existing `run_redteam_async`. Workstream C isolates the two factors with a 2x2 factorial driver and a `factorial_2x2` analysis with separated bypass/compliance reporting. The final task wires the new entry points into the README.

Net-new code is confined to the runner, experiments, analysis, and `pyrit_wiring` import surface (Requirement 6.6); PyRIT is called, never reimplemented, and no custom Crescendo YAML is written. Tests assert behaviour and structure (construction, signatures, dataclass shape, pure analysis math) without contacting a model; Crescendo construction is asserted from the built attack object's configuration using real targets built by `make_target` (construction only, no model call).

## Tasks

- [x] 1. Workstream A: remove shipped harmful banks, fix the import surface, decouple search and jailbreak
  - [x] 1.1 Remove the CipherChat banks and repair the public surface
    - In `smuggle/context_templates.py` delete `load_cipherchat_prepend`, `load_cipherchat_category_bank`, `DEFAULT_CIPHERCHAT_PREPEND_PATH`, `DEFAULT_CIPHERCHAT_CATEGORY_PATH`, and `LLAMA_GUARD_TEXT_CATEGORIES`; keep `load_context_templates`, `make_priming_provider`, `DEFAULT_TEMPLATES_PATH`, and `Context`
    - In `smuggle/experiments.py` delete `make_cipherchat_prepend`, `make_cipherchat_prepend_async`, `make_category_cipherchat_prepend`, and `make_category_cipherchat_prepend_async` (leave `make_jailbreak_priming`/`make_jailbreak_priming_async` and `make_decode_priming*` intact; `_resolve_category` and the remaining in-function CipherChat references are removed with the search/jailbreak refactors in 1.4 and 1.6)
    - In `smuggle/__init__.py` remove those nine names from the `from .context_templates import (...)` and `from .experiments import (...)` blocks and from `__all__`
    - Confirm the package still imports (no shipped CipherChat content file is read at import, which already holds because no such files exist on disk)
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 6.1_

  - [x]* 1.2 Write the removed-symbols import property test
    - **Property 1: Removed symbols are unimportable**
    - Parametrize over the nine removed names; assert each `from smuggle import <name>` raises `ImportError` and `getattr(smuggle, <name>)` raises `AttributeError`
    - Tag the test `**Feature: crescendo-unicode-arm, Property 1: Removed symbols are unimportable**`
    - New file `tests/test_public_surface.py`
    - **Validates: Requirements 1.1, 1.2**

  - [x]* 1.3 Write the export-list resolution property test
    - **Property 2: The public export list fully resolves**
    - Iterate `smuggle.__all__`; assert every name resolves on the package to a present, non-`None` attribute; include a fresh-import smoke assertion that `import smuggle` succeeds with no CipherChat path access
    - Tag the test `**Feature: crescendo-unicode-arm, Property 2: The public export list fully resolves**`
    - Add to `tests/test_public_surface.py`
    - **Validates: Requirements 1.3**

  - [x] 1.4 Decouple `run_adaptive_unicode_search(_async)` from the category bank
    - In `smuggle/experiments.py` remove the `categories` and `template_path` parameters and the `_resolve_category` helper; delete the `load_cipherchat_category_bank(...)` and `make_category_cipherchat_prepend_async(...)` calls
    - Add an optional `prepend=None` parameter resolved per objective via `_R._as_prepended(prepend, prompt)`; when `prepend is None` deliver each candidate with no prepended conversation (branch the `execute_async` call like `run_async` does, passing `prepended_conversation` only when a prepend is injected); add an optional `prepend_label=""` recorded on each attempt
    - Set `carrier_values = list(carriers) if carriers else [None]` (no carrier unless injected)
    - Keep candidate generation, byte-dedup, Llama Guard `unsafe_score` ranking, `target_budget`, strict task-completion stop, and the `end_to_end = success and bypassed` gate unchanged
    - Make `category` optional (default `""`) on `AdaptiveUnicodeOutcome` and `AdaptiveUnicodeAttempt`, and `prepend_label` default `""` on the attempt; because dataclass defaults must follow non-default fields, move these fields to the end of each field list (all construction sites use keyword arguments, so field order is safe); the search now records `category=""`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 6.1_

  - [x]* 1.5 Write the decoupled-search behaviour test
    - With stub target/guardrail/judge (no live model, following the `FakeGuardrail` pattern in `tests/test_triggers.py` and a recording fake target), assert `run_adaptive_unicode_search_async` completes when called with neither `categories` nor `prepend`, that each target send used `prepended_conversation=None`, and that `AdaptiveUnicodeOutcome.category == ""`
    - Assert via `inspect.signature` that `categories` is gone, `prepend` exists with default `None`, and that an injected `prepend` is delivered on top of every candidate
    - New file `tests/test_search_decoupled.py`
    - _Requirements: 2.2, 2.3, 2.4, 2.5_

  - [x] 1.6 Rewire `run_jailbreak_prepend(_async)` onto injected demonstrations
    - In `smuggle/experiments.py` replace the `template_path` parameter and the `_CT.load_cipherchat_prepend(...)` load with a required `demonstrations: Sequence[tuple[str, str]]` parameter and an optional injected `system_prompt=None`
    - Raise `ValueError` up front when `demonstrations` is empty or `None`, before any target send
    - For each target x family x variant, build the prepend via `make_jailbreak_priming_async(family, demonstrations=demonstrations, carrier=<per-variant>, system_prompt=system_prompt, encode_assistant=..., assistant_carrier=...)` and deliver through the unchanged `_R.run_async(prepend=prepend, whole_prompt=whole_prompt, prepend_label=prepend_label, ...)` path; default `prepend_label="jailbreak_prepend"`
    - _Requirements: 3.1, 3.2, 3.3, 6.1_

  - [x]* 1.7 Write the no-demonstrations rejection property test
    - **Property 3: Jailbreak prepend without demonstrations is rejected**
    - For random family selections and empty/`None` demonstrations, assert both `make_jailbreak_priming_async` and `run_jailbreak_prepend_async` raise `ValueError` before any prompt is sent (use a fake target that records sends and assert it was never called)
    - Tag the test `**Feature: crescendo-unicode-arm, Property 3: Jailbreak prepend without demonstrations is rejected**`
    - New file `tests/test_jailbreak_prepend.py`
    - **Validates: Requirements 3.2**

- [x] 2. Checkpoint - Workstream A
  - Ensure all offline tests pass and `import smuggle` is clean, ask the user if questions arise.

- [x] 3. Workstream B: Crescendo-Unicode multi-turn runner
  - [x] 3.1 Add `CrescendoAttack` to the single PyRIT import surface
    - In `smuggle/pyrit_wiring.py` add `CrescendoAttack` to the `from pyrit.executor.attack import (...)` block and to `__all__` (`SeedPrompt` is already exported, so the variant loader needs nothing new)
    - _Requirements: 4.1, 6.6_

  - [x] 3.2 Add `CrescendoOutcome` and `run_crescendo(_async)` to the runner
    - In `smuggle/runner.py` add the frozen `CrescendoOutcome` dataclass (`seed_id, objective, family, mode, unicode_on, executed_turns, backtrack_count, judge_success, last_response, model, adversarial_model, max_turns, max_backtracks, variant_label="", outcome_reason=""`)
    - Add `run_crescendo_async` modelled on `run_redteam_async`: when `attack` is set, build a targeted-inline `MutationEngine` (no `whole_prompt`) and attach it via `P.AttackConverterConfig(request_converters=ConverterConfiguration.from_converters([engine]))`; when `attack is None`, set the converter config to `None` so no converter is attached (plaintext Crescendo)
    - Pass the injected judge as `P.AttackScoringConfig(objective_scorer=judge)` (never let Crescendo build its own default scale scorer); accept and forward `adversarial`, `judge`, `attack` (family), `max_turns`, and `max_backtracks`
    - When `variant_path` is given, load it with `P.SeedPrompt.from_yaml_file(variant_path)` and pass it as `P.AttackAdversarialConfig(target=adversarial, system_prompt=<SeedPrompt>)`, recording `variant_label=str(variant_path)`; when omitted, use `P.AttackAdversarialConfig(target=adversarial)` so PyRIT resolves its shipped `crescendo_variant_1.yaml`, with `variant_label=""`
    - Extract results like `run_redteam_async`: `outcome.name == "SUCCESS"`, `last_response.get_value()`, `executed_turns`, plus `backtrack_count`; set `family=()` and `mode="plaintext"` when Unicode is off
    - In the docstring, document that both targets must be natively capable (objective: editable history + multi-turn; adversarial: multi-turn + system prompt) and that where the adversarial endpoint lacks JSON-schema enforcement (e.g. Ollama `supports_json_schema=False`) the adversarial chat must be a model capable of producing the Crescendo variant JSON by instruction-following
    - Add the thin synchronous wrapper `run_crescendo(**kwargs)` delegating via `asyncio.run`
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 6.2, 6.6, 6.7_

  - [x] 3.3 Export the Crescendo runner from the package
    - In `smuggle/__init__.py` import `run_crescendo`, `run_crescendo_async`, and `CrescendoOutcome` from `.runner` and add them to `__all__`
    - _Requirements: 1.3, 4.8_

  - [x]* 3.4 Write the converter-attachment property test
    - **Property 4: Crescendo attaches a targeted-inline mutation engine**
    - Build the attack with `run_crescendo`'s wiring using real `make_target` `OpenAIChatTarget`s (construction only, no model call); for each family in `FAMILIES`, introspect the built attack's request-converter configuration and assert exactly one `MutationEngine` configured in targeted mode with whole-prompt encoding off; assert `attack=None` attaches no request converter
    - Tag the test `**Feature: crescendo-unicode-arm, Property 4: Crescendo attaches a targeted-inline mutation engine**`
    - New file `tests/test_crescendo_runner.py`
    - **Validates: Requirements 4.2, 4.3**

  - [x]* 3.5 Write the injected-judge identity property test
    - **Property 5: The Crescendo objective scorer is the injected judge**
    - Using a judge object accepted by `CrescendoAttack` (the project composite judge or a minimal `TrueFalseScorer`), assert the built attack uses that exact object as its objective scorer (identity), never PyRIT's default scale scorer
    - Tag the test `**Feature: crescendo-unicode-arm, Property 5: The Crescendo objective scorer is the injected judge**`
    - Add to `tests/test_crescendo_runner.py`
    - **Validates: Requirements 4.4, 6.2**

  - [x]* 3.6 Write the `CrescendoOutcome` shape and variant-path test
    - Assert all `CrescendoOutcome` fields are present with correct defaults, that `family == ()` and `mode == "plaintext"` when `unicode_on` is false, and that `variant_label` defaults to `""`
    - Assert that a supplied `variant_path` sets `AttackAdversarialConfig.system_prompt` to the loaded `SeedPrompt`, and that omitting it leaves PyRIT's default (no system prompt set), both observed from the built attack's adversarial configuration without a model call
    - Add to `tests/test_crescendo_runner.py`
    - _Requirements: 4.6, 4.7_

  - [x]* 3.7 Write the composite-judge truth-table property test
    - **Property 9: The composite judge is success iff achieved and not refused**
    - Build `make_judge` with stub task-achieved and refusal sub-scorers; enumerate the `(achieved, refused)` truth table and assert the composite scores success exactly when `achieved AND NOT refused`
    - Tag the test `**Feature: crescendo-unicode-arm, Property 9: The composite judge is success iff achieved and not refused**`
    - New file `tests/test_crescendo_support.py`
    - **Validates: Requirements 6.2**

  - [x]* 3.8 Write the reproducibility and import-surface smoke test
    - Assert `make_target(...)` pins `temperature == 0.0` on the built target
    - Assert the runner reaches PyRIT only through the `pyrit_wiring` surface (`P`) plus the one locally-imported `ConverterConfiguration` already used by `run_redteam_async` (static check over `smuggle/runner.py` for direct `pyrit` imports)
    - Add to `tests/test_crescendo_support.py`
    - _Requirements: 6.4, 6.6_

- [x] 4. Checkpoint - Workstream B
  - Ensure all tests pass and the Crescendo attack constructs offline, ask the user if questions arise.

- [x] 5. Workstream C: 2x2 factorial and separated reporting
  - [x] 5.1 Add `run_crescendo_factorial(_async)` to the experiments layer
    - In `smuggle/experiments.py` add `run_crescendo_factorial_async` returning `{cell_label: [records]}` for the four conditions keyed by `seed_id`: `single_plain` (no-converter `PromptSendingAttack` on the raw Goal, scored by the injected judge via `_R._score`, optionally guardrail-scored, recorded as a `VariantOutcome` with `family=("plaintext",)`, `mode="plaintext"`, `encoded_text=text`, `imperceptibility=1.0`); `single_enc` (reuse `_R.run_async(attack=family, mode="targeted", ...)` so the payload is identical to Experiment 1); `crescendo_plain` (`run_crescendo_async(attack=None, ...)`); `crescendo_enc` (`run_crescendo_async(attack=family, mode="targeted", ...)`)
    - Hold `family`, prompts, and model constant across cells; keep the existing `(rows, errors)` failure convention per cell; add the synchronous `run_crescendo_factorial` wrapper via `asyncio.run`
    - Consume injected prompts as JBB Goals through `_R._normalise_prompts`; document that a real multi-hour run should call `init_pyrit_sqlite` before building targets so transcripts persist
    - _Requirements: 5.1, 6.1, 6.3, 6.5, 6.6_

  - [x] 5.2 Add `factorial_2x2`, `FactorialResult`, and `format_factorial_table` to the analysis layer
    - In `smuggle/analysis.py` add the frozen `FactorialResult` dataclass and `factorial_2x2(cells, *, aggregate="any")` reusing the existing primitives with no change to them
    - Compliance ASR per cell via behaviour-level best-of-N (`asr_over_prompts`/`per_prompt_success` with `aggregate="any"`) plus Wilson CI; three paired contrasts via `mcnemar_over_prompts` (`single_enc` vs `single_plain`, `crescendo_plain` vs `single_plain`, `crescendo_enc` vs `single_plain`)
    - Interaction as difference-in-differences with labels `a=single_plain, b=single_enc, c=crescendo_plain, d=crescendo_enc`: `unicode_main = b - a`, `crescendo_main = c - a`, `interaction = (d - c) - (b - a)`
    - Bypass vs compliance as separate columns via `bypass_and_asr`, with `bypass_rate = None` for the two Crescendo cells (no comparable single-payload guardrail verdict); `format_factorial_table` renders ASR and bypass in separate columns, mirroring `format_report_table`
    - _Requirements: 5.2, 5.3, 5.4, 5.5, 6.6_

  - [x] 5.3 Export the factorial driver and analysis from the package
    - In `smuggle/__init__.py` import and add to `__all__`: `run_crescendo_factorial`, `run_crescendo_factorial_async` (from `.experiments`) and `factorial_2x2`, `FactorialResult`, `format_factorial_table` (from `.analysis`)
    - _Requirements: 1.3, 5.1_

  - [x]* 5.4 Write the best-of-N property test
    - **Property 6: Best-of-N success is the maximum over draws**
    - For random per-prompt draw sets, assert `per_prompt_success(aggregate="any")` for each prompt equals the logical OR over that prompt's draw outcomes
    - Tag the test `**Feature: crescendo-unicode-arm, Property 6: Best-of-N success is the maximum over draws**`
    - New file `tests/test_factorial.py`
    - **Validates: Requirements 5.5**

  - [x]* 5.5 Write the interaction difference-in-differences property test
    - **Property 7: The factorial interaction is the difference-in-differences**
    - For random four-cell success maps, assert `factorial_2x2` reports `unicode_main == b - a`, `crescendo_main == c - a`, and `interaction == (d - c) - (b - a)`
    - Tag the test `**Feature: crescendo-unicode-arm, Property 7: The factorial interaction is the difference-in-differences**`
    - Add to `tests/test_factorial.py`
    - **Validates: Requirements 5.3**

  - [x]* 5.6 Write the separated bypass/compliance property test
    - **Property 8: Bypass and compliance are reported separately**
    - For random outcome sets with and without guardrail verdicts, assert compliance ASR and guardrail-bypass rate are two distinct values never collapsed, and that the bypass rate is `None` exactly when no outcome in the group carried a guardrail verdict (so both Crescendo cells report `None` bypass)
    - Tag the test `**Feature: crescendo-unicode-arm, Property 8: Bypass and compliance are reported separately**`
    - Add to `tests/test_factorial.py`
    - **Validates: Requirements 5.4**

  - [x]* 5.7 Write the `factorial_2x2` unit test on synthetic cells
    - On four synthetic cell lists of minimal outcome-shaped records (no model), assert the per-cell ASR, the two main effects and the interaction, the three McNemar contrasts, that both Crescendo cells carry `None` bypass while the single-turn cells carry a real bypass rate, that compliance and bypass stay in separate columns, and that `format_factorial_table` renders them in separate columns
    - Add to `tests/test_factorial.py`
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 6. Checkpoint - Workstream C
  - Ensure all tests pass and the factorial analysis math is verified, ask the user if questions arise.

- [x] 7. Final integration and documentation
  - [x] 7.1 Update the README for the new and decoupled entry points
    - Add runnable usage snippets for `run_crescendo` / `run_crescendo_async` and `run_crescendo_factorial` / `run_crescendo_factorial_async`, and update the examples for the decoupled `run_adaptive_unicode_search` (no `categories`, optional `prepend`) and `run_jailbreak_prepend` (required injected `demonstrations`)
    - Document the adversarial-must-be-capable caveat (an endpoint with `supports_json_schema=False` needs an adversarial model able to emit the Crescendo variant JSON by instruction-following) and the reproducibility stance (`make_target` pins `temperature=0.0`; initialise the SQLite file store before building targets for multi-hour runs)
    - _Requirements: 4.8, 6.4, 6.5, 6.7_

- [x] 8. Final checkpoint
  - Ensure all offline tests pass and the public surface is consistent, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test sub-tasks and can be skipped for a faster path; core implementation tasks are never marked optional.
- Ordering is lowest-risk first (A clears the import surface, B adds the runner, C adds the factorial on top), so each task is independently verifiable and leaves the package importable.
- Property tests cover the nine correctness properties in the design; each is its own sub-task, tagged `**Feature: crescendo-unicode-arm, Property {n}: {title}**`, and annotated with the requirement clauses it validates. Randomised properties (3, 6, 7, 8) run a minimum of 100 iterations.
- All tests assert behaviour or structure without a live model: removals via import errors, signatures and dataclass shape via introspection, Crescendo wiring via the built attack object's configuration (construction only), and the factorial via pure analysis math on synthetic cells.
- Net-new code stays in `runner.py`, `experiments.py`, `analysis.py`, and the `pyrit_wiring.py` import surface (Requirement 6.6); PyRIT's `CrescendoAttack`, configs, scorers, and `SeedPrompt` loader are called, never reimplemented, and no custom Crescendo escalation YAML is written.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "3.1"] },
    { "id": 1, "tasks": ["1.2", "1.4", "3.2"] },
    { "id": 2, "tasks": ["1.3", "1.5", "1.6", "3.3"] },
    { "id": 3, "tasks": ["1.7", "3.4", "3.7", "5.1", "5.2"] },
    { "id": 4, "tasks": ["3.5", "3.8", "5.3", "5.4"] },
    { "id": 5, "tasks": ["3.6", "5.5"] },
    { "id": 6, "tasks": ["5.6"] },
    { "id": 7, "tasks": ["5.7"] },
    { "id": 8, "tasks": ["7.1"] }
  ]
}
```
