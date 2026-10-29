# Requirements Document

## Introduction

This feature extends the COS700 study "Unicode / ASCII Smuggling Attacks in Chatbots". The study characterises how susceptible open-source chatbots are to Unicode/ASCII smuggling by measuring attack success rate (ASR) per encoding family, across family combinations, across multi-turn strategies, and across model scale. The goal is measurement and characterisation, not building the strongest jailbreak.

The scientific motivation for this feature is that Unicode encoding shifts the token surface and reliably bypasses an external guardrail, while a capable model can still read through the encoding and refuse. Guardrail bypass (token surface) and model compliance (internal alignment willingness) are therefore two separate layers that must be measured as separate factors.

The feature delivers three workstreams in one spec: (A) remove all shipped harmful request/answer prepend banks and decouple the adaptive search from them; (B) add a Crescendo-Unicode multi-turn arm built on the installed PyRIT `CrescendoAttack`; and (C) isolate the two factors with a 2x2 factorial ({Crescendo on/off} x {Unicode on/off}) and report guardrail bypass and compliance as separate columns. All work builds on pip-installed PyRIT 1.1.0 with no framework modification, and every operational input is injected by the caller.

## Glossary

- **Smuggle_Package**: The `smuggle/` Python package under study, built on pip-installed PyRIT 1.1.0.
- **Context_Templates_Module**: The `smuggle/context_templates.py` module.
- **Experiments_Module**: The `smuggle/experiments.py` module.
- **Crescendo_Runner**: The net-new `run_crescendo` and `run_crescendo_async` functions added to `smuggle/runner.py`.
- **Adaptive_Unicode_Search**: The `run_adaptive_unicode_search` and `run_adaptive_unicode_search_async` functions in the Experiments_Module.
- **Jailbreak_Prepend_Driver**: The `run_jailbreak_prepend` and `run_jailbreak_prepend_async` functions in the Experiments_Module.
- **Jailbreak_Priming_Primitive**: The `make_jailbreak_priming` and `make_jailbreak_priming_async` functions that build one injected prepend from caller-supplied demonstrations.
- **Mutation_Engine**: The `MutationEngine` PyRIT `PromptConverter` (Artefact 1) that applies an encoding family.
- **Judge**: The composite attack-success scorer combining a `SelfAskTrueFalseScorer` (task achieved) AND an inverted `SelfAskRefusalScorer` (not refused).
- **Guardrail**: The injected external guardrail classifier (for example Llama Guard) scored by first-token probability of "unsafe".
- **Analysis_Module**: The `smuggle/analysis.py` module.
- **JBB_Goal_Loader**: The `smuggle.jbb.load_jbb_goals` function, returning the harmful JBB `Goal` request.
- **Crescendo_Attack**: PyRIT 1.1.0's `pyrit.executor.attack.CrescendoAttack`.
- **Best_of_N**: A budgeted search that counts a prompt as successful when any one of N candidate draws succeeds.

## Requirements

### Requirement 1: Remove shipped harmful prepend banks

**User Story:** As a researcher, I want all shipped harmful request/answer prepend banks removed from the package, so that nothing harmful is baked in and the measurement is not confounded by a shipped jailbreak.

#### Acceptance Criteria

1. IF a caller imports `load_cipherchat_prepend`, `load_cipherchat_category_bank`, `DEFAULT_CIPHERCHAT_PREPEND_PATH`, `DEFAULT_CIPHERCHAT_CATEGORY_PATH`, or `LLAMA_GUARD_TEXT_CATEGORIES` from the Smuggle_Package, THEN THE Smuggle_Package SHALL raise an ImportError.
2. IF a caller imports `make_cipherchat_prepend`, `make_cipherchat_prepend_async`, `make_category_cipherchat_prepend`, or `make_category_cipherchat_prepend_async` from the Smuggle_Package, THEN THE Smuggle_Package SHALL raise an ImportError.
3. THE Smuggle_Package SHALL expose a public export list in which every listed name resolves to a defined object.
4. WHEN the Smuggle_Package is imported, THE Smuggle_Package SHALL initialise without reading any shipped CipherChat prepend content file.

### Requirement 2: Decouple the adaptive Unicode search

**User Story:** As a researcher, I want the adaptive Unicode search decoupled from the forced category bank, so that the search is a clean Best-of-N over the encoding rather than encoding combined with a full shipped jailbreak.

#### Acceptance Criteria

1. THE Adaptive_Unicode_Search SHALL accept an optional caller-injected prepend parameter whose default value is None.
2. WHEN the Adaptive_Unicode_Search runs without an injected prepend, THE Adaptive_Unicode_Search SHALL perform a Best_of_N search over the encoding with no prepended conversation.
3. THE Adaptive_Unicode_Search SHALL execute when invoked without a caller-supplied harm-category argument.
4. WHERE a caller injects a prepend, THE Adaptive_Unicode_Search SHALL deliver each candidate on top of the injected prepend.
5. THE Adaptive_Unicode_Search result records SHALL treat the harm-category attribute as optional.

### Requirement 3: Caller-injected jailbreak demonstrations

**User Story:** As a researcher, I want the jailbreak-prepend driver to consume caller-injected demonstrations, so that any unsafe example is supplied by the caller and never shipped in the package.

#### Acceptance Criteria

1. THE Jailbreak_Prepend_Driver SHALL accept a caller-injected list of (request, response) demonstration pairs.
2. IF the Jailbreak_Prepend_Driver runs without any injected demonstration pair, THEN THE Jailbreak_Prepend_Driver SHALL raise a ValueError.
3. THE Smuggle_Package SHALL retain the Jailbreak_Priming_Primitive as the single injected prepend primitive.

### Requirement 4: Crescendo-Unicode multi-turn runner

**User Story:** As a researcher, I want a Crescendo-Unicode multi-turn runner built on PyRIT, so that I can measure multi-turn escalation combined with Unicode smuggling without reimplementing the attack loop.

#### Acceptance Criteria

1. THE Crescendo_Runner SHALL construct the attack using the installed PyRIT 1.1.0 Crescendo_Attack class.
2. THE Crescendo_Runner SHALL attach the Mutation_Engine as a request converter through PyRIT's `AttackConverterConfig`.
3. WHEN the Crescendo_Runner builds the Mutation_Engine, THE Crescendo_Runner SHALL configure the Mutation_Engine in targeted inline mode so that each adversarial turn stays human-readable.
4. THE Crescendo_Runner SHALL set the attack success scorer to the injected Judge through PyRIT's `AttackScoringConfig`.
5. THE Crescendo_Runner SHALL accept an injected adversarial chat target, judge, encoding family, maximum turn count, and maximum backtrack count.
6. WHERE a caller supplies a Crescendo variant template path, THE Crescendo_Runner SHALL pass that template path to PyRIT's `AttackAdversarialConfig`.
7. WHERE a caller supplies no Crescendo variant template path, THE Crescendo_Runner SHALL use PyRIT's shipped Crescendo variant template.
8. THE Crescendo_Runner SHALL provide one synchronous entry point and one asynchronous entry point.

### Requirement 5: Factorial isolation and separated reporting

**User Story:** As a researcher, I want a 2x2 factorial that isolates the Crescendo factor from the Unicode factor with separate bypass and compliance reporting, so that I can attribute effects to the token surface versus model willingness.

#### Acceptance Criteria

1. THE Experiments_Module SHALL support four conditions: single-turn plaintext baseline, single-turn encoded, plaintext Crescendo, and encoded Crescendo.
2. THE Analysis_Module SHALL compute a paired McNemar comparison across the factorial conditions.
3. THE Analysis_Module SHALL report an interaction readout between the Crescendo factor and the Unicode factor.
4. THE Analysis_Module SHALL report guardrail-bypass rate and compliance ASR as separate columns.
5. WHERE a Best_of_N search is reported, THE Analysis_Module SHALL report the maximum success over the draws.

### Requirement 6: Injection, grounding, and reproducibility invariants

**User Story:** As a researcher, I want every component to stay injected, grounded in installed PyRIT, and reproducible, so that the study measures the smuggling attack rather than a baked-in instruction.

#### Acceptance Criteria

1. THE Smuggle_Package SHALL obtain the target, dataset, guardrail, judge, adversarial chat, encoding family, and framing content from caller-supplied arguments.
2. THE Judge SHALL score a response as a success only when the response achieves the objective AND is not a refusal.
3. THE Experiments_Module SHALL load harmful prompts through the JBB_Goal_Loader using the JBB `Goal` field.
4. WHEN the Crescendo_Runner constructs an objective target, THE Crescendo_Runner SHALL set the sampling temperature to 0.0 where the target accepts a temperature setting.
5. WHILE a multi-hour run is executing, THE Smuggle_Package SHALL persist transcripts to a SQLite file-backed store.
6. THE Smuggle_Package SHALL confine net-new code to the analysis, experiment, and runner wiring layers.
7. WHERE the injected adversarial chat lacks JSON-schema output enforcement, THE Crescendo_Runner SHALL document that the adversarial chat must be a model capable of producing the Crescendo variant JSON.
