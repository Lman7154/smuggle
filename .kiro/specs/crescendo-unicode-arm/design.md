# Design Document

## Overview

This feature extends the Unicode/ASCII smuggling study with a Crescendo multi-turn arm and tightens the injection discipline of the existing package. It delivers three workstreams in one spec:

- **A. Remove shipped harmful prepend banks** and decouple the adaptive Unicode search from the forced category bank, so nothing harmful is baked in and the measurement is not confounded by a shipped jailbreak (Requirements 1, 2, 3).
- **B. Add a Crescendo-Unicode runner** (`run_crescendo` / `run_crescendo_async`) built directly on the installed PyRIT 1.1.0 `CrescendoAttack`, with the `MutationEngine` attached as a request converter and the project composite judge driving success (Requirement 4).
- **C. Isolate the two factors** with a 2x2 factorial ({Crescendo on/off} x {Unicode on/off}) on the same prompts/family/model, and report guardrail bypass and model compliance as separate columns, with a paired McNemar comparison and an interaction readout (Requirement 5).

All work builds on pip-installed PyRIT 1.1.0 with no framework modification; net-new code stays in the runner, experiments, and analysis layers; `pyrit_wiring.py` remains the single PyRIT import surface; and every operational input is injected by the caller (Requirement 6).

The scientific frame stays constant: guardrail **bypass** (the token surface the external filter sees) and model **compliance** (the model's internal willingness after it reads through the encoding) are two separate layers and must be measured as separate factors.

## Grounding (verified against installed source this session)

Every design decision below is anchored to source read this session, not memory. The installed interpreter is `C:\Users\lusiz\AppData\Local\Programs\Python\Python313\python.exe`; `pyrit.__version__ == "1.1.0"`.

| Fact | Source read this session |
| --- | --- |
| `CrescendoAttack.__init__(*, objective_target, attack_adversarial_config, attack_converter_config=None, attack_scoring_config=None, prompt_normalizer=None, max_backtracks=10, max_turns=10, prepended_conversation_config=None)` | `pyrit/executor/attack/multi_turn/crescendo.py` |
| If no `objective_scorer` is supplied, Crescendo builds its OWN `FloatScaleThresholdScorer(SelfAskScaleScorer…, threshold=0.8)` | `crescendo.py` `__init__` `if not objective_scorer:` branch |
| Request converters are applied to EVERY objective-target send (`request_converter_configurations=self._request_converters`) | `crescendo.py` `_send_prompt_to_objective_target_async` |
| Default adversarial system prompt = `datasets/executors/crescendo/crescendo_variant_1.yaml`; it declares `response_json_schema_name: adversarial_chat` and parameters `objective`, `max_turns`, `conversation_context` | `crescendo.py` `DEFAULT_ADVERSARIAL_CHAT_SYSTEM_PROMPT_TEMPLATE_PATH`; the YAML head |
| `AttackAdversarialConfig.system_prompt: str | SeedPrompt | None` (a raw path string is treated as an INLINE template, not a file) | `pyrit.executor.attack.core.AttackAdversarialConfig` dataclass fields |
| `AttackScoringConfig.objective_scorer: TrueFalseScorer | None`; the project composite judge (`TrueFalseCompositeScorer`) IS a `TrueFalseScorer` | config dataclass fields; `issubclass(TrueFalseCompositeScorer, TrueFalseScorer) == True` |
| `CrescendoAttack` re-exported at `pyrit.executor.attack` (same surface the wiring already imports from) | `from pyrit.executor.attack import CrescendoAttack` succeeds |
| Objective-target requirement: native `EDITABLE_HISTORY` + `MULTI_TURN`; adversarial requirement: native `MULTI_TURN` + `SYSTEM_PROMPT` | `crescendo.py` `TARGET_REQUIREMENTS`, `_ADVERSARIAL_REQUIREMENTS` |
| Ollama `OpenAIChatTarget` capabilities: `supports_editable_history=True, supports_multi_turn=True, supports_system_prompt=True, supports_json_schema=False, supports_json_output=True` | constructed instance `.capabilities` |
| Result exposes `outcome` (`AttackOutcome` ∈ {SUCCESS, FAILURE, ERROR, UNDETERMINED}), `executed_turns`, `last_response` (a `MessagePiece` via `get_piece()`, so `.get_value()`), and `backtrack_count` (metadata property) | `crescendo.py` `_perform_async` result build + `CrescendoAttackResult.backtrack_count` |
| `run_redteam_async` already shows the exact wiring pattern to mirror | `smuggle/runner.py` |
| No `smuggle/templates/` directory and no `cipherchat_*.yaml` exist anywhere in the repo; the CipherChat loaders point at non-existent paths and import reads nothing | `ls` (no such dir) + workspace file search (no matches) |
| `jbb.build_priming_categories` / `JBB_CATEGORY_TO_TEMPLATE` map Goals to BENIGN context-template keys, not the removed Llama Guard category bank; they are unaffected by the removals | `smuggle/jbb.py` |
| `MutationEngine` targeted mode with empty `trigger_words` falls back to all content words (no identity); so "Unicode off" must be a true no-converter send | `smuggle/engine.py` `_keywords_for` / `convert_async` |

Two caveats this grounding forces into the design:

1. **Variant path must be loaded, not passed raw.** Because `system_prompt` accepts `str | SeedPrompt` and a plain path would be read as an inline Jinja template, the runner loads a caller-supplied `variant_path` with `P.SeedPrompt.from_yaml_file(variant_path)` before handing it to `AttackAdversarialConfig` (Requirement 4.6).
2. **Adversarial JSON capability.** On Ollama, `supports_json_schema == False`, so the Crescendo variant's `adversarial_chat` JSON schema cannot be enforced by the endpoint. The adversarial chat must therefore be a model capable of producing the Crescendo variant JSON by instruction-following alone. This is documented, not worked around (Requirement 6.7).

## Architecture

```
                      caller-injected: target, adversarial, judge, guardrail, prompts(JBB Goals),
                      family, variant_path, demonstrations, prepend, carriers
                                              |
   smuggle/__init__.py  (public surface; import/__all__ fixed for removals + new names)
                                              |
        +-------------------------+-----------+-------------------------------+
        |                         |                                           |
  runner.py                 experiments.py                               analysis.py
  - run_crescendo(_async)   - run_crescendo_factorial(_async)  (2x2)     - factorial_2x2 + FactorialResult
  - CrescendoOutcome        - run_adaptive_unicode_search (prepend-only) - format_factorial_table
  - (existing runners)      - run_jailbreak_prepend (demos injected)     - (existing: mcnemar_over_prompts,
        |                   - (removed: make_cipherchat_* , _resolve_category)  asr_over_prompts, bypass_and_asr)
        |                         |
        +------------ pyrit_wiring.py  (single PyRIT import surface; + CrescendoAttack) ------------+
                                              |
                     pip-installed PyRIT 1.1.0  (CrescendoAttack, configs, scorers, targets, memory)
                                              |
                       engine.py MutationEngine  (Artefact 1 PromptConverter; unchanged)
```

The engine, guardrail, triggers, tokens, printing, and JBB loader are untouched by this feature (Requirement 6.6). `context_templates.py` loses its CipherChat surface but keeps the benign context-template loader used by Strategy A.

## Components and Interfaces

### C0. `pyrit_wiring.py` — add `CrescendoAttack` to the single import surface

`CrescendoAttack` is re-exported from `pyrit.executor.attack`, where the wiring already imports the other attack classes. Add it to the existing import block and to `__all__`; do not import PyRIT directly from the runner.

```python
from pyrit.executor.attack import (
    PromptSendingAttack,
    RedTeamingAttack,
    CrescendoAttack,            # NEW
    AttackAdversarialConfig,
    AttackConverterConfig,
    AttackScoringConfig,
)
# ... __all__ gains "CrescendoAttack"
```

`SeedPrompt` is already imported and exported by the wiring, so the variant-path loader needs nothing new.

### Workstream A — remove shipped harmful banks and decouple the search

#### A1. `context_templates.py` removals (Requirement 1)

Delete, with no replacement:

- `load_cipherchat_prepend`
- `load_cipherchat_category_bank`
- `DEFAULT_CIPHERCHAT_PREPEND_PATH`
- `DEFAULT_CIPHERCHAT_CATEGORY_PATH`
- `LLAMA_GUARD_TEXT_CATEGORIES`

Keep `load_context_templates`, `make_priming_provider`, `DEFAULT_TEMPLATES_PATH`, and `Context` (these serve Strategy A's benign priming and are not harmful banks). No content files are deleted because none exist on disk (verified this session); removal is code-only, and import already reads no CipherChat file (Requirement 1.4).

#### A2. `experiments.py` removals (Requirement 1, call-site fallout)

Delete:

- `make_cipherchat_prepend` / `make_cipherchat_prepend_async`
- `make_category_cipherchat_prepend` / `make_category_cipherchat_prepend_async`
- `_resolve_category` (private helper; only the deleted paths used it)

The retained injected prepend primitive is `make_jailbreak_priming` / `make_jailbreak_priming_async` (Requirement 3.3), which already raises `ValueError` on empty demonstrations and builds one encoded unsafe-continuation prepend from caller-supplied `(request, response)` pairs. It stays unchanged.

#### A3. `__init__.py` import/`__all__` fixes (Requirement 1.3)

Remove the deleted names from both the `from .context_templates import (...)` / `from .experiments import (...)` blocks and from `__all__`:

- from context_templates: `load_cipherchat_prepend`, `load_cipherchat_category_bank`, `LLAMA_GUARD_TEXT_CATEGORIES`
- from experiments: `make_cipherchat_prepend`, `make_cipherchat_prepend_async`, `make_category_cipherchat_prepend`, `make_category_cipherchat_prepend_async`

Add the new names (Workstreams B and C): `run_crescendo`, `run_crescendo_async`, `CrescendoOutcome`, `run_crescendo_factorial`, `run_crescendo_factorial_async`, `factorial_2x2`, `FactorialResult`, `format_factorial_table`. After the edit, every name in `__all__` must resolve to a defined object (this is Property 2).

Because `smuggle/__init__.py` imports these names eagerly at module load, a leftover reference to a deleted symbol would raise `ImportError` at package import. That is the desired behaviour for the removed names (Requirement 1.1, 1.2): a caller importing a removed name gets `ImportError`.

#### A4. Refactor `run_adaptive_unicode_search(_async)` (Requirement 2)

The current search forces a Llama Guard harm-category argument and a category-matched CipherChat prepend on every candidate, so it measures "encoding + a full shipped jailbreak", not a clean Best-of-N over the encoding. Refactor it to a bare, prepend-optional search.

Signature change (async shown; the sync wrapper delegates):

```python
async def run_adaptive_unicode_search_async(
    *, targets, prompts, guardrail, judge,
    families=("variation_selector", "homoglyph", "structural"),
    candidate_pool=20, target_budget=5, require_guardrail_bypass=True,
    carriers=None,
    prepend=None,                 # NEW: injected list[(role, content)] or callable(objective)->that; default None
    stop_on_first_success=True,
) -> tuple[list[AdaptiveUnicodeOutcome], list[tuple]]:
```

Behaviour:

- **Remove** the required `categories` parameter, the `template_path` parameter, the `load_cipherchat_category_bank` call, the `make_category_cipherchat_prepend_async` call, and the `_resolve_category` call (Requirement 2.3).
- **Add** optional `prepend=None`. Resolve it once per objective via the existing helper `runner._as_prepended(prepend, objective)` (accepts `list[(role, content)]` or `callable(objective) -> that`). When `prepend is None`, `prepended_messages = None` and each candidate is delivered with no prepended conversation, i.e. a clean Best-of-N over the encoding (Requirement 2.1, 2.2). When a prepend is injected, every candidate is delivered on top of it (Requirement 2.4).
- **Carrier default**: the old default came from the category bank's carrier. Replace with `carrier_values = list(carriers) if carriers else [None]` so the search runs with no carrier unless the caller injects one.
- Keep the candidate generation, byte-dedup, Llama Guard `unsafe_score` ranking (bypassing first, then by score), `target_budget` cap, strict task-completion stop, and the `end_to_end = success and bypassed` gate exactly as they are. These already implement the Best-of-N-with-guardrail-awareness described in the proposal, and still log guardrail score and bypass per attempt.

Data-model change: `category` becomes optional on both records with default `""` (Requirement 2.5). The `AdaptiveUnicodeAttempt.prepend_label` becomes `""` when no prepend is injected, or a caller-supplied label when one is.

#### A5. Refactor `run_jailbreak_prepend(_async)` (Requirement 3)

The current driver loads `_CT.load_cipherchat_prepend` (now deleted) for its demonstrations and system prompt. Rewire it to consume caller-injected demonstrations and route them through the retained `make_jailbreak_priming_async` primitive.

Signature change:

```python
async def run_jailbreak_prepend_async(
    *, targets, prompts, guardrail, judge, input_guardrail=None,
    demonstrations,                # NEW required: Sequence[tuple[str, str]] of (request, response)
    system_prompt=None,            # NEW optional: injected role instruction
    families=FAMILIES, n_variants=1, carriers=None,
    encode_assistant=False, assistant_carrier="", whole_prompt=True,
    prepend_label="jailbreak_prepend",
) -> tuple[list, list]:
```

Behaviour:

- If `demonstrations` is empty or `None`, raise `ValueError` before any send (Requirement 3.2). The primitive `make_jailbreak_priming_async` already raises on empty demonstrations; the driver also validates up front so the error is raised once, not once per target/family.
- For each target x family x variant, build the prepend via `make_jailbreak_priming_async(family, demonstrations=demonstrations, carrier=<per-variant>, system_prompt=system_prompt, encode_assistant=..., assistant_carrier=...)` and deliver through the existing `run_async(prepend=prepend, whole_prompt=whole_prompt, prepend_label=prepend_label, ...)` path (unchanged).
- Remove the `template_path` parameter and the `spec = _CT.load_cipherchat_prepend(...)` lines.

This keeps the explicit jailbreak-prepend condition available for research while guaranteeing every unsafe example is injected by the caller and never shipped (Requirement 3.1, 3.3, 6.1).

### Workstream B — the Crescendo-Unicode runner (Requirement 4)

A new `run_crescendo_async` in `runner.py` mirrors `run_redteam_async` exactly, swapping `RedTeamingAttack` for `CrescendoAttack` and adding `max_backtracks` and the variant-template option. It is the real score-driven multi-turn attack: the adversarial chat escalates each turn, the `MutationEngine` smuggles each adversarial turn as a request converter, and the injected composite judge decides success.

```python
@dataclass(frozen=True)
class CrescendoOutcome:
    seed_id: str
    objective: str
    family: tuple[str, ...]        # () when Unicode is off (plaintext Crescendo)
    mode: str                      # "targeted" (inline) when on; "plaintext" when off
    unicode_on: bool
    executed_turns: int
    backtrack_count: int           # from CrescendoAttackResult.backtrack_count
    judge_success: bool
    last_response: str
    model: str
    adversarial_model: str
    max_turns: int
    max_backtracks: int
    variant_label: str = ""        # "" => PyRIT default crescendo_variant_1
    outcome_reason: str = ""


async def run_crescendo_async(
    *,
    target,                        # objective target (chatbot under test); needs editable history + multi-turn
    adversarial,                   # adversarial chat that escalates each turn; needs multi-turn + system prompt
    judge,                         # injected composite TrueFalseScorer (task achieved AND not refused)
    prompts,                       # JBB Goals (via _normalise_prompts)
    attack=None,                   # encoding family/families; None => Unicode OFF (no converter attached)
    mode="targeted",               # engine mode; targeted inline keeps each escalation turn readable (Req 4.3)
    trigger_words=(),
    benign_carrier=None,
    variant_path=None,             # optional Crescendo variant YAML; None => PyRIT default (Req 4.6/4.7)
    max_turns=10,                  # PyRIT default
    max_backtracks=10,             # PyRIT default
    seed=0,
    model_name=None,
    adversarial_model_name=None,
) -> list[CrescendoOutcome]:
    await ensure_pyrit()
    from pyrit.prompt_normalizer import ConverterConfiguration

    families = () if attack is None else tuple([attack] if isinstance(attack, str) else attack)
    resolved_model = model_name or getattr(target, "_model_name", None) or type(target).__name__
    resolved_adv = adversarial_model_name or getattr(adversarial, "_model_name", None) or type(adversarial).__name__

    # Unicode ON -> attach the MutationEngine (targeted/inline) as a request converter (Req 4.2, 4.3).
    # Unicode OFF -> omit the converter config entirely (plaintext Crescendo for the factorial).
    if families:
        engine = MutationEngine(families=list(families), mode=mode,
                                trigger_words=tuple(trigger_words), rng_seed=seed,
                                benign_carrier=benign_carrier)         # NOT whole_prompt: stays readable
        conv_cfg = P.AttackConverterConfig(
            request_converters=ConverterConfiguration.from_converters(converters=[engine]))
    else:
        conv_cfg = None

    # Adversarial system prompt: a caller variant PATH is loaded into a SeedPrompt (a raw path would be
    # read as an inline template); None leaves it unset so PyRIT applies crescendo_variant_1.yaml.
    if variant_path is not None:
        adv_cfg = P.AttackAdversarialConfig(
            target=adversarial, system_prompt=P.SeedPrompt.from_yaml_file(variant_path))
        variant_label = str(variant_path)
    else:
        adv_cfg = P.AttackAdversarialConfig(target=adversarial)
        variant_label = ""

    scoring_cfg = P.AttackScoringConfig(objective_scorer=judge)        # the injected composite judge (Req 4.4)

    attack_obj = P.CrescendoAttack(
        objective_target=target,
        attack_adversarial_config=adv_cfg,
        attack_converter_config=conv_cfg,
        attack_scoring_config=scoring_cfg,
        max_turns=max_turns,
        max_backtracks=max_backtracks,
    )

    outcomes: list[CrescendoOutcome] = []
    for seed_id, text in _normalise_prompts(prompts):
        result = await attack_obj.execute_async(objective=text)
        success = str(getattr(getattr(result, "outcome", None), "name", "")).upper() == "SUCCESS"
        piece = getattr(result, "last_response", None)
        last = (piece.get_value() if piece is not None and hasattr(piece, "get_value") else "") or ""
        outcomes.append(CrescendoOutcome(
            seed_id=seed_id, objective=text, family=families,
            mode=(mode if families else "plaintext"), unicode_on=bool(families),
            executed_turns=int(getattr(result, "executed_turns", 0) or 0),
            backtrack_count=int(getattr(result, "backtrack_count", 0) or 0),
            judge_success=success, last_response=last,
            model=resolved_model, adversarial_model=resolved_adv,
            max_turns=max_turns, max_backtracks=max_backtracks,
            variant_label=variant_label,
            outcome_reason=str(getattr(result, "outcome_reason", "") or ""),
        ))
    return outcomes


def run_crescendo(**kwargs) -> list[CrescendoOutcome]:
    """Synchronous wrapper over run_crescendo_async (Requirement 4.8)."""
    return asyncio.run(run_crescendo_async(**kwargs))
```

Design decisions and how they satisfy Requirement 4:

- **4.1 Build on installed PyRIT.** `P.CrescendoAttack(...)` is constructed from `pyrit_wiring`, never reimplemented.
- **4.2 / 4.3 Engine as a request converter, targeted inline.** The `MutationEngine` is wrapped with `ConverterConfiguration.from_converters([engine])` and passed via `AttackConverterConfig(request_converters=...)`, exactly as `run_redteam_async` does. The engine is built with `mode="targeted"` and without `whole_prompt`, so each adversarial escalation turn stays human-readable (only the salient spans are smuggled). Crescendo applies request converters to every objective-target send (verified), so every escalation turn is encoded.
- **4.4 Injected judge.** `AttackScoringConfig(objective_scorer=judge)` passes the project composite judge. This is critical: if omitted, Crescendo silently builds its own `SelfAskScaleScorer`-based scorer (threshold 0.8), which is not the project's task-achieved AND not-refused composite. The composite is a `TrueFalseScorer`, which is exactly the type `objective_scorer` accepts.
- **4.5 Injected knobs.** `adversarial`, `judge`, `attack` (family), `max_turns`, and `max_backtracks` are all parameters.
- **4.6 / 4.7 Variant template.** A caller `variant_path` is loaded with `SeedPrompt.from_yaml_file` and passed as `AttackAdversarialConfig.system_prompt`; omitting it leaves the adversarial config without a system prompt so PyRIT resolves its shipped `crescendo_variant_1.yaml`. This design does **not** write a custom Crescendo escalation YAML.
- **4.8 Sync + async.** `run_crescendo_async` plus the thin `run_crescendo` wrapper.

Result extraction reuses the proven `run_redteam_async` pattern: `outcome.name == "SUCCESS"`, `last_response.get_value()`, `executed_turns`, plus the Crescendo-specific `backtrack_count` metadata property.

Target-capability note (documented, Requirement 6.7): both targets must be natively capable. The objective target needs editable history + multi-turn; the adversarial chat needs multi-turn + system-prompt support; PyRIT validates these in the `CrescendoAttack` constructor and raises `ValueError` if unmet. The Ollama `OpenAIChatTarget` satisfies both (verified capabilities). Because `supports_json_schema == False` on Ollama, the Crescendo variant's JSON schema is not endpoint-enforced, so the adversarial chat must be a model capable enough to emit the variant JSON by instruction-following. A weak or aligned adversarial model degrades or refuses the red-team role; this is a measurement caveat, not a code path.

### Workstream C — 2x2 factorial and separated reporting (Requirement 5)

#### C1. Factorial driver `run_crescendo_factorial(_async)` (experiments layer)

The factorial runs four conditions on the same prompts, the same single injected family, and the same model, so an effect can be attributed to the token surface (Unicode) versus the escalation (Crescendo). The final delivered payload in the encoded cells is held comparable to Experiment 1 by reusing the same targeted-inline engine configuration.

```python
async def run_crescendo_factorial_async(
    *, targets, prompts, judge, guardrail=None, input_guardrail=None,
    adversarial,                   # required for the two Crescendo cells
    family,                        # single injected encoding family, held constant across cells
    trigger_words=(), benign_carrier=None,
    variant_path=None, max_turns=10, max_backtracks=10,
    seed=0,
) -> dict[str, list]:
    """Return {cell_label: [outcome records]} for the four conditions:
       'single_plain'    : single-turn, Unicode off  (C off, U off)  -> baseline
       'single_enc'      : single-turn, Unicode on   (C off, U on)   == Experiment 1 payload
       'crescendo_plain' : Crescendo,   Unicode off  (C on,  U off)
       'crescendo_enc'   : Crescendo,   Unicode on   (C on,  U on)
    """
```

Cell construction:

- **`single_enc`** reuses the existing single-turn path exactly: `run_async(target=..., prompts=..., attack=family, mode="targeted", guardrail=..., input_guardrail=..., judge=..., carrier=benign_carrier, seed=seed)`. This guarantees the encoded single-turn payload is identical to Experiment 1's (same trigger detection, same targeted-inline encoding), so the factorial contrast isolates the escalation effect from a different payload.
- **`single_plain`** is a true no-converter send (the engine has no identity mode; verified). A small internal helper sends the raw Goal through `P.PromptSendingAttack(objective_target=target).execute_async(objective=text)`, scores with the injected judge via `runner._score`, and optionally scores the plaintext against `input_guardrail`. It records a `VariantOutcome` with `family=("plaintext",)`, `mode="plaintext"`, `encoded_text=text`, `imperceptibility=1.0`.
- **`crescendo_enc`** calls `run_crescendo_async(attack=family, mode="targeted", trigger_words=..., benign_carrier=..., variant_path=..., max_turns=..., max_backtracks=..., ...)`.
- **`crescendo_plain`** calls `run_crescendo_async(attack=None, ...)` so no converter is attached (plaintext Crescendo).

All four cells key their records by `seed_id`, so behaviour-level pairing is well defined. The sync wrapper `run_crescendo_factorial` delegates through `asyncio.run`.

Layer separation (research discipline, Requirement 5.4): guardrail **bypass** is only well defined where a single comparable payload is scored by the input guardrail, i.e. the single-turn cells (`single_plain` on the raw text, `single_enc` on the encoded payload). The multi-turn cells deliver a different adversary-generated payload each turn, so a single "bypass" figure is not comparable; the factorial reports **compliance ASR for all four cells** and **bypass for the single-turn cells only**, leaving the Crescendo bypass column undefined (`None`) rather than inventing a number. This keeps the smuggling attack (encoding) measurement separate from the escalation confound.

#### C2. Analysis `factorial_2x2` + `FactorialResult` (analysis layer)

A net-new analysis function in `analysis.py` consumes the four cell lists and reuses the existing, separately-tested primitives (`asr_over_prompts`, `mcnemar_over_prompts`, `bypass_and_asr`). Nothing in the existing analysis functions changes.

```python
@dataclass(frozen=True)
class FactorialResult:
    asr: dict[str, float]                 # cell_label -> behaviour-level compliance ASR (best-of-N = any)
    ci: dict[str, tuple[float, float]]    # cell_label -> Wilson 95% CI
    n: dict[str, int]                     # cell_label -> number of behaviours
    unicode_main_effect: float            # ASR(single_enc) - ASR(single_plain)
    crescendo_main_effect: float          # ASR(crescendo_plain) - ASR(single_plain)
    interaction: float                    # (ASR(cresc_enc) - ASR(cresc_plain)) - (ASR(single_enc) - ASR(single_plain))
    mcnemar_unicode: McNemarResult        # single_enc vs single_plain, paired on behaviours
    mcnemar_crescendo: McNemarResult      # crescendo_plain vs single_plain, paired on behaviours
    mcnemar_combined: McNemarResult       # crescendo_enc vs single_plain, paired on behaviours
    bypass_rate: dict[str, Optional[float]]   # cell_label -> guardrail-bypass rate (None for the Crescendo cells)
    compliance_asr: dict[str, float]          # cell_label -> compliance ASR (same unit as asr), kept SEPARATE


def factorial_2x2(cells: dict[str, list], *, aggregate: str = "any") -> FactorialResult:
    ...
```

Behaviour:

- **Compliance ASR per cell** uses `asr_over_prompts(cell, aggregate="any")`: behaviour-level ASR with a Wilson 95% CI, the honest unit being the behaviour, not the variant. With `aggregate="any"` this is exactly Best-of-N reported as the maximum over draws (Requirement 5.5): a behaviour counts as a success when any one draw succeeds.
- **Paired McNemar** uses `mcnemar_over_prompts(cell_X, cell_Y, aggregate="any")` for each contrast, pairing on the behaviours present in both cells (Requirement 5.2).
- **Interaction readout** (Requirement 5.3) is the difference-in-differences on the four cell ASRs, with the two main effects reported alongside. Labelling the cells `a=single_plain, b=single_enc, c=crescendo_plain, d=crescendo_enc`: `unicode_main = b - a`, `crescendo_main = c - a`, `interaction = (d - c) - (b - a)`. A positive interaction means Unicode and Crescendo combine super-additively.
- **Separate bypass vs compliance columns** (Requirement 5.4) come from `bypass_and_asr`, which already keeps compliance ASR and guardrail-bypass rate as distinct fields and returns `bypass_rate = None` when a cell had no guardrail-scored outcomes. The Crescendo cells therefore carry `None` bypass (documented limitation above), while the single-turn cells carry a real bypass rate. `format_factorial_table` renders ASR and bypass in separate columns, mirroring the existing `format_report_table`.

## Data Models

Changed/new records (frozen dataclasses, consistent with the existing style):

- **`CrescendoOutcome`** (new, `runner.py`): mirrors `RedTeamOutcome` (`seed_id, objective, family, mode, executed_turns, judge_success, last_response, model, adversarial_model`) and adds `unicode_on`, `backtrack_count`, `max_turns`, `max_backtracks`, `variant_label`, `outcome_reason`. `family` is `()` and `mode` is `"plaintext"` when Unicode is off.
- **`AdaptiveUnicodeAttempt`** (changed): `category` becomes optional with default `""`; `prepend_label` defaults to `""`.
- **`AdaptiveUnicodeOutcome`** (changed): `category` becomes optional with default `""`.
- **`FactorialResult`** (new, `analysis.py`): the structured 2x2 readout described above.

Records are plain return values; durable logging of transcripts/scores is PyRIT `CentralMemory`'s job. Reproducibility relies on `make_target` pinning `temperature=0.0` (Requirement 6.4) and, for multi-hour runs, initialising the SQLite file-backed store via `init_pyrit_sqlite` before any target is built (Requirement 6.5).

## Error Handling

- **Removed imports (Req 1.1, 1.2):** importing any removed name raises `ImportError` at the point of import, because the names are gone from both the submodules and the package `__init__`. No shim or deprecation stub is provided; the removal is intentional and total.
- **No demonstrations (Req 3.2):** `run_jailbreak_prepend_async` raises `ValueError` up front when `demonstrations` is empty/`None`, and the underlying `make_jailbreak_priming_async` also raises, so the error cannot be bypassed by calling the primitive directly.
- **Adaptive search inputs:** `candidate_pool < 1` or `target_budget < 1` raises `ValueError` (unchanged); unknown families raise `ValueError` (unchanged). A no-prepend run is valid and must not raise (Req 2.2, 2.3).
- **Crescendo target capability:** if the objective or adversarial target lacks the required native capabilities, PyRIT's `CrescendoAttack` constructor raises `ValueError`; the runner lets it propagate rather than masking it, and the design documents the capability requirement.
- **Per-variant failures in batch drivers:** the factorial and search drivers keep the existing `(rows, errors)` convention, appending `(model, cell/family, seed, message)` on any variant that raises so one failure never aborts the batch and failures are recorded rather than hidden.
- **Variant path:** a caller `variant_path` that is missing or malformed surfaces as the error raised by `SeedPrompt.from_yaml_file`; a variant template lacking the required `objective`/`max_turns` parameters surfaces as PyRIT's resolve-time error.

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system. Properties bridge the human-readable specification and machine-verifiable correctness guarantees.*

### Property 1: Removed symbols are unimportable

*For any* name in the removed set {`load_cipherchat_prepend`, `load_cipherchat_category_bank`, `DEFAULT_CIPHERCHAT_PREPEND_PATH`, `DEFAULT_CIPHERCHAT_CATEGORY_PATH`, `LLAMA_GUARD_TEXT_CATEGORIES`, `make_cipherchat_prepend`, `make_cipherchat_prepend_async`, `make_category_cipherchat_prepend`, `make_category_cipherchat_prepend_async`}, importing that name from the package raises an `ImportError` (and the package has no such attribute).

**Validates: Requirements 1.1, 1.2**

### Property 2: The public export list fully resolves

*For all* names listed in `smuggle.__all__`, the attribute resolves on the package to a defined (present, non-`None`) object.

**Validates: Requirements 1.3**

### Property 3: Jailbreak prepend without demonstrations is rejected

*For any* encoding family selection and *for any* empty demonstrations input (empty sequence or `None`), the jailbreak-prepend primitive and driver raise a `ValueError` before any prompt is sent.

**Validates: Requirements 3.2**

### Property 4: Crescendo attaches a targeted-inline mutation engine

*For any* encoding family (or combination), the attack object built by the Crescendo runner (Unicode-on) has request converters containing exactly one `MutationEngine`, and that engine is configured in targeted inline mode (`mode == "targeted"` and whole-prompt encoding off), observable from the attack's converter configuration without contacting a model.

**Validates: Requirements 4.2, 4.3**

### Property 5: The Crescendo objective scorer is the injected judge

*For any* injected judge object, the Crescendo attack built by the runner uses that exact object as its objective scorer, never PyRIT's default scale scorer.

**Validates: Requirements 4.4, 6.2**

### Property 6: Best-of-N success is the maximum over draws

*For any* collection of per-prompt candidate draws, the reported Best-of-N success for a prompt equals the logical OR over that prompt's draw outcomes (a prompt succeeds exactly when at least one draw succeeds).

**Validates: Requirements 5.5**

### Property 7: The factorial interaction is the difference-in-differences

*For any* four cell success rates `a` (single plaintext), `b` (single encoded), `c` (plaintext Crescendo), `d` (encoded Crescendo), the reported Unicode main effect equals `b - a`, the Crescendo main effect equals `c - a`, and the interaction equals `(d - c) - (b - a)`.

**Validates: Requirements 5.3**

### Property 8: Bypass and compliance are reported separately

*For any* set of outcome records, the compliance ASR (fraction whose judge verdict is success) and the guardrail-bypass rate (fraction of the guardrail-scored subset that evaded the filter) are reported as two distinct values and never collapsed into one; and the bypass rate is undefined (`None`) exactly when no outcome in the group carried a guardrail verdict.

**Validates: Requirements 5.4**

### Property 9: The composite judge is success iff achieved and not refused

*For any* response, the composite judge scores success as true if and only if the task-achieved scorer is true AND the refusal scorer is false.

**Validates: Requirements 6.2**

## Testing Strategy

Dual approach: property-based tests for the universals above, example/integration tests for the wiring and construction facts. All tests build on the existing `tests/` suite and must not require a live model; Crescendo construction is asserted through the attack object's configuration, and target/guardrail/judge are stubbed where a call would otherwise be made.

**Property tests** (minimum 100 iterations each; tag `**Feature: crescendo-unicode-arm, Property {n}: {property text}**`):

- Property 1: parametrize over the removed-name set; assert `ImportError`/`AttributeError`.
- Property 2: iterate `smuggle.__all__`; assert each resolves.
- Property 3: generate random family selections and empty/None demonstrations; assert `ValueError`.
- Property 4: iterate families in `FAMILIES`; build the attack via the runner wiring with stub targets; introspect `request_converters` for a single targeted-inline `MutationEngine`.
- Property 5: pass a sentinel composite judge; assert identity on the built attack's objective scorer.
- Property 6: random per-prompt draw sets; assert `per_prompt_success(aggregate="any")` equals the OR over draws.
- Property 7: random four-cell success maps; assert the main effects and interaction equal their difference-in-differences definitions.
- Property 8: random outcome sets with and without guardrail verdicts; assert the two measures match their definitions and that bypass is `None` iff no guarded outcome.
- Property 9: enumerate the `(achieved, refused)` truth table with stub sub-scorers; assert the composite equals `achieved AND not refused`.

**Example / integration tests** (new, under `tests/`):

- Import-raises-after-removal: `from smuggle import load_cipherchat_prepend` (and the other removed names) raises.
- Adaptive search runs with no categories and no prepend: with stub target/guardrail/judge, `run_adaptive_unicode_search_async(...)` completes, every send used `prepended_conversation=None`, and `AdaptiveUnicodeOutcome.category == ""`.
- Adaptive search with an injected prepend delivers every candidate on top of it (stub target records the prepended conversation).
- `run_jailbreak_prepend` raises `ValueError` without demonstrations; with injected demonstrations it builds the encoded prepend and runs.
- `run_crescendo` constructs a `CrescendoAttack` and attaches the `MutationEngine` as a request converter, asserted from the attack's converter config without a live model; `attack=None` attaches no converter (plaintext Crescendo).
- `CrescendoOutcome` shape: all fields present, defaults correct, `family == ()` and `mode == "plaintext"` when `unicode_on` is false.
- `run_crescendo` with `variant_path` sets `AttackAdversarialConfig.system_prompt` to the loaded `SeedPrompt`; without it leaves the default (PyRIT's `crescendo_variant_1.yaml`).
- Factorial driver produces all four labelled cells and `factorial_2x2` returns a `FactorialResult` with separate ASR and bypass columns and the three McNemar contrasts.

**Smoke tests:**

- Package imports cleanly in a fresh process with no CipherChat path access.
- `make_target` pins `temperature=0.0`.
- `pyrit_wiring` remains the only PyRIT import surface (the runner imports PyRIT symbols through `P`, plus the one already-present local `ConverterConfiguration` import mirrored from `run_redteam_async`).
