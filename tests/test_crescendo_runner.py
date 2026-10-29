"""Offline property + coverage tests for the Crescendo-Unicode runner's wiring.

Two correctness properties plus the task-3.6 shape/variant-path coverage are asserted here,
all by INTROSPECTING the built ``CrescendoAttack`` or the config objects the runner passes to
it (construction only, no model call):

* **Property 4** -- the targeted-inline ``MutationEngine`` is attached as a request converter.
* **Property 5** -- the INJECTED judge is the attack's objective scorer (identity), never the
  default scale/threshold scorer Crescendo would build for itself if none were supplied.
* **Task 3.6 (Requirements 4.6, 4.7)** -- the ``CrescendoOutcome`` dataclass shape/defaults and
  the Unicode-off invariant (``family == ()`` / ``mode == "plaintext"`` when ``unicode_on`` is
  False), plus that a supplied ``variant_path`` is loaded into the adversarial
  ``AttackAdversarialConfig.system_prompt`` as a ``SeedPrompt`` while omitting it leaves that
  field ``None`` (PyRIT then resolves its shipped ``crescendo_variant_1.yaml``). See the section
  above those tests for the installed-source anchors.

No model is ever contacted. The Crescendo attack object is CONSTRUCTED with real
``make_target`` ``OpenAIChatTarget`` endpoints (a localhost Ollama-style base_url; no
network happens at construction time) and then INTROSPECTED -- the design's chosen
verification method for this property ("Crescendo wiring via the built attack object's
configuration (construction only)", design Testing Strategy).

How the built attack is obtained (real wiring, no model call)
-------------------------------------------------------------
Each case invokes the REAL ``smuggle.runner.run_crescendo_async`` with an EMPTY prompt
list. ``run_crescendo_async`` builds the ``CrescendoAttack`` BEFORE its per-prompt loop,
and with ``prompts=[]`` that loop never runs, so ``execute_async`` is never called and no
model is contacted (verified: the call returns ``[]``). We capture the constructed attack
by wrapping ``P.CrescendoAttack`` for the duration of the test (``monkeypatch``, reverted
afterwards; no source is modified). This exercises the genuine runner construction path --
the real engine build and the real ``AttackConverterConfig`` /
``ConverterConfiguration.from_converters`` wiring -- rather than a re-implementation.

Where the asserted attribute names come from (installed PyRIT 1.1.0, read this session)
--------------------------------------------------------------------------------------
* ``pyrit/executor/attack/multi_turn/crescendo.py`` (lines 190-191)::

      attack_converter_config = attack_converter_config or AttackConverterConfig()
      self._request_converters = attack_converter_config.request_converters

  so the built ``CrescendoAttack`` stores its request converters on
  ``._request_converters`` as a ``list[ConverterConfiguration]``; when no converter config
  is supplied the fallback ``AttackConverterConfig()`` has an EMPTY ``request_converters``
  (``pyrit/executor/core/config.py`` -> ``StrategyConverterConfig.request_converters =
  field(default_factory=list)``).
* ``pyrit/prompt_normalizer/converter_configuration.py``::

      ConverterConfiguration.converters: list[Converter]

  so the converter instances are reached via ``cfg.converters``.
* ``pyrit/executor/attack/multi_turn/crescendo.py`` ``__init__`` (the
  ``if not objective_scorer:`` branch)::

      objective_scorer = attack_scoring_config.objective_scorer
      if not objective_scorer:
          objective_scorer = FloatScaleThresholdScorer(scorer=SelfAskScaleScorer..., threshold=0.8)
      self._objective_scorer = objective_scorer

  so the built ``CrescendoAttack`` stores its objective scorer on ``._objective_scorer``, and a
  default ``FloatScaleThresholdScorer`` (``pyrit.score``) is fabricated ONLY when no scorer is
  injected. ``AttackScoringConfig.__post_init__`` additionally rejects a non-``TrueFalseScorer``
  objective scorer and -- because its guard reads ``if self.objective_scorer and ...`` -- a falsy
  one would be silently swapped for that default. Property 5 therefore asserts identity with the
  injected (truthy) judge AND, as a control, confirms the default appears when none is injected.

The ``MutationEngine`` mode / whole-prompt flags are read from the engine's own public
attributes (``smuggle/engine.py`` -> ``MutationEngine.mode`` and
``MutationEngine.whole_prompt``), not assumed names.

Iteration strategy (honours the tasks.md ">=100 iterations" guidance for randomised
properties): the invariant is asserted (a) for EVERY family in ``smuggle.FAMILIES`` (full
per-family coverage via ``parametrize``) and (b) over 200 randomised draws of
``(family, seed, trigger_words)`` -- including two-family combinations -- from a seeded,
reproducible RNG.
"""
from __future__ import annotations

import asyncio
import dataclasses
import random
import string
import textwrap

import pytest

import smuggle
from smuggle import make_judge, make_target, runner
from smuggle.engine import MutationEngine
from smuggle.runner import CrescendoOutcome

# Property 5 anchors, imported from the INSTALLED PyRIT 1.1.0 surface (read this session):
#   FloatScaleThresholdScorer -> pyrit.score  (pyrit/score/true_false/float_scale_threshold_scorer.py)
#     == the default objective scorer Crescendo builds for itself when none is injected.
#   TrueFalseScorer -> pyrit/score/true_false/true_false_scorer.py
#     == the base AttackScoringConfig.__post_init__ requires an injected objective scorer to subclass.
from pyrit.score import FloatScaleThresholdScorer
from pyrit.score.true_false.true_false_scorer import TrueFalseScorer

#: Localhost Ollama-style endpoint. Used for CONSTRUCTION ONLY -- never contacted, because
#: every attack is built with an empty prompt list so no turn is ever sent.
OLLAMA_BASE_URL = "http://localhost:11434/v1"

#: Randomised-property iteration count. The tasks.md Notes require randomised properties to
#: run a minimum of 100 iterations; 200 comfortably exceeds that while staying sub-second.
RANDOM_ITERATIONS = 200


# --------------------------------------------------------------------------- #
# Fixtures: build the real endpoints + judge ONCE (construction only, no model),
# then expose a builder that drives the REAL runner construction path per case.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def endpoints():
    """Real ``OpenAIChatTarget`` objective/adversarial targets plus the project composite
    judge, all constructed offline.

    ``init_pyrit`` sets up the ephemeral in-memory PyRIT store that ``OpenAIChatTarget``
    construction needs (idempotent: a no-op if memory already exists); a localhost base_url
    means no network is touched at construction time. The judge is the project composite
    ``TrueFalseScorer`` (``make_judge``) -- exactly the type ``AttackScoringConfig`` requires
    -- built from a dummy chat target (also construction only).
    """
    smuggle.init_pyrit()
    target = make_target("llama3.1:8b", base_url=OLLAMA_BASE_URL, api_key="ollama")
    adversarial = make_target("llama3.1:70b", base_url=OLLAMA_BASE_URL, api_key="ollama")
    judge = make_judge(make_target("judge", base_url=OLLAMA_BASE_URL, api_key="ollama"))
    return target, adversarial, judge


@pytest.fixture
def build_attack(endpoints, monkeypatch):
    """Return ``build(*, attack, judge_override=None, ...)`` -> the ``CrescendoAttack`` the
    runner constructs.

    ``P.CrescendoAttack`` is wrapped once (for this test only) so the attack
    ``run_crescendo_async`` builds internally is captured; the runner is then called with
    ``prompts=[]`` so its per-prompt ``execute_async`` loop never runs and no model is hit.

    ``judge_override`` lets a case inject a SPECIFIC judge object (used by Property 5 to assert
    identity); when ``None`` the module-scoped project composite judge from ``endpoints`` is
    threaded through, so the Property 4 cases are unchanged.

    Task 3.6 additions (backward compatible): ``build`` accepts an optional ``variant_path``
    (default ``None`` -- the pre-existing behaviour) threaded straight to
    ``run_crescendo_async``, and the wrapper now ALSO records the exact kwargs dict the runner
    passes to ``P.CrescendoAttack`` on the latest call, exposed as
    ``build_attack.captured_kwargs[-1]`` so a case can read the ``attack_adversarial_config`` the
    runner built -- the config object carrying the adversarial ``system_prompt`` -- without a
    model call. Property 4/5 ignore both additions, so they are unaffected.
    """
    target, adversarial, judge = endpoints
    captured: list = []
    captured_kwargs: list[dict] = []
    real_crescendo = runner.P.CrescendoAttack  # genuine class, read before patching

    def _capture(**kwargs):
        # Record the kwargs the runner passed BEFORE constructing (so they survive even a raise),
        # then build the real attack and capture it -- the original path is otherwise unchanged.
        captured_kwargs.append(kwargs)
        obj = real_crescendo(**kwargs)
        captured.append(obj)
        return obj

    monkeypatch.setattr(runner.P, "CrescendoAttack", _capture)

    def build(*, attack, mode="targeted", trigger_words=(), seed=0, judge_override=None,
              variant_path=None):
        captured.clear()
        captured_kwargs.clear()
        outcomes = asyncio.run(
            runner.run_crescendo_async(
                target=target,
                adversarial=adversarial,
                judge=judge if judge_override is None else judge_override,
                prompts=[],  # empty => no execute_async => no model call
                attack=attack,
                mode=mode,
                trigger_words=tuple(trigger_words),
                seed=seed,
                variant_path=variant_path,  # 3.6: None (default) keeps Property 4/5 behaviour
            )
        )
        assert outcomes == [], "empty prompt list must produce no outcomes (no model was called)"
        assert len(captured) == 1, f"expected exactly one CrescendoAttack construction, got {len(captured)}"
        return captured[-1]

    #: kwargs the runner passed to ``P.CrescendoAttack`` on the latest ``build(...)`` call.
    build.captured_kwargs = captured_kwargs
    return build


# --------------------------------------------------------------------------- #
# Introspection helpers, anchored to installed PyRIT 1.1.0 source (see module docstring).
# --------------------------------------------------------------------------- #
def _request_converter_instances(attack_obj) -> list:
    """Converter instances configured on the built attack's REQUEST path.

    ``attack_obj._request_converters`` is a ``list[ConverterConfiguration]``
    (crescendo.py:191); each ``ConverterConfiguration.converters`` is a ``list[Converter]``
    (converter_configuration.py). Flatten to the converter instances.
    """
    configs = attack_obj._request_converters
    return [conv for cfg in configs for conv in cfg.converters]


def _assert_single_targeted_inline_engine(attack_obj, *, label: str) -> None:
    """The Unicode-on invariant: exactly one targeted-inline ``MutationEngine`` attached."""
    engines = _request_converter_instances(attack_obj)
    assert len(engines) == 1, (
        f"[{label}] expected exactly one request converter (MutationEngine), found "
        f"{len(engines)}: {[type(e).__name__ for e in engines]}"
    )
    engine = engines[0]
    assert isinstance(engine, MutationEngine), (
        f"[{label}] the attached request converter is {type(engine).__name__}, not MutationEngine"
    )
    # Targeted INLINE mode (Req 4.3): only the salient spans are smuggled, so each escalation
    # turn stays human-readable.
    assert engine.mode == "targeted", f"[{label}] engine.mode == {engine.mode!r}, expected 'targeted'"
    # Whole-prompt encoding OFF: a readable turn, not a fully-encoded one (Req 4.3).
    assert engine.whole_prompt is False, (
        f"[{label}] engine.whole_prompt is {engine.whole_prompt!r}, expected False"
    )


# --------------------------------------------------------------------------- #
# Property 4 -- full per-family coverage (parametrize over every family).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("family", smuggle.FAMILIES)
def test_each_family_attaches_single_targeted_inline_engine(build_attack, family: str) -> None:
    """**Feature: crescendo-unicode-arm, Property 4: Crescendo attaches a targeted-inline mutation engine**

    For every encoding family the Unicode-on runner must attach exactly one
    ``MutationEngine`` as a request converter, in targeted inline mode with whole-prompt
    encoding off -- observable from the built attack's converter configuration without
    contacting a model.

    **Validates: Requirements 4.2, 4.3**
    """
    attack_obj = build_attack(attack=family)
    _assert_single_targeted_inline_engine(attack_obj, label=family)


# --------------------------------------------------------------------------- #
# Property 4 -- randomised >= 100 iterations (family + seed + trigger_words).
# --------------------------------------------------------------------------- #
def _random_trigger_words(rng: random.Random) -> tuple[str, ...]:
    """A reproducible random 0-4 word trigger set (arbitrary lowercase tokens)."""
    count = rng.randint(0, 4)
    return tuple(
        "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(2, 8)))
        for _ in range(count)
    )


def test_randomised_family_seed_triggers_attach_invariant(build_attack) -> None:
    """**Feature: crescendo-unicode-arm, Property 4: Crescendo attaches a targeted-inline mutation engine**

    The attachment invariant is independent of which family, RNG seed, or trigger words are
    chosen. Over 200 randomised draws (seeded RNG, reproducible) of
    ``(family, seed, trigger_words)`` -- mixing single families with two-family combinations
    so the property's "or combination" clause is exercised -- the built attack always carries
    exactly one targeted-inline ``MutationEngine`` on its request path, with whole-prompt
    encoding off.

    **Validates: Requirements 4.2, 4.3**
    """
    rng = random.Random(0xC0FFEE)
    saw_combination = False
    for i in range(RANDOM_ITERATIONS):
        if rng.random() < 0.3:  # ~30% two-family combinations
            fam = rng.sample(list(smuggle.FAMILIES), 2)
            label = "+".join(fam)
            saw_combination = True
        else:
            fam = rng.choice(smuggle.FAMILIES)
            label = fam
        attack_obj = build_attack(
            attack=fam,
            mode="targeted",
            trigger_words=_random_trigger_words(rng),
            seed=rng.randint(0, 2**31 - 1),
        )
        _assert_single_targeted_inline_engine(attack_obj, label=f"iter{i}:{label}")
    assert saw_combination, "randomised draws should have included at least one family combination"


# --------------------------------------------------------------------------- #
# Property 4 -- attack=None attaches NO request converter (plaintext Crescendo).
# --------------------------------------------------------------------------- #
def test_attack_none_attaches_no_request_converter(build_attack) -> None:
    """**Feature: crescendo-unicode-arm, Property 4: Crescendo attaches a targeted-inline mutation engine**

    With ``attack=None`` (Unicode off, the plaintext-Crescendo factorial cell) the runner
    passes ``attack_converter_config=None``; PyRIT then falls back to a default
    ``AttackConverterConfig()`` whose ``request_converters`` default is empty (installed
    source, see module docstring), so the built attack has NO request converter attached --
    in particular no ``MutationEngine``.

    **Validates: Requirements 4.2, 4.3**
    """
    attack_obj = build_attack(attack=None)
    assert attack_obj._request_converters == [], (
        f"attack=None must attach no request-converter config, got {attack_obj._request_converters!r}"
    )
    assert _request_converter_instances(attack_obj) == [], "attack=None must attach no converter instances"


# --------------------------------------------------------------------------- #
# Property 5 -- the injected judge IS the attack's objective scorer.
#
# Attribute anchored to installed PyRIT 1.1.0 (read this session),
# pyrit/executor/attack/multi_turn/crescendo.py __init__ ``if not objective_scorer:`` branch::
#
#     objective_scorer = attack_scoring_config.objective_scorer
#     if not objective_scorer:
#         objective_scorer = FloatScaleThresholdScorer(scorer=SelfAskScaleScorer..., threshold=0.8)
#     self._objective_scorer = objective_scorer
#
# so the built attack exposes its objective scorer on ``._objective_scorer`` and only
# fabricates a ``FloatScaleThresholdScorer`` when none was injected.
# --------------------------------------------------------------------------- #
class _StubObjectiveScorer(TrueFalseScorer):
    """A real ``TrueFalseScorer`` subclass with a known identity and no model/network.

    Subclassing the installed base is what makes ``AttackScoringConfig`` accept it
    (``__post_init__`` rejects a non-``TrueFalseScorer`` objective scorer) and leaves the
    instance truthy, so Crescendo keeps it instead of building its own default. It is never
    scored here (the attack is built with ``prompts=[]``); the abstract methods exist only so
    the class is concrete. Pattern mirrors ``_StubLeafScorer`` in tests/test_crescendo_support.py.
    """

    def _build_identifier(self):
        return self._create_identifier(params={"stub": type(self).__name__})

    async def _score_scorable_async(self, *, scorable, expectation):  # pragma: no cover - never called
        return []


def _objective_scorer_of(attack_obj):
    """The objective scorer stored on the built attack (crescendo.py: ``self._objective_scorer``)."""
    return attack_obj._objective_scorer


@pytest.mark.parametrize("attack", [*smuggle.FAMILIES, None])
def test_objective_scorer_is_the_injected_composite_judge(build_attack, endpoints, attack) -> None:
    """**Feature: crescendo-unicode-arm, Property 5: The Crescendo objective scorer is the injected judge**

    The runner must thread the INJECTED composite judge straight through to ``CrescendoAttack``
    as its objective scorer. For every encoding family AND for the Unicode-off (``attack=None``)
    cell, the built attack stores the EXACT judge object the ``endpoints`` fixture handed the
    runner (identity, ``is``) -- not a copy, and not PyRIT's own default scale/threshold scorer.
    Independence from ``attack`` shows the judge wiring is orthogonal to the converter wiring
    (Property 4).

    **Validates: Requirements 4.4, 6.2**
    """
    _, _, judge = endpoints
    attack_obj = build_attack(attack=attack)
    stored = _objective_scorer_of(attack_obj)
    assert stored is judge, (
        f"objective scorer is {stored!r}, expected the injected composite judge {judge!r}"
    )
    # Discriminating: the injected judge is NOT the default Crescendo scale/threshold scorer.
    assert not isinstance(stored, FloatScaleThresholdScorer), (
        "objective scorer is PyRIT's default FloatScaleThresholdScorer, not the injected judge"
    )


def test_objective_scorer_is_the_exact_injected_stub(build_attack) -> None:
    """**Feature: crescendo-unicode-arm, Property 5: The Crescendo objective scorer is the injected judge**

    With a minimal caller-controlled ``TrueFalseScorer`` injected as the judge, the built attack
    stores THAT precise instance as its objective scorer. Controlling the object end to end
    (rather than relying on ``make_judge``) makes the identity assertion unambiguous: PyRIT
    neither wraps nor replaces the injected scorer.

    **Validates: Requirements 4.4, 6.2**
    """
    stub = _StubObjectiveScorer()
    attack_obj = build_attack(attack="homoglyph", judge_override=stub)
    stored = _objective_scorer_of(attack_obj)
    assert stored is stub, f"objective scorer is {stored!r}, expected the injected stub {stub!r}"
    assert not isinstance(stored, FloatScaleThresholdScorer)


def test_default_scale_scorer_built_only_when_none_injected_control(endpoints) -> None:
    """**Feature: crescendo-unicode-arm, Property 5: The Crescendo objective scorer is the injected judge**

    CONTROL / falsifier (research-rigor s3): proves the identity assertions above are not
    vacuous. Constructing ``CrescendoAttack`` DIRECTLY with no scoring config exercises the
    ``if not objective_scorer:`` branch, so PyRIT fabricates its own ``FloatScaleThresholdScorer``
    -- exactly the object the runner must NEVER leave in place. The runner, by contrast, always
    injects the judge (asserted above), so this default never reaches a real run. Construction
    only; the localhost adversarial target is never contacted.

    **Validates: Requirements 4.4, 6.2**
    """
    target, adversarial, _ = endpoints
    adv_cfg = runner.P.AttackAdversarialConfig(target=adversarial)
    attack_obj = runner.P.CrescendoAttack(
        objective_target=target,
        attack_adversarial_config=adv_cfg,
        attack_scoring_config=None,  # nothing injected => PyRIT builds its own default
    )
    assert isinstance(_objective_scorer_of(attack_obj), FloatScaleThresholdScorer), (
        "with no injected scorer Crescendo should build its default FloatScaleThresholdScorer"
    )


# --------------------------------------------------------------------------- #
# Task 3.6 -- CrescendoOutcome shape/defaults + variant-path wiring.
#
# These are NOT numbered correctness properties; they are the task-3.6 coverage for
# Requirements 4.6 (optional Crescendo variant template) and 4.7 (PyRIT default when omitted).
# Every assertion is structural or construction-only -- no model is contacted.
#
# Installed-source anchors (PyRIT 1.1.0, read this session):
# * ``pyrit/executor/attack/core/attack_config.py`` -- ``@dataclass AttackAdversarialConfig``
#   declares ``system_prompt: str | SeedPrompt | None = None`` (the field NAME asserted on and
#   the ``None`` default the "omitted" case relies on). Its ``resolve_adversarial_system_prompt``
#   validates an explicitly-provided ``SeedPrompt`` against the strategy's required parameters.
# * ``pyrit/executor/attack/multi_turn/crescendo.py`` ``__init__`` resolves that config with
#   ``system_prompt_required_parameters=["objective", "max_turns"]`` -- so the minimal variant
#   YAML below MUST declare those two parameters or construction raises (that resolution runs
#   inside the real runner build these tests drive via ``build_attack``).
# * ``pyrit/models/seeds/seed.py`` -- ``Seed.value: str`` is required and the model is
#   ``extra="forbid"``; ``pyrit/models/seeds/yaml_seed_loader.py`` sets ``is_jinja_template=True``
#   and canonicalises a scalar ``parameters`` into a list, so the YAML below loads cleanly via
#   ``SeedPrompt.from_yaml_file`` (confirmed in-test before it is threaded through the runner).
# --------------------------------------------------------------------------- #

#: Unique, Jinja-free marker embedded in the minimal variant's ``value`` so a test can prove the
#: LOADED ``SeedPrompt`` (not the raw path, not PyRIT's shipped default) is what the runner
#: threaded into the adversarial ``system_prompt``.
_MIN_VARIANT_MARKER = "MINIMAL_TEST_VARIANT_MARKER_7f3a1c"

#: A minimal, real, loadable Crescendo variant YAML written to ``tmp_path`` per test. This is the
#: PREFERRED approach over a monkeypatched sentinel (task-3.6 note): it declares ``value``
#: (required) and the ``objective``/``max_turns`` parameters ``CrescendoAttack`` validates, and
#: pins ``data_type: text``. No Jinja placeholders are used, so the marker survives the silent
#: load-time render verbatim and the assertions do not depend on placeholder reproduction.
_MIN_VARIANT_YAML = textwrap.dedent(
    f"""\
    name: Minimal Test Crescendo Variant
    description: Minimal offline-test variant; declares the params CrescendoAttack requires.
    data_type: text
    parameters:
      - objective
      - max_turns
    value: |
      {_MIN_VARIANT_MARKER} minimal crescendo adversarial system prompt for the offline wiring test.
    """
)

#: The ``CrescendoOutcome`` field contract (names + definition order) read from the runner source
#: this session (``smuggle/runner.py`` -> ``@dataclass(frozen=True) class CrescendoOutcome``).
_EXPECTED_OUTCOME_FIELDS = (
    "seed_id", "objective", "family", "mode", "unicode_on", "executed_turns",
    "backtrack_count", "judge_success", "last_response", "model", "adversarial_model",
    "max_turns", "max_backtracks", "variant_label", "outcome_reason",
)


def test_crescendo_outcome_shape_defaults_and_unicode_off_invariant() -> None:
    """**Feature: crescendo-unicode-arm, Task 3.6: CrescendoOutcome shape + Unicode-off invariant**

    Structural and fully offline (no model, no runner call). Enumerates ``CrescendoOutcome`` via
    ``dataclasses.fields`` and asserts (a) every expected field is present with the exact name
    AND definition order the runner declares, (b) the two defaulted fields ``variant_label`` and
    ``outcome_reason`` each default to ``""`` while every other field is required (no default),
    and (c) the runner's Unicode-off invariant holds on a constructed instance: ``family == ()``
    and ``mode == "plaintext"`` when ``unicode_on`` is False. The runner sets exactly these in its
    ``attack is None`` branch (``family=()`` and ``mode=(mode if families else "plaintext")``),
    whose converter-level consequence Property 4's ``attack=None`` test already exercises.

    **Validates: Requirements 4.6, 4.7**
    """
    field_map = {f.name: f for f in dataclasses.fields(CrescendoOutcome)}

    # (a) exact field set -- names AND definition order.
    assert tuple(field_map) == _EXPECTED_OUTCOME_FIELDS, (
        f"CrescendoOutcome fields are {tuple(field_map)!r}, expected {_EXPECTED_OUTCOME_FIELDS!r}"
    )

    # (b) defaults: the two documented defaulted fields are "", everything else is required.
    assert field_map["variant_label"].default == "", "variant_label must default to ''"
    assert field_map["outcome_reason"].default == "", "outcome_reason must default to ''"
    for name in _EXPECTED_OUTCOME_FIELDS:
        if name in {"variant_label", "outcome_reason"}:
            continue
        assert field_map[name].default is dataclasses.MISSING, (
            f"field {name!r} must be required (no default); found default {field_map[name].default!r}"
        )

    # (c) Unicode-off invariant on a constructed instance (the shape the runner emits for
    #     attack=None): empty family tuple, the plaintext mode sentinel, and the two defaulted
    #     fields falling back to "" when not supplied.
    off = CrescendoOutcome(
        seed_id="p0", objective="irrelevant", family=(), mode="plaintext", unicode_on=False,
        executed_turns=0, backtrack_count=0, judge_success=False, last_response="",
        model="target-model", adversarial_model="adversarial-model",
        max_turns=10, max_backtracks=10,
    )
    assert off.unicode_on is False
    assert off.family == (), "Unicode-off outcome must carry an empty family tuple"
    assert off.mode == "plaintext", "Unicode-off outcome must carry mode == 'plaintext'"
    assert off.variant_label == "", "variant_label must default to '' (PyRIT default variant)"
    assert off.outcome_reason == ""


def test_variant_path_threads_loaded_seedprompt_into_adversarial_system_prompt(
    build_attack, endpoints, tmp_path
) -> None:
    """**Feature: crescendo-unicode-arm, Task 3.6: variant_path -> AttackAdversarialConfig.system_prompt**

    A supplied ``variant_path`` must cause the runner to LOAD the YAML into a ``SeedPrompt`` and
    place THAT object on the ``AttackAdversarialConfig.system_prompt`` it passes to
    ``CrescendoAttack`` -- observed from the captured constructor kwargs, with no model call. The
    real minimal variant YAML (preferred over a sentinel) is first confirmed loadable via the
    same ``SeedPrompt.from_yaml_file`` loader the runner uses, and the embedded marker proves the
    threaded object is the loaded template rather than the raw path or PyRIT's shipped default.

    **Validates: Requirements 4.6, 4.7**
    """
    _, adversarial, _ = endpoints
    variant = tmp_path / "min_variant.yaml"
    variant.write_text(_MIN_VARIANT_YAML, encoding="utf-8")

    # Sanity: the minimal YAML is loadable by the SAME loader the runner calls (P.SeedPrompt).
    loaded = runner.P.SeedPrompt.from_yaml_file(str(variant))
    assert isinstance(loaded, runner.P.SeedPrompt)
    assert _MIN_VARIANT_MARKER in (loaded.value or ""), "sanity: marker survived the YAML load"

    # Drive the REAL runner build with the variant path (prompts=[] => no model call).
    build_attack(attack="homoglyph", variant_path=str(variant))
    adv_cfg = build_attack.captured_kwargs[-1]["attack_adversarial_config"]

    assert isinstance(adv_cfg, runner.P.AttackAdversarialConfig)
    assert adv_cfg.target is adversarial, "runner must pass the injected adversarial target through"
    assert isinstance(adv_cfg.system_prompt, runner.P.SeedPrompt), (
        f"system_prompt is {type(adv_cfg.system_prompt).__name__}, expected a loaded SeedPrompt"
    )
    assert _MIN_VARIANT_MARKER in (adv_cfg.system_prompt.value or ""), (
        "the adversarial system_prompt is not the SeedPrompt loaded from variant_path"
    )


def test_omitting_variant_path_leaves_adversarial_system_prompt_none(build_attack) -> None:
    """**Feature: crescendo-unicode-arm, Task 3.6: omitted variant_path leaves system_prompt None**

    Omitting ``variant_path`` must leave the ``AttackAdversarialConfig.system_prompt`` the runner
    passes UNSET (``None``) -- the installed dataclass default
    (``attack_config.py``: ``system_prompt: str | SeedPrompt | None = None``) -- so PyRIT later
    resolves its shipped ``crescendo_variant_1.yaml`` itself. Observed from the captured
    constructor kwargs, with no model call.

    **Validates: Requirements 4.6, 4.7**
    """
    build_attack(attack="homoglyph")  # no variant_path => runner default None
    adv_cfg = build_attack.captured_kwargs[-1]["attack_adversarial_config"]

    assert isinstance(adv_cfg, runner.P.AttackAdversarialConfig)
    assert adv_cfg.system_prompt is None, (
        "omitting variant_path must leave system_prompt unset (None) so PyRIT resolves its "
        f"shipped crescendo_variant_1.yaml; got {adv_cfg.system_prompt!r}"
    )
