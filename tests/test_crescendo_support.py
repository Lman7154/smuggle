r"""Offline property test for the composite attack-success judge (task 3.7).

Property 9 (design.md): the composite judge scores success as true if and only if
the task-achieved scorer is true AND the refusal scorer is false. This validates
Requirement 6.2 ("THE Judge SHALL score a response as a success only when the
response achieves the objective AND is not a refusal").

No model and no network. ``make_judge`` (smuggle.runner) builds PyRIT's canonical
composite:

    objective   = SelfAskTrueFalseScorer(chat_target=...)              # task achieved?
    not_refused = TrueFalseInverterScorer(SelfAskRefusalScorer(...))   # not a refusal?
    judge       = TrueFalseCompositeScorer(AND, [objective, not_refused])

Only the two Self-Ask LEAF scorers are stubbed (so no model is contacted); the
inverter, the composite and the AND aggregator are the REAL pip-installed PyRIT
1.1.0 classes, so this exercises the real composition rather than a reimplementation.

Interface anchored to installed source read this session under
``...\Python313\Lib\site-packages\pyrit`` (pinned 1.1.0, NOT GitHub main):
  * score/true_false/true_false_composite_scorer.py - the composite forwards each
    child via ``child._score_nested_async(scorable=, expectation=)`` and reduces the
    one-score-per-child results with the aggregator in ``_build_aggregate_score``.
  * score/true_false/true_false_inverter_scorer.py - the inverter flips a determinate
    child verdict (True<->False), leaving an undetermined one unchanged.
  * score/true_false/true_false_score_aggregator.py - ``TrueFalseScoreAggregator.AND``
    reduces verdicts with a three-valued ``_and`` (one definite False settles to False),
    which for two determinate booleans is ordinary boolean AND.
  * score/true_false/true_false_scorer.py + score/scorer.py - children are reached
    through ``_score_nested_async`` -> the abstract ``_score_scorable_async``; both the
    composite and the inverter ``isinstance``-check ``TrueFalseScorer`` on their
    children, so the stub subclasses the REAL base and implements the abstract
    ``_score_scorable_async`` / ``_build_identifier``. ``Score.get_value()`` parses a
    ``true_false`` score value ("true"/"false") into a bool (models/score/score.py).

The public ``score_text_async`` (what runner._score calls) is a thin wrapper over the
composite's ``_score_scorable_async`` that ADDITIONALLY persists the aggregate to
CentralMemory. Property 9 concerns only the aggregation verdict, so this test calls
``_score_scorable_async`` directly to stay fully offline with no memory side-effects,
matching the hermetic style of the rest of ``tests/``.

Property 9 (task 3.7) is implemented first; the task 3.8 reproducibility (Requirement 6.4)
and import-surface (Requirement 6.6) smoke tests follow at the end of this file. Those two are
structure/construction-only - no model, no network - matching the hermetic style above.
"""
from __future__ import annotations

import asyncio
import inspect
import re
from pathlib import Path

import pytest
from pyrit.models import ContentScorable, Score, ScoreStatus, ScoringExpectation

# The composite and inverter require their children to be real ``TrueFalseScorer``
# instances (isinstance check). The base is not part of the ``pyrit_wiring`` surface,
# so import it from its installed module directly.
from pyrit.score.true_false.true_false_scorer import TrueFalseScorer

import smuggle.runner  # for smuggle.make_target and the runner source path (task 3.8)
from smuggle import pyrit_wiring as P  # the SAME module object runner.py binds as ``P``
from smuggle.runner import make_judge


class _StubLeafScorer(TrueFalseScorer):
    """A real ``TrueFalseScorer`` that returns a fixed verdict, with no model or network.

    Subclassing PyRIT's real base is what lets ``TrueFalseInverterScorer`` and
    ``TrueFalseCompositeScorer`` accept it (both ``isinstance``-check ``TrueFalseScorer``).
    It overrides ``_score_scorable_async`` - the exact method both reach through
    ``_score_nested_async`` in PyRIT 1.1.0 - to return one completed ``true_false``
    ``Score`` carrying ``verdict``. ``chat_target`` is accepted because ``make_judge``
    passes it, and is ignored.
    """

    def __init__(self, *, verdict: bool, chat_target: object | None = None) -> None:
        super().__init__()
        self._verdict = bool(verdict)

    def _build_identifier(self):
        # Mirror the real leaf scorers (self_ask_*_scorer.py): identity via _create_identifier.
        return self._create_identifier(params={"stub": type(self).__name__, "verdict": self._verdict})

    async def _score_scorable_async(self, *, scorable, expectation):
        return [Score(score_value=str(self._verdict), score_type="true_false", status=ScoreStatus.COMPLETE)]


def _composite_verdict(monkeypatch: pytest.MonkeyPatch, *, achieved: bool, refused: bool) -> bool:
    """Build ``make_judge`` with the two leaf scorers stubbed to ``(achieved, refused)`` and
    return the composite's boolean verdict for a dummy response/objective.

    ``make_judge`` reads ``P.SelfAskTrueFalseScorer`` / ``P.SelfAskRefusalScorer`` where
    ``P`` is ``smuggle.runner.P`` is ``smuggle.pyrit_wiring``; ``monkeypatch`` restores the
    real classes on teardown so later tests in this file see the genuine PyRIT surface.
    """
    monkeypatch.setattr(P, "SelfAskTrueFalseScorer",
                        lambda *, chat_target=None: _StubLeafScorer(verdict=achieved))
    monkeypatch.setattr(P, "SelfAskRefusalScorer",
                        lambda *, chat_target=None: _StubLeafScorer(verdict=refused))

    judge = make_judge(object())  # chat_target is a dummy; the stub leaf scorers ignore it

    scores = asyncio.run(judge._score_scorable_async(
        scorable=ContentScorable(value="dummy response"),
        expectation=ScoringExpectation(objective="dummy objective"),
    ))
    assert len(scores) == 1, f"the composite must return exactly one score, got {scores!r}"
    return bool(scores[0].get_value())


# The full (achieved, refused) truth table. Deterministic enumeration of all four rows is
# complete coverage: Property 9 is a fixed truth table, not one of the randomised properties
# {3, 6, 7, 8}, so no 100-iteration minimum applies.
_TRUTH_TABLE = [(achieved, refused) for achieved in (True, False) for refused in (True, False)]


@pytest.mark.parametrize("achieved, refused", _TRUTH_TABLE)
def test_composite_is_achieved_and_not_refused(achieved: bool, refused: bool, monkeypatch) -> None:
    """**Feature: crescendo-unicode-arm, Property 9: The composite judge is success iff achieved and not refused**

    For every ``(achieved, refused)`` combination the composite judge's verdict equals
    ``achieved AND NOT refused``. A deviation on any row (refusal not inverted, or OR used
    instead of AND) is a real judge-composition defect in ``make_judge`` - not a test
    artefact - and must surface here.

    **Validates: Requirements 6.2**
    """
    expected = achieved and not refused
    assert _composite_verdict(monkeypatch, achieved=achieved, refused=refused) is expected


def test_exactly_one_truth_table_row_succeeds(monkeypatch) -> None:
    """**Feature: crescendo-unicode-arm, Property 9: The composite judge is success iff achieved and not refused**

    Enumerate the whole four-row table and assert success occurs for EXACTLY the single
    row ``(achieved=True, refused=False)``. This is the discriminating check: an
    OR-composition or a missing refusal inversion would light up a different set of rows.

    **Validates: Requirements 6.2**
    """
    successes = {
        (achieved, refused)
        for (achieved, refused) in _TRUTH_TABLE
        if _composite_verdict(monkeypatch, achieved=achieved, refused=refused)
    }
    assert successes == {(True, False)}, (
        f"composite judged success on {sorted(successes)}; "
        "expected success on only (achieved=True, refused=False)"
    )


# --------------------------------------------------------------------------- #
# Task 3.8 smoke tests: reproducibility (Requirement 6.4) and import-surface
# discipline (Requirement 6.6). Both are structure/construction-only - no model
# and no network - so they run in the same hermetic style as Property 9 above.
# --------------------------------------------------------------------------- #


def test_make_target_pins_temperature_zero() -> None:
    """Reproducibility: ``make_target`` pins deterministic decoding (``temperature == 0.0``).

    Requirement 6.4 ("set the sampling temperature to 0.0 where the target accepts a
    temperature setting"). ``make_target`` builds the objective target with a pinned
    ``temperature=0.0`` default; a non-deterministic default (e.g. a provider's 0.7) would
    make every run unreproducible and must surface here.

    Construction only: a localhost ``base_url`` is passed and NOTHING is sent (no model call).
    The pinned value is read off the CONSTRUCTED object's ``_temperature`` attribute, which
    installed PyRIT 1.1.0 sets verbatim from the ``temperature`` argument in
    ``OpenAIChatTarget.__init__`` (``self._temperature = temperature``) - read this session at
    ``...\\site-packages\\pyrit\\prompt_target\\openai\\openai_chat_target.py`` (pinned 1.1.0,
    NOT GitHub main). This asserts the real value on the built target, not a reimplementation.

    **Validates: Requirements 6.4**
    """
    target = smuggle.make_target(
        "any-model", base_url="http://localhost:11434/v1", api_key="ollama"
    )  # construction only: localhost endpoint, no prompt is ever sent
    assert type(target).__name__ == "OpenAIChatTarget"

    # Primary check: the pinned value lives on the constructed object (installed-source attr).
    assert target._temperature == 0.0, (
        f"make_target must pin temperature=0.0 on the built target; got {target._temperature!r}"
    )

    # Reinforcing check: the pin IS the make_target default, so a caller that passes no
    # temperature gets 0.0. This documents the source-level default and catches a changed
    # default even if PyRIT were ever to relocate the attribute.
    default_temperature = inspect.signature(smuggle.make_target).parameters["temperature"].default
    assert default_temperature == 0.0, (
        f"make_target's temperature default must be 0.0; signature reports {default_temperature!r}"
    )


# Allowlist for the import-surface check: the EXACT set of direct ``pyrit`` import statements
# present in smuggle/runner.py this session (enumerated by reading the whole file). Every one is
# a LOCAL (in-function) import; the sole module-scope PyRIT access is ``from . import
# pyrit_wiring as P``. Why each name is allowlisted:
#   * ``from pyrit.models import Message`` - run_async / _as_prepended / _adv_priming_turn build
#     PyRIT Messages for a prepended conversation (three in-function sites).
#   * ``from pyrit.models import SeedPrompt`` - _load_priming_template loads the adversarial
#     system-prompt template (one in-function site).
#   * ``from pyrit.prompt_normalizer import ConverterConfiguration`` - run_redteam_async and
#     run_crescendo_async attach the MutationEngine as a request converter (two in-function
#     sites). This is the one local import the task explicitly permits.
# Anything else importing a PyRIT class straight into runner.py must go through ``P`` instead.
_ALLOWED_RUNNER_PYRIT_IMPORTS = frozenset({
    "from pyrit.models import Message",
    "from pyrit.models import SeedPrompt",
    "from pyrit.prompt_normalizer import ConverterConfiguration",
})

# Matches a DIRECT pyrit-package import at any indentation (``from pyrit...`` / ``import pyrit``).
# The trailing ``\bpyrit\b`` boundary deliberately EXCLUDES ``from . import pyrit_wiring as P``:
# the ``_`` in ``pyrit_wiring`` is a word character, so there is no word boundary after ``pyrit``,
# and that line is the allowed wiring surface rather than a direct package import.
_DIRECT_PYRIT_IMPORT = re.compile(r"^\s*(?:from|import)\s+pyrit\b")


def test_runner_reaches_pyrit_only_through_wiring_surface() -> None:
    """Import-surface discipline: runner.py reaches PyRIT only through ``P`` plus allowlisted locals.

    Requirement 6.6 (net-new code confined to the wiring layers). ``smuggle/runner.py`` must reach
    PyRIT only through the single ``pyrit_wiring`` surface (bound as ``P``) plus the handful of
    local imports ``run_redteam_async`` already established. A static scan of the runner SOURCE
    TEXT (no import side effects, no model) asserts:

      (a) every direct ``pyrit`` import is in the allowlist - catching the target regression where
          a PyRIT class is imported straight into runner.py instead of going through ``P``; and
      (b) no direct ``pyrit`` import sits at module scope - so the only module-scope PyRIT symbol
          stays ``from . import pyrit_wiring as P``.

    **Validates: Requirements 6.6**
    """
    source = Path(smuggle.runner.__file__).read_text(encoding="utf-8")
    direct = [
        (n, line)
        for n, line in enumerate(source.splitlines(), start=1)
        if _DIRECT_PYRIT_IMPORT.match(line)
    ]

    # Guard against a vacuous pass: the runner DOES import PyRIT names locally, so the scan must
    # find them. Finding none means the regex or the file layout changed and this test is no
    # longer checking what it claims to.
    assert direct, (
        "expected runner.py to contain the allowlisted local pyrit imports, but the scan found "
        "none - the regex or the runner layout changed and this guard must be revisited."
    )

    # (a) No NEW direct pyrit surface beyond what run_redteam_async established.
    unexpected = {line.strip() for _, line in direct} - _ALLOWED_RUNNER_PYRIT_IMPORTS
    assert not unexpected, (
        f"runner.py has direct pyrit import(s) outside the allowlist: {sorted(unexpected)}. "
        "Route PyRIT access through smuggle.pyrit_wiring (imported as P) rather than importing "
        "the PyRIT class directly in runner.py."
    )

    # (b) Module scope reaches PyRIT only via ``P``: every allowlisted import is LOCAL (indented
    #     inside a function), so none of them appears at column 0.
    module_scope = [(n, line.strip()) for n, line in direct if line == line.lstrip()]
    assert not module_scope, (
        f"runner.py has a module-scope direct pyrit import: {module_scope}. Module scope must "
        "reach PyRIT only through `from . import pyrit_wiring as P`."
    )
