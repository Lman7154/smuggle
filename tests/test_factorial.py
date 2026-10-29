"""Property + teeth tests for the Workstream-C factorial / Best-of-N analysis math.

This module hosts the pure-analysis tests for Workstream C of the crescendo-unicode-arm
spec. It opens with Property 6 (Best-of-N), then Property 7 (the difference-in-differences
interaction readout); the separated bypass/compliance property (Property 8) and the
``factorial_2x2`` unit test are appended by later tasks (5.6-5.7). Shared record-building
helpers live at the top so those tests can reuse them, and each property is kept under its
own banner so additions slot in cleanly.

**Feature: crescendo-unicode-arm, Property 6: Best-of-N success is the maximum over draws**
**Validates: Requirements 5.5**

Offline only: pure analysis math on synthetic outcome-shaped records. NO model, NO network,
NO PyRIT. The function under test -- ``smuggle.analysis.per_prompt_success`` -- is reused
exactly as shipped (never reimplemented); it reads ``getattr(o, "seed_id", "")`` and
``bool(o.judge_success)`` on each record and, for ``aggregate="any"``, collapses a
behaviour's draws with the builtin ``any`` (best-of-N).

Scientific framing (Requirement 5.5 / design Property 6): WHERE a Best-of-N search is
reported, the Analysis_Module reports the MAXIMUM success over the draws -- i.e. a prompt
(seed_id) counts as a success exactly when at least one of its draws succeeded, which is the
logical OR over that prompt's draw outcomes.

Randomisation strategy (documented choice, per tasks.md Notes = minimum 100 iterations):
Hypothesis is installed under the pinned interpreter, so it is used here (the project's
preferred engine; same convention as ``tests/test_jailbreak_prepend.py``). That sibling test
draws from a tiny ``sampled_from`` space -- which Hypothesis exhausts after ~36 examples -- so
it needed a wide-integer draw to still reach 100. The strategy here is different: a
lists-of-lists of booleans (each inner list is one prompt's draws), whose space is
astronomically large, so Hypothesis runs the full ``max_examples`` with NO early exhaustion
and no wide-integer trick is required. We set ``max_examples=200`` (margin over the 100
minimum) and additionally COUNT the real property-body executions in a module-level counter
that ``test_property6_ran_minimum_iterations`` asserts is >= 100, so the floor is checked, not
assumed. Had Hypothesis been absent, the fallback would be a deterministic seeded ``random``
loop of >= 100 iterations (not needed here).

Teeth (falsification): ``test_property6_teeth_any_differs_from_all_on_mixed_set`` shows the
``any`` aggregator genuinely differs from ``all`` on a mixed draw set, so the Property-6
assertion is not vacuous -- wiring the wrong aggregator (e.g. ``all``) would fail it.
"""
from __future__ import annotations

from itertools import zip_longest
from types import SimpleNamespace

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

import smuggle
from smuggle.analysis import factorial_2x2, per_prompt_success

#: >= 100 iterations (tasks.md Notes); 200 leaves margin. The lists-of-lists strategy below
#: has a huge input space, so no wide-integer trick is needed to avoid early exhaustion.
_MAX_EXAMPLES = 200
_MIN_ITERATIONS = 100

#: Counts real property-body executions so the >= 100-iteration floor is asserted, not
#: assumed. Checked by ``test_property6_ran_minimum_iterations``, which is defined AFTER the
#: property so pytest's in-file definition order (no pytest-randomly; single process -- the
#: prescribed run passes no ``-n``) runs it last.
_iteration_count = {"n": 0}


def _records_from(draws_by_prompt: dict[str, list[bool]], *, interleave: bool = True) -> list[SimpleNamespace]:
    """Build minimal outcome-shaped records from a ``{seed_id: [draw booleans]}`` mapping.

    Each record is a ``SimpleNamespace`` carrying only the two attributes
    ``per_prompt_success`` reads: ``seed_id`` and ``judge_success``. No model is involved.
    With ``interleave=True`` (default) the records for one seed_id are NOT contiguous
    (round-robin across prompts), which exercises ``per_prompt_success``'s group-by-key
    contract rather than letting a naive block-grouping implementation pass.
    """
    columns = [
        [SimpleNamespace(seed_id=sid, judge_success=bool(b)) for b in draws]
        for sid, draws in draws_by_prompt.items()
    ]
    if not interleave:
        return [rec for group in columns for rec in group]
    records: list[SimpleNamespace] = []
    for row in zip_longest(*columns, fillvalue=None):
        records.extend(rec for rec in row if rec is not None)
    return records


# --------------------------------------------------------------------------- #
# Property 6: Best-of-N success is the maximum over draws
# **Feature: crescendo-unicode-arm, Property 6: Best-of-N success is the maximum over draws**
# **Validates: Requirements 5.5**
# --------------------------------------------------------------------------- #
#: Each inner list is ONE prompt's draw outcomes (>= 1 boolean draws per the task); the outer
#: list is the set of prompts. Generous bounds keep the space large (no early exhaustion).
_draws_by_prompt = st.lists(
    st.lists(st.booleans(), min_size=1, max_size=8),
    min_size=1,
    max_size=8,
)


@settings(max_examples=_MAX_EXAMPLES, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(draws=_draws_by_prompt)
@example(draws=[[True]])                                  # single succeeding draw -> True
@example(draws=[[False]])                                 # single failing draw    -> False
@example(draws=[[True, False]])                           # mixed draws            -> True (teeth vs "all")
@example(draws=[[False, False], [True, False], [False]])  # several mixed prompts
def test_property6_best_of_n_is_or_over_draws(draws: list[list[bool]]) -> None:
    """**Feature: crescendo-unicode-arm, Property 6: Best-of-N success is the maximum over draws**

    **Validates: Requirements 5.5**

    For random per-prompt draw sets (each prompt has >= 1 boolean draws),
    ``per_prompt_success(records, aggregate="any")`` returns, for every prompt, the logical OR
    over that prompt's draw outcomes -- the Best-of-N maximum (Requirement 5.5).
    """
    _iteration_count["n"] += 1

    # One generated inner list == one prompt (one seed_id); the index gives a distinct key.
    draws_by_prompt = {f"p{i}": d for i, d in enumerate(draws)}
    records = _records_from(draws_by_prompt)

    # Ground truth: a prompt succeeds exactly when at least one of its draws succeeded.
    expected = {sid: any(d) for sid, d in draws_by_prompt.items()}

    assert per_prompt_success(records, aggregate="any") == expected


def test_property6_ran_minimum_iterations() -> None:
    """The randomised Property-6 run met the >= 100-iteration floor (tasks.md Notes)."""
    assert _iteration_count["n"] >= _MIN_ITERATIONS, (
        f"Property 6 ran only {_iteration_count['n']} iterations; need >= {_MIN_ITERATIONS}"
    )


def test_property6_uses_the_shipped_reexported_function() -> None:
    """The function under test is exactly the one the pipeline uses (the package re-export),
    not a shadow copy, so the property pins real shipped behaviour."""
    assert smuggle.per_prompt_success is per_prompt_success


def test_property6_teeth_any_differs_from_all_on_mixed_set() -> None:
    """Teeth / falsification: on a MIXED draw set the ``any`` (Best-of-N) verdict differs from
    ``all``, so the Property-6 OR assertion is not vacuous -- a wrong aggregator would fail it.
    """
    records = _records_from({"p0": [True, False]})

    best_of_n = per_prompt_success(records, aggregate="any")   # OR  over draws
    all_draws = per_prompt_success(records, aggregate="all")   # AND over draws

    assert best_of_n == {"p0": True}
    assert all_draws == {"p0": False}
    assert best_of_n != all_draws   # the OR assertion would FAIL under aggregate="all"


# --------------------------------------------------------------------------- #
# Property 7: The factorial interaction is the difference-in-differences
# **Feature: crescendo-unicode-arm, Property 7: The factorial interaction is the difference-in-differences**
# **Validates: Requirements 5.3**
# --------------------------------------------------------------------------- #
# Offline only: pure analysis math on synthetic outcome-shaped records. NO model, NO network,
# NO PyRIT. The function under test -- ``smuggle.analysis.factorial_2x2`` -- is reused exactly
# as shipped (the difference-in-differences is NOT reimplemented here). It collapses each cell
# to a behaviour-level ASR via ``per_prompt_success(aggregate="any")`` (Best-of-N over draws),
# then, with the four cell ASRs labelled ``a=single_plain, b=single_enc, c=crescendo_plain,
# d=crescendo_enc``, reports ``unicode_main_effect = b - a``, ``crescendo_main_effect = c - a``,
# and ``interaction = (d - c) - (b - a)`` (Requirement 5.3 / design Property 7).
#
# Randomisation strategy (documented choice, per tasks.md Notes = minimum 100 iterations):
# Hypothesis is used (same engine and ``_MAX_EXAMPLES=200`` / ``_MIN_ITERATIONS=100`` budget as
# Property 6). The strategy draws ONE prompt count ``k`` and then, for each of the four cells,
# ``k`` per-prompt draw-lists, so every cell lands on the SAME prompt ids ``p0..p{k-1}`` -- the
# ASRs are directly comparable and the McNemar pairing inside ``factorial_2x2`` is well defined.
# That lists-of-lists-across-four-cells space is astronomically large, so Hypothesis runs the
# full ``max_examples`` with no early exhaustion (no wide-integer trick needed). The real
# property-body executions are COUNTED in a module-level counter that
# ``test_property7_ran_minimum_iterations`` asserts is >= 100, so the floor is checked, not
# assumed (mirroring Property 6). Had Hypothesis been absent, the fallback would be a
# deterministic seeded ``random`` loop of >= 100 iterations (not needed here).
#
# Teeth (falsification): ``test_property7_teeth_wrong_interaction_formula_would_fail`` shows, on
# a hand-picked asymmetric 2x2, that the difference-in-differences differs from the common wrong
# wirings (sign flip, '+' instead of '-'), so the Property-7 assertions are not vacuous.

#: Cell labels in difference-in-differences order a,b,c,d (stated explicitly rather than imported
#: from the analysis module's private constant, so the test documents the contract it pins).
_CELL_LABELS = ("single_plain", "single_enc", "crescendo_plain", "crescendo_enc")

#: Counts real Property-7 body executions so the >= 100-iteration floor is asserted, not assumed
#: (separate from Property 6's counter). Checked by ``test_property7_ran_minimum_iterations``,
#: defined AFTER the property so pytest's in-file definition order runs it last.
_p7_iteration_count = {"n": 0}


@st.composite
def _four_cell_draws(draw: st.DrawFn) -> dict[str, list[list[bool]]]:
    """Random best-of-N draws for the four factorial cells over a SHARED prompt-id set.

    Draws a single prompt count ``k`` (>= 1), then, for EACH of the four cells, ``k`` independent
    per-prompt draw-lists (each >= 1 boolean). Because every cell gets exactly ``k`` prompts, the
    four cells share the ids ``p0..p{k-1}``: their ASRs are directly comparable and the three
    paired-McNemar contrasts inside ``factorial_2x2`` are well defined."""
    k = draw(st.integers(min_value=1, max_value=6))
    per_prompt = st.lists(st.booleans(), min_size=1, max_size=5)
    return {label: draw(st.lists(per_prompt, min_size=k, max_size=k)) for label in _CELL_LABELS}


@settings(max_examples=_MAX_EXAMPLES, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(cells=_four_cell_draws())
@example(cells={                                    # 1 prompt, SUB-additive: a=0,b=1,c=1,d=1 -> interaction=-1
    "single_plain":    [[False]],
    "single_enc":      [[True]],
    "crescendo_plain": [[True]],
    "crescendo_enc":   [[True]],
})
@example(cells={                                    # 2 prompts, ADDITIVE: a=0,b=.5,c=.5,d=1 -> interaction=0
    "single_plain":    [[False], [False]],
    "single_enc":      [[True],  [False]],
    "crescendo_plain": [[False], [True]],
    "crescendo_enc":   [[True],  [True]],
})
@example(cells={                                    # 2 prompts, SUPER-additive: a=b=c=0,d=1 -> interaction=+1
    "single_plain":    [[False], [False]],
    "single_enc":      [[False], [False]],
    "crescendo_plain": [[False], [False]],
    "crescendo_enc":   [[True],  [True]],
})
def test_property7_interaction_is_difference_in_differences(cells: dict[str, list[list[bool]]]) -> None:
    """**Feature: crescendo-unicode-arm, Property 7: The factorial interaction is the difference-in-differences**

    **Validates: Requirements 5.3**

    For random four-cell success maps over a shared prompt-id set, ``factorial_2x2`` reports the
    two main effects and the interaction as the difference-in-differences on the four cell ASRs,
    with labels ``a=single_plain, b=single_enc, c=crescendo_plain, d=crescendo_enc``:
    ``unicode_main_effect == b - a``, ``crescendo_main_effect == c - a``, and
    ``interaction == (d - c) - (b - a)`` (Requirement 5.3).
    """
    _p7_iteration_count["n"] += 1

    # The strategy guarantees all four cells share the same prompt count; assert it so the
    # shared-id contract (comparable ASRs, well-defined McNemar pairing) is explicit, not implied.
    lengths = {len(draws_lists) for draws_lists in cells.values()}
    assert len(lengths) == 1, "the four factorial cells must share one prompt-id set"
    k = lengths.pop()

    cell_records: dict[str, list] = {}
    expected_asr: dict[str, float] = {}
    for label, draws_lists in cells.items():
        # Shared ids p0..p{k-1} across every cell; one record per draw (reusing _records_from).
        draws_by_prompt = {f"p{i}": draws_lists[i] for i in range(k)}
        cell_records[label] = _records_from(draws_by_prompt)
        # Independent ground truth: a prompt succeeds iff ANY draw succeeds (best-of-N), so the
        # cell ASR = (#succeeding prompts)/(#prompts) -- computed here with plain any()/sum, NOT
        # via the function under test.
        successes = sum(1 for draws in draws_lists if any(draws))
        expected_asr[label] = successes / k

    result = factorial_2x2(cell_records)

    # (1) The function's per-cell ASR matches our independent ground truth (best-of-N over draws,
    #     divided by the number of prompts) -- this pins the a,b,c,d the effects are built from.
    #     ASRs are ratios, so compare with pytest.approx.
    for label in _CELL_LABELS:
        assert result.asr[label] == pytest.approx(expected_asr[label])

    # (2) The two main effects and the interaction equal the difference-in-differences formulas
    #     on OUR independently-computed a,b,c,d (Requirement 5.3). A wrong wiring (swapped cells,
    #     sign flip, or '+' instead of '-') would break this for almost every random input.
    a = expected_asr["single_plain"]
    b = expected_asr["single_enc"]
    c = expected_asr["crescendo_plain"]
    d = expected_asr["crescendo_enc"]
    assert result.unicode_main_effect == pytest.approx(b - a)
    assert result.crescendo_main_effect == pytest.approx(c - a)
    assert result.interaction == pytest.approx((d - c) - (b - a))


def test_property7_ran_minimum_iterations() -> None:
    """The randomised Property-7 run met the >= 100-iteration floor (tasks.md Notes)."""
    assert _p7_iteration_count["n"] >= _MIN_ITERATIONS, (
        f"Property 7 ran only {_p7_iteration_count['n']} iterations; need >= {_MIN_ITERATIONS}"
    )


def test_property7_uses_the_shipped_reexported_function() -> None:
    """The function under test is exactly the one the pipeline uses (the package re-export),
    not a shadow copy, so the property pins real shipped behaviour."""
    assert smuggle.factorial_2x2 is factorial_2x2


def test_property7_teeth_wrong_interaction_formula_would_fail() -> None:
    """Teeth / falsification: on a hand-picked ASYMMETRIC 2x2 the difference-in-differences
    differs from the common wrong wirings, so the Property-7 interaction assertion is not vacuous.

    Four prompts, best-of-N draws chosen so a=0/4, b=1/4, c=3/4, d=2/4:
      unicode_main   = b - a           =  0.25
      crescendo_main = c - a           =  0.75
      interaction    = (d - c) - (b - a) = (0.50 - 0.75) - (0.25 - 0.00) = -0.50
    """
    cells = {
        "single_plain":    _records_from({"p0": [False], "p1": [False], "p2": [False], "p3": [False]}),  # 0/4
        "single_enc":      _records_from({"p0": [True],  "p1": [False], "p2": [False], "p3": [False]}),  # 1/4
        "crescendo_plain": _records_from({"p0": [True],  "p1": [True],  "p2": [True],  "p3": [False]}),  # 3/4
        "crescendo_enc":   _records_from({"p0": [True],  "p1": [True],  "p2": [False], "p3": [False]}),  # 2/4
    }
    a, b, c, d = 0.0, 0.25, 0.75, 0.50

    result = factorial_2x2(cells)

    assert result.asr == pytest.approx({
        "single_plain": a, "single_enc": b, "crescendo_plain": c, "crescendo_enc": d,
    })
    assert result.unicode_main_effect == pytest.approx(b - a)       #  0.25
    assert result.crescendo_main_effect == pytest.approx(c - a)     #  0.75
    assert result.interaction == pytest.approx((d - c) - (b - a))   # -0.50

    # The correct interaction (-0.50) is NOT what the common wrong wirings would produce:
    assert result.interaction != pytest.approx((b - a) - (d - c))   # sign flip  -> +0.50
    assert result.interaction != pytest.approx((d - c) + (b - a))   # '+' not '-' ->  0.00
    # ...and it is a genuinely distinct number from the two main effects on this asymmetric 2x2:
    assert result.interaction != pytest.approx(result.unicode_main_effect)
    assert result.interaction != pytest.approx(result.crescendo_main_effect)


# --------------------------------------------------------------------------- #
# Property 8: Bypass and compliance are reported separately
# **Feature: crescendo-unicode-arm, Property 8: Bypass and compliance are reported separately**
# **Validates: Requirements 5.4**
# --------------------------------------------------------------------------- #
# Offline only: pure analysis math on synthetic outcome-shaped records. NO model, NO network,
# NO PyRIT. The function under test -- ``smuggle.analysis.factorial_2x2`` -- is reused exactly
# as shipped (nothing reimplemented); it is the same object as ``smuggle.factorial_2x2`` (pinned
# by ``test_property7_uses_the_shipped_reexported_function`` above). For each cell it reports two
# SEPARATE scalars, each from its own field:
#   * compliance ASR  via ``_cell_prompt_asr`` -> ``per_prompt_success`` (best-of-N over
#     ``judge_success``, keyed by seed_id: a PER-SEED fraction, independent of the guardrail);
#   * guardrail bypass via ``_cell_bypass_rate`` over the subset with ``input_flagged is not
#     None``: the PER-RECORD fraction whose ``input_flagged`` is falsy, or ``None`` when that
#     subset is empty.
# The two use different fields AND different denominators, so matching each to its own
# independent ground truth proves they are never collapsed into one number (Requirement 5.4 /
# design Property 8). ``factorial_2x2`` stores ``compliance_asr=dict(asr_d)`` (a COPY) and
# ``bypass_rate=bypass_d``, so the two are distinct dict objects on the result.
#
# Scientific framing (research discipline): guardrail BYPASS (the token surface the external
# filter sees) and model COMPLIANCE (willingness after reading through the encoding) are two
# layers that must be measured separately; conflating them would measure neither. The two
# Crescendo cells deliver a different adversary-generated payload each turn, so a single
# comparable bypass figure is undefined there -- their ``CrescendoOutcome`` records carry no
# ``input_flagged`` attribute at all, so ``getattr(o, "input_flagged", None)`` is ``None`` and
# their bypass is reported as ``None`` (undefined), never a fabricated zero.
#
# Randomisation strategy (documented choice, per tasks.md Notes = minimum 100 iterations):
# Hypothesis is used (same engine and ``_MAX_EXAMPLES=200`` / ``_MIN_ITERATIONS=100`` budget as
# Properties 6/7, same counter-based floor assertion). Per example it draws all four cells; each
# cell independently draws a ``guarded`` boolean (the "with / without guardrail verdict" split
# the task calls for) and then a non-empty list of ``(seed_id, judge_success, flag)`` records.
# An UNGUARDED cell draws flags only from {``absent``, ``none``} (so it carries NO verdict ->
# bypass ``None``); a GUARDED cell draws from all four and is guaranteed >= 1 of {``flagged``,
# ``clear``} (so it carries a verdict -> bypass float). Both unguarded encodings are exercised:
# ``none`` sets ``input_flagged=None`` explicitly, while ``absent`` omits the attribute entirely
# (the Crescendo-record shape, hitting the ``getattr`` default-None path) -- the task asks for
# both. That lists-of-records-across-four-cells space is astronomically large, so Hypothesis runs
# the full ``max_examples`` with no early exhaustion (no wide-integer trick needed), and the real
# property-body executions are COUNTED so the >= 100 floor is asserted, not assumed.
#
# Teeth (falsification): ``test_property8_teeth_bypass_is_not_a_copy_of_compliance`` forces
# compliance and bypass to opposite extremes (ASR 1.0 / bypass 0.0 and ASR 0.0 / bypass 1.0) and
# includes two Crescendo-style cells (built with ``_records_from``, i.e. no ``input_flagged``),
# so a collapse (bypass aliased to ASR, or either column fabricated) would fail the ``!=`` and the
# ``None`` assertions -- the property is not vacuous.

#: Small seed-id pool so some seeds get multiple draws (exercising best-of-N over judge_success
#: within a cell, where the bypass denominator is per-record, not per-seed).
_SEED_POOL = ("p0", "p1", "p2", "p3")

#: The four flag tokens a generated record can carry. 'absent' => build a record with NO
#: ``input_flagged`` attribute (identical shape to ``_records_from``'s records, i.e. the
#: ``CrescendoOutcome`` / Crescendo-cell shape -> ``getattr`` default None); 'none' =>
#: ``input_flagged=None`` (explicitly unscored); 'flagged' => ``True`` (filter blocked it, so
#: NOT a bypass); 'clear' => ``False`` (filter let it through, so a bypass).
_GUARDED_FLAGS = ("flagged", "clear")
_UNGUARDED_FLAGS = ("absent", "none")

#: Counts real Property-8 body executions so the >= 100-iteration floor is asserted, not assumed
#: (separate from Properties 6/7). Checked by ``test_property8_ran_minimum_iterations``, defined
#: AFTER the property so pytest's in-file definition order runs it last.
_p8_iteration_count = {"n": 0}


def _guarded_records_from(specs: list[tuple[str, bool, str]]) -> list[SimpleNamespace]:
    """Build outcome-shaped records from ``(seed_id, judge_success, flag)`` specs.

    Extends ``_records_from``'s ``(seed_id, judge_success)`` shape with an optional
    ``input_flagged`` per the ``flag`` token (see ``_GUARDED_FLAGS`` / ``_UNGUARDED_FLAGS``).
    The ``'absent'`` branch omits the attribute entirely, which is byte-for-byte the record
    shape ``_records_from`` produces -- the Crescendo-cell shape that drives ``getattr(o,
    "input_flagged", None) -> None``. No model is involved."""
    out: list[SimpleNamespace] = []
    for sid, js, flag in specs:
        if flag == "absent":
            out.append(SimpleNamespace(seed_id=sid, judge_success=bool(js)))
        elif flag == "none":
            out.append(SimpleNamespace(seed_id=sid, judge_success=bool(js), input_flagged=None))
        elif flag == "flagged":
            out.append(SimpleNamespace(seed_id=sid, judge_success=bool(js), input_flagged=True))
        elif flag == "clear":
            out.append(SimpleNamespace(seed_id=sid, judge_success=bool(js), input_flagged=False))
        else:  # pragma: no cover - guards the generator contract
            raise AssertionError(f"unknown flag {flag!r}")
    return out


def _has_verdict(specs: list[tuple[str, bool, str]]) -> bool:
    """True iff at least one record in the cell carried a guardrail verdict (a non-None
    ``input_flagged``), i.e. a {flagged, clear} flag -- the exact condition under which
    ``_cell_bypass_rate`` returns a float rather than ``None``."""
    return any(flag in _GUARDED_FLAGS for _, _, flag in specs)


def _expected_bypass(specs: list[tuple[str, bool, str]]) -> float | None:
    """Independent ground truth for the guardrail-bypass rate: over the guarded subset
    ({flagged, clear}), the per-record fraction that EVADED the filter ('clear' =>
    ``input_flagged`` False); ``None`` when the subset is empty. Plain Python, NOT the function
    under test."""
    guarded = [flag for _, _, flag in specs if flag in _GUARDED_FLAGS]
    if not guarded:
        return None
    return sum(1 for flag in guarded if flag == "clear") / len(guarded)


def _expected_best_of_n_asr(specs: list[tuple[str, bool, str]]) -> float:
    """Independent ground truth for compliance ASR: best-of-N over ``judge_success`` keyed by
    seed_id (a seed succeeds iff ANY of its draws did), divided by the number of distinct seeds.
    Plain Python (any()/sum()), independent of ``input_flagged`` -- a different field AND a
    different denominator from the bypass rate."""
    by_seed: dict[str, list[bool]] = {}
    for sid, js, _flag in specs:
        by_seed.setdefault(sid, []).append(bool(js))
    if not by_seed:
        return 0.0
    return sum(1 for draws in by_seed.values() if any(draws)) / len(by_seed)


@st.composite
def _cell_spec(draw: st.DrawFn) -> list[tuple[str, bool, str]]:
    """One cell's records: a ``guarded`` boolean then a non-empty list of ``(seed_id,
    judge_success, flag)``. Unguarded cells draw flags only from {absent, none} (NO verdict);
    guarded cells draw from all four and are guaranteed >= 1 {flagged, clear} (a verdict), so
    both branches of the None-iff-no-verdict invariant are covered across examples."""
    guarded = draw(st.booleans())
    flag_choices = list(_UNGUARDED_FLAGS) + list(_GUARDED_FLAGS) if guarded else list(_UNGUARDED_FLAGS)
    record = st.tuples(st.sampled_from(_SEED_POOL), st.booleans(), st.sampled_from(flag_choices))
    specs = draw(st.lists(record, min_size=1, max_size=8))
    if guarded and not any(flag in _GUARDED_FLAGS for _, _, flag in specs):
        # Guarantee the guarded cell truly carries a verdict (append, never mutate a drawn value).
        specs = specs + [(
            draw(st.sampled_from(_SEED_POOL)),
            draw(st.booleans()),
            draw(st.sampled_from(_GUARDED_FLAGS)),
        )]
    return specs


@st.composite
def _four_cells(draw: st.DrawFn) -> dict[str, list[tuple[str, bool, str]]]:
    """All four factorial cells, each independently guarded or unguarded (``factorial_2x2``
    requires every label to be present)."""
    return {label: draw(_cell_spec()) for label in _CELL_LABELS}


@settings(max_examples=_MAX_EXAMPLES, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(cell_specs=_four_cells())
@example(cell_specs={                                   # single cells guarded (ASR != bypass); both Crescendo cells = None bypass
    "single_plain":    [("p0", True, "flagged"), ("p1", True, "flagged")],   # asr 1.0, bypass 0.0
    "single_enc":      [("p0", False, "clear"),  ("p1", True,  "clear")],    # asr 0.5, bypass 1.0
    "crescendo_plain": [("p0", True, "absent"),  ("p1", False, "absent")],   # no attr -> bypass None, asr 0.5
    "crescendo_enc":   [("p0", False, "absent")],                            # no attr -> bypass None, asr 0.0
})
@example(cell_specs={                                   # explicit None path + mixed flagged/clear on one seed (best-of-N vs per-record bypass)
    "single_plain":    [("p0", True, "none"), ("p1", False, "none")],        # input_flagged=None -> bypass None
    "single_enc":      [("p0", True, "flagged"), ("p0", False, "clear")],    # one seed: asr 1.0; guarded {1 flagged,1 clear} -> bypass 0.5
    "crescendo_plain": [("p0", False, "absent")],
    "crescendo_enc":   [("p0", True, "absent")],
})
@example(cell_specs={                                   # guarded cell MIXING unguarded + verdict records (bypass over the guarded subset only)
    "single_plain":    [("p0", True, "absent"), ("p1", True, "clear"), ("p2", False, "flagged")],  # guarded subset {clear,flagged} -> bypass 0.5
    "single_enc":      [("p0", True, "clear")],
    "crescendo_plain": [("p0", True, "absent")],
    "crescendo_enc":   [("p0", False, "none")],                              # 'none' -> unguarded -> bypass None
})
def test_property8_bypass_and_compliance_reported_separately(
    cell_specs: dict[str, list[tuple[str, bool, str]]],
) -> None:
    """**Feature: crescendo-unicode-arm, Property 8: Bypass and compliance are reported separately**

    **Validates: Requirements 5.4**

    For random outcome sets with and without guardrail verdicts, ``factorial_2x2`` (the shipped
    ``smuggle.factorial_2x2``) reports compliance ASR and guardrail-bypass rate as two DISTINCT
    values that each match their own independent definition and are never collapsed into one; and
    the bypass rate is ``None`` EXACTLY when no outcome in that cell carried a guardrail verdict
    (so Crescendo-style cells, whose records have no ``input_flagged``, report ``None`` bypass).
    """
    _p8_iteration_count["n"] += 1

    cells = {label: _guarded_records_from(specs) for label, specs in cell_specs.items()}
    result = factorial_2x2(cells)

    # Separateness at the container level: compliance ASR and guardrail bypass are two DISTINCT
    # dict fields (not the same object), and compliance_asr is its OWN object (a copy of asr), so
    # ASR is never aliased into the bypass column.
    assert result.compliance_asr is not result.bypass_rate
    assert result.compliance_asr is not result.asr
    assert result.compliance_asr == result.asr

    for label, specs in cell_specs.items():
        has_verdict = _has_verdict(specs)
        expected_bypass = _expected_bypass(specs)
        expected_asr = _expected_best_of_n_asr(specs)

        # (3) None-iff-no-verdict, EXACTLY (the core invariant): bypass is None precisely when the
        #     cell carried no guardrail verdict, and a float precisely when it did.
        assert (result.bypass_rate[label] is None) == (not has_verdict)

        if has_verdict:
            # (1) bypass is a float in [0,1] equal to the independent guarded-subset fraction.
            assert isinstance(result.bypass_rate[label], float)
            assert 0.0 <= result.bypass_rate[label] <= 1.0
            assert result.bypass_rate[label] == pytest.approx(expected_bypass)
        else:
            # (2) no verdict -> bypass None (compliance is still defined, checked below).
            assert result.bypass_rate[label] is None

        # compliance ASR is ALWAYS a float and equals the independent best-of-N ASR over
        # judge_success -- computed from a different field and denominator than the bypass rate.
        assert isinstance(result.compliance_asr[label], float)
        assert result.compliance_asr[label] == pytest.approx(expected_asr)

        # (4) never collapsed: wherever the two definitions genuinely disagree, the reported
        #     numbers disagree too. A collapse (reporting one where the other is wanted) would
        #     force them equal and fail here.
        if has_verdict and expected_bypass != pytest.approx(expected_asr):
            assert result.compliance_asr[label] != pytest.approx(result.bypass_rate[label])


def test_property8_ran_minimum_iterations() -> None:
    """The randomised Property-8 run met the >= 100-iteration floor (tasks.md Notes)."""
    assert _p8_iteration_count["n"] >= _MIN_ITERATIONS, (
        f"Property 8 ran only {_p8_iteration_count['n']} iterations; need >= {_MIN_ITERATIONS}"
    )


def test_property8_teeth_bypass_is_not_a_copy_of_compliance() -> None:
    """Teeth / falsification: force compliance ASR and bypass rate to OPPOSITE extremes and add
    two Crescendo-style cells with no verdict. If the two columns were ever collapsed (bypass
    aliased to ASR, or either fabricated), the ``!=`` and ``None`` assertions would fail -- so the
    Property-8 separateness claim is not vacuous. The Crescendo cells are built with
    ``_records_from`` (no ``input_flagged`` attribute), the concrete "both Crescendo cells report
    None bypass" case from the design."""
    cells = {
        # every judge True (compliance ASR 1.0) but every record flagged/blocked (bypass 0.0)
        "single_plain": _guarded_records_from([("p0", True, "flagged"), ("p1", True, "flagged")]),
        # every judge False (compliance ASR 0.0) but every record clear/evaded (bypass 1.0)
        "single_enc":   _guarded_records_from([("p0", False, "clear"), ("p1", False, "clear")]),
        # Crescendo cells: records with NO input_flagged attribute (exactly _records_from's shape)
        "crescendo_plain": _records_from({"p0": [True, False], "p1": [False]}),   # best-of-N asr = 1/2
        "crescendo_enc":   _records_from({"p0": [False]}),                        # best-of-N asr = 0/1
    }

    result = factorial_2x2(cells)

    # Single-turn cells: compliance and bypass are genuinely DIFFERENT numbers (never collapsed).
    assert result.compliance_asr["single_plain"] == pytest.approx(1.0)
    assert result.bypass_rate["single_plain"] == pytest.approx(0.0)
    assert result.compliance_asr["single_plain"] != pytest.approx(result.bypass_rate["single_plain"])

    assert result.compliance_asr["single_enc"] == pytest.approx(0.0)
    assert result.bypass_rate["single_enc"] == pytest.approx(1.0)
    assert result.compliance_asr["single_enc"] != pytest.approx(result.bypass_rate["single_enc"])

    # Both Crescendo cells report None bypass (no comparable single-payload verdict), while
    # compliance ASR stays a real float (best-of-N over judge_success).
    assert result.bypass_rate["crescendo_plain"] is None
    assert result.bypass_rate["crescendo_enc"] is None
    assert isinstance(result.compliance_asr["crescendo_plain"], float)
    assert isinstance(result.compliance_asr["crescendo_enc"], float)
    assert result.compliance_asr["crescendo_plain"] == pytest.approx(0.5)
    assert result.compliance_asr["crescendo_enc"] == pytest.approx(0.0)

    # The two columns are distinct dict objects on the result (compliance is a copy of asr).
    assert result.compliance_asr is not result.bypass_rate
    assert result.compliance_asr is not result.asr

# --------------------------------------------------------------------------- #
# 5.7 factorial_2x2 + format_factorial_table end-to-end unit test (DETERMINISTIC)
# **Validates: Requirements 5.1, 5.2, 5.3, 5.4**
# --------------------------------------------------------------------------- #
# Offline only: pure analysis math on FIXED synthetic outcome-shaped records. NO model, NO
# network, NO PyRIT. ``smuggle.analysis.factorial_2x2`` / ``format_factorial_table`` are reused
# exactly as shipped (nothing reimplemented). Unlike Properties 6-8 (randomised), this is a
# single DETERMINISTIC unit test over four hand-built cells, so EVERY expected number below is
# computed by hand and documented in comments; it pins the whole contract end-to-end on one
# concrete 2x2: the four conditions (Req 5.1), the three paired-McNemar contrasts (Req 5.2), the
# difference-in-differences interaction (Req 5.3), and the SEPARATE bypass-vs-compliance columns
# (Req 5.4), including that ``format_factorial_table`` renders them in separate columns.
#
# The four cells SHARE one prompt-id set {p0,p1,p2,p3} (4 behaviours) so the three McNemar
# contrasts pair cleanly. The two single-turn cells are built with ``_guarded_records_from`` (each
# record carries ``input_flagged`` -> a REAL per-record bypass rate); the two Crescendo cells are
# built with ``_records_from`` (no ``input_flagged`` -> bypass ``None``). One seed in EVERY cell
# carries MULTIPLE draws, so best-of-N (any-over-draws) is genuinely exercised and the per-record
# bypass denominator (5 records) is visibly DIFFERENT from the per-seed ASR denominator (4
# behaviours) -- the concrete reason compliance ASR and guardrail bypass cannot be one number.
#
# Per-seed outcomes AFTER best-of-N collapse (logical OR over that seed's draws):
#     seed | a=single_plain | b=single_enc | c=crescendo_plain | d=crescendo_enc
#     p0   |       F        |      T       |        T          |       T
#     p1   |       F        |      T       |        T          |       T
#     p2   |       F        |      T       |        F          |       T
#     p3   |       T        |      F       |        F          |       F
# Cell ASR = (#succeeding seeds)/4:   a=1/4=0.25  b=3/4=0.75  c=2/4=0.50  d=3/4=0.75
# Main effects / interaction (labels a,b,c,d as above, Req 5.3):
#     unicode_main   = b - a             = 0.75 - 0.25          =  0.50
#     crescendo_main = c - a             = 0.50 - 0.25          =  0.25
#     interaction    = (d - c) - (b - a) = (0.75-0.50)-(0.75-0.25) = 0.25 - 0.50 = -0.25
# McNemar (b = success in A & fail in B; c = fail in A & success in B; paired on sorted p0..p3):
#     unicode   A=single_enc      vs B=single_plain: p0 TF,p1 TF,p2 TF -> b=3 ; p3 FT -> c=1
#     crescendo A=crescendo_plain vs B=single_plain: p0 TF,p1 TF       -> b=2 ; p3 FT -> c=1  (p2 FF)
#     combined  A=crescendo_enc   vs B=single_plain: p0 TF,p1 TF,p2 TF -> b=3 ; p3 FT -> c=1
# Bypass (single cells only; per-record fraction with input_flagged==False over the GUARDED subset):
#     single_plain: records clear,flagged,clear,flagged,flagged -> 2 clear / 5 = 0.40
#     single_enc  : records clear,flagged,flagged,flagged,flagged -> 1 clear / 5 = 0.20
#     crescendo_plain / crescendo_enc: NO input_flagged on any record -> None (undefined, NOT 0.0)
from smuggle.analysis import McNemarResult, format_factorial_table

#: Hand-computed expectations for the fixed 2x2 above (see the banner for the arithmetic).
_U57_EXPECTED_ASR = {"single_plain": 0.25, "single_enc": 0.75, "crescendo_plain": 0.50, "crescendo_enc": 0.75}
_U57_EXPECTED_BYPASS = {"single_plain": 0.40, "single_enc": 0.20, "crescendo_plain": None, "crescendo_enc": None}
_U57_EXPECTED_N = 4


def _synthetic_factorial_cells() -> dict[str, list]:
    """Build the four FIXED synthetic cells described in the banner (no model, no randomness).

    ``single_*`` use ``_guarded_records_from`` (``input_flagged`` -> a real bypass rate);
    ``crescendo_*`` use ``_records_from`` (no ``input_flagged`` -> bypass ``None``). All four share
    the seed set {p0,p1,p2,p3}, and each cell has ONE multi-draw seed so best-of-N is exercised and
    the per-record bypass denominator differs from the per-seed ASR denominator."""
    return {
        # a = single_plain: successes {p3} -> ASR 1/4 = 0.25; 2 clear of 5 guarded records -> bypass 0.40.
        # p3 has two draws [F, T] -> best-of-N True (the seed succeeds although its first draw failed).
        "single_plain": _guarded_records_from([
            ("p0", False, "clear"),
            ("p1", False, "flagged"),
            ("p2", False, "clear"),
            ("p3", False, "flagged"),
            ("p3", True,  "flagged"),
        ]),
        # b = single_enc: successes {p0,p1,p2} -> ASR 3/4 = 0.75; 1 clear of 5 guarded -> bypass 0.20.
        # p0 has two draws [T, F] -> best-of-N True; its two records split clear/flagged, so the
        # per-record bypass denominator (5) visibly exceeds the per-seed ASR denominator (4).
        "single_enc": _guarded_records_from([
            ("p0", True,  "clear"),
            ("p0", False, "flagged"),
            ("p1", True,  "flagged"),
            ("p2", True,  "flagged"),
            ("p3", False, "flagged"),
        ]),
        # c = crescendo_plain: successes {p0,p1} -> ASR 2/4 = 0.50; no input_flagged -> bypass None.
        "crescendo_plain": _records_from({
            "p0": [True, False],   # -> T
            "p1": [False, True],   # -> T
            "p2": [False, False],  # -> F
            "p3": [False],         # -> F
        }),
        # d = crescendo_enc: successes {p0,p1,p2} -> ASR 3/4 = 0.75; no input_flagged -> bypass None.
        "crescendo_enc": _records_from({
            "p0": [True],              # -> T
            "p1": [True, False, False],# -> T
            "p2": [False, True],       # -> T
            "p3": [False, False],      # -> F
        }),
    }


def _row_for(rendered: str, label: str) -> str:
    """Return the single rendered table ROW that STARTS WITH ``label``.

    The cell rows are left-justified with the label at column 0, while the footer lines start with
    'McNemar' / a capitalised word ('Unicode', 'Crescendo', 'Interaction') and the header with
    'cell' -- so a prefix match isolates the cell row without depending on exact whitespace."""
    rows = [ln for ln in rendered.splitlines() if ln.startswith(label)]
    assert len(rows) == 1, f"expected exactly one rendered row starting with {label!r}, got {len(rows)}"
    return rows[0]


def test_factorial_2x2_unit_per_cell_asr_and_n() -> None:
    """**Validates: Requirements 5.1, 5.5**

    (1) Each cell's per-cell ASR (and the separate ``compliance_asr`` copy) equals the
    hand-computed best-of-N value, over the shared 4-behaviour set -- pinning the a,b,c,d the
    effects are built from, for all four factorial conditions (Req 5.1)."""
    result = factorial_2x2(_synthetic_factorial_cells())
    for label in _CELL_LABELS:
        assert result.n[label] == _U57_EXPECTED_N
        assert result.asr[label] == pytest.approx(_U57_EXPECTED_ASR[label])
        assert result.compliance_asr[label] == pytest.approx(_U57_EXPECTED_ASR[label])


def test_factorial_2x2_unit_main_effects_and_interaction() -> None:
    """**Validates: Requirements 5.3**

    (2) The two main effects and the difference-in-differences interaction equal the hand-computed
    values on the four cell ASRs (a=0.25, b=0.75, c=0.50, d=0.75)."""
    result = factorial_2x2(_synthetic_factorial_cells())
    a, b, c, d = 0.25, 0.75, 0.50, 0.75
    assert result.unicode_main_effect == pytest.approx(b - a)       #  0.50
    assert result.crescendo_main_effect == pytest.approx(c - a)     #  0.25
    assert result.interaction == pytest.approx((d - c) - (b - a))   # -0.25
    # Concrete hand values (not only the formula), so a cell swap/sign flip is caught numerically:
    assert result.unicode_main_effect == pytest.approx(0.50)
    assert result.crescendo_main_effect == pytest.approx(0.25)
    assert result.interaction == pytest.approx(-0.25)


def test_factorial_2x2_unit_mcnemar_contrasts() -> None:
    """**Validates: Requirements 5.2**

    (3) The three McNemar contrasts carry the expected condition_a/condition_b labels and the
    hand-computed discordant b/c counts (b = success in A & fail in B; c = fail in A & success in
    B; each contrast is paired against the single-turn plaintext baseline on the shared seeds)."""
    result = factorial_2x2(_synthetic_factorial_cells())

    assert isinstance(result.mcnemar_unicode, McNemarResult)
    assert (result.mcnemar_unicode.condition_a, result.mcnemar_unicode.condition_b) == ("single_enc", "single_plain")
    assert (result.mcnemar_unicode.b, result.mcnemar_unicode.c) == (3, 1)

    assert (result.mcnemar_crescendo.condition_a, result.mcnemar_crescendo.condition_b) == ("crescendo_plain", "single_plain")
    assert (result.mcnemar_crescendo.b, result.mcnemar_crescendo.c) == (2, 1)

    assert (result.mcnemar_combined.condition_a, result.mcnemar_combined.condition_b) == ("crescendo_enc", "single_plain")
    assert (result.mcnemar_combined.b, result.mcnemar_combined.c) == (3, 1)


def test_factorial_2x2_unit_bypass_none_for_crescendo_real_for_single() -> None:
    """**Validates: Requirements 5.4**

    (4) BOTH Crescendo cells carry ``None`` bypass (no comparable single-payload guardrail
    verdict) while BOTH single-turn cells carry a REAL float bypass rate equal to the hand-computed
    per-record guarded-subset fraction."""
    result = factorial_2x2(_synthetic_factorial_cells())

    assert result.bypass_rate["crescendo_plain"] is None
    assert result.bypass_rate["crescendo_enc"] is None

    for label in ("single_plain", "single_enc"):
        assert isinstance(result.bypass_rate[label], float)
        assert result.bypass_rate[label] == pytest.approx(_U57_EXPECTED_BYPASS[label])
    assert result.bypass_rate["single_plain"] == pytest.approx(0.40)
    assert result.bypass_rate["single_enc"] == pytest.approx(0.20)


def test_factorial_2x2_unit_bypass_and_compliance_are_separate_columns() -> None:
    """**Validates: Requirements 5.4**

    (5) Compliance ASR and guardrail bypass live in SEPARATE dict fields (distinct objects;
    ``compliance_asr`` is its own copy of ``asr``, never the bypass dict), and wherever the two
    definitions disagree numerically the reported numbers disagree too -- never collapsed."""
    result = factorial_2x2(_synthetic_factorial_cells())

    # Distinct containers; compliance_asr is its own object (a copy of asr), never the bypass dict.
    assert result.compliance_asr is not result.bypass_rate
    assert result.compliance_asr is not result.asr
    assert result.compliance_asr == result.asr

    # single_plain: ASR 0.25 vs bypass 0.40 -> genuinely different numbers (not collapsed).
    assert result.compliance_asr["single_plain"] == pytest.approx(0.25)
    assert result.bypass_rate["single_plain"] == pytest.approx(0.40)
    assert result.compliance_asr["single_plain"] != pytest.approx(result.bypass_rate["single_plain"])

    # single_enc: ASR 0.75 vs bypass 0.20 -> genuinely different numbers (not collapsed).
    assert result.compliance_asr["single_enc"] == pytest.approx(0.75)
    assert result.bypass_rate["single_enc"] == pytest.approx(0.20)
    assert result.compliance_asr["single_enc"] != pytest.approx(result.bypass_rate["single_enc"])


def test_factorial_2x2_unit_format_table_renders_separate_columns() -> None:
    """**Validates: Requirements 5.4**

    (6) ``format_factorial_table`` renders ASR and bypass as SEPARATE columns: the header names
    both; each cell row shows its hand-computed ASR percent AND its bypass token; and the two
    Crescendo rows render ``n/a`` (undefined) rather than a fabricated bypass number. Assertions
    are on token presence within the located row, so they are robust to exact spacing."""
    result = factorial_2x2(_synthetic_factorial_cells())
    rendered = format_factorial_table(result)

    # Header advertises two distinct columns.
    header = rendered.splitlines()[0]
    assert "ASR" in header
    assert "bypass" in header

    # Single-turn rows: a real ASR% AND a real bypass% (distinct tokens in the same row), not n/a.
    # A collapse (bypass column aliased to ASR) would drop the bypass token and fail these.
    sp = _row_for(rendered, "single_plain")
    assert "25%" in sp and "40%" in sp          # ASR 0.25 | bypass 0.40 -> separate columns
    assert "n/a" not in sp

    se = _row_for(rendered, "single_enc")
    assert "75%" in se and "20%" in se          # ASR 0.75 | bypass 0.20
    assert "n/a" not in se

    # Crescendo rows: a real ASR% but bypass rendered as 'n/a' (undefined), never a fabricated number.
    cp = _row_for(rendered, "crescendo_plain")
    assert "50%" in cp                          # ASR 0.50
    assert "n/a" in cp

    ce = _row_for(rendered, "crescendo_enc")
    assert "75%" in ce                          # ASR 0.75
    assert "n/a" in ce
