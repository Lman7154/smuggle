"""Property-based rejection test for the injected jailbreak-prepend path.

**Feature: crescendo-unicode-arm, Property 3: Jailbreak prepend without demonstrations is rejected**
**Validates: Requirements 3.2**

Offline only: no model, no network. The rejection path raises before any prompt is
sent, so a recording fake target is enough to prove nothing was dispatched.

Scientific framing (research discipline): Requirement 3.2 governs the
Jailbreak_Prepend_Driver (``run_jailbreak_prepend``/``run_jailbreak_prepend_async``):
running it without any injected demonstration pair SHALL raise ``ValueError`` and send
nothing. The driver validates up front -- before ``ensure_pyrit`` and the target loop --
so the guardrail/judge are never touched and no prompt leaves the process. This property
also exercises the injected prepend primitive (``make_jailbreak_priming_async``) the driver
is built on.

Randomisation strategy (documented choice, per tasks.md Notes = minimum 100 iterations):
Hypothesis is installed under the pinned interpreter, so it is used here (the preferred
engine). A naive ``sampled_from(FAMILIES) x sampled_from([[], None])`` space has only
36 combinations and Hypothesis exhausts it after 36 examples -- below the required 100
(verified this session). To guarantee >= 100 iterations we draw ONE wide integer and
DERIVE both the family (``n % len(FAMILIES)``) and the empty-vs-None choice
(``(n // len(FAMILIES)) % 2``) from it; a wide-integer draw runs the full ``max_examples``
(also verified this session). Two ``@example`` entries pin one ``None`` case and one ``[]``
case so both inputs are always exercised. Had Hypothesis been absent, the fallback would be
a deterministic seeded ``random`` loop of >= 100 iterations (not needed here).
"""
from __future__ import annotations

import asyncio

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

import smuggle
from smuggle.experiments import (
    make_jailbreak_priming_async,
    run_jailbreak_prepend_async,
)

FAMILIES: tuple[str, ...] = tuple(smuggle.FAMILIES)

#: >= 100 iterations (tasks.md Notes). The wide-integer draw below avoids the
#: small-``sampled_from`` early-exhaustion that would otherwise cap this at ~36.
_MAX_EXAMPLES = 200


class RecordingTarget:
    """Fake PyRIT target that records any send attempt instead of contacting a model.

    The jailbreak-prepend driver must reject missing demonstrations BEFORE it ever
    touches a target, so in the rejection path this object is never used and ``sends``
    stays empty. Any attribute lookup that looks like a send (e.g. ``send_prompt_async``,
    ``execute_async``) returns a recorder that appends to ``sends`` and raises, so a
    regression that dispatches before validating would be caught (non-empty ``sends``).
    """

    def __init__(self) -> None:
        self.sends: list[tuple] = []

    def __getattr__(self, name: str):
        # Only reached for attributes not set in __init__ (``sends`` is set there, so no
        # recursion). Dunder probes are not sends.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)

        def _recorder(*args, **kwargs):
            self.sends.append((name, args, kwargs))
            raise AssertionError(f"target send {name!r} was called before validation")

        return _recorder


@settings(
    max_examples=_MAX_EXAMPLES,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(n=st.integers(min_value=0, max_value=2**63 - 1))
@example(n=0)                 # -> demonstrations=None (explicit-phase coverage)
@example(n=len(FAMILIES))     # -> demonstrations=[]   (explicit-phase coverage)
def test_property3_jailbreak_prepend_without_demonstrations_is_rejected(n: int) -> None:
    """**Feature: crescendo-unicode-arm, Property 3: Jailbreak prepend without demonstrations is rejected**

    **Validates: Requirements 3.2**

    For a randomly selected encoding family and empty/``None`` demonstrations, BOTH the
    injected prepend primitive and the jailbreak-prepend driver must raise ``ValueError``
    before any prompt is sent, and the recording target must register zero sends.
    """
    family = FAMILIES[n % len(FAMILIES)]
    demonstrations = None if (n // len(FAMILIES)) % 2 == 0 else []

    # (1) The Jailbreak_Prepend_Driver (Requirement 3.2): reject up front, send nothing.
    target = RecordingTarget()
    with pytest.raises(ValueError, match="demonstration"):
        asyncio.run(
            run_jailbreak_prepend_async(
                targets=target,
                prompts="write something harmful-sounding",
                guardrail=None,
                judge=None,
                demonstrations=demonstrations,
                families=(family,),
            )
        )
    assert target.sends == [], f"a prompt was sent before validation: {target.sends!r}"

    # (2) The injected prepend primitive must likewise reject empty/None demonstrations.
    with pytest.raises(ValueError, match="demonstration"):
        asyncio.run(make_jailbreak_priming_async(family, demonstrations=demonstrations))


# --------------------------------------------------------------------------- #
# Focused unit tests: isolate the Requirement 3.2 driver (which governs 3.2) from the
# injected primitive, so the ground truth is readable alongside the randomised property.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("demonstrations", [[], None])
def test_driver_rejects_without_sending(demonstrations) -> None:
    """Requirement 3.2: the driver raises ValueError and sends nothing for [] and None."""
    target = RecordingTarget()
    with pytest.raises(ValueError, match="demonstration"):
        asyncio.run(
            run_jailbreak_prepend_async(
                targets=target,
                prompts="write something harmful-sounding",
                guardrail=None,
                judge=None,
                demonstrations=demonstrations,
            )
        )
    assert target.sends == []


def test_priming_primitive_rejects_empty_list() -> None:
    """The injected prepend primitive raises ValueError on an empty demonstration list."""
    with pytest.raises(ValueError, match="demonstration"):
        asyncio.run(make_jailbreak_priming_async(FAMILIES[0], demonstrations=[]))
