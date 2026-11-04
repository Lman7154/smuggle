from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_experiment
import smuggle
import smuggle.analysis as analysis


def _variant(*, goal: str, encoded: str, success: bool, flagged: bool = True):
    return SimpleNamespace(
        model="model-a",
        family=("homoglyph",),
        mode="targeted",
        seed_id=goal,
        encoded_text=encoded,
        judge_success=success,
        input_flagged=flagged,
        imperceptibility=1.0,
        used_fallback=False,
    )


def test_delivery_report_never_calculates_or_exposes_wilson_ci(monkeypatch) -> None:
    rows = [
        _variant(goal="p0", encoded="first", success=True, flagged=False),
        _variant(goal="p0", encoded="second", success=False, flagged=True),
    ]

    def fail_wilson(*_args, **_kwargs):
        raise AssertionError("delivery-level reporting must not calculate a Wilson interval")

    monkeypatch.setattr(analysis, "wilson_interval", fail_wilson)
    report = analysis.bypass_and_asr(rows, eligible_seed_ids={"p0"})[0]

    assert (report.successes, report.n, report.asr) == (1, 2, 0.5)
    assert (report.bypasses, report.n_guarded, report.bypass_rate) == (1, 2, 0.5)
    assert (report.joint_successes, report.joint_rate) == (1, 0.5)
    assert not hasattr(report, "asr_ci")
    assert not hasattr(report, "bypass_ci")
    assert not hasattr(report, "joint_ci")

    rendered = analysis.format_report_table([report])
    assert "N_deliv" in rendered
    assert "1/2 (50%)" in rendered
    assert "CI" not in rendered
    assert "[0." not in rendered


def test_goal_level_asr_collapses_variants_before_wilson() -> None:
    rows = [
        _variant(goal="p0", encoded="a", success=True),
        _variant(goal="p0", encoded="b", success=False),
        _variant(goal="p1", encoded="c", success=False),
    ]
    first = analysis.asr_over_prompts(rows)[0]
    expected = analysis.wilson_interval(1, 2)
    assert (first.n_prompts, first.n_variants, first.asr) == (2, 3, 0.5)
    assert (first.ci_low, first.ci_high) == pytest.approx(expected)

    second = analysis.asr_over_prompts(
        rows + [_variant(goal="p0", encoded="d", success=False)]
    )[0]
    assert (second.n_prompts, second.n_variants, second.asr) == (2, 4, 0.5)
    assert (second.ci_low, second.ci_high) == pytest.approx(expected)


def test_conversation_pools_are_descriptive_without_ci(tmp_path, monkeypatch) -> None:
    rows = []
    for model in ("model-a", "model-b"):
        for family in ("homoglyph", "tag_block"):
            for goal, success in (("p0", True), ("p1", False)):
                rows.append(
                    SimpleNamespace(
                        model=model,
                        family=(family,),
                        seed_id=goal,
                        pre_probe_success=success,
                        final_probe_success=False,
                        judge_success=success,
                        turn_depth_at_first_success=2 if success else None,
                        input_flagged=True,
                    )
                )

    monkeypatch.setattr(run_experiment.smuggle, "save_jsonl", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        run_experiment.smuggle,
        "format_conversations",
        lambda _items: "private transcript omitted in test\n",
    )
    monkeypatch.setattr(
        run_experiment.smuggle,
        "wilson_interval",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("repeated-Goal pools must not calculate Wilson intervals")
        ),
    )

    run_experiment.save_everything_conversations(rows, [], tmp_path, "report")
    report = (tmp_path / "report_report.txt").read_text(encoding="utf-8")
    assert "pooled families/repeated Goals (descriptive; no CI)" in report
    assert "pooled models/repeated Goals" in report
    assert "95% CI" not in report
    assert "[0." not in report


def test_exp3a_cell_rates_reject_repeated_goal_rows() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "make_report_tables.py"
    exp3a_cell_rates = runpy.run_path(str(script))["exp3a_cell_rates"]
    row = {
        "model": "model-a",
        "family": "homoglyph",
        "goal_id": "p0",
        "pre_probe_success": "0",
        "final_probe_success": "0",
        "overall_success": "0",
    }
    with pytest.raises(ValueError, match="repeated Goal rows.*p0"):
        exp3a_cell_rates([row, dict(row)])
