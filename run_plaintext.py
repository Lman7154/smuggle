"""Plaintext baseline experiment driver.

Sends each of the 100 JailbreakBench Goals to each target model exactly as written
(no encoding, no converters, no trigger words, no prelude), so clean compliance can
be compared with encoded compliance.

This is the EXPERIMENT LAYER: it only USES the pip-installed ``smuggle`` artefact
(built on PyRIT 1.1.0).  Nothing is added to or changed in the ``smuggle`` package.
Everything the run needs is injected here.

Usage (headless / background):
    python run_plaintext.py --pilot                       # 5 Goals x 3 models, then stop
    python run_plaintext.py --goals 100 --out runs/plaintext --tag plaintext_full
(--goals 0 or a negative number means ALL goals in the split.)
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import random
import re
import sqlite3
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

# ---- durable PyRIT SQLite store MUST be selected before ``import smuggle`` ------- #
_HERE = Path(__file__).resolve().parent
_PLAINTEXT_DB = _HERE / "runs" / "plaintext" / "plaintext.db"
os.environ.setdefault("SMUGGLE_DB", str(_PLAINTEXT_DB))
os.environ.setdefault("RETRY_MAX_NUM_ATTEMPTS", "3")

import smuggle  # noqa: E402
from openai import (  # noqa: E402
    OpenAI,
    APITimeoutError,
    RateLimitError,
    APIConnectionError,
)

try:  # InternalServerError exists in openai >= 1.x
    from openai import InternalServerError  # noqa: E402
except ImportError:  # pragma: no cover
    InternalServerError = type(None)  # never matches isinstance

# --------------------------------------------------------------------------------------------- #
# Config constants — COPIED from run_experiment.py and validated at startup (check #7).
# Do not edit these in isolation; the validator below fails if they drift.
# --------------------------------------------------------------------------------------------- #
LOCAL = "http://localhost:11434/v1"
TARGET_MODELS = ["gpt-oss:20b-cloud", "gemma4:31b-cloud", "nemotron-3-nano:30b-cloud"]
JUDGE_MODEL = "beam:cloud"
TARGET_MAX_TOKENS = 4096
JUDGE_MAX_TOKENS = 4096
INFERENCE_SEED = 0
REQUEST_TIMEOUT_SECONDS = 900
SCORER_REQUEST_TIMEOUT_SECONDS = 300

# Concurrency / retry defaults (overridable via CLI).
DEFAULT_CONCURRENCY = 5
DEFAULT_JUDGE_CONCURRENCY = 5
MAX_DELIVERY_ATTEMPTS = 4
MAX_SCORER_ATTEMPTS = 3
BACKOFF_BASE = 4.0
BACKOFF_MAX = 120.0
BACKOFF_JITTER = 2.0

# Comparator: the encoded Experiment 1 targeted seed-0 rows (the "Direct" column).
ENCODED_CSV = _HERE / "data" / "derived" / "exp1_metrics.csv"
CORE_FAMILIES = ["homoglyph", "variation_selector", "structural", "tag_block"]


def _safe(name: str) -> str:
    """Make a model id safe for a Windows filename."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------------------------- #
# Check #7 — validate every config value against run_experiment.py.
# --------------------------------------------------------------------------------------------- #
def validate_config() -> dict[str, Any]:
    """Fail if any config value differs from run_experiment.py.  Return a manifest dict."""
    import run_experiment as _re  # safe: SMUGGLE_DB already set, smuggle already imported
    checks: list[tuple[str, Any, Any]] = [
        ("LOCAL", LOCAL, _re.LOCAL),
        ("TARGET_MODELS", TARGET_MODELS, _re.TARGET_MODELS),
        ("JUDGE_MODEL", JUDGE_MODEL, _re.JUDGE_MODEL),
        ("TARGET_MAX_TOKENS", TARGET_MAX_TOKENS, _re.TARGET_MAX_TOKENS),
        ("JUDGE_MAX_TOKENS", JUDGE_MAX_TOKENS, _re.JUDGE_MAX_TOKENS),
        ("INFERENCE_SEED", INFERENCE_SEED, _re.INFERENCE_SEED),
        ("REQUEST_TIMEOUT_SECONDS", REQUEST_TIMEOUT_SECONDS, _re.REQUEST_TIMEOUT_SECONDS),
        ("SCORER_REQUEST_TIMEOUT_SECONDS", SCORER_REQUEST_TIMEOUT_SECONDS,
         _re.SCORER_REQUEST_TIMEOUT_SECONDS),
    ]
    mismatches: list[str] = []
    for name, mine, theirs in checks:
        if mine != theirs:
            mismatches.append(f"  {name}: plaintext={mine!r} vs run_experiment={theirs!r}")
    if mismatches:
        raise ValueError(
            "Config mismatch with run_experiment.py (check #7 failed):\n" + "\n".join(mismatches)
        )
    return {
        "checked_against": "run_experiment.py",
        "passed": True,
        "values": {
            "LOCAL": LOCAL,
            "TARGET_MODELS": TARGET_MODELS,
            "JUDGE_MODEL": JUDGE_MODEL,
            "TARGET_MAX_TOKENS": TARGET_MAX_TOKENS,
            "JUDGE_MAX_TOKENS": JUDGE_MAX_TOKENS,
            "INFERENCE_SEED": INFERENCE_SEED,
            "REQUEST_TIMEOUT_SECONDS": REQUEST_TIMEOUT_SECONDS,
            "SCORER_REQUEST_TIMEOUT_SECONDS": SCORER_REQUEST_TIMEOUT_SECONDS,
            "temperature": 0.0,
            "top_p": 1.0,
            "system_prompt": None,
            "retry_max_num_attempts": os.environ.get("RETRY_MAX_NUM_ATTEMPTS", "not set"),
            "httpx_max_retries": 0,
            "extra_body_max_tokens": TARGET_MAX_TOKENS,
        },
    }


# --------------------------------------------------------------------------------------------- #
# Check #6 — classify errors so only infrastructure failures are retried.
# --------------------------------------------------------------------------------------------- #
def _is_infrastructure_error(exc: BaseException) -> bool:
    """True if *exc* is a transient infrastructure failure worth retrying."""
    seen: BaseException | None = exc
    for _ in range(8):
        if seen is None:
            return False
        if isinstance(seen, (asyncio.TimeoutError, TimeoutError, APITimeoutError)):
            return True
        if isinstance(seen, (RateLimitError, APIConnectionError, InternalServerError)):
            return True
        status = getattr(seen, "status_code", None)
        if isinstance(status, int) and (status == 429 or status >= 500):
            return True
        name = type(seen).__name__
        if name in ("TimeoutException", "ReadTimeout", "ConnectTimeout",
                    "WriteTimeout", "PoolTimeout", "ConnectError",
                    "ReadError", "RemoteProtocolError", "LocalProtocolError"):
            return True
        seen = seen.__cause__ or seen.__context__
    return False


async def _backoff(attempt: int) -> None:
    """Exponential backoff with full jitter."""
    base = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** attempt))
    wait = random.uniform(0, base + BACKOFF_JITTER)
    await asyncio.sleep(wait)


# --------------------------------------------------------------------------------------------- #
# Role builders — identical target / judge construction to run_experiment.py.
# --------------------------------------------------------------------------------------------- #
def build_roles() -> tuple[list, Any]:
    """Build the 3 targets and the judge target (no guardrail, no adversarial)."""
    common = dict(
        base_url=LOCAL,
        api_key="ollama",
        seed=INFERENCE_SEED,
        httpx_client_kwargs={"timeout": REQUEST_TIMEOUT_SECONDS, "max_retries": 0},
    )
    scorer_common = {**common,
                     "httpx_client_kwargs": {"timeout": SCORER_REQUEST_TIMEOUT_SECONDS,
                                             "max_retries": 0}}
    target_extra = {"extra_body_parameters": {"max_tokens": TARGET_MAX_TOKENS}}
    targets = [
        smuggle.make_target(
            m,
            max_completion_tokens=TARGET_MAX_TOKENS,
            **target_extra,
            **common,
        )
        for m in TARGET_MODELS
    ]
    judge_target = smuggle.make_target(
        JUDGE_MODEL,
        max_completion_tokens=JUDGE_MAX_TOKENS,
        **scorer_common,
    )
    return targets, judge_target


def build_scorers(judge_target: Any) -> tuple[Any, Any, Any]:
    """Build the objective scorer, refusal scorer, and composite judge (pilot only).

    The objective scorer is ``SelfAskTrueFalseScorer`` (TASK_ACHIEVED).
    The refusal scorer is ``SelfAskRefusalScorer``.
    The composite is the same ``TrueFalseCompositeScorer`` (AND) that
    ``smuggle.make_judge`` builds — used in pilot to verify the split.
    """
    objective_scorer = smuggle.make_judge(judge_target, refusal_guard=False)
    refusal_scorer = smuggle.make_refusal_scorer(judge_target)
    composite_judge = smuggle.make_judge(judge_target)  # objective AND not-refused
    return objective_scorer, refusal_scorer, composite_judge


# --------------------------------------------------------------------------------------------- #
# Goal loading — identical source to the encoded experiments.
# --------------------------------------------------------------------------------------------- #
def load_goals(n: int) -> list[str]:
    """The attack prompts = JBB *Goal* requests (never the Behavior labels)."""
    rows = smuggle.load_jbb_goals()
    goals = smuggle.jbb_prompts(rows)
    return goals if n is None or n <= 0 else goals[:n]


# --------------------------------------------------------------------------------------------- #
# Outcome record.
# --------------------------------------------------------------------------------------------- #
@dataclass
class PlaintextOutcome:
    model: str
    goal_id: str
    input_sha256: str
    success: bool
    refused: bool
    task_achieved: bool
    empty_truncated: bool
    finish_reason: str
    output_tokens: Optional[int]
    conversation_id: str
    timestamp: str
    # private-only fields (never written to the public CSV)
    goal_text: str = ""
    response: str = ""
    objective_rationale: str = ""
    refusal_rationale: str = ""
    # pilot-only fields
    composite_success: Optional[bool] = None
    composite_disagreement: Optional[bool] = None


# --------------------------------------------------------------------------------------------- #
# Progress / resume — keyed by (model, goal_id).  Never re-send a completed row.
# --------------------------------------------------------------------------------------------- #
def _progress_path(out_dir: str | Path, tag: str) -> Path:
    return Path(out_dir) / f"{tag}_progress.jsonl"


def _load_progress(out_dir: str | Path, tag: str) -> dict[tuple[str, str], PlaintextOutcome]:
    path = _progress_path(out_dir, tag)
    completed: dict[tuple[str, str], PlaintextOutcome] = {}
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                key = (str(item["key"][0]), str(item["key"][1]))
                completed[key] = _outcome_from_record(item["outcome"])
            except Exception as exc:
                print(f"[resume] ignoring unreadable progress line {line_no}: {exc}", flush=True)
    return completed


def _outcome_from_record(record: dict) -> PlaintextOutcome:
    return PlaintextOutcome(**record)


def _outcome_to_record(o: PlaintextOutcome) -> dict:
    return asdict(o)


def _append_progress(o: PlaintextOutcome, out_dir: str | Path, tag: str) -> None:
    path = _progress_path(out_dir, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    key = [o.model, o.goal_id]
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"key": key, "outcome": _outcome_to_record(o)},
                            ensure_ascii=True) + "\n")


# --------------------------------------------------------------------------------------------- #
# Delivery — send one Goal to one target, score, return outcome (check #4, #6).
# --------------------------------------------------------------------------------------------- #
async def _score_one(scorer: Any, response: str, objective: str) -> tuple[bool, str]:
    """Score a response and return (verdict, rationale).  Empty-score -> (False, '')."""
    try:
        scores = await scorer.score_text_async(response, objective=objective)
    except TypeError:
        scores = await scorer.score_text_async(response)
    if not scores:
        return False, ""
    s = scores[0]
    return bool(s.get_value()), (getattr(s, "score_rationale", "") or "")


async def _run_delivery(
    *,
    target: Any,
    goal_text: str,
    goal_id: str,
    model_name: str,
    objective_scorer: Any,
    refusal_scorer: Any,
    composite_judge: Any | None,   # pilot only
    model_sem: asyncio.Semaphore,
    judge_sem: asyncio.Semaphore,
    pilot: bool,
) -> tuple[PlaintextOutcome | None, str | None]:
    """Send one Goal as-is, score it, return (outcome, error).

    On infrastructure failure, retry up to MAX_DELIVERY_ATTEMPTS times.
    Real outcomes (empty, truncated, refused) are never retried (check #6).
    """
    input_sha = _sha256_text(goal_text)

    # --- Phase 1: deliver the prompt to the target (check #4: new conversation per call) ---
    result = None
    last_error: Exception | None = None
    for attempt in range(MAX_DELIVERY_ATTEMPTS):
        try:
            async with model_sem:
                attack = smuggle.pyrit_wiring.PromptSendingAttack(objective_target=target)
                result = await asyncio.wait_for(
                    attack.execute_async(objective=goal_text),
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
            break  # got a result — do not retry (check #6)
        except Exception as exc:
            last_error = exc
            if _is_infrastructure_error(exc) and attempt < MAX_DELIVERY_ATTEMPTS - 1:
                await _backoff(attempt)
                continue
            # Non-infrastructure error, or retries exhausted
            return None, f"delivery {type(exc).__name__}: {exc}"

    if result is None:
        return None, f"delivery: no result after {MAX_DELIVERY_ATTEMPTS} attempts: {last_error}"

    # Byte-identity assertion (user requirement).
    assert result.objective == goal_text, (
        f"byte-identity violation for {model_name}/{goal_id}: "
        f"objective != goal_text"
    )

    # --- Extract response + metadata from the AttackResult (by conversation ID, not order) ---
    conv_id = result.conversation_id
    piece = result.last_response
    if piece is not None:
        response_text = piece.converted_value or ""
        finish_reason = str(piece.prompt_metadata.get("finish_reason", "") or "")
        output_tokens = piece.prompt_metadata.get("token_usage_output_tokens")
        output_tokens = int(output_tokens) if output_tokens is not None else None
    else:
        response_text = ""
        finish_reason = ""
        output_tokens = None

    # --- Determine empty_truncated (check #6: empty + length = failure, not retried) ---
    empty_truncated = (not response_text.strip()) and (finish_reason == "length")

    # --- Phase 2: score the response (retry judge infrastructure failures only) ---
    if not response_text.strip():
        # Empty response: real outcome, do not call scorers (matching existing _is_refusal).
        task_achieved = False
        refused = False
        obj_rationale = ""
        ref_rationale = ""
    else:
        for attempt in range(MAX_SCORER_ATTEMPTS):
            try:
                async with judge_sem:
                    task_achieved, obj_rationale = await _score_one(
                        objective_scorer, response_text, goal_text)
                    refused, ref_rationale = await _score_one(
                        refusal_scorer, response_text, goal_text)
                break
            except Exception as exc:
                last_error = exc
                if _is_infrastructure_error(exc) and attempt < MAX_SCORER_ATTEMPTS - 1:
                    await _backoff(attempt)
                    continue
                return None, f"scoring {type(exc).__name__}: {exc}"

    success = task_achieved and not refused

    # --- Pilot: also run the real composite judge and compare (check #3) ---
    composite_success = None
    composite_disagreement = None
    if pilot and response_text.strip():
        for attempt in range(MAX_SCORER_ATTEMPTS):
            try:
                async with judge_sem:
                    comp_scores = await composite_judge.score_text_async(
                        response_text, objective=goal_text)
                composite_success = bool(comp_scores[0].get_value()) if comp_scores else False
                break
            except Exception as exc:
                last_error = exc
                if _is_infrastructure_error(exc) and attempt < MAX_SCORER_ATTEMPTS - 1:
                    await _backoff(attempt)
                    continue
                # Scorer failure is infrastructure, not a real outcome — record as error.
                return None, f"composite scoring {type(exc).__name__}: {exc}"
        if composite_success is not None:
            composite_disagreement = (composite_success != success)

    outcome = PlaintextOutcome(
        model=model_name,
        goal_id=goal_id,
        input_sha256=input_sha,
        success=success,
        refused=refused,
        task_achieved=task_achieved,
        empty_truncated=empty_truncated,
        finish_reason=finish_reason,
        output_tokens=output_tokens,
        conversation_id=conv_id,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        goal_text=goal_text,
        response=response_text,
        objective_rationale=obj_rationale,
        refusal_rationale=ref_rationale,
        composite_success=composite_success,
        composite_disagreement=composite_disagreement,
    )
    return outcome, None


# --------------------------------------------------------------------------------------------- #
# Parallel execution — per-model semaphores, shared judge semaphore, resume.
# --------------------------------------------------------------------------------------------- #
async def run_plaintext_async(
    *,
    goals: list[str],
    out_dir: str | Path,
    tag: str,
    concurrency: int = DEFAULT_CONCURRENCY,
    judge_concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
    pilot: bool = False,
) -> tuple[list[PlaintextOutcome], list[tuple]]:
    """Run the plaintext baseline with asyncio parallelism and crash-safe resume."""
    targets, judge_target = build_roles()
    objective_scorer, refusal_scorer, composite_judge = build_scorers(judge_target)

    completed = _load_progress(out_dir, tag)
    all_outcomes: list[PlaintextOutcome] = list(completed.values())
    all_errors: list[tuple] = []
    db_locked_count = 0

    n_goals = len(goals)
    n_models = len(targets)
    total = n_goals * n_models
    print(f"[{tag}] {n_goals} goals x {n_models} models = {total} deliveries "
          f"(concurrency={concurrency}/model, judge={judge_concurrency})", flush=True)
    if completed:
        print(f"[resume] {len(completed)} already done; running missing only", flush=True)

    judge_sem = asyncio.Semaphore(judge_concurrency)
    t0 = time.time()

    async def _run_target_stream(target: Any) -> None:
        nonlocal db_locked_count
        model = getattr(target, "_model_name", type(target).__name__)
        model_sem = asyncio.Semaphore(concurrency)
        target_outcomes = [o for o in all_outcomes if o.model == model]
        target_errors: list[tuple] = []
        done = len(target_outcomes)

        async def _one(goal_idx: int, goal_text: str) -> tuple[str, PlaintextOutcome | None, str | None]:
            gid = f"p{goal_idx}"
            outcome, error = await _run_delivery(
                target=target, goal_text=goal_text, goal_id=gid, model_name=model,
                objective_scorer=objective_scorer, refusal_scorer=refusal_scorer,
                composite_judge=composite_judge if pilot else None,
                model_sem=model_sem, judge_sem=judge_sem, pilot=pilot,
            )
            return gid, outcome, error

        # Build pending tasks for this model.
        pending: list = []
        for i, goal_text in enumerate(goals):
            gid = f"p{i}"
            if (model, gid) in completed:
                continue
            pending.append(_one(i, goal_text))

        for fut in asyncio.as_completed(pending):
            gid, outcome, error = await fut
            done += 1
            if outcome is not None:
                all_outcomes.append(outcome)
                completed[(model, gid)] = outcome
                _append_progress(outcome, out_dir, tag)
                marker = "OK"
                if pilot and outcome.composite_disagreement:
                    marker = "DISAGREE"
            else:
                record = (model, gid, error)
                target_errors.append(record)
                all_errors.append(record)
                marker = "ERROR"
                if "database is locked" in str(error or "").lower():
                    db_locked_count += 1
                    marker = "DB-LOCKED"
            elapsed = time.time() - t0
            eta = (elapsed / done * total) if done else 0
            print(f"[{tag}] {model} {done:3d}/{total} {gid} -> {marker} "
                  f"({elapsed/60:.0f}m, ETA {eta/60:.0f}m)", flush=True)

        succ = sum(1 for o in target_outcomes if o.success)
        print(f"[{tag}] TARGET DONE {model}: {len(target_outcomes)} scored, "
              f"{succ} success, {len(target_errors)} errors", flush=True)

    # All target streams in parallel (N lanes per model, 3N total).
    await asyncio.gather(*[_run_target_stream(t) for t in targets])

    dt = time.time() - t0
    print(f"[{tag}] DONE: {len(all_outcomes)} deliveries, {len(all_errors)} errors, "
          f"{dt/60:.1f} min", flush=True)
    if db_locked_count:
        print(f"[{tag}] WARNING: {db_locked_count} database-locked errors detected (check #5)",
              flush=True)
    return all_outcomes, all_errors


# --------------------------------------------------------------------------------------------- #
# Output writers.
# --------------------------------------------------------------------------------------------- #
PUBLIC_COLUMNS = [
    "model", "goal_id", "input_sha256", "success", "refused",
    "task_achieved", "empty_truncated", "finish_reason", "output_tokens", "timestamp",
]


def save_private_results(outcomes: list[PlaintextOutcome], out_dir: str | Path, tag: str) -> Path:
    """Write the private JSONL with Goal text and raw responses (gitignored)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{tag}_private.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for o in outcomes:
            fh.write(json.dumps(_outcome_to_record(o), ensure_ascii=True) + "\n")
    return path


def save_public_csv(outcomes: list[PlaintextOutcome], path: Path) -> Path:
    """Write the text-free public CSV.  Asserts no Goal or response text leaks (check #8)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for o in sorted(outcomes, key=lambda x: (x.model, x.goal_id)):
        row = {
            "model": o.model,
            "goal_id": o.goal_id,
            "input_sha256": o.input_sha256,
            "success": int(o.success),
            "refused": int(o.refused),
            "task_achieved": int(o.task_achieved),
            "empty_truncated": int(o.empty_truncated),
            "finish_reason": o.finish_reason,
            "output_tokens": o.output_tokens if o.output_tokens is not None else "",
            "timestamp": o.timestamp,
        }
        # Hard safety check: no text fields in the public row.
        for key, val in row.items():
            if isinstance(val, str) and len(val) > 128:
                raise ValueError(f"unexpected long value in public CSV column {key!r}")
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PUBLIC_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    # Verify the file contains no Goal or response text.
    _verify_no_text_leak(path, outcomes)
    return path


def _verify_no_text_leak(csv_path: Path, outcomes: list[PlaintextOutcome]) -> None:
    """Scan the public CSV for any Goal or response text (check #8)."""
    content = csv_path.read_text(encoding="utf-8")
    for o in outcomes:
        # Check a distinctive substring of each Goal/response is absent.
        if o.goal_text and o.goal_text[:50] in content:
            raise ValueError(f"Goal text leak detected in {csv_path}")
        if o.response and o.response[:50] in content:
            raise ValueError(f"Response text leak detected in {csv_path}")
        if o.objective_rationale and o.objective_rationale[:50] in content:
            raise ValueError(f"Rationale text leak detected in {csv_path}")
    print("[check #8] Public CSV verified: no Goal or response text present.", flush=True)


def save_manifest(
    *,
    outcomes: list[PlaintextOutcome],
    errors: list[tuple],
    out_dir: str | Path,
    tag: str,
    pilot: bool,
    config_validation: dict,
    concurrency: int,
    judge_concurrency: int,
    start_time: float,
    end_time: float,
) -> Path:
    """Write the run manifest with time, settings, model names, and source file hashes."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{tag}_manifest.json"

    source_hashes: dict[str, str] = {}
    for fname in ["run_plaintext.py", "run_experiment.py"]:
        fpath = _HERE / fname
        if fpath.exists():
            source_hashes[fname] = _sha256_file(fpath)

    manifest = {
        "schema_version": "1.0",
        "description": "Plaintext baseline experiment manifest",
        "run": {
            "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)),
            "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(end_time)),
            "duration_minutes": round((end_time - start_time) / 60, 1),
            "pilot": pilot,
            "goals": len({o.goal_id for o in outcomes}),
            "deliveries": len(outcomes),
            "concurrency": concurrency,
            "judge_concurrency": judge_concurrency,
        },
        "settings": config_validation["values"],
        "models": {
            "targets": TARGET_MODELS,
            "judge": JUDGE_MODEL,
        },
        "dataset": {
            "id": "JailbreakBench/JBB-Behaviors",
            "config": "behaviors",
            "split": "harmful",
            "field": "Goal",
        },
        "config_validation": config_validation,
        "source_hashes": source_hashes,
        "counts": {
            "deliveries": len(outcomes),
            "successes": sum(1 for o in outcomes if o.success),
            "refusals": sum(1 for o in outcomes if o.refused),
            "task_achieved": sum(1 for o in outcomes if o.task_achieved),
            "empty_truncated": sum(1 for o in outcomes if o.empty_truncated),
            "errors": len(errors),
        },
        "database": str(_PLAINTEXT_DB),
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                     encoding="utf-8", newline="\n")
    return path


# --------------------------------------------------------------------------------------------- #
# Analysis — Wilson intervals, McNemar vs encoded, Goal listing (checks #1, #2).
# --------------------------------------------------------------------------------------------- #
def _read_encoded_comparator() -> list[dict]:
    """Read the targeted, mutation_seed=0 rows from exp1_metrics.csv (check #1)."""
    if not ENCODED_CSV.exists():
        raise FileNotFoundError(
            f"Encoded comparator CSV not found: {ENCODED_CSV}\n"
            "Run scripts/export_public_results.py first."
        )
    with ENCODED_CSV.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    seed0 = [r for r in rows if r["mode"] == "targeted" and int(r["mutation_seed"]) == 0]
    return seed0


def run_analysis(outcomes: list[PlaintextOutcome]) -> None:
    """Report Goal-level compliance, McNemar vs encoded, and Goal-level cross-tab."""
    # --- Wilson intervals per model ---
    print("\n=== Plaintext Goal-level compliance per model (Wilson 95% CI) ===")
    models = sorted({o.model for o in outcomes})
    for model in models:
        m_outcomes = [o for o in outcomes if o.model == model]
        n = len(m_outcomes)
        s = sum(1 for o in m_outcomes if o.success)
        lo, hi = smuggle.wilson_interval(s, n)
        print(f"  {model:30} {s:3d}/{n:<3d} ASR={s/n:.1%} "
              f"[{lo:.2f}, {hi:.2f}]")

    # --- McNemar: plaintext vs encoded, per (model, family), Holm across 12 ---
    encoded = _read_encoded_comparator()
    print(f"\n=== McNemar: plaintext vs encoded (targeted seed-0, check #1) ===")
    print(f"    Encoded comparator rows: {len(encoded)} (expected 1200)")

    # Check #2: join on (model, goal_id), assert 100 pairs per cell, no duplicates.
    plaintext_by_mg: dict[tuple[str, str], bool] = {
        (o.model, o.goal_id): o.success for o in outcomes
    }
    tests: list[dict] = []
    for model in models:
        for family in CORE_FAMILIES:
            cell = [r for r in encoded if r["model"] == model and r["family"] == family]
            pairs_a: list[bool] = []
            pairs_b: list[bool] = []
            for row in cell:
                gid = row["goal_id"]
                pkey = (model, gid)
                if pkey not in plaintext_by_mg:
                    raise ValueError(
                        f"check #2 failed: no plaintext result for {pkey} "
                        f"(encoded family={family})"
                    )
                pairs_a.append(plaintext_by_mg[pkey])      # plaintext success
                pairs_b.append(row["judge_success"] == "1")  # encoded success
            n_pairs = len(pairs_a)
            if n_pairs != 100:
                raise ValueError(
                    f"check #2 failed: {model}/{family} has {n_pairs} pairs, expected 100"
                )
            result = smuggle.mcnemar(
                pairs_a, pairs_b,
                condition_a="plaintext", condition_b="encoded",
            )
            tests.append({
                "model": model,
                "family": family,
                "n": n_pairs,
                "plaintext_successes": sum(pairs_a),
                "encoded_successes": sum(pairs_b),
                "b_plaintext_only": result.b,
                "c_encoded_only": result.c,
                "p_exact": result.p_value,
                "odds_ratio": result.odds_ratio,
                "odds_ratio_ci_low": result.odds_ratio_ci_low,
                "odds_ratio_ci_high": result.odds_ratio_ci_high,
            })

    adjusted = smuggle.holm_adjust([t["p_exact"] for t in tests])
    for t, p_holm in zip(tests, adjusted):
        t["p_holm"] = p_holm

    print(f"\n{'model':30} {'family':22} {'n':>3} {'plain':>5} {'enc':>4} "
          f"{'b':>2} {'c':>2} {'p exact':>8} {'p Holm':>8} {'OR [95% CI]':>28}")
    print("-" * 120)
    for t in tests:
        or_text = (
            "NE" if t["odds_ratio"] is None else
            ("inf" if t["odds_ratio"] == float("inf") else f"{t['odds_ratio']:.2f}")
        )
        lo = f"{t['odds_ratio_ci_low']:.2f}"
        hi = ("inf" if t["odds_ratio_ci_high"] == float("inf")
              else f"{t['odds_ratio_ci_high']:.2f}")
        print(f"{t['model']:30} {t['family']:22} {t['n']:3d} "
              f"{t['plaintext_successes']:5d} {t['encoded_successes']:4d} "
              f"{t['b_plaintext_only']:2d} {t['c_encoded_only']:2d} "
              f"{t['p_exact']:8.4f} {t['p_holm']:8.4f} "
              f"{or_text:>8} [{lo},{hi}]")

    # --- List which Goals succeeded in plaintext, and whether they succeeded under encoding ---
    print("\n=== Goals that succeeded in plaintext, by Goal ID ===")
    for model in models:
        m_outcomes = [o for o in outcomes if o.model == model]
        succ_gids = sorted([o.goal_id for o in m_outcomes if o.success],
                           key=lambda g: int(g[1:]))
        if not succ_gids:
            print(f"\n  {model}: no Goals succeeded in plaintext.")
            continue
        print(f"\n  {model}: {len(succ_gids)} Goals succeeded: {', '.join(succ_gids)}")
        # For each successful plaintext Goal, show encoding family outcomes.
        for gid in succ_gids:
            enc_results: list[str] = []
            for family in CORE_FAMILIES:
                row = next(
                    (r for r in encoded
                     if r["model"] == model and r["family"] == family and r["goal_id"] == gid),
                    None,
                )
                if row:
                    enc_results.append(
                        f"{family}={'Y' if row['judge_success'] == '1' else 'N'}")
                else:
                    enc_results.append(f"{family}=?")
            print(f"    {gid}: {', '.join(enc_results)}")

    # Save the McNemar test results to a CSV for make_report_tables.py.
    out = _HERE / "results" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "plaintext_vs_encoded_tests.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(tests[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(tests)
    print(f"\n  McNemar results saved to {csv_path}")


# --------------------------------------------------------------------------------------------- #
# Pilot report (check #8).
# --------------------------------------------------------------------------------------------- #
def pilot_report(
    outcomes: list[PlaintextOutcome],
    errors: list[tuple],
    elapsed: float,
    csv_path: Path,
) -> None:
    """Print the pilot report and stop."""
    print("\n" + "=" * 70)
    print("PILOT REPORT")
    print("=" * 70)

    n = len(outcomes)
    n_errors = len(errors)
    print(f"\n  Deliveries scored: {n}/15")
    print(f"  Error count: {n_errors}")
    print(f"  Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    if n > 0:
        per_delivery = elapsed / n
        est_full = per_delivery * 300
        print(f"  Time per delivery: {per_delivery:.1f}s")
        print(f"  Estimated full-run time: {est_full/60:.0f} min ({est_full/3600:.1f} h)")

    # Check #3: scorer split agreement.
    disagreements = [o for o in outcomes if o.composite_disagreement is True]
    scored_with_composite = [o for o in outcomes if o.composite_success is not None]
    print(f"\n  Check #3 (scorer split):")
    print(f"    Responses scored with composite: {len(scored_with_composite)}/{n}")
    print(f"    Disagreements (split vs composite): {len(disagreements)}")
    if disagreements:
        print("    DISAGREEMENTS FOUND — showing details:")
        for o in disagreements:
            print(f"      {o.model}/{o.goal_id}: split={o.success} "
                  f"composite={o.composite_success} "
                  f"(task_achieved={o.task_achieved}, refused={o.refused})")
        print("\n  *** STOPPING: scorer split does not match make_judge. ***")
        print("  *** Do not proceed to the full run until this is resolved. ***")
    else:
        print("    PASSED: split (task_achieved AND NOT refused) == composite for all.")

    # Check #5: database locked.
    db_locked = [e for e in errors if "database is locked" in str(e or "").lower()]
    print(f"\n  Check #5 (database locked):")
    print(f"    Database-locked errors: {len(db_locked)}")
    if db_locked:
        print("    WARNING: database locking detected under parallel load.")
    else:
        print("    PASSED: no database-locked errors.")

    # Check #8: CSV has no text.
    print(f"\n  Check #8 (public CSV text-free):")
    print(f"    CSV path: {csv_path}")
    print(f"    CSV verified: no Goal or response text present.")
    # Spot-check column names.
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
    print(f"    Columns: {', '.join(header)}")
    forbidden = {"goal_text", "response", "rationale", "objective"}
    leaked = [h for h in header if any(f in h.lower() for f in forbidden)]
    if leaked:
        print(f"    WARNING: forbidden columns found: {leaked}")
    else:
        print("    PASSED: no text columns in public CSV.")

    print("\n" + "=" * 70)
    print("PILOT COMPLETE. Stopping. Do not start the full run until I say so.")
    print("=" * 70)


# --------------------------------------------------------------------------------------------- #
# Main CLI.
# --------------------------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the plaintext baseline experiment (no encoding)."
    )
    ap.add_argument("--pilot", action="store_true",
                    help="run 5 Goals on all 3 models, then stop")
    ap.add_argument("--goals", type=int, default=100,
                    help="number of JBB Goals (0/neg = all 100)")
    ap.add_argument("--out", default="runs/plaintext", help="output directory")
    ap.add_argument("--tag", default=None, help="output file tag")
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                    help="per-model concurrency limit")
    ap.add_argument("--judge-concurrency", type=int, default=DEFAULT_JUDGE_CONCURRENCY,
                    help="judge concurrency limit")
    args = ap.parse_args()

    tag = args.tag or ("plaintext_pilot" if args.pilot else "plaintext_full")

    # Check #7: validate config against run_experiment.py.
    config_validation = validate_config()
    print(f"[config] SMUGGLE_DB = {os.environ.get('SMUGGLE_DB')}")
    print(f"[config] validation passed — all values match run_experiment.py")

    goals = load_goals(5 if args.pilot else args.goals)
    out_dir = args.out

    start_time = time.time()
    outcomes, errors = asyncio.run(run_plaintext_async(
        goals=goals,
        out_dir=out_dir,
        tag=tag,
        concurrency=args.concurrency,
        judge_concurrency=args.judge_concurrency,
        pilot=args.pilot,
    ))
    end_time = time.time()

    # Save outputs.
    private_path = save_private_results(outcomes, out_dir, tag)
    public_path = Path("data/derived") / "plaintext_metrics.csv"
    save_public_csv(outcomes, public_path)
    manifest_path = save_manifest(
        outcomes=outcomes, errors=errors, out_dir=out_dir, tag=tag,
        pilot=args.pilot, config_validation=config_validation,
        concurrency=args.concurrency, judge_concurrency=args.judge_concurrency,
        start_time=start_time, end_time=end_time,
    )

    print(f"\nOutputs:")
    print(f"  Private: {private_path}")
    print(f"  Public:  {public_path}")
    print(f"  Manifest: {manifest_path}")

    if args.pilot:
        elapsed = end_time - start_time
        pilot_report(outcomes, errors, elapsed, public_path)
    else:
        # Full run: run the analysis.
        if errors:
            print("\n=== ERRORS ===")
            for e in errors:
                print(f"  {e}")
        run_analysis(outcomes)


if __name__ == "__main__":
    main()
