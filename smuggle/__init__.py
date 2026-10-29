"""Plug-and-play Unicode/ASCII smuggling artefact on pip-installed PyRIT.

One call combines Artefact 1 (the MutationEngine PromptConverter) with PyRIT's
attack + judge + memory. Everything is injected; nothing is hardcoded. PyRIT
memory is initialised automatically on import, so you can build targets and call
``run`` straight away:

    import smuggle
    from pyrit.prompt_target import OpenAIChatTarget

    local = "http://localhost:11434/v1"
    target = smuggle.make_target("llama3.1:8b", base_url=local, api_key="ollama")  # pinned decoding config
    guard  = smuggle.make_guardrail(model="llama-guard3:latest", base_url=local, api_key="ollama")
    judge  = smuggle.make_judge(OpenAIChatTarget(model_name="gemma2:9b", endpoint=local, api_key="ollama"))

    result = smuggle.run(target=target, prompts="...", attack="variation_selector",
                         guardrail=guard, judge=judge)
    print(result.summary())
"""
from .engine import MutationEngine, FAMILIES, ALL_FAMILIES, imperceptibility
from .tokens import token_fragmentation
from .guardrail import GuardrailAdapter
from .triggers import TriggerWordDetector, TriggerResult
from .runner import (
    run, run_async, SmuggleResult, VariantOutcome, make_judge, make_guardrail, make_target,
    make_refusal_scorer, init_pyrit, init_pyrit_sqlite, init_pyrit_sqlite_async,
    run_multiturn_a, run_multiturn_a_async, run_multiturn_b, run_multiturn_b_async, MultiTurnOutcome,
    run_redteam, run_redteam_async, RedTeamOutcome,
)
from . import export
from .export import save_jsonl, load_jsonl
from .analysis import (
    asr, wilson_interval, mcnemar, family_ranking,
    bypass_and_asr, format_report_table,
    asr_over_prompts, per_prompt_success, fallback_rate,
    mcnemar_over_prompts, targeted_vs_random,
    AsrEstimate, McNemarResult, FamilyModeReport, PromptLevelAsr,
)
from . import printing
from .printing import (
    print_result, format_result, print_conversations, format_conversations,
    format_variant, format_conversation,
)
from . import context_templates
from .context_templates import (
    load_context_templates, make_priming_provider,
)
from . import jbb
from .jbb import load_jbb_goals, jbb_prompts, build_priming_categories, JBB_CATEGORY_TO_TEMPLATE


from .experiments import (
    run_experiment_1, run_experiment_1_async, run_experiment_2, run_experiment_2_async,
    run_experiment_3a, run_experiment_3a_async, run_experiment_3b, run_experiment_3b_async,
    run_decode_probe, run_decode_probe_async,
    make_decode_priming, make_decode_priming_async,
    make_jailbreak_priming, make_jailbreak_priming_async,
    run_jailbreak_prepend, run_jailbreak_prepend_async,
    run_adaptive_unicode_search, run_adaptive_unicode_search_async,
    AdaptiveUnicodeAttempt, AdaptiveUnicodeOutcome,
    FAMILY_PAIRS,
)


def _auto_init() -> None:
    """Initialise PyRIT CentralMemory once, automatically, at import time.

    Targets cannot be constructed until PyRIT memory exists, so we set it up here
    so callers never have to. Safe to no-op if already initialised, and falls back
    to a direct in-memory store when an event loop is already running (e.g. Jupyter).
    """
    import asyncio
    import logging
    from . import pyrit_wiring as P

    # Quiet PyRIT's import-time env-loading notice (keep real errors).
    for _name in ("pyrit.setup.environment_loading", "pyrit.setup", "pyrit"):
        logging.getLogger(_name).setLevel(logging.ERROR)

    import os

    try:
        P.CentralMemory.get_memory_instance()
        return  # already initialised
    except Exception:
        pass
    if P.initialize_pyrit_async is None:
        return
    # Durable, file-backed store when the caller injects SMUGGLE_DB (proposal 4.9);
    # otherwise an ephemeral in-memory store for quick throwaway probes.
    db_path = os.environ.get("SMUGGLE_DB")

    def _set_in_memory():
        try:
            from pyrit.memory import CentralMemory, SQLiteMemory
            CentralMemory.set_memory_instance(SQLiteMemory(db_path=db_path or ":memory:"))
        except Exception:
            pass

    # If an event loop is already running (e.g. Jupyter), we cannot call asyncio.run - set an
    # in-memory store directly instead of creating a coroutine we would never await (which emits a
    # RuntimeWarning). For a durable file store under Jupyter, call init_pyrit_sqlite() yourself.
    try:
        asyncio.get_running_loop()
        _set_in_memory()
        return
    except RuntimeError:
        pass  # no running loop -> safe to use asyncio.run below

    try:
        if db_path and P.SQLITE is not None:
            from pathlib import Path
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            asyncio.run(P.initialize_pyrit_async(P.SQLITE, db_path=Path(db_path), silent=True))
        else:
            asyncio.run(P.initialize_pyrit_async(P.IN_MEMORY, silent=True))
    except Exception:
        _set_in_memory()


_auto_init()

__all__ = [
    "run", "run_async", "SmuggleResult", "VariantOutcome",
    "init_pyrit", "init_pyrit_sqlite", "init_pyrit_sqlite_async",
    "make_judge", "make_guardrail", "make_target", "make_refusal_scorer",
    "export", "save_jsonl", "load_jsonl",
    "run_multiturn_a", "run_multiturn_a_async", "run_multiturn_b", "run_multiturn_b_async", "MultiTurnOutcome",
    "MutationEngine", "FAMILIES", "ALL_FAMILIES", "imperceptibility", "token_fragmentation",
    "GuardrailAdapter", "TriggerWordDetector", "TriggerResult",
    "asr", "wilson_interval", "mcnemar", "family_ranking",
    "bypass_and_asr", "format_report_table",
    "asr_over_prompts", "per_prompt_success", "fallback_rate",
    "mcnemar_over_prompts", "targeted_vs_random",
    "AsrEstimate", "McNemarResult", "FamilyModeReport", "PromptLevelAsr",
    "printing", "print_result", "format_result", "print_conversations",
    "format_conversations", "format_variant", "format_conversation",
    "context_templates", "load_context_templates", "make_priming_provider",
    "jbb", "load_jbb_goals", "jbb_prompts", "build_priming_categories", "JBB_CATEGORY_TO_TEMPLATE",
    "run_experiment_1", "run_experiment_1_async", "run_experiment_2", "run_experiment_2_async",
    "run_experiment_3a", "run_experiment_3a_async", "run_experiment_3b", "run_experiment_3b_async",
    "run_decode_probe", "run_decode_probe_async",
    "make_decode_priming", "make_decode_priming_async",
    "make_jailbreak_priming", "make_jailbreak_priming_async",
    "run_jailbreak_prepend", "run_jailbreak_prepend_async",
    "run_adaptive_unicode_search", "run_adaptive_unicode_search_async",
    "AdaptiveUnicodeAttempt", "AdaptiveUnicodeOutcome",
    "FAMILY_PAIRS",
]
