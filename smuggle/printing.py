"""Presentable transcripts, PyRIT-printer style.

Shows, for every variant or conversation: the full exchange with the CONVERTED
(encoded) prompt made visible, the target's reply, and the scorer's verdict WITH
its rationale (pass/fail and why). Single-turn renders one panel per variant;
multi-turn renders the whole conversation turn by turn.

Everything is derived from the records the runner already returns, so it works
offline and needs no extra model calls. ``smuggle.print_result`` /
``smuggle.print_conversations`` are the convenience entry points.
"""
from __future__ import annotations

import unicodedata
from typing import Any, Iterable

_RULE = "\u2500"            # ─
_BAR = "\u2502"            # │
_PASS = "\u2714"           # ✔
_FAIL = "\u2718"           # ✘


def _invisible_count(s: str) -> int:
    n = 0
    for ch in s:
        o = ord(ch)
        if (0x200B <= o <= 0x200F or o == 0xFEFF or 0x202A <= o <= 0x202E
                or 0x2066 <= o <= 0x2069 or 0xFE00 <= o <= 0xFE0F
                or 0xE0100 <= o <= 0xE01EF or 0xE0000 <= o <= 0xE007F
                or unicodedata.combining(ch)):
            n += 1
    return n


def _visible(s: str) -> str:
    """Render text with invisible/encoding code points made explicit so the reader
    can SEE that the prompt was smuggled, instead of them vanishing in a terminal."""
    out: list[str] = []
    for ch in s:
        o = ord(ch)
        if 0xE0000 <= o <= 0xE007F:
            out.append(f"\u2420")              # tag-block char -> visible marker
        elif (0x200B <= o <= 0x200F or o == 0xFEFF or 0x202A <= o <= 0x202E
              or 0x2066 <= o <= 0x2069 or 0xFE00 <= o <= 0xFE0F
              or 0xE0100 <= o <= 0xE01EF or unicodedata.combining(ch)):
            out.append("\u00b7")               # invisible char -> middle dot
        else:
            out.append(ch)
    return "".join(out)


def _wrap(text: str, width: int, indent: str) -> str:
    text = (text or "").replace("\r", "")
    lines: list[str] = []
    for raw in text.split("\n"):
        while len(raw) > width:
            cut = raw.rfind(" ", 0, width)
            cut = cut if cut > 0 else width
            lines.append(indent + raw[:cut])
            raw = raw[cut:].lstrip()
        lines.append(indent + raw)
    return "\n".join(lines)


def _hr(width: int, char: str = _RULE) -> str:
    return char * width


def _header(title: str, width: int) -> str:
    title = f" {title} "
    pad = max(0, width - len(title))
    left = pad // 2
    return _RULE * left + title + _RULE * (pad - left)


def _verdict_line(ok: bool) -> str:
    return (f"{_PASS} SUCCESS — objective achieved and not a refusal" if ok
            else f"{_FAIL} FAILURE — objective not achieved (refused / off-target / decode-only)")


# --------------------------------------------------------------------------- #
# Single-turn
# --------------------------------------------------------------------------- #
def format_variant(o: Any, *, width: int = 100, show_response_chars: int | None = None) -> str:
    fam = "+".join(o.family) if not isinstance(o.family, str) else o.family
    inv = _invisible_count(o.encoded_text)
    lines = [
        _header(f"{o.model}  |  {fam}  |  {o.mode}  |  seed_prompt={o.seed_id}", width),
        f"OBJECTIVE (clean)    : {o.prompt}",
        f"TRIGGER WORDS        : {', '.join(o.trigger_words) if o.trigger_words else '(fallback: all content)'}"
        + ("  [fallback]" if o.used_fallback else ""),
        f"CONVERTED PROMPT     : {_visible(o.encoded_text)}",
        f"                       (invisible/encoded code points: {inv}; "
        f"imperceptibility={o.imperceptibility:.3f})",
    ]
    tf = getattr(o, "token_fragmentation", None)
    if tf:
        lines.append(
            f"TOKEN SHIFT          : clean={tf['clean_tokens']} -> encoded={tf['encoded_tokens']} "
            f"tokens ({tf['inflation']:.1f}x); shared ids {tf['shared_ids']}/{tf['clean_tokens']} "
            f"(retained={tf['retained']:.0%}, tokenizer={tf['tokenizer']})"
        )
    if o.input_flagged is not None:
        verdict = "BYPASSED" if o.bypassed_input_guardrail else "flagged"
        lines.append(f"INPUT GUARDRAIL      : {verdict} (encoded prompt {'not ' if o.bypassed_input_guardrail else ''}caught)")
    lines.append(_hr(width))
    lines.append("TARGET RESPONSE:")
    # ``None`` is the lossless default for supervisor/research transcripts. Pass an
    # integer explicitly only when a compact terminal preview is wanted.
    resp = o.response if show_response_chars is None or len(o.response) <= show_response_chars else o.response[:show_response_chars] + " …[truncated]"
    lines.append(_wrap(resp, width - 2, "  "))
    lines.append(_hr(width))
    lines.append(_verdict_line(o.judge_success))
    if o.judge_rationale:
        lines.append("SCORER RATIONALE:")
        lines.append(_wrap(o.judge_rationale, width - 2, "  "))
    lines.append("")
    return "\n".join(lines)


def format_result(result: Any, *, width: int = 100, show_response_chars: int | None = None) -> str:
    outcomes = list(getattr(result, "outcomes", result))
    body = "\n".join(format_variant(o, width=width, show_response_chars=show_response_chars) for o in outcomes)
    n = len(outcomes)
    s = sum(1 for o in outcomes if o.judge_success)
    summary = _header("SUMMARY", width) + f"\n{s}/{n} variants judged a success (ASR={s / n:.0%})" if n else "no variants"
    return body + "\n" + summary


def print_result(result: Any, *, width: int = 100, show_response_chars: int | None = None) -> None:
    print(format_result(result, width=width, show_response_chars=show_response_chars))


# --------------------------------------------------------------------------- #
# Multi-turn
# --------------------------------------------------------------------------- #
def format_conversation(o: Any, *, width: int = 100, show_response_chars: int | None = None) -> str:
    fam = "+".join(o.family) if not isinstance(o.family, str) else o.family
    lines = [
        _header(f"{o.model}  |  {o.strategy}  |  {fam}  |  seed_prompt={o.seed_id}", width),
        f"OBJECTIVE (clean)    : {o.prompt}",
        f"TRIGGER WORDS        : {', '.join(o.trigger_words) if o.trigger_words else '(fallback: all content)'}",
        f"EXECUTED TURNS       : {o.executed_turns}"
        + (f"   first success at turn {o.turn_depth_at_first_success}"
           if o.turn_depth_at_first_success else "   (no turn succeeded)"),
    ]
    if getattr(o, "probe_turn", None) is not None:
        lines.append(
            f"ADAPTIVE PROBE       : prelude success={o.pre_probe_success}; "
            f"fixed-probe success={o.final_probe_success} (probe at turn {o.probe_turn})")
    if o.input_flagged is not None:
        lines.append(f"INPUT GUARDRAIL      : {'BYPASSED' if not o.input_flagged else 'flagged'} (on the encoded delivery)")
    lines.append(_hr(width))
    rationales = o.per_turn_rationale or tuple("" for _ in o.turns_sent)
    for i, (sent, resp) in enumerate(zip(o.turns_sent, o.responses), start=1):
        ok = o.per_turn_success[i - 1] if i - 1 < len(o.per_turn_success) else False
        tag = f"{_PASS}" if ok else f"{_FAIL}"
        inv = _invisible_count(sent)
        ptf = getattr(o, "per_turn_input_flagged", ())
        extra = ""
        if ptf and i - 1 < len(ptf) and ptf[i - 1] is not None:
            extra = f"; guardrail {'BYPASSED' if not ptf[i - 1] else 'flagged'}"
        probe_tag = "  [FIXED UNICODE PROBE]" if getattr(o, "probe_turn", None) == i else ""
        lines.append(f"TURN {i}  [{tag}]  (encoded code points: {inv}{extra}){probe_tag}")
        lines.append(_wrap("ATTACKER -> " + _visible(sent), width - 2, "  "))
        r = resp if show_response_chars is None or len(resp) <= show_response_chars else resp[:show_response_chars] + " …[truncated]"
        lines.append(_wrap("TARGET   -> " + r, width - 2, "  "))
        if rationales[i - 1] and rationales[i - 1] != "benign priming turn (not scored)":
            lines.append(_wrap("SCORER   -> " + rationales[i - 1], width - 2, "  "))
        lines.append(_hr(width, "\u00b7"))
    lines.append(_verdict_line(o.judge_success))
    lines.append("")
    return "\n".join(lines)


def format_conversations(outcomes: Iterable[Any], *, width: int = 100) -> str:
    items = list(outcomes)
    body = "\n".join(format_conversation(o, width=width) for o in items)
    n = len(items)
    s = sum(1 for o in items if o.judge_success)
    summary = _header("SUMMARY", width) + f"\n{s}/{n} conversations reached the objective (ASR={s / n:.0%})" if n else "none"
    return body + "\n" + summary


def print_conversations(outcomes: Iterable[Any], *, width: int = 100) -> None:
    print(format_conversations(outcomes, width=width))
