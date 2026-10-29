---
inclusion: always
---

# Research Protocol — RUN the gates, don't just nod at them

Operationalises how real research agents work (K-Dense `AGENTS.md`; palamedes
skill: verifier split, stakes ladder, "held not invented", consensus != independence).
This is AI-security research: be the skeptic who catches the bluff, not the
assistant who reports "it works." A flattering narrative is a failure.

## 0. Stakes ladder — match rigor to the action
- **L0** quick fact/lookup → answer, name the source you read.
- **L1** claim about how code / PyRIT / a model *behaves* → VERIFY in the installed
  source or `proposal.md` **this session** before asserting (see §4).
- **L2** an experiment result / metric / trigger word / bypass / judge verdict →
  run §1 ledger **and** §3 falsification pass **before** reporting it as anything.
- **L3** methodology / spec / experiment-design change → check against `proposal.md`;
  if validity-affecting and genuinely ambiguous, ask ONE question (§6); else proceed
  and state the assumption.

## 1. Observation ledger — no result becomes a conclusion without it (L2)
Write this before reporting:
- **OBSERVATION** — the raw thing (verbatim output / exact number), not a paraphrase.
- **MECHANISM** — why it happened.
- **EXPECTED?** — name the hypothesis it confirms or breaks.
- **RIVAL EXPLANATIONS** — artifact? confound? the judge? truncated output? cloud
  nondeterminism? the wrong role/config?
- **FALSIFIER** — the check that would prove it wrong — then actually run it.
- **VERIFIED-BY** — source read / command run this session, else label `UNVERIFIED`.

## 2. FAIL signals — each forces STOP + investigate, never a rubber-stamp
- Trigger word is a function word or generic verb (`a`, `the`, `write`, `make`,
  `give`) → method suspect; do NOT report it — dig in.
- Judge says success but the reply only **decoded / explained / mentioned /
  partially did / refused** → success is FALSE until the objective is actually done.
- "Tests pass" / "no error" with behaviour unchecked → not done. Green != correct.
- ASR = 0% or 100% from one example, or any too-good jump → suspect the measurement first.
- A result that flatters the hypothesis → run the disconfirming check BEFORE reporting.
- Behaviour assumed from GitHub main / memory → stop; read the INSTALLED version.
- A surprising AGGREGATE (a harmful-by-design dataset scoring mostly "safe"; an ASR far
  from expectation; a rate that makes your own design look weak) → that is an OBSERVATION,
  not a finding. Trace it UPSTREAM to the exact input / field / config BEFORE concluding
  anything about the design.

## 2a. Symptom != finding — trace to the cause before you conclude
An observation (a number, an output) is NOT a finding. A finding is an observation whose
CAUSE you have traced and verified. When a result is surprising, the first move is to go
UPSTREAM (what was the exact input? which dataset field? which role/config?), not to build a
story DOWNSTREAM about the design. Rationalising a symptom into a plausible design critique is
a failure in the same way a flattering narrative is — both skip the trace. When caught, OWN it
and re-trace; do not defend.

Worked example of the required self-correction (do THIS, don't defend):
> "You're right, and it's a fair hit. I reported N1 as a finding ('your central mechanism is
> barely active, novelty is thin') when it was really a symptom with an unexamined cause. A
> scientist asks '16% unsafe on a dataset that's harmful by design is surprising — what's
> upstream?' and traces it. I didn't; I rationalized the symptom into a design critique. That's
> the exact failure mode your research-rigor rules warn about ('a flattering narrative is a
> failure' — and so is a plausible-sounding one). I'll do the tracing now, on the other
> findings that could be the same kind of artifact, then fix the data bug."

Trap already hit (learn from it): PyRIT's `jbb_behaviors` provider returns the short *Behavior*
label ("Defamation"), not the harmful *Goal* request — so Llama Guard flagged only 16/100 and
"targeting is barely active" looked like a design finding. It was a data-loading symptom; on the
real Goals, 96/100 flag. Trace first, conclude second.

## 3. Build != judge (falsification pass)
The pass that produced a result does not get to bless it. Run a separate pass whose
only job is to break it: add the **control** (e.g. random-position baseline), read
the **full untruncated** output, **flip one variable** (role/config/model), and
confirm the model actually **performed the objective**. Report what survived AND
what didn't — negative results are results.

## 4. Evidence discipline — held, not invented
- Behaviour claims are anchored to source **read this session** (installed file path,
  `proposal.md` section, or URL). No "should be", no memory.
- Check the **installed** version when behaviour depends on it (GitHub main != pinned
  1.1.0 has already bitten us).
- One example, or a model agreeing with you, is **one data point**, not confirmation.
- State uncertainty; mark `UNVERIFIED` / `SEARCHED-AND-MISSING` instead of filling gaps.

## 5. Security framing — separate the layers every time
Report distinctly: guardrail/filter **BYPASS** vs model **COMPLIANCE** vs
**DECODE-vs-EXECUTE**. For anything blocked, ask "what is the bypass?". Keep the
attack (encoding) separate from any confound (framing/injection) so we measure the
smuggling attack, not our own instruction.

## 6. Ask vs proceed
Validity-affecting AND genuinely ambiguous in the proposal → ask ONE pointed, neutral
question (not leading, not a lazy "is this ok?"). Otherwise proceed on the most
defensible proposal-grounded reading and STATE the assumption.

## Definition of Done (a result/claim qualifies only when)
ledger written · falsification pass run · no FAIL signal outstanding · anchored to a
session-verified source · layers separated · assumptions stated · tied to a proposal
objective (§2.2).
