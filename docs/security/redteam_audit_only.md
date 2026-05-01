# Red-team — audit-only attention residual hole

**Status.** Initial pass (A0 + A1) on the toy bridge harness.
**Date.** 2026-04-25.
**Test.** `crates/verilm-verify/tests/redteam_fake_a.rs`.

The kept product path verifies attention inputs and wiring, but does not
verify arbitrary-position attention outputs. This document measures how
detectable a consistency-constrained `a` substitution is against the
audit-only verifier — i.e. how much of an answer-change attack surface the
residual hole actually exposes.

The framing is: *harder to game and more detectable*, not *verified*.
Hard numbers below.

## Threat model

The adversary keeps everything that is bound:

- Captured GPU logits (sampled-decode binding).
- KV roots / KV provenance.
- Wiring: GQA mapping, RoPE config hash, causal-mask structure.
- Witnessed scores (when retained — last generated token only).
- The full residual chain after the targeted layer.

The adversary is free to pick a fake `a_L` (post-attention output, before
`Wo`) at one or more layers, and then runs the rest of the model honestly
from that fake `a_L`. The downstream Wo/FFN/residual/next-attention values
are re-derived consistently; the receipt is not internally inconsistent.

The audit-only verifier never replays attention to bind `a` to
`softmax(QKᵀ/√d)·V`, so Wo Freivalds — the only check downstream of `a` —
holds by construction. There is no other check that rejects fake `a`.

## A0 — sanity baseline (post-capture mutation)

| | |
|---|---|
| Method | Mutate `response.retained.layers[L].a[0]` *after* `open_v4`. |
| Expected | Verifier rejects (Merkle leaf hash diverges from committed root). |
| Result | Both layers (`L=0`, `L=1`) rejected. ✅ |

This is the "is the verifier even doing anything" check. It must pass.

## A1 — consistent fake-`a` (the real measurement)

The perturbation is injected *before* the downstream computation. The toy
bridge re-derives Wo / residual / next x_attn / next a / lm_head honestly
from the fake `a_L`. The committed token is whatever the fake state
actually produces.

Sweep:

- **Layer L** ∈ {0, 1} — toy model has 2 layers; this covers first/last.
- **Attack mode** — five variants ranging from a single-element nudge to a
  full zero-out of `a_L`.
- **Position** — generated-token only (single-token toy run).

Measured against the audit-only profile (`AuditedInputsOnly` +
`ExactTokenIdentity` decode + full QKV Freivalds).

| layer | attack       | fake answer | answer changed | verifier |
| ----- | ------------ | ----------- | -------------- | -------- |
| 0     | `single+1`   | 32          | false          | Pass     |
| 0     | `single+16`  | 32          | false          | Pass     |
| 0     | `shift+16`   | **58**      | **true**       | Pass     |
| 0     | `shift+64`   | 32          | false          | Pass     |
| 0     | `zero`       | **24**      | **true**       | Pass     |
| 1     | `single+1`   | 32          | false          | Pass     |
| 1     | `single+16`  | 32          | false          | Pass     |
| 1     | `shift+16`   | 32          | false          | Pass     |
| 1     | `shift+64`   | **33**      | **true**       | Pass     |
| 1     | `zero`       | **49**      | **true**       | Pass     |

(Honest answer = token 32 over a 64-token vocab.)

### Headline metric

| | count | rate |
|---|---|---|
| Verifier passed | 10 | **100%** |
| Undetected answer change (passed *and* answer flipped) | 4 | **40%** |

**Reading.** Every consistent fake-`a` substitution clears the audit-only
verifier — including ones that change the emitted token. The verifier
catches *inconsistent* tampering (A0) but cannot catch *consistent* `a`
substitution (A1). This is the residual hole, not a verifier bug.

## What this measurement does *not* claim

- The toy harness uses a synthetic single-token attention
  (`a = expand(V)`), no witnessed scores, no KV transcript, no per-step
  Q. The audited surface in this configuration is bridge Freivalds +
  embedding proof + structural checks. On the production audit-only path
  (Llama / Qwen `*-w8a8-audited`), the audit additionally enforces score
  anchoring on the last generated token, KV provenance over the opened
  range, GQA / RoPE / causal-mask wiring, and captured-logits decode
  binding. None of those bind arbitrary-position `a`, so the residual
  hole is the same — but the adversary's surface is narrower in
  practice.
- This is a pass/fail count over a single seed / single prompt / 2-layer
  toy. It is not a security argument. The claim is bounded by the
  honest framing in the paper: *"audited inputs / wiring; arbitrary-
  position attention outputs not verified."*
- The fake answers are not "good" answers — they are whatever the toy
  lm_head produces from the perturbed final residual. Real attackers
  optimize their fake `a` to land on a chosen target answer. The fact
  that it is even possible to flip the answer with a uniform shift is
  the measurement; landing on a *specific* target with consistent
  forgery is a strictly easier problem with more compute.

## What this measurement *does* establish

- The consistency-constrained adversary has a real residual hole. The
  audit-only verifier does not close it.
- Inconsistent / lazy tampering is reliably caught (A0 baseline). Real
  attackers who skip downstream re-derivation get rejected immediately.
- Any future tightening (per-step Q retention → arbitrary-token score
  audit, or a deterministic attention kernel) would close this hole.
  Both are explicitly deferred (see
  `project_witness_per_step_q_followup.md` and the deterministic-kernel
  roadmap).

## Future passes (not run yet)

- **A2** — sparse multi-layer fake-`a`. Same shape, multiple layers.
- **A3** — last-token-witness evasion. Pick a fake `a` that still passes
  the score-anchor audit at `t = N - 1`. Requires the witness path to be
  active in the harness (production profile, witnessed scores, retained
  Q for the last step).
- **A4** — canary prompts. Track answer-change rate on a curated set
  where the honest answer is high-confidence (top-1 margin large).
- **A5** — local replay smoke (B4 token-0 only). Optional, cheap
  regression check; not a security claim.

The A0 / A1 measurement is sufficient to fix the public claim
("audited but not verified") with hard numbers. A2–A5 sharpen the
picture but do not move the headline.
