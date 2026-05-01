# Adversarial Methodology: From Defender Testing to Attacker Simulation

The current red team suite proves the verifier catches **obvious** attacks (single-field mutations). It does not test whether a **smart** adversary with full protocol knowledge can forge receipts that pass verification. This document defines the research agenda to close that gap.

---

## 1. Goal-oriented attack design

Organize attacks by **what the adversary wants**, not by which field to flip.

### 1a. Economic substitution

The dominant real-world attack: serve a cheaper model, pocket the margin.

| Attack | Why it matters |
|--------|---------------|
| Same family, smaller size (Qwen-7B → Qwen-1.5B) | Hardest to detect — same architecture, same tokenizer, different weights |
| Same model, lower quantization (W8A8 → W4A16) | Provider saves VRAM/throughput, receipts may still look plausible |
| Same model, early exit (skip 30% of layers) | Partial computation, partial receipts — does the verifier notice missing layers? |
| LoRA swap (serve base, claim adapter) | Adapter identity is hashed but weights aren't independently verified yet |

### 1b. Replay and caching attacks

Provider precomputes or caches honest responses to avoid real-time computation.

| Attack | Why it matters |
|--------|---------------|
| Replay honest receipt for known prompt | Without freshness binding (#22), cached receipts verify indefinitely |
| Prefix-sharing forgery | Precompute receipt for common prefix, graft onto new suffix |
| Selective honest/dishonest | Honest for audited fraction, dishonest for the rest — directly attacks sampling rate |

### 1c. Selective layer cheating

Cheat on layers unlikely to be audited; be honest on opened layers.

| Attack | Why it matters |
|--------|---------------|
| Cheat on 1/28 layers | What's the detection probability at each audit tier? |
| Cheat on unopened layers only | If the adversary can predict which layers are opened, they cheat on the rest |
| Cheat on attention only (not shell) | Attention corridor has tolerance — can the adversary exploit the gap? |

---

## 2. Adaptive adversary with white-box access

The adversary knows the verification algorithm. Test whether they can construct passing forgeries.

### 2a. Freivalds evasion

The Freivalds check verifies `r·(Wx - y) = 0` for random vector r. Probability of false positive is 1/p per check.

**Research questions:**
- What is the effective field size p in our INT8 implementation? Is it the prime modulus, or is there a smaller effective space?
- Can the adversary find y' ≠ y such that `r·(Wx - y') = 0` with probability > 1/p? (e.g., by exploiting structure in W or alignment in the accumulator)
- If the adversary controls the model weights (serves a different model), they control W — does this change the Freivalds bound?
- What happens with batched Freivalds (multiple checks with the same W)? Does correlation between checks reduce effective soundness?

### 2b. Merkle forgery

**Research questions:**
- Is the hash domain-separated? (length prefix, type tags)
- Can a second-preimage be found through padding or length-extension?
- Are all committed fields included in the leaf? Can the adversary find uncommitted fields that affect verification?

### 2c. Tolerance zone exploitation

The attention corridor allows L∞ ≤ τ (empirically τ=8 for Qwen, τ=9 for Llama). This is the protocol's weakest point.

**Research questions:**
- Can the adversary craft attention outputs within ±τ of honest values that correspond to a **completely different** model's computation?
- What is the maximum divergence in final output achievable while staying within corridor bounds at every layer?
- Amplification: if each layer contributes ±τ error, does the error compound across L layers? What's the worst-case output divergence after 28 layers of ±8 perturbation?
- Can the adversary use the tolerance to perform a "model morphing" attack — gradually shift computation toward a cheaper model while staying within bounds?

### 2d. Stock-mode attention forgery under the current audit-only contract

The current product path does **not** verify arbitrary-position attention outputs.
It does bind and/or audit:

- pre-attention boundary / `x_attn` where captured
- KV transcript / opened `K/V`
- witnessed pre-softmax scores (currently for the last generated token)
- GQA / RoPE / causal-mask wiring
- post-attention boundary `a`
- downstream shell / decode / logits

This means the live attacker problem is no longer "forge arbitrary attention with
no constraints." It is:

> Can the adversary precommit a fake post-attention output `a` that survives
> score anchoring, KV provenance, wiring audits, and exact downstream decode?

That is the stock-mode attention red-team target.

#### Concrete attack families

| Attack | Constraint | Why it matters |
|--------|------------|----------------|
| Single-layer fake-`a` | Keep `Q/K/V`, scores, and wiring honest; replace one layer's `a`; run downstream honestly | Measures the exact remaining hole |
| Sparse layer cheating | Fake `a` on 1, 2, 4, or N layers only | Tests detectability under realistic audit sampling |
| Early-vs-late layer cheating | Cheat only on early or late layers | Quantifies which layers are economically dangerous |
| Last-token-only evasion | Behave honestly on the last generated token (where score anchoring is live), cheat elsewhere | Measures how much the current witness scope limits detectability |
| Local-window evasion | Behave honestly on token-0 / local replay smoke windows, cheat outside them | Measures whether smoke checks catch meaningful attacks |
| Downstream-consistent forgery | Pick fake `a`, then recompute the rest of the model honestly | Strongest realistic attacker under current product assumptions |
| Missing-evidence attack | Suppress scores / wiring metadata / KV evidence and hope the verifier soft-skips | Must become fail-closed once rollout is complete |

#### Metrics to report

Do not report only pass/fail. Report:

- token-flip rate under fake-`a` attacks
- detection rate under the current live audit policy
- smallest `||delta a||` that flips the accepted token
- earliest / latest cheating layer that changes the answer
- which audit catches the attack first:
  - score anchoring
  - KV provenance
  - wiring
  - local replay smoke
  - downstream decode / logits
- attack success conditioned on:
  - number of audited generated tokens
  - number of audited layers per token
  - whether score witness is present
  - whether local replay smoke is enabled

#### Immediate test campaign

Phase this as an actual test plan, not an open-ended research note:

1. **Offline fake-`a` harness**
   - take an honest committed trace
   - patch one `(token, layer)` post-attention `a`
   - run the rest of the model honestly from the patched boundary
   - verify whether the receipt still passes and whether the final token changes

2. **Sparse cheating sweep**
   - cheat on `1, 2, 4, 8` layers
   - separate early-only and late-only attacks
   - record detection probability under current sampled audits

3. **Witness-scope evasion**
   - force honest behavior on the last generated token
   - cheat on earlier generated tokens
   - quantify how much the current last-token score witness limits detection

4. **Local replay evasion**
   - cheat inside the token-0 / local smoke window
   - cheat outside it
   - measure incremental value of B4 if enabled

5. **Canary prompts**
   - long-range retrieval
   - masking-sensitive prompts
   - position-sensitive prompts
   - cache-indexing-sensitive prompts

#### Product interpretation rule

If these tests show that small sparse fake-`a` attacks frequently flip outputs
without being caught, the product claim must stay narrow and the roadmap should
prioritize stronger anti-cheat bindings (for example score-root commitment or
broader audit sampling) before broadening any trust language.

If they show that useful attacks require large precommitted perturbations and are
caught reliably under practical audit rates, then the right claim is still not
"verified attention" but rather:

> attention cheating is materially constrained and operationally detectable under
> the shipped audit policy.

---

## 3. Composition attacks

Current tests mutate one field at a time. Real attacks combine subtle changes.

| Attack | Methodology |
|--------|-------------|
| Multi-layer perturbation | Perturb shell openings on N layers (each within noise floor). Does the combination mask a substitution? |
| Weight swap + Merkle recomputation | Swap model weights, recompute valid Merkle trees for new weights. Is model identity the only binding, or do other checks catch it? |
| Scale drift | Change quantization scale slightly on every layer. Each change is tiny; accumulated effect is a different model. |
| Attention + shell coordination | Craft attention outputs within corridor AND perturb shell openings within Freivalds false-positive probability. Joint probability? |

---

## 4. Probabilistic security analysis

Replace "does this attack get caught?" with "what fraction of cheating strategies get caught at what audit rate?"

### Methodology

1. Define a cheating strategy as: (set of layers cheated, perturbation magnitude per layer, fields modified)
2. Sample 10,000 random cheating strategies with varying aggressiveness
3. For each strategy, simulate audit at sampling rates: 1%, 5%, 10%, 25%, 50%, 100%
4. Record: detected/undetected, which check caught it, perturbation magnitude
5. Plot: detection probability vs. cheating fraction vs. audit rate

### Deliverables

- **Security curve**: P(detect) as a function of (cheating fraction, audit rate)
- **Break-even analysis**: at what cheating fraction does the expected gain exceed the expected penalty?
- **Minimum audit rate**: for a given target detection probability (e.g., 99%), what audit rate is required?
- Feed results into #69 (cheating-incentive analysis) and #3 (formal security argument)

---

## 5. Verification path coverage

### 5a. Code coverage under adversarial inputs

- Instrument the Rust verifier with coverage tracking
- Run the full red team suite, measure which verification branches are exercised
- Any unexercised error path is a potential blind spot — craft an adversarial input that reaches it

### 5b. Boundary value analysis

| Boundary | Test |
|----------|------|
| Corridor threshold | L∞ = τ-1, τ, τ+1 — is the bound off-by-one? |
| Token count | 0 tokens, 1 token, max_tokens, max_tokens+1 |
| Layer count | Fewer layers than expected, more layers, duplicate layer indices |
| Merkle tree depth | Minimum depth, maximum depth, mismatched depth |
| Field sizes | INT8 overflow, scale = 0, scale = max, NaN-equivalent in INT8 |

### 5c. Degenerate inputs

- Empty prompt, empty response
- Prompt longer than context window
- All-padding tokens
- Single repeated token for entire response

---

## 6. Protocol-level attacks

### 6a. Concurrent request confusion

Two requests in flight simultaneously. Can retained state from request A contaminate request B's receipt? Does the capture layer properly isolate per-request state?

### 6b. Commitment ordering races

- Commit arrives before generation finishes
- Audit requested during generation
- Commit for request N arrives after commit for request N+1
- Protocol state machine race conditions

### 6c. Hot-swap attack

Provider loads correct weights at startup (passes identity check). During serving, hot-swaps to cheaper weights. Do capture hooks detect the weight change? Is model identity re-verified after startup?

### 6d. Gradient-based forgery

Given white-box access to the verifier, use optimization (gradient descent or search) to find the minimal perturbation to a dishonest receipt that makes it pass all checks. If this is computationally feasible, the protocol has a fundamental hole.

---

## 7. Implementation plan

### Phase 1: Foundation (blocks publication claims)

- [ ] Smaller-model substitution campaign (Qwen-7B → Qwen-1.5B)
- [ ] Selective layer cheating with detection probability measurement
- [ ] Corridor boundary testing (exact threshold, amplification across layers)
- [ ] Verifier code coverage under adversarial inputs
- [ ] Property-based receipt mutation generator (thousands of random mutations)

### Phase 2: Adaptive adversary (blocks strong security claims)

- [ ] Freivalds evasion analysis (effective field size, structured false positives)
- [ ] Tolerance zone exploitation (maximum output divergence within corridor bounds)
- [ ] Composition attacks (multi-field, multi-layer coordinated perturbations)
- [ ] Probabilistic security curve (10K strategies × 6 audit rates)
- [ ] Stock-mode attention fake-`a` campaign (single-layer, sparse-layer, early/late)
- [ ] Last-token-witness evasion campaign
- [ ] Local replay smoke incremental-value measurement

### Phase 3: Protocol-level (blocks production deployment)

- [ ] Concurrent request isolation testing
- [ ] Hot-swap detection testing
- [ ] Commitment ordering race conditions
- [ ] Gradient-based forgery feasibility study
- [ ] Fail-closed rollout for missing audited attention evidence

---

## References

- Roadmap #2: Adversarial testing (in progress)
- Roadmap #3: Formal security argument
- Roadmap #16: Fuzz binary parsers
- Roadmap #69: Cheating-incentive analysis
- Roadmap #15: Adversarial methodology research (this document)
- `redteam/attack_matrix.md`: current attack coverage inventory
