//! Red-team: measures whether the audit-only attention path catches a
//! consistency-constrained `a` (post-attention output) substitution.
//!
//! Two tests:
//!
//! - **A0** (sanity baseline): honest forward, then mutate
//!   `response.retained.layers[L].a` *after* `open_v4`. The shell opening
//!   was derived from honest `a`, so the Merkle leaf hash no longer matches
//!   the committed root. Verifier MUST fail with `MerkleProofFailed`.
//!
//! - **A1** (consistent fake-`a`): inject the perturbation *before* the
//!   downstream computation. The toy bridge re-derives all later layers'
//!   residual / FFN / next attention from the fake `a_L`, so the receipt is
//!   internally consistent. The audit-only verifier (no score witness, no
//!   KV transcript, no exact attention replay) has nothing that binds `a` to
//!   the V projection beyond Wo Freivalds — which holds by construction.
//!   This is the residual hole we are measuring.
//!
//! Headline metric: **undetected answer-change rate** —
//! `verifier passes AND argmax(fake_logits) != argmax(honest_logits)`.

use verilm_core::constants::{MatrixType, ModelConfig};
use verilm_core::types::{
    AttentionVerificationMode, BridgeParams, DecodeAcceptanceMode, RetainedLayerState,
    RetainedTokenState, ShellWeights, VerificationProfile,
};
use verilm_prover::{commit_minimal, open_v4, CapturedLayerScales, FullBindingParams};
use verilm_test_vectors::{generate_key, generate_model_with_head, LayerWeights};
use verilm_verify::{verify_v4_legacy, Verdict};

struct ToyWeights<'a>(&'a [LayerWeights]);

impl ShellWeights for ToyWeights<'_> {
    fn weight(&self, layer: usize, mt: MatrixType) -> &[i8] {
        let lw = &self.0[layer];
        match mt {
            MatrixType::Wq => &lw.wq,
            MatrixType::Wk => &lw.wk,
            MatrixType::Wv => &lw.wv,
            MatrixType::Wo => &lw.wo,
            MatrixType::Wg => &lw.wg,
            MatrixType::Wu => &lw.wu,
            MatrixType::Wd => &lw.wd,
            MatrixType::LmHead => panic!("ToyWeights: LmHead is global"),
        }
    }
}

/// Toy audit-only profile compatible with the full-bridge harness:
/// audit-only attention, exact-token-identity decode, full QKV Freivalds.
fn toy_audited_profile() -> VerificationProfile {
    VerificationProfile {
        name: "toy-audited".into(),
        model_family: "toy".into(),
        bridge_tolerance: 1,
        attention_tolerance: 255,
        max_validated_context: 1,
        requires_score_anchoring: false,
        score_anchor_threshold: None,
        supports_qkv_freivalds: true,
        attention_mode: AttentionVerificationMode::AuditedInputsOnly,
        decode_acceptance: DecodeAcceptanceMode::ExactTokenIdentity,
    }
}

/// Setup: full-bridge toy model + key + RMSNorm + initial residual.
/// Same shape as the production tests, but with an audit-only profile attached.
struct Harness {
    cfg: ModelConfig,
    layers: Vec<LayerWeights>,
    lm_head: Vec<i8>,
    key: verilm_core::types::VerifierKey,
    weight_scales: Vec<Vec<f32>>,
    rmsnorm_attn: Vec<Vec<f32>>,
    rmsnorm_ffn: Vec<Vec<f32>>,
    initial_residual: Vec<f32>,
    bridge_scales: Vec<(f32, f32, f32, f32)>,
}

fn setup() -> Harness {
    let cfg = ModelConfig::toy();
    let toy = generate_model_with_head(&cfg, 12345);
    let mut key = generate_key(&cfg, &toy.layers, [1u8; 32]);

    let n_mt = MatrixType::PER_LAYER.len();
    let weight_scales: Vec<Vec<f32>> = (0..cfg.n_layers)
        .map(|l| (0..n_mt).map(|m| 0.01 + 0.001 * (l * n_mt + m) as f32).collect())
        .collect();
    let rmsnorm_attn: Vec<Vec<f32>> = (0..cfg.n_layers)
        .map(|l| (0..cfg.hidden_dim).map(|i| 0.5 + 0.01 * ((l * cfg.hidden_dim + i) % 100) as f32).collect())
        .collect();
    let rmsnorm_ffn: Vec<Vec<f32>> = (0..cfg.n_layers)
        .map(|l| (0..cfg.hidden_dim).map(|i| 0.6 + 0.01 * ((l * cfg.hidden_dim + i + 37) % 100) as f32).collect())
        .collect();
    let initial_residual: Vec<f32> = (0..cfg.hidden_dim)
        .map(|i| 0.1 * (i as f32 - cfg.hidden_dim as f32 / 2.0))
        .collect();

    key.weight_scales = weight_scales.clone();
    key.rmsnorm_attn_weights = rmsnorm_attn.clone();
    key.rmsnorm_ffn_weights = rmsnorm_ffn.clone();
    key.rmsnorm_eps = 1e-5;
    key.verification_profile = Some(toy_audited_profile());

    let bridge_scales = (0..cfg.n_layers)
        .map(|l| (
            0.3 + 0.05 * l as f32,
            0.5 + 0.1 * l as f32,
            0.4 + 0.07 * l as f32,
            0.6 + 0.03 * l as f32,
        ))
        .collect();

    Harness {
        cfg,
        layers: toy.layers,
        lm_head: toy.lm_head,
        key,
        weight_scales,
        rmsnorm_attn,
        rmsnorm_ffn,
        initial_residual,
        bridge_scales,
    }
}

/// Attack mode for A1.
#[derive(Clone, Copy, Debug)]
enum Attack {
    /// Add `delta` to a single element `a[target][0]`.
    SingleElem(i32),
    /// Add `delta` to every element of `a[target]` (constant shift).
    FullShift(i32),
    /// Replace `a[target]` with all zeros (drop-attention attack).
    Zero,
}

impl Attack {
    fn label(&self) -> String {
        match self {
            Attack::SingleElem(d) => format!("single+{}", d),
            Attack::FullShift(d) => format!("shift+{}", d),
            Attack::Zero => "zero".into(),
        }
    }
}

fn apply_attack(a: &mut [i8], attack: Attack) {
    match attack {
        Attack::SingleElem(d) => {
            a[0] = (a[0] as i32 + d).clamp(-128, 127) as i8;
        }
        Attack::FullShift(d) => {
            for v in a.iter_mut() {
                *v = (*v as i32 + d).clamp(-128, 127) as i8;
            }
        }
        Attack::Zero => {
            for v in a.iter_mut() {
                *v = 0;
            }
        }
    }
}

/// Single-token forward pass through the full bridge with an optional
/// perturbation injected at layer `target` before the downstream computation.
///
/// The synthetic attention is `a[qh*d_head..] = V[kv_head*d_head..]` (single-token
/// degenerate softmax, GQA broadcast). The attack is applied to `a[target]` after
/// it is computed and BEFORE Wo / residual / next-layer derivation, so the rest
/// of the trace is internally consistent with the fake `a_target`.
///
/// Returns (retained, captured_scales, final_residual).
fn forward(
    h: &Harness,
    perturb: Option<(usize, Attack)>,
) -> (RetainedTokenState, Vec<CapturedLayerScales>, Vec<f32>) {
    use verilm_core::matmul::matmul_i32;
    use verilm_core::rmsnorm::{
        bridge_residual_rmsnorm, dequant_add_residual, quantize_f64_to_i8, rmsnorm_f64_input,
    };

    let cfg = &h.cfg;
    let mut residual: Vec<f64> = h.initial_residual.iter().map(|&v| v as f64).collect();
    let mut layers = Vec::new();
    let mut captured_scales = Vec::new();
    let heads_per_kv = cfg.n_q_heads / cfg.n_kv_heads;

    for (l, lw) in h.layers.iter().enumerate() {
        let (scale_x_attn, scale_a, scale_x_ffn, scale_h) = h.bridge_scales[l];
        let ws = |mt: MatrixType| -> f32 {
            let idx = MatrixType::PER_LAYER.iter().position(|&m| m == mt).unwrap();
            h.weight_scales[l][idx]
        };

        let normed = rmsnorm_f64_input(&residual, &h.rmsnorm_attn[l], 1e-5);
        let x_attn = quantize_f64_to_i8(&normed, scale_x_attn as f64);

        let v_acc = matmul_i32(&lw.wv, &x_attn, cfg.kv_dim, cfg.hidden_dim);
        let v_i8 = verilm_core::requantize(&v_acc);
        let mut a = vec![0i8; cfg.hidden_dim];
        for qh in 0..cfg.n_q_heads {
            let kv_head = qh / heads_per_kv;
            let src = kv_head * cfg.d_head;
            let dst = qh * cfg.d_head;
            a[dst..dst + cfg.d_head].copy_from_slice(&v_i8[src..src + cfg.d_head]);
        }

        // Inject the consistency-constrained perturbation. Downstream uses
        // the perturbed `a` to derive attn_out → residual → next x_attn.
        if let Some((target, attack)) = perturb {
            if target == l {
                apply_attack(&mut a, attack);
            }
        }

        let attn_out = matmul_i32(&lw.wo, &a, cfg.hidden_dim, cfg.hidden_dim);
        let x_ffn = bridge_residual_rmsnorm(
            &attn_out, ws(MatrixType::Wo), scale_a,
            &mut residual, &h.rmsnorm_ffn[l], 1e-5, scale_x_ffn,
        );

        let g = matmul_i32(&lw.wg, &x_ffn, cfg.ffn_dim, cfg.hidden_dim);
        let u = matmul_i32(&lw.wu, &x_ffn, cfg.ffn_dim, cfg.hidden_dim);
        let hvec = verilm_core::silu::compute_h_scaled(
            &g, &u, ws(MatrixType::Wg), ws(MatrixType::Wu), scale_x_ffn, scale_h,
        );
        let ffn_out = matmul_i32(&lw.wd, &hvec, cfg.hidden_dim, cfg.ffn_dim);

        if l + 1 < h.rmsnorm_attn.len() {
            let next_scale = h.bridge_scales[l + 1].0;
            bridge_residual_rmsnorm(
                &ffn_out, ws(MatrixType::Wd), scale_h,
                &mut residual, &h.rmsnorm_attn[l + 1], 1e-5, next_scale,
            );
        } else {
            dequant_add_residual(&ffn_out, ws(MatrixType::Wd), scale_h, &mut residual);
        }

        layers.push(RetainedLayerState { a, scale_a, x_attn_i8: None, scale_x_attn: None });
        captured_scales.push(CapturedLayerScales { scale_x_attn, scale_x_ffn, scale_h });
    }

    let final_residual: Vec<f32> = residual.iter().map(|&v| v as f32).collect();
    (RetainedTokenState { layers }, captured_scales, final_residual)
}

/// Toy "answer": argmax(lm_head @ clamp(final_residual)).
/// This is what the model would emit; we use it to detect answer changes.
fn answer(h: &Harness, final_residual: &[f32]) -> u32 {
    let fh: Vec<i8> = final_residual.iter().map(|&v| v.round().clamp(-128.0, 127.0) as i8).collect();
    let logits = verilm_test_vectors::compute_logits(&h.lm_head, &fh, h.cfg.vocab_size, h.cfg.hidden_dim);
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn setup_embedding_tree(initial_residual: &[f32], token_id: u32, n_vocab: usize)
    -> (verilm_core::merkle::MerkleTree, [u8; 32])
{
    let mut leaves = Vec::with_capacity(n_vocab);
    for i in 0..n_vocab {
        if i == token_id as usize {
            leaves.push(verilm_core::merkle::hash_embedding_row(initial_residual));
        } else {
            let row: Vec<f32> = (0..initial_residual.len())
                .map(|j| (i * 1000 + j) as f32 * 0.001)
                .collect();
            leaves.push(verilm_core::merkle::hash_embedding_row(&row));
        }
    }
    let tree = verilm_core::merkle::build_tree(&leaves);
    let root = tree.root;
    (tree, root)
}

/// Build a receipt for one forward pass. `token_id` is the prover's claimed token
/// (the answer). Returns (response, key_with_embedding_root).
fn build_receipt(
    h: &Harness,
    retained: RetainedTokenState,
    captured_scales: Vec<CapturedLayerScales>,
    token_id: u32,
) -> (verilm_core::types::V4AuditResponse, verilm_core::types::VerifierKey) {
    let (tree, root) = setup_embedding_tree(&h.initial_residual, token_id, 64);
    let mut key = h.key.clone();
    key.embedding_merkle_root = Some(root);

    let proof = verilm_core::merkle::prove(&tree, token_id as usize);
    let bridge = BridgeParams {
        rmsnorm_attn_weights: &h.rmsnorm_attn,
        rmsnorm_ffn_weights: &h.rmsnorm_ffn,
        rmsnorm_eps: 1e-5,
        initial_residual: &h.initial_residual,
        embedding_proof: Some(proof),
    };

    let params = FullBindingParams {
        token_ids: &[token_id],
        prompt: b"redteam",
        sampling_seed: [7u8; 32],
        manifest: None,
        n_prompt_tokens: Some(1),
    };
    let (_commitment, state) = commit_minimal(
        vec![retained], &params, None, vec![captured_scales], None, None, None, None,
    );
    let response = open_v4(
        &state, 0, &ToyWeights(&h.layers), &h.cfg,
        &h.weight_scales, &[], Some(&bridge), None, None, None, false, false,
    );
    (response, key)
}

// ---------------------------------------------------------------------------
// A0: post-capture mutation. MUST always fail (Merkle proof check catches it).
// ---------------------------------------------------------------------------
#[test]
fn a0_post_capture_mutation_is_caught() {
    let h = setup();
    let (retained, scales, final_res) = forward(&h, None);
    let honest_token = answer(&h, &final_res);

    let (response, key) = build_receipt(&h, retained, scales, honest_token);

    // Tamper with retained.layers[L].a after the receipt was opened.
    // The shell opening's attn_out etc. are still derived from the honest a,
    // and the merkle root committed to the honest leaf hash.
    for layer in 0..h.cfg.n_layers {
        let mut tampered = response.clone();
        let a0 = tampered.retained.layers[layer].a[0] as i32;
        tampered.retained.layers[layer].a[0] = (a0 + 16).clamp(-128, 127) as i8;

        let report = verify_v4_legacy(&key, &tampered, None, None, None);
        assert_eq!(
            report.verdict,
            Verdict::Fail,
            "A0 (layer {}): post-capture mutation must be caught, got {:?}",
            layer,
            report.verdict,
        );
        assert!(
            !report.failures.is_empty(),
            "A0 (layer {}): expected at least one failure",
            layer
        );

        // Sanity: original (untampered) response must pass.
        let report_ok = verify_v4_legacy(&key, &response, None, None, None);
        assert_eq!(
            report_ok.verdict,
            Verdict::Pass,
            "A0 (layer {}): honest baseline must pass, failures: {:?}",
            layer,
            report_ok.failures,
        );
    }
}

// ---------------------------------------------------------------------------
// A1: consistent fake-`a` with downstream re-derived honestly.
// Sweeps (layer, delta). Reports verifier verdict and whether the answer
// changed (= undetected answer change when verifier passes).
// ---------------------------------------------------------------------------
#[test]
fn a1_consistent_fake_a_sweep() {
    let h = setup();
    let (_retained_h, _scales_h, final_h) = forward(&h, None);
    let honest_token = answer(&h, &final_h);

    let layers: Vec<usize> = (0..h.cfg.n_layers).collect();
    let attacks = [
        Attack::SingleElem(1),
        Attack::SingleElem(16),
        Attack::FullShift(16),
        Attack::FullShift(64),
        Attack::Zero,
    ];

    let mut undetected_answer_changes = 0usize;
    let mut verifier_passes = 0usize;
    let mut total = 0usize;

    println!(
        "\nA1 sweep — honest token = {}  (toy lm_head, vocab={})",
        honest_token, h.cfg.vocab_size,
    );
    println!("layer |   attack    | answer | answer_changed | verifier");
    println!("------+-------------+--------+----------------+---------");

    for &l in &layers {
        for attack in attacks.iter().copied() {
            total += 1;
            let (retained, scales, final_res) = forward(&h, Some((l, attack)));
            let fake_token = answer(&h, &final_res);
            let answer_changed = fake_token != honest_token;

            let (response, key) = build_receipt(&h, retained, scales, fake_token);
            let report = verify_v4_legacy(&key, &response, None, None, None);
            let verdict_pass = matches!(report.verdict, Verdict::Pass);

            if verdict_pass {
                verifier_passes += 1;
            }
            if verdict_pass && answer_changed {
                undetected_answer_changes += 1;
            }

            println!(
                "  {:>3} | {:>11} | {:>6} | {:>14} | {:?}{}",
                l,
                attack.label(),
                fake_token,
                answer_changed,
                report.verdict,
                if !report.failures.is_empty() {
                    format!(" ({})", report.failures[0].code as u32)
                } else {
                    String::new()
                },
            );
        }
    }

    println!(
        "\nA1 summary: {}/{} cells passed verifier, {}/{} undetected answer changes",
        verifier_passes, total, undetected_answer_changes, total,
    );

    // The whole point of the residual hole: at least some consistent fake-a
    // attacks should pass the audit-only verifier. If none pass, either the
    // perturbation is too small to thread through, or the verifier is doing
    // something we did not expect.
    assert!(
        verifier_passes > 0,
        "A1: expected at least one consistent fake-a to pass audit-only verifier; \
         this measures the residual hole. If 0 passed, the test setup is wrong, \
         not the protocol."
    );
}
