# Mismatch triage

Enter from any stage when `gen_kl_error` misses its gate. The goal is to find which side (AutoModel, vLLM, or NeMo-RL's refit between them) produces the difference, and which module, before changing anything.

## What to expect, and what to chase first

AutoModel and vLLM are not expected to agree exactly, and `gen_kl_error` is not expected to reach zero. They fuse kernels differently and round from FP32 to BF16 at different points; vLLM's own prefill and decode paths can disagree for the same reason. These differences usually stay around the `1e-3` gate and do not destabilize 100 steps of training. Do not try to remove them.

Chase, in this order:

1. Comparison bugs (step 0): the most common cause of a large number.
2. Implementation and refit bugs: a wrong operator, mask, layout or missing tensor. These give a KL far above the gate, usually from the first step.
3. Differences that grow during training (a rising `gen_kl_error`), such as a rollout precision the trainer does not reproduce.
4. Only after those: precision and engine-path differences, and only when the KL is still well above the gate or a long run is unstable.

Step 0 and the refit audit (step 2) are cheap; layer bisection and sensitivity measurement (steps 3 and 4) are expensive and are needed only when the cause is not found by then.

## What the metric measures

`gen_kl_error` is a sampled k3 estimate on the generated tokens: with `d = logp_trainer - logp_rollout` per valid token, the mean of `exp(d) - 1 - d`. It is not a full-vocabulary KL and not the reference-policy KL penalty. `token_mult_prob_error` is the related probability-ratio metric (near 1 is good). Both compare the trainer's recomputed logprobs with the rollout engine's logprobs on the same tokens, under the same weights.

## Step 0: rule out comparison bugs

A large value is often not a model difference. Check first:

- Both sides see the same tokens: same chat template and template arguments (thinking on or off), same special tokens, no off-by-one shift between positions and logprobs.
- Logprob processing matches: the rollout returns logprobs under the same temperature the trainer divides by; `top_p = 1` and no top-k when comparing.
- The metric is produced: it is reported at every step, over a non-zero number of tokens.
- Masks agree: padding and prompt tokens are excluded on both sides.

## Step 1: isolate the side

Save one step's tokens with both logprobs (the per-step training data dumps contain the rollout logprobs and the trainer's previous logprobs), then score the same tokens with independent forwards:

| Forward | How |
|---|---|
| Reference | the official inference code or the HF model, loaded from the checkpoint on disk |
| vLLM from disk | a fresh vLLM engine loading the checkpoint directly, scoring with prompt logprobs |
| AutoModel fresh | a fresh AutoModel load, no optimizer, same parallel layout |

| Observation | Points to |
|---|---|
| vLLM from disk ≈ reference, but rollout logprobs after refit differ from vLLM from disk | NeMo-RL refit (names, layout, dtype, missing tensors) |
| AutoModel fresh ≈ reference, vLLM from disk differs | vLLM model or kernels |
| vLLM from disk ≈ reference, AutoModel fresh differs | AutoModel model definition or its precision policy |
| Every side differs from every other by a similar amount | model sensitivity; go to step 4 before blaming code |
| vLLM prefill differs from vLLM decode on the same tokens by far more than the gate | vLLM decode or cache path (chunked prefill state, cache quantization, prefix caching). A small prefill/decode difference is normal and not a finding |

NeMo-RL ships diagnostics for the vLLM-side checks under `tools/model_diagnostics/` (decode vs prefill, prefix-caching NaN, precision and compilation, routed-expert completeness).

## Step 2: audit refit

When the refit is suspected, export the weights exactly as the policy worker does, through the same parameter generator and HF adapter, and compare every tensor with the checkpoint on disk: missing and extra names, shapes, dtypes, transposes, fused-order swaps (gate and up, Q/K/V), expert ordering. Run first with a small expert-parallel size, then at the production size. Also check what vLLM's loader does with each received tensor (cast, fuse, quantize).

## Step 3: bisect to the first divergence

On fixed tokens and the proxy:

- Compare hidden states layer by layer (relative L2 or RMS) between the two sides; the first layer whose error jumps is the place to read.
- Within that layer, compare per operator: norms, projections, attention, router scores and selected experts, expert outputs, residual mixing.
- Compare per position to find the first divergent token; a divergence at a window or block boundary points to masking or cache logic.

Read the inference source (the vLLM model file and the official inference code), not the paper or the HF modeling file, when deciding what the operator should compute.

## Step 4: measure sensitivity before blaming code

A model with discrete routing and quantized caches can turn a one-ULP input change into a large logprob difference. Before calling a difference a bug, measure:

- Self-perturbation: perturb one element by one BF16 ULP on one side and score again. If that alone produces a difference of the same size, the comparison cannot separate a bug from noise on this input.
- Expert choice: MoE runs already use R3 (the trainer reuses the experts vLLM selected for each token), so different expert choices are not the cause. If R3 is on and the KL is still high, first confirm the replayed experts are actually applied (every MoE layer, every token, no missing-route fallback), then look elsewhere. If a run had R3 off, turn it on before anything else.
- Quantization boundaries: replace cache or activation fake quantization with identity in a diagnostic run to see how much of the difference crosses rounding midpoints.

In one bring-up, a 12-layer slice gave a full-vocabulary KL of 0.56 from a one-ULP perturbation; R3 and removing cache quantization together brought it to 0.0026. The slice amplified noise, so its large KL was not by itself evidence of a refit or cache bug.

## Root-cause classes

Classify every finding; each class has a different owner and fix. The priority column reflects past bring-ups: high means it usually explains a KL far above the gate or a collapse.

| Class | Examples from past bring-ups | Priority | Owner |
|---|---|---|---|
| Implementation | residual mixing matrix applied in the wrong orientation; compressed sparse-attention mask dropped position 0; RoPE on a compressed path ignored YaRN scaling; SwiGLU clamp missing in one path | high | AutoModel or vLLM, whichever side is wrong |
| Integration | integer buffers cast during refit; missing tensors in the refit manifest; names or fused order wrong | high | NeMo-RL |
| Rollout quantization the trainer does not reproduce | rollout rounds to FP8 and training does not; looks fine early, drifts after about 100 steps | high when the rollout quantizes | train-side fake quantization matching the rollout's format and scales |
| Discrete decisions | near-tied router scores select different experts; sparse-attention selection differs | experts: covered by R3, which is on by default for MoE; other selections (sparse attention): medium | experts: R3 (`policy.router_replay`); other selections: alignment of score precision |
| Precision | FP32 parameters or a gate computed in BF16 on one side only | low in general; high for a module that is known to be FP32-sensitive (see below) | AutoModel precision policy, or the refit dtype |
| Engine | decode and prefill disagree slightly; kernel not batch-invariant; different fusion or rounding points | low; expected residual | vLLM, only if the difference is far above the gate |

FP32 to BF16: in most models, a parameter or computation dropped from FP32 to BF16 changes the KL little and does not affect short-term stability. Some modules are exceptions: in past models a few FP32-sensitive modules produced a large KL and a collapse once downcast. Treat a downcast as the cause only when the per-layer or per-operator comparison (step 3) points at that module, or when restoring FP32 for that one module removes most of the gap. The reference implementation keeping a module in FP32 is a hint that it may be sensitive, not proof.

## Rules

- Never change tokens, weights, sequence lengths or precision to make the number pass. A diagnostic replacement (calling the other side's kernel, removing a quantization) identifies a cause; it is not a training configuration unless it is adopted deliberately and documented. R3 is not such a replacement: it is the default training setting for MoE models.
- Stop when `gen_kl_error` is around the gate and flat. The remaining residual from fusion and rounding differences is expected; reducing it further is consistency work, outside this skill.
- Every conclusion records its scope: model, slice or full, commits, image, hardware, precision path and input lengths. A slice's KL does not extrapolate to the full model in either direction.
- Fix it on the owner's per-model branch and record it in the change list with the minimal reproduction from step 3; a human decides whether to file it upstream ([record, do not file](branches-and-upstreaming.md#upstream-problems-record-do-not-file)).
