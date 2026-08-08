# Bitwise-deterministic true on-policy GRPO on the inference-optimized Megatron engine (EP8/TP1)

Generation logprobs bitwise-identical to the training/scoring forward:
TensorBoard `train/gen_kl_error == 0.000000e+00` over full-lr GRPO
(18/18 steps, prod bed seq4096/GBS512, 5,217 gen tok/s/GPU = 1.02x tax vs
matched non-deterministic; small bed 20/20 at 1,368 tok/s/GPU).

## Megatron-LM dependency
This work requires the companion Megatron-LM branch (deterministic
inference-optimized engine, certified stack):

    https://github.com/utkarsh530/Megatron-LM/tree/det-inference-ep8tp1-certified

(base: upstream `2463dbe89`, the version this stack was certified against).
Upstream Megatron master has since gained `batch_invariant_mode`
(and PR #5700 fixed the InferenceTopKRouter wiring); migration of this stack
onto that framework is planned as a follow-up
(`batch-invariant-inference-ep8tp1` branch).

## What this PR adds (NeMo-RL side)
- fail-fast validation for colocated-vs-dedicated mcore generation config
- non-colocated (dedicated-node) megatron generation support
- deterministic scoring path: fused logprob (`float() -> log_softmax ->
  gather`), per-EP-rank fp64 combine mirror, refit param-gather sync barrier
- batch-invariant activation hooks in the megatron model setup
- certified recipe: `examples/configs/recipes/grpo_math_qwen30ba3b_megatron_det_ep8tp1.yaml`
- determinism CI gate: `examples/gen_kl_harness.py` (run_grpo minus the
  training loop; asserts bitwise-exact gen-vs-score, ~8 min on 1 node)

## Scope
EP=8, TP=1, decode-only CUDA graphs. Router replay and MoE config pinning are
OFF (proven unnecessary: bitwise logits imply identical routes).
