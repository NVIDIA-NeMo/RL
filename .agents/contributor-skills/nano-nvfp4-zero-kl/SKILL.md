---
name: nano-nvfp4-zero-kl
description: Checklist to verify a NeMo-RL checkout can run Nemotron Nano v3 with per-token NVFP4 routed experts on FlashInfer and zero train/generation mismatch (commits, submodule pins, packages, recipe knobs, known failures, pass criteria).
when_to_use: Setting up, reproducing or debugging Nano per-token NVFP4 zero-KL; the zero-kl-nvfp4-flashinfer recipe or nightly; moe_inference_training_forward; nonzero gen_kl_error with FlashInfer NVFP4 experts; 'nano nvfp4 zero kl', 'pertoken nvfp4 zero kl', 'flashinfer zero kl'.
---

# Skill: check that a repo can run Nemotron Nano v3 per-token NVFP4 with zero train/generation KL

Generation quantizes the routed experts to NVFP4 weights with per-token NVFP4 activations and
runs them on FlashInfer. Training keeps BF16 weights and BF16 refit. Each training MoE forward
(`moe_inference_training_forward`) runs the same FlashInfer kernel on the same quantization,
rebuilt from the live parameters on every call. The backward comes from a TE recompute of the
MoE layer in BF16 (a straight-through estimator). Attention, the shared experts and the Mamba
mixers stay BF16.

This builds on the Nano MXFP8 zero-KL setup: everything in `nano-mxfp8-zero-kl` sections 3-6
still applies, except where this skill says otherwise. Go through the checklist top to bottom.
Every item is required; if one fails, that is what is missing. Commands run from the NeMo-RL root.

```bash
B=3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
M=$B/3rdparty/Megatron-LM
R=examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n4g-megatron_generation-noncolocated-zero-kl-nvfp4-flashinfer.yaml
T=tests/test_suites/llm/grpo-nanov3-30BA3B-2n4g-megatron_generation-noncolocated-zero-kl-nvfp4-flashinfer.sh
```

## 1. NeMo-RL code

Required: branch `yigongq/nano-minf-training-forward` (NVIDIA-NeMo/RL), which is the Nano
MXFP8 zero-KL branch plus `0b6624437` (recipe, `moe_inference_training_forward` config and
validator), `c84cedcfa` (FlashInfer 0.7.0.post1) and `412216136` (worker-side fixes). If the
commits were cherry-picked, use the marker checks.

| Item | Check (must print `ok`) |
|---|---|
| Nano MXFP8 zero-KL prerequisites | every check in `nano-mxfp8-zero-kl` section 1 |
| `moe_inference_training_forward` plumbing and token bound (`0b6624437`) | `rg -q 'def moe_inference_training_max_tokens_per_rank' nemo_rl/models/megatron/setup.py && ls $R $T >/dev/null && echo ok` |
| Deterministic Triton tuning on Megatron-LM with `megatron.core.tuning` (`412216136`) | `rg -q _enable_mcore_deterministic_tuning nemo_rl/models/megatron/zero_train_gen_mismatch.py && echo ok` |
| Generation workers skip the training-side FlashInfer rule (`412216136`) | `rg -q 'generation_only=is_refit_destination' nemo_rl/models/policy/workers/megatron_policy_worker.py && echo ok` |
| `te_precision_config_file: null` accepted (`412216136`) | `rg -q 'te_precision_config_file: NotRequired\[str \| None\]' nemo_rl/models/policy/__init__.py && echo ok` |

## 2. Submodule pins

The Megatron-Bridge and Megatron-LM commits this needs are not on their `main` branches yet.
NeMo-RL's `.gitmodules` points the Bridge submodule at the fork branch
`YigongQin/Megatron-Bridge:yigongq/nano-pertoken-nvfp4-zero-kl`, and that Bridge commit points its
Megatron-LM submodule at `YigongQin/Megatron-LM:yigongq/nano-pertoken-nvfp4-zero-kl` (`f2f8fab`).
After cloning: `git submodule sync --recursive && git submodule update --init --recursive`.

Megatron-LM (`$M`) must be `f2f8fab` or a descendant. It contains:

| PR | Check (must print `ok`) |
|---|---|
| [NVIDIA/Megatron-LM#7798](https://github.com/NVIDIA/Megatron-LM/pull/7798): FlashInfer train-gen parity (`moe_inference_training_forward`) and NVFP4 refit | `rg -q 'def refresh_flashinfer_nvfp4_weights' $M/megatron/core/transformer/moe/experts.py && rg -q moe_inference_training_forward $M/megatron/core/transformer/transformer_config.py && echo ok` |
| [NVIDIA/Megatron-LM#8262](https://github.com/NVIDIA/Megatron-LM/pull/8262): per-token NVFP4 squared-ReLU MoE via FlashInfer (branch `pertoken-nvfp4`) | `rg -q inference_moe_nvfp4_recipe $M/megatron/core/transformer/transformer_config.py && echo ok` |
| [NVIDIA/Megatron-LM#7562](https://github.com/NVIDIA/Megatron-LM/pull/7562) and [#8195](https://github.com/NVIDIA/Megatron-LM/pull/8195) | the two checks in `nano-mxfp8-zero-kl` section 2 |
| [NVIDIA/Megatron-LM#6076](https://github.com/NVIDIA/Megatron-LM/pull/6076): `megatron.core.tuning` (present on `main`; it is why section 1 needs `_enable_mcore_deterministic_tuning`) | `test -f $M/megatron/core/tuning/policy.py && echo ok` |

Megatron-Bridge (`$B`) is `ec83553` plus one commit with:

| Change | Check (must print `ok`) |
|---|---|
| Megatron-LM pin moved to `f2f8fab` | `[ "$(git -C $M rev-parse --short=7 HEAD)" = f2f8fab ] && echo ok` (or a descendant) |
| FlashInfer 0.7.0.post1, same as NeMo-RL's `mcore` extra | `rg -q '"flashinfer-python==0.7.0.post1"' $B/pyproject.toml && rg -q '"flashinfer-cubin==0.7.0.post1"' $B/pyproject.toml && echo ok` |
| Gemma 4 imports `te_checkpoint` from TE, not `moe_layer` (Bridge upstream `02fb8d3`) | `rg -q '^from megatron.core.extensions.transformer_engine import te_checkpoint' $B/src/megatron/bridge/models/gemma/modeling_gemma4.py && echo ok` |

Without the Gemma 4 fix, every Megatron worker dies at import with `cannot import name
'te_checkpoint' from 'megatron.core.transformer.moe.moe_layer'`: Megatron-LM `main` no longer
exports it there, and importing any Bridge model imports Gemma 4.

## 3. Packages and worker venv

| Item | Check (must print `ok`) |
|---|---|
| NeMo-RL `mcore` extra pins FlashInfer 0.7.0.post1 (per-token NVFP4 needs >= 0.7.0) | `rg -q '"flashinfer-python==0.7.0.post1"' pyproject.toml && echo ok` |
| `uv.lock` agrees with both pins | `uv lock --check` with the repo's uv version (`docker/Dockerfile` `ARG UV_VERSION`); a newer uv rewrites the lock revision |
| Built worker venv has them, plus FA4 | `SP=$(ls -d <venv-dir>/*MegatronPolicyWorker/lib/python3.*/site-packages); ls -d $SP/flashinfer_python-0.7.0.post1.dist-info $SP/flashinfer_cubin-0.7.0.post1.dist-info $SP/flash_attn_4-4.0.0b19.dist-info >/dev/null && echo ok` |

`flashinfer-jit-cache` is deliberately not installed: 0.7.0.post1 is a meta-package over
per-architecture wheels, and Megatron-Core skips the jit-cache check for NVFP4. The FlashInfer
MXFP8 and BF16 MoE paths do check it, so they fail in this venv.

Build the worker venv **once, on one node**, into a shared directory, then reuse it (see
`nano-mxfp8-zero-kl` section 3 for `NEMO_RL_VENV_DIR`). A fresh build started by a 2-node job
races: NeMo-RL's per-node builder treats `bin/python` as ready, and `uv venv` creates it
seconds into a ~15-minute install, so the other node's workers start on a half-installed venv
(`No module named 'ray'`, then `No module named 'megatron.bridge'`). One way, inside the
container on a single node:

```bash
NEMO_RL_VENV_DIR=/lustre/.../venvs/<name> uv run --no-sync python -m nemo_rl.utils.prefetch_venvs MegatronPolicyWorker
```

If the image is older than the checkout, also refresh the driver's system venv on every node
before Ray starts (`uv sync --inexact --frozen`); a stale one fails with errors such as
`cannot import name 'span_attributes' from 'nemo.lens'`.

## 4. Recipe knobs

`$R` inherits `grpo-nanov3-30BA3B-2n4g-megatron_generation-noncolocated-zero-kl-mxfp8-flashinfer.yaml`,
which inherits the Nano MXFP8 zero-KL recipe. Check the **resolved** recipe. Every knob in
`nano-mxfp8-zero-kl` section 4 applies (including `env_vars.MAMBA_DETERMINISTIC: '1'`) except the
precision and grouped-GEMM rows, which are replaced by these:

`policy.megatron_cfg` (training):

| Knob | Required value | |
|---|---|---|
| `transformer_impl` | `inference_optimized` | validated |
| `inference_grouped_gemm_backend` | `flashinfer` | validated |
| `inference_moe_token_dispatcher_type` | `nvls`, same as generation | validated (must match) |
| `moe_inference_training_forward` | `true` | validated (FlashInfer NVFP4 without it is rejected) |
| `activation_checkpointing` / `recompute_granularity` / `recompute_modules` | `true` / `selective` / contains `moe` | validated: the backward comes from the TE recompute |
| `cuda_graph_impl` | not `local` (recipe: `none`) | validated |
| `inference_moe_nvfp4_recipe` | `per_token`, same as generation | validated (must match) |
| `expert_model_parallel_size` | same as generation (`4`) | validated (must match) |
| `fp8_cfg.enabled` | `false` | per-token NVFP4 needs BF16 training weights |
| `te_precision_config_file` | `null` (clears the parent's MXFP8 file) | **not validated** |
| `moe_inference_training_max_tokens_per_rank` | unset (derived from micro-batch × padded sequence length, or packing `mb_tokens`) | sizes the NVLS symmetric heap, which cannot grow |

`policy.generation.mcore_generation_config`:

| Knob | Required value | |
|---|---|---|
| `inference_grouped_gemm_backend` | `flashinfer` | validated |
| `fp8_cfg.enabled` | `false` (this block replaces training's `fp8_cfg` wholesale; the parent sets MXFP8) | **not validated** |

Generation never runs `moe_inference_training_forward`: `merged_inference_megatron_cfg` clears it.
The dedicated generation workers then validate with `generation_only=True`; without the
`412216136` fix they fail with `requires policy.megatron_cfg.moe_inference_training_forward=true`.

## 5. Determinism on current Megatron-LM

With Megatron-LM #6076, `use_deterministic_mode()` no longer reads `MAMBA_DETERMINISTIC`; it
follows `megatron.core.tuning.set_deterministic_mode()` or PyTorch's deterministic flag.
Megatron's training script sets it from `--deterministic-mode`, which NeMo-RL does not use. So
when `MAMBA_DETERMINISTIC=1`, `_enable_mcore_deterministic_tuning` calls
`set_deterministic_mode(True)` and `tuning.install(deterministic=True)` in every worker. That
pins each in-scope Triton autotuner (`mamba_ssm`, `transformer_engine`, `megatron.core`) to one
config per architecture, kernel and shape, without timing.

- Missing: model build fails with `Batch invariant mode on a hybrid model requires
  MAMBA_DETERMINISTIC=1` **even though the variable is set** (the assert message predates #6076).
- `TRITON_CACHE_AUTOTUNING` no longer affects pinned kernels. The validator still requires it.
- Pinned kernels without a tuned-table entry use the cheapest config, so the Mamba kernels can
  be slower than with timed autotuning.

## 6. Confirm with a run

Run `$T` (2 GB200 nodes, 5 steps, sync GRPO). It is listed in `tests/test_suites/disabled.txt`
until the submodule pins land upstream. It passes only if, on every step:
- `train/gen_kl_error` is exactly `0` and `train/token_mult_prob_error` exactly `1` (the script
  asserts these);
- `train/probs_ratio_min` and `train/probs_ratio_max` are exactly `1` (the recipe trains one
  optimizer step per GRPO step: `train_global_batch_size` 32 = 8 prompts × 4 generations);
- `ray-driver.log` contains `[zero_train_gen_mismatch] batch-invariant kernels enabled` from both
  the training and the generation workers.

Reference: GB200, NeMo-RL `412216136`, Megatron-LM `f2f8fab`, Bridge `ec83553` + the section 2
commit, FlashInfer 0.7.0.post1: job 1120793 passed with all of the above on all 5 steps
(W&B run `kxb7fmkk`, about 31 minutes including the model load).
