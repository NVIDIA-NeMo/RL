---
name: nano-mxfp8-zero-kl
description: Checklist to verify a NeMo-RL checkout can run Nemotron Nano v3 MXFP8 with zero train/generation mismatch (commits, Megatron-LM PRs, recipe knobs, precision file, pass criteria).
when_to_use: Setting up or debugging Nano MXFP8 zero-KL; nonzero gen_kl_error or probs_ratio != 1 with zero_train_gen_mismatch; reviewing the nano zero-kl-mxfp8 recipe; 'nano mxfp8 zero kl', 'zero train gen mismatch nano'.
---

# Skill: check that a repo can run Nemotron Nano v3 MXFP8 with zero train/generation KL

Go through the checklist top to bottom. Every item is required; if one fails, that is what
is missing. Commands run from the NeMo-RL root.

```bash
M=3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM
R=examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n4g-megatron_generation-noncolocated-zero-kl-mxfp8.yaml
P=examples/te_precision/nanov3_routed_experts_mxfp8.yaml
```

## 1. NeMo-RL code

Required: NeMo-RL `main` with [#3208](https://github.com/NVIDIA-NeMo/RL/pull/3208), plus the
four commits of branch `yigongq/nano-mxfp8-main`, whose latest commit is `aabe7131a`
("add nano mxfp8 example"). If the commits were cherry-picked (new hashes), use the marker
checks.

| Item | Check (must print `ok`) |
|---|---|
| #3208 | `[ -n "$(git log --grep='#3208' -1 --format=%h)" ] && echo ok` |
| Branch head in tree | `git merge-base --is-ancestor aabe7131a HEAD && echo ok` (skip if cherry-picked) |
| MXFP8 allowed under zero-KL (`1ea8f59ce`) | `rg -q "supports FP8 only with fp8_recipe='mxfp8'" nemo_rl/models/megatron/zero_train_gen_mismatch.py && rg -q override_nonquantized_autocast nemo_rl/models/megatron/setup.py && echo ok` |
| Hybrid Mamba allowed under zero-KL (`f54da4d18`) | `rg -qU '"multi_latent_attention",\n\s*"experimental_attention_variant",' nemo_rl/models/megatron/zero_train_gen_mismatch.py && echo ok` |
| Nano recipe and hybrid env check (`aabe7131a`) | `rg -q _HYBRID_MAMBA_ENV_VARS nemo_rl/models/megatron/zero_train_gen_mismatch.py && ls $R $P >/dev/null && echo ok` |

## 2. Megatron-LM PRs in the tree

Both PRs must be in the Megatron-LM that NeMo-RL builds against (`$M`). If workers import
Megatron from somewhere else (a prebuilt venv or a code snapshot), run the same checks on that
copy: its path appears in `ray-driver.log` warnings as `.../Megatron-LM/megatron/core/...`.

| PR | Check (must print `ok`) |
|---|---|
| [NVIDIA/Megatron-LM#7562](https://github.com/NVIDIA/Megatron-LM/pull/7562) (in `main` since `1abb04855`) | `rg -q 'requires MAMBA_DETERMINISTIC=1' $M/megatron/core/transformer/transformer_config.py && echo ok` |
| [NVIDIA/Megatron-LM#8195](https://github.com/NVIDIA/Megatron-LM/pull/8195) | `rg -q 'acc = tl.zeros\(\[BLOCK_H\], dtype=tl.float64\)' $M/megatron/core/inference/moe/batch_invariant.py && echo ok` |

To add a missing PR, inside `$M`: `git cherry-pick -s -S 1abb04855` for #7562, and
`curl -sL https://github.com/NVIDIA/Megatron-LM/pull/8195.diff | git apply --index` for #8195.
Then bump the submodule pins.

## 3. Worker venv with flash-attn-4

If a worker fails with `flash-attn-4 is not installed (mcore extra)`, the image's prebuilt
venv is older than this repo's FA4 pin. Rebuild the worker venvs once into a shared
directory, then reuse them:

```bash
# first run: build
NEMO_RL_VENV_DIR=/lustre/.../venvs/<name> NRL_FORCE_REBUILD_VENVS=true uv run examples/run_grpo.py ...
# every later run: reuse
NEMO_RL_VENV_DIR=/lustre/.../venvs/<name> uv run examples/run_grpo.py ...
```

Set both on the command that runs inside the container: the image sets its own
`NEMO_RL_VENV_DIR`, which beats values exported on the Slurm side. Run later jobs from the same code path as the build: the venv's editable installs point at
it. The check must print `ok`:

```bash
ls -d /lustre/.../venvs/<name>/*/lib/python3.*/site-packages/flash_attn_4-4.0.0b19.dist-info >/dev/null && echo ok
```

## 4. Recipe knobs

Check these values in the **resolved** recipe `$R`, after `defaults:` inheritance. The
validator does not catch the items marked **not validated**: if one of them is wrong, the
run trains normally but the KL is nonzero.

`policy.megatron_cfg` (training):

| Knob | Required value | |
|---|---|---|
| `zero_train_gen_mismatch` | `true` | validated |
| `batch_invariant_mode` | `true` | validated |
| `batch_invariant_backend` / `batch_invariant_collective` | `te_native` / `ordered` | validated |
| `attention_backend` / `flash_attention_version` | `flash` / `4` | validated |
| `moe_permute_fusion` | `false` | validated |
| `tensor_model_parallel_size` / `context_parallel_size` | `1` / `1` | validated |
| `env_vars.CUBLASLT_WORKSPACE_SIZE` / `env_vars.CUBLAS_WORKSPACE_CONFIG` | `'0'` / `':0:0'` | validated |
| `env_vars.MAMBA_DETERMINISTIC` | `'1'` | validated. Megatron-LM with [#6076](https://github.com/NVIDIA/Megatron-LM/pull/6076) (`megatron/core/tuning/`) ignores the variable; NeMo-RL then turns on deterministic mode itself (`_enable_mcore_deterministic_tuning`). Without that, model build fails with `requires MAMBA_DETERMINISTIC=1` even when it is set |
| `env_vars.TRITON_CACHE_AUTOTUNING` | `'1'` | validated by `aabe7131a`; without it KL is about 0.003 |
| `fp8_cfg.fp8_recipe` (with `enabled: true`) | `mxfp8` | validated |
| `fp8_cfg.fp8_param` | `true` | validated indirectly: rejected if `false` while the precision file sets `fp8_param: true` |
| `fp8_cfg.fp8` | `hybrid` | **not validated** |
| `use_fused_linear_logprobs` | unset or `false` | validated |
| `te_precision_config_file` | `examples/te_precision/nanov3_routed_experts_mxfp8.yaml` | **not validated** |
| `use_fused_weighted_squared_relu` | `true` | **not validated** |
| `moe_router_dtype` | `fp32` | **not validated** |
| `moe_token_dispatcher_type` | `alltoall` | **not validated** |
| `expert_model_parallel_size` | same as generation (verified: `4`) | **not validated** |

`policy.generation` and `policy.generation.mcore_generation_config` (generation):

| Knob | Required value | |
|---|---|---|
| `backend` | `megatron` | validated |
| `colocated.enabled` | `false` | validated |
| `transformer_impl` | `inference_optimized` | validated |
| `logprobs_mode` | `raw_logprobs` | validated |
| `temperature` / `top_p` / `top_k` | `1.0` / `1.0` / `null` | validated |
| `enable_chunked_prefill` | `true`, `max_tokens: 256`, `max_requests: 32` | hybrid needs `max_tokens` > Mamba chunk size (128); MCore needs `max_tokens >= max_requests` |
| `tensor_model_parallel_size` | same as training (`1`) | validated |
| `fp8_cfg` | **identical to the training block**, including `fp8: hybrid` | only `enabled` and `fp8_recipe` must match training (validated); `fp8` and `fp8_param` are **not validated** (this block replaces the training one wholesale, and the parent recipe has `e4m3`) |
| `inference_grouped_gemm_backend` | `torch` | **not validated** (`flashinfer` is rejected unless training sets `moe_inference_training_forward`; that setup is in `nano-nvfp4-zero-kl`) |
| `inference_moe_token_dispatcher_type` | `nvls` | **not validated** |
| `moe_router_dtype` | `fp32` | **not validated** |
| `enable_prefix_caching` | `false` | **not validated** |
| `expert_model_parallel_size` | same as training (`4`) | **not validated** |

Top level:

| Knob | Required value | |
|---|---|---|
| `policy.precision` | `bfloat16` | validated |
| `policy.sequence_packing.enabled` | `false` | validated |
| `policy.dynamic_batching.enabled` | `false` (the verified value) | not required: allowed if `sequence_length_round` is a multiple of 64, but untested with this recipe |
| `policy.make_sequence_length_divisible_by` | multiple of `64` | validated |
| `cluster.segment_size` | `null` (with 1-node training and generation clusters) | fails at startup |

Not required: synchronous GRPO. `grpo.async_grpo` is not validated, and async works (it
helps convergence). Section 6 says which checks apply to sync and async.

## 5. Precision file (`$P`)

Required matchers, in this order (first match wins):

| Order | Pattern | Precision |
|---|---|---|
| 1 | `*linear_qkv*` | BF16 |
| 2 | `*linear_proj*` | BF16 |
| 3 | `*.mlp.shared_experts.*` | BF16 |
| 4 | `*.mlp.experts.linear_fc1` | MXFP8 |
| 5 | `*.mlp.experts.linear_fc2` | MXFP8 |
| 6 | `*` | BF16 (includes the Mamba `mixer.in_proj` / `mixer.out_proj`) |

The BF16 config is `training_recipe: {}`. The MXFP8 config must set all of the following in
**both** `training_recipe` and `evaluation_recipe`:
- `fp8_quantization_recipe: mxfp8`
- `fp8_format: hybrid`
- `fp8_param: true`
- `override_quantized_autocast: true`
- `override_nonquantized_autocast: true`

Validated in `setup.py`: both override flags in every MXFP8 block present, and
`fp8_param: true` requires `fp8_cfg.fp8_param: true`. **Not validated**: the matchers and
their order, `fp8_format`, and a missing `evaluation_recipe` (an absent block is skipped).

## 6. Confirm with a run

Run `tests/test_suites/llm/grpo-nanov3-30BA3B-2n4g-megatron_generation-noncolocated-zero-kl-mxfp8.sh`
(2 GB200 nodes, 5 steps), sync or async. It passes only if:
- `train/probs_ratio_min` and `train/probs_ratio_max` are both exactly `1` on every step.
  This holds only with one optimizer step per GRPO step, i.e.
  `policy.train_global_batch_size` = `grpo.num_prompts_per_step` ×
  `grpo.num_generations_per_prompt` (the recipe has 32 = 8 × 4). With more steps, the
  minibatches after the first train on updated weights and their ratio is not 1, even with
  zero mismatch. The check is also meaningless with `loss_fn.force_on_policy_ratio: true`,
  where the ratio is 1 by construction;
- sync runs only (async adds policy staleness to these): `train/gen_kl_error` is exactly
  `0` and `train/token_mult_prob_error` is exactly `1` on every step. These are what the
  nightly script asserts;
- `ray-driver.log` does **not** contain `Deterministic mode: set TRITON_CACHE_AUTOTUNING=1`.

Reference (GB200, Megatron-LM `b4d72b7` + #7562 + #8195): with everything above, KL was
0 on all 5 steps; the same recipe without `TRITON_CACHE_AUTOTUNING` gave about 0.003 from
step 1.
