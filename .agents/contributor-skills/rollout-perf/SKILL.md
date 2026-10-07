---
name: rollout-perf
description: Rollout (generation) performance for NeMo-RL recipes and CI tests. Covers diagnosing the generation bottleneck from metrics.json and vLLM logs, the validated vLLM settings (CUDA graphs, parallel layout), a same-window A/B with logprob-consistency gates, and updating the test's time budget.
when_to_use: Adding or changing a nightly/release/performance test or recipe that generates with vLLM; a test is slow or close to its NUM_MINUTES; a recipe sets `enforce_eager`, `vllm_cfg` or `vllm_kwargs`; reviewing such a PR; 'rollout is slow', 'generation time', 'cuda graph', 'enforce_eager', 'speed up nightly test', 'GPU hours budget'.
---

# Rollout Performance

Every test or recipe that generates with vLLM should have its rollout
diagnosed before it lands. Generation is often most of a GRPO/DAPO step, and
the default eager configuration can waste most of it.

Five rules:

1. **Diagnose before tuning, and decide by A/B.** The right knob depends on the
   bottleneck. Version-specific rules (for example "PIECEWISE only") expire as
   vLLM moves, so re-measure them instead of copying them.
2. **Change one knob (or one declared bundle) at a time.**
3. **Prove the setting took effect** from the engine log, not from the YAML.
4. **Gate on logprob consistency and accuracy.** A faster rollout that
   changes `token_mult_prob_error` or accuracy is not a win.
5. **Compare in the same time window**, with the same container and code.

## 1. Diagnose

Run the helper on the test's artifacts. CI and `tools/launch` runs write
`tests/test_suites/<domain>/<exp>/metrics.json` and
`<jobid>-logs/ray-driver.log` inside the code snapshot.

```bash
uv run --no-project python .agents/contributor-skills/rollout-perf/rollout_perf_report.py \
  <snapshot>/tests/test_suites/llm/<exp>/metrics.json \
  --log <snapshot>/<jobid>-logs/ray-driver.log
```

Read the result with this table:

| Signal | Where | Meaning |
| :-- | :-- | :-- |
| `timing/train/generation` / `timing/train/total_step_time` ≥ 50% | metrics.json | Generation-bound: rollout tuning moves wall time |
| `train/max_gen_tokens_per_sample` hits `max_new_tokens` on most steps, and generation time varies < 10% | metrics.json | Decode-tail-bound: the longest response sets the step, so per-step decode latency is the lever |
| `train/mean_prompt_length` ≫ generated length, or multi-turn | metrics.json | Prefill-heavy: prefix reuse and scheduler token budget matter |
| Engine line `'cudagraph_mode': <CUDAGraphMode.NONE: 0>` or `enforce_eager=True` | ray-driver.log | vLLM runs eager: every decode step pays kernel-launch overhead. Ignore the `Overrides:` and `MasterConfig(` echo lines; they show the request, not what ran |
| `Maximum concurrency for N tokens per request: Xx` vs per-replica load | ray-driver.log | Memory knobs only help when X is below the load |
| `timing/validation/total_validation_time` | metrics.json | Validation uses the same engine and speeds up with it |

Per-replica load = (prompts × generations per prompt × DAPO `batch_multiplier`)
/ number of vLLM replicas. The number of replicas is the generation GPUs
divided by TP × PP.

## 2. Levers by regime

| Regime | Lever | Notes |
| :-- | :-- | :-- |
| Eager + decode-tail-bound (most math/code GRPO/DAPO tests) | **CUDA graphs**: drop `enforce_eager: true` (the inherited default is `False`) and let vLLM use its default `cudagraph_mode` (FULL_AND_PIECEWISE). A/B it against eager and against `cudagraph_mode=PIECEWISE` | Biggest single lever. Pick the mode by A/B, not by rule. Keep the default capture sizes unless the batch exceeds them |
| Graph mode depends on the vLLM version | vLLM 0.20: **PIECEWISE only**, plus `pass_config.fuse_allreduce_rms=false`. Newer vLLM (0.29 measured): A/B FULL_AND_PIECEWISE first | vLLM 0.20 had an accuracy bug with FULL graphs. In internal Nano V3.5 GRPO runs, 271–2,548 of 8,192 sequences were masked per step, against 1–5 with PIECEWISE. On vLLM 0.29 (cw_dfw H100), `grpo-nanov3-30BA3B-1n8g-fsdp2.v2` and `grpo-nemotron3-super-120BA12B-16n8g-megatron` had the same `token_mult_prob_error` and `gen_kl_error` under FULL_AND_PIECEWISE, and decoded 2.0× and 1.5× faster than under PIECEWISE |
| FULL_AND_PIECEWISE is not always the winner | Keep PIECEWISE when the A/B says so | Across 11 cw_dfw tests, FULL_AND_PIECEWISE won 6 and PIECEWISE won 5. In `grpo-math-qwen3-30ba3b-megatron-tp4-32k` and `dapo-qwen2.5-7b.v2`, FULL_AND_PIECEWISE added `token_mult_prob_error` spikes, and on `dapo-qwen2.5-7b.v2` it was also slower. `dapo-nanov3.5-30BA3B-4n8g-automodel`: generation was the same (100.4 vs 99.8 s per step), but `token_mult_prob_error` spiked on 6/20 steps against 1/20 with PIECEWISE. `grpo-qwen3.5-35ba3b-2n8g-megatron-ep16tp2-fp8`: FULL graphs' extra memory left 833 Mamba cache blocks for `max_num_seqs` 1024, so vLLM failed at startup. The driver then sat idle until the job was cancelled, so watch for a missing first step |
| Logprob gates regress with graphs on | `compilation_config.backend=eager` | This keeps graphs but replays vLLM's custom kernels instead of Inductor-compiled ones. With Inductor, `grpo-gspo-deepscaler-1.5b-8K` went from `token_mult_prob_error` median 1.013 to 1.040 and failed its `gen_kl_error` gate. With `backend=eager` (FULL_AND_PIECEWISE) it kept 1.013 / 0.00044 and still cut generation by 49% and wall time by 40%. The `grpo-deepscaler-1.5b-*` recipes already do this |
| Tail-bound with many replicas | Keep more, smaller replicas | Nano DAPO at 4n8g: TP8 × 4 was ~25% slower than TP4 × 8 with graphs on |
| Prefill-heavy, multi-turn | Prefix caching, `max_num_batched_tokens` sweep, session-sticky routing | Budget semantics differ per engine and topology. Sweep instead of copying a number |
| KV-bound | `gpu_memory_utilization`, `max_num_seqs` | Colocated runs share memory with training, so check training headroom |

Do not use these:

- **MTP speculative decoding as a free speedup.** When the policy does not
  train the MTP head, NeMo-RL loads the drafter from the checkpoint and never
  refits it (`load_mtp_weights_from_disk` in
  `nemo_rl/models/generation/vllm/vllm_worker.py`). The frozen drafter drifts
  from the trained policy and acceptance falls over training. Check
  `train/vllm/spec_acceptance_rate` across the whole run before adopting it.
- **Prefix caching with router replay (R3).** R3 needs routes for every
  prompt token.
- **KV/weight precision changes (FP8 etc.)** as a "perf knob". They change
  numerics and need their own convergence study.
- **Shared-lustre JIT caches.** Pointing `TRITON_CACHE_DIR`,
  `TORCHINDUCTOR_CACHE_DIR`, `VLLM_CACHE_ROOT` or `XDG_CACHE_HOME` at a shared
  lustre directory made a 32-rank Mamba Triton warmup take 20 min instead of
  48 s. `ray.sub` uses `--no-container-mount-home`, so the container defaults
  are safe.

## 3. A/B on Slurm

1. **Freeze the container.** `rl.nightly.sqsh` is rebuilt daily, so use an
   immutable image (e.g. `rl.<build-id>.sqsh` or
   `nvcr.io/nvidian/nemo-rl:nightly-YYYY-MM-DD`). Old runs on another image
   are not a valid control.
2. **Launch both arms together** from the same commit. Give them separate
   snapshot dirs so neither resumes the other's checkpoints:

   ```bash
   export CONTAINER=<immutable.sqsh> ACCOUNT=<account> PARTITION=batch
   export HF_HOME=<hf_home> HF_DATASETS_CACHE=<hf_datasets_cache>  # required by tools/launch
   CODE_SNAPSHOT_DIRNAME=code_snapshots_perf_base \
     EXTRA_SCRIPT_ARGS="logger.wandb_enabled=False" \
     tools/launch tests/test_suites/llm/<test>.sh
   CODE_SNAPSHOT_DIRNAME=code_snapshots_perf_full \
     EXTRA_SCRIPT_ARGS="logger.wandb_enabled=False policy.generation.vllm_cfg.enforce_eager=false" \
     tools/launch tests/test_suites/llm/<test>.sh
   CODE_SNAPSHOT_DIRNAME=code_snapshots_perf_pw \
     EXTRA_SCRIPT_ARGS="logger.wandb_enabled=False policy.generation.vllm_cfg.enforce_eager=false ++policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=PIECEWISE" \
     tools/launch tests/test_suites/llm/<test>.sh
   ```

3. **Prove the setting.** Each graph arm's `ray-driver.log` must show the
   engine lines `enforce_eager=False` and
   `'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: ...>` (or
   `PIECEWISE`), plus the matching `Capturing CUDA graphs (...)` lines. The control must show
   `<CUDAGraphMode.NONE: 0>`. The helper reads these lines and skips the
   `Overrides:` echo.
4. **Compare paired steps.** The same seed gives the same data order:

   ```bash
   uv run --no-project python .agents/contributor-skills/rollout-perf/rollout_perf_report.py \
     --base code_snapshots_perf_base/<exp>/tests/test_suites/llm/<exp>/metrics.json \
     --treat code_snapshots_perf_full/<exp>/tests/test_suites/llm/<exp>/metrics.json
   ```

## 4. Gates (treatment against the same-window control)

- `train/token_mult_prob_error`: the mean over tokens of
  exp(|logp_vLLM − logp_train|), where 1.0 is a perfect match. The median and
  spike count must not get worse. Spikes come from a handful of tokens that
  are tens of nats apart, which is common in MoE without R3.
- `train/gen_kl_error` mean: unchanged.
- `train/mean_gen_tokens_per_sample`: unchanged, otherwise the workload
  changed rather than the engine.
- Validation accuracy: compare against the run-to-run noise. Step-0 accuracy
  uses identical weights in both arms, so its difference estimates the noise
  of a single run (about ±0.025 on DAPOMathAIME2024).
- The test's own `check_metrics.py` gates. If the control also fails a gate,
  the failure is pre-existing; say so in the PR.

## 5. Land it

- Put the setting in the recipe YAML, with a one-line comment that gives the
  measured effect. Recipes inherit `enforce_eager: False` from the base config,
  so delete the `enforce_eager: true` line instead of setting it to false.
  `tools/config_cli.py minimize-check` rejects keys that only repeat a
  default.
- If `enforce_eager: true` must stay, add a comment explaining why
  (correctness bug, unsupported model). Reviewers should ask for that comment.
- Lower `NUM_MINUTES` in the test script to about 2× the new measured wall
  time. Nightly budget = NUM_RUNS × NUM_NODES × GPUS_PER_NODE × NUM_MINUTES/60,
  and the total is asserted in `tests/unit/test_recipes_and_test_suites.py`.
- If the recipe sets `checkpointing.checkpoint_must_save_by` (a `DD:HH:MM:SS`
  duration, for example `00:03:45:00`, after which training saves and stops),
  lower it below the new Slurm limit. Leave room for the checkpoint save and for
  Ray startup and setup, which happen before this timer starts. Otherwise Slurm
  kills the job before the timeout checkpoint is written.
- In the PR description, give a before/after table: generation per step, step
  time, validation, setup, wall time, gates, container and job IDs.

## Reference result

`dapo-nanov3.5-30BA3B-4n8g-automodel` (Nemotron-3.5 30B-A3B, 4n8g H100,
vLLM 0.29, TP4 × 8), and its router-replay variant from #4458. The change was
from eager to the bundle `enforce_eager=false`, `cudagraph_mode=PIECEWISE`,
`fuse_allreduce_rms=false`. Both arms ran in the same window on the same
container.

| Metric | Change |
| :-- | :-- |
| Generation per step | 325 → 101 s (−69%) |
| Validation | −79% |
| Setup | +21 s |
| Wall time | 152 → 65 min (−57%) |
| `token_mult_prob_error` median and `gen_kl_error` | Unchanged |

Same-window A/Bs on the cw_dfw nightly tests (H100, container `rl.71856875.sqsh`,
eager → PIECEWISE). Every arm passed its own `check_metrics` gates, with
`token_mult_prob_error` and `gen_kl_error` unchanged:

| Test | Generation per step | Wall time |
| :-- | :-- | :-- |
| `grpo-qwen3.5-35ba3b-2n8g-megatron-ep16tp2-fp8` | −85% | 174 → 79 min |
| `grpo-qwen3.5-35ba3b-2n8g-megatron-ep16tp2cp2` | −79% | 153 → 83 min |
| `vlm_grpo-nemotron-omni-30ba3b-clevr-2n8g-megatron-tp8ep8.v1` | −78% | 51 → 29 min |
| `grpo-math-qwen3-30ba3b-megatron-tp4-32k` | −74% | 123 → 57 min |
| `vlm_grpo-nemotron-omni-30ba3b-mmpr-4n8g-automodel-ep8.v1` | −72% | 81 → 41 min |
| `grpo-qwen3-8b-base-dapo-2n8g-long-megatron-qa-nvfp4-w4a16` | −71% | 164 → 59 min |
| `vlm_grpo-nemotron-omni-30ba3b-clevr-1n8g-automodel-ep8.v2` | −65% | 95 → 79 min |
| `vlm_grpo-gemma4-e4b-geo3k-1n8g-automodel` | −60% | 59 → 45 min |
