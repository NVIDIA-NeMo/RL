# Block JustGRPO

Research implementation on upstream NeMo-RL `main` (`612d5274059c821dc09c2c90767b546b6f2d50b5`), based on the [diffusion design document](https://docs.google.com/document/d/1HHzRpLfsRalQkqLGFe0vljZjRO7dwKaKWH43jZPxCu4/edit) and the [Flow GRPO research structure](https://github.com/triple-mu/RL/pull/1). The algorithm, 6x6 Sudoku generator, and few-shot prompt come from the Block JustGRPO reference implementation.

**Training uses Megatron only.** Generation can use the reference diffusion vLLM runtime or the in-process Megatron diffusion decoder. Both use the same driver, denoising schedule, loss, and single validation mode inherited from training. All experiment defaults live in YAML, inheriting `examples/configs/grpo_math_1B.yaml`.

## Upstream controller

`train.py` imports `setup()`, `grpo_train()`, and `async_grpo_train()` from `nemo_rl.algorithms.grpo`. Upstream owns Ray worker placement, data loading, rollout collection, GRPO advantages, validation, logging, and the training loop. Research code supplies the Sudoku data/environment and a `MegatronDiffusionGeneration` implementation of the existing `GenerationInterface`. vLLM uses upstream `VllmGeneration` directly. Setup constructs upstream `Policy` directly; the recipe selects `MegatronDiffusionPolicyWorker` through the existing `policy.worker_extension_cls_fqn` setting. `train.py` registers its runtime and passes the research configuration to the worker. For Megatron inference, `train.py` replaces the native generation wrapper after ordinary upstream setup and attaches a native colocated weight synchronizer. HTTP serving is disabled, so setup has not started an AR engine. There is no generation-factory hook in core. vLLM retains its native setup path. Rollout logprobs are included in scoring inputs for Fast token selection. Previous-policy logprobs are always recomputed through upstream `Policy.get_logprobs()`.

## Diffusion training

`BlockJustGRPOSchedule` produces one denoising level at a time. Each view supplies clean token IDs, masked positions, selected targets, and their original rollout coordinates. `prepare_diffusion_microbatch` constructs `[noisy | clean]` inputs with repeated positions. The native Megatron diffusion mask lets noisy queries see their noisy block and earlier clean blocks; clean queries attend causally to clean keys.

Prompts are left-padded to block boundaries with **MASK tokens** before generation. These fixed, attended context tokens are excluded from corruption and loss. Generation and training receive exactly the same prompt IDs. This prefix changes conditioning from an unpadded prompt. Ungenerated positions in a trailing response block stay masked.

`MegatronDiffusionPolicyWorker` extends the existing `MegatronPolicyWorkerImpl`. It supplies diffusion input preparation and loss/logprob postprocessors, while the upstream worker owns the model, optimizer, scheduler, reference weights, and forward/backward execution. Training uses:

1. `begin_train_step()` once per rollout batch.
2. `train_microbatch(level)` for each level from `DenoisingSchedule.iter_levels(data)`. This calls upstream `get_microbatch_iterator()`.
3. `finish_train_step()` once, normalizing with upstream's accumulated token-mask counts.

Each selected response token appears in exactly one level's loss mask. Gradients accumulate across levels before one reduction and optimizer update; source sample counts are corrected only for reporting. There is no expanded batch or custom iterator factory. Policy and reference scoring use `aggregate_diffusion_logprobs` to collect selected scores into the original `[batch, sequence]` layout. Reference weights remain loaded across the entire schedule.

`DiffusionLossPostProcessor` and `DiffusionLogprobsPostProcessor` share `selected_diffusion_logprobs`. The loss processor calls core `ClippedPGLossFn` with `shift_labels=False`; scores and metadata remain at the same positions without dummy-column padding. Temperature scaling and microbatch loss scaling remain in the upstream forward and postprocessor.

Supported training layouts are dense BF16 TP/DP with PP=CP=EP=1, unpacked fixed-width sequences, and synchronous gradient/parameter collectives. Sequence parallelism, PEFT, quantization, dynamic batching, and resumable checkpoints are not supported by this research driver. Megatron generation shares live training weights; vLLM uses native weight synchronization.

## Generation

Generation code lives in `just_grpo/generation/`. `megatron_generation.py` implements the existing `GenerationInterface` and the research denoising loop. Synchronous requests delegate to upstream `Policy.generate()`. Async requests await native Ray worker calls on TP=1 DP replicas, distributing inference microbatches across replicas and yielding rows with their original indices. This also handles single-row requests when DP exceeds the batch size. `MegatronDiffusionPolicyWorker` runs upstream `model_forward()` on its already-loaded model. Native `MegatronWeightSynchronizer` handles the colocated lifecycle, and `prepare_for_lp_inference()` restores model parameters, enters eval mode, and releases training buffers. Generation sees optimizer updates directly; there is no second model, checkpoint export, subprocess, or custom attention implementation.

The Megatron decoder uses uncached full forwards with the same fixed-width `[noisy | clean]` attention layout as training. Each current block starts fully masked and reveals positions from left to right. Sampling records temperature-scaled logprobs at commitment and stops independently per sample at EOS. Sampled MASK IDs count as committed tokens. Prompts are grouped by length, microbatched using `policy.logprob_batch_size`, and returned in their original order. This first implementation supports TP=1; KV caching and TP>1 decoding are deferred. It prioritizes matching the training forward, not inference throughput.

Use `policy.generation.backend: megatron`, `just_grpo.runtime: megatron`, and `generation_python: null` for this decoder. The driver constructs the research `GenerationInterface` after upstream setup; it does not initialize Megatron's AR inference engine. Colocated setup uses `refit_transport: mcore` and `refit_backend: gloo` to satisfy native refit configuration; weights are shared rather than transferred. Its recipe disables generation CUDA graphs. No vLLM/SGLang installation is needed for decoding.

`reference_vllm.py` extends native `VllmGenerationWorkerImpl` and `VllmAsyncGenerationWorkerImpl`, sharing the reference fork's temperature/logprob contract and diffusion context check. The recipe selects it with `policy.generation.worker_extension_cls_fqn`; upstream `VllmGeneration` owns worker placement, lifecycle, ports, output packing, and weight synchronization. The reference fork must support leftmost diffusion and sampled-token logprobs at commitment. Stock upstream vLLM/SGLang is not assumed to implement that contract; `upstream_vllm` and `upstream_sglang` remain rejected.

## Async collection

The Megatron decoder extends the native `MegatronGeneration` wrapper so upstream recognizes its colocated lifecycle. It implements `generate_async()` even with synchronous GRPO, because upstream selects async rollout dispatch for every Megatron generation backend. It yields completed microbatches without blocking the event loop and drains submitted calls before propagating an error or cancellation.

Set `grpo.async_grpo.enabled: true` to select upstream `async_grpo_train()` and its trajectory collector. The wrapper reports `blocks_training()` and `wake_carries_weight_updates()` as true. With `in_flight_weight_updates: false`, upstream drains pending rollouts before previous/reference scoring and training, then wakes the shared model and advances the collector's weight version. Generation and training alternate on the same model; this does not provide concurrent model execution. Importance-sampling correction is required. KV-cache recomputation, HTTP serving, and the async data plane are unsupported. Reference vLLM also supports upstream async GRPO using the native async engine and dedicated generation GPUs. Dedicated async configurations can allocate two training nodes and two generation nodes (32 GPUs total), with no in-flight weight updates. Training can overlap generation on the separate GPUs; refits drain pending requests first. Both runtimes must load the same NCCL version for dedicated weight synchronization. The DFW async vLLM submissions preload the existing vLLM runtime's NCCL library on both sides.

The retained Megatron recipes support async collection with the CLI override `grpo.async_grpo.enabled=true`. See `VERIFICATION.md` for GPU verification status.

## Recipes and launch

Four 32-GPU long-run recipes are published in `research/just_grpo/configs/recipes/`:

| Experiment | Recipe |
| --- | --- |
| Standard JustGRPO, TP=1/DP=32 | `just_grpo-sudoku6x6-4n8g-megatron-inference-long.yaml` |
| Fast JustGRPO, 25% per block, TP=1/DP=32 | `just_grpo-sudoku6x6-4n8g-megatron-inference-fast-long.yaml` |
| Async JustGRPO, TP=1/DP=32 | `just_grpo-sudoku6x6-4n8g-megatron-inference-async-long.yaml` |
| Fast Async JustGRPO, 25% per block, TP=1/DP=32 | `just_grpo-sudoku6x6-4n8g-megatron-inference-fast-async-long.yaml` |

The standard recipe inherits upstream `examples/configs/grpo_math_1B.yaml`. Fast inherits the standard recipe and overrides the token fraction and run name. Async inherits the same standard recipe and enables async GRPO with a separate log directory and run name. Fast Async inherits Async and overrides the token fraction and logging names. The functional smoke script applies two-GPU, three-update overrides to the standard recipe.

From the checkout root, with the NeMo-RL `mcore` environment:

```bash
export PYTHONPATH="$PWD:$PWD/research/just_grpo${PYTHONPATH:+:$PYTHONPATH}"
uv run --extra mcore python research/just_grpo/run_just_grpo.py \
  --config "$PWD/research/just_grpo/configs/recipes/just_grpo-sudoku6x6-4n8g-megatron-inference-long.yaml" \
  --model /path/to/Nemotron-Labs-Diffusion-3B \
  --output-dir /path/to/fresh/run
```

For vLLM, `--generation-python` selects the native Ray generation worker interpreter; it must include the NeMo-RL worker dependencies as well as the reference fork. If omitted, the upstream vLLM actor environment is used. For Megatron generation, select its recipe and omit `--generation-python`. `tools/dfw_distributed.sbatch` handles both smoke and long runs; choose the allocation with `sbatch --nodes` and `--gpus-per-node`, matching the YAML cluster settings. Set `REPO_DIR`, `POLICY_PYTHON`, `MODEL`, `OUTPUT_DIR`, `CONFIG`, `CONTAINER_IMAGE`, and `CUDA_COMPAT_DIR`. `GENERATION_PYTHON` is optional. `POLICY_EXTRA_PYTHONPATH` supports a provisioned dependency overlay. The driver starts or connects to Ray; multi-node jobs use upstream `ray.sub`. Compilation caches use node-local storage. Native vLLM workers own port isolation. Megatron decoding executes inside the training workers.

`tests/functional/sudoku.sh` launches a two-GPU Ray smoke and checks native TensorBoard metrics for three updates, finite gradients, generation/training KL, timing metrics, and initial/final validation.

## Fast JustGRPO

For every sample and response block, select the lowest-confidence positions using detached sampled-token rollout logprobs. The budget is `ceil(training_token_fraction * block_size)`, capped by valid positions. Ties prefer earlier positions; prompts, padding, and excluded samples never enter selection. This is a confidence proxy, not entropy.

With block size 16 and fraction 0.25, the schedule supplies four views instead of sixteen. Each sample and block can choose a different offset in a view. Actual tokens before the selected offset are revealed, including unselected tokens; the selected offset and all later positions remain masked. Earlier-block context comes from the clean half, preserving ordinary JustGRPO conditioning at selected positions.

The objective is normalized by selected-token count, without selection-probability correction. Previous-policy logprobs are always recomputed before the update and serve as the PPO denominator. Generation logprobs remain separate for confidence selection, importance-sampling correction, and generation/training diagnostics. Previous-policy scoring covers all response positions for upstream sequence-level diagnostics; Fast selection reduces the training and reference-scoring views. Reference scoring remains enabled when reference KL is nonzero. `loss_fn.force_on_policy_ratio: true` is rejected because it can skip previous-policy scoring. Reduced view count does not guarantee proportional end-to-end speedup.

## Sudoku and the matched experiment

The custom 6x6 environment uses the reference generator, 2x3 boxes, few-shot prompt, and **blank-cell accuracy** reward. `just_grpo/environments/sudoku.py` contains the dataset, prompts, scoring, and NeMo-RL environment interface. `sudoku6x6_generator.py` implements puzzle generation; `sudoku6x6_fewshot.txt` holds the reference system prompt. Only this 6x6 task is supported; the 4x4 Reasoning Gym adapter and its dependency have been removed.

The 32-GPU recipe matches [cmp0913_sudoku6x6_block_justgrpo_val_t1_extend0914](https://wandb.ai/nemo-llm-service/diffusion_rl/runs/sw1o4ibd): 128 prompts x 8 completions, 100 updates, LR 3e-7, 13 warmup updates starting at 3e-8, weight decay 0.01, reward normalization off, reference k3 KL 0.01, TIS capped at 2, and 512 output tokens. Block size is 16 and training temperature is 1.0. Validation uses the training decoder at temperature 1.0 every ten updates, on 256 puzzles repeated four times.

The reference seeds (training 1, validation 2) produce overlapping puzzles because the generator seeds each puzzle with `seed + index`. These matched splits are **not held out**. Runs start fresh from the base model, without the reference optimizer state or an identical RNG/data-order replay.

## Metrics and verification

W&B uses upstream GRPO names, including `train/reward`, `train/kl_penalty`, `train/gen_kl_error`, `train/loss`, `train/grad_norm`, and ratio/importance metrics. `validation/accuracy` reports the mean environment reward. Validation has one mode inherited from training.

Metrics and timers come directly from upstream GRPO, including `timing/train/total_step_time`, generation, scoring, training, validation, and worker timings. `global_valid_toks` counts selected training tokens. Upstream writes rollout JSONL files; `wandb_url.txt` contains the tracking link. The former custom `metrics.json` format is replaced by upstream logging.

See [VERIFICATION.md](VERIFICATION.md) for recorded Megatron checks and their scope.
