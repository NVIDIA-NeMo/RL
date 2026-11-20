# Single-Controller Simulator

`examples/run_single_controller_simulator.py` runs one side of the [Single-Controller](single-controller.md) loop with the other side replaced by a stand-in that uses no GPU, on the same pumps, data plane, sampler and refit lifecycle as a training run. Use it to profile and tune one side on its own, with the recipe a training run uses.

Only the `rollout_only` mode is implemented: generation, the environments (NeMo-Gym or native) and the TransferQueue data plane are real, while the trainer is a `NoOpTrainer` and the weight transfer is a `NoOpWeightSynchronizer`. The `train_only` mode and the `cpu` rollout backend are tracked in [issue #4860](https://github.com/NVIDIA-NeMo/RL/issues/4860).

## Run a Rollout-Only Simulation

The launcher takes two YAML files. `--config` is an SC recipe, loaded as `run_grpo_single_controller.py` loads it. `--simulator-config` describes the stand-ins and defaults to [`examples/configs/simulator/rollout_only.yaml`](../../examples/configs/simulator/rollout_only.yaml). Hydra-style overrides apply to the recipe, or to the simulator config when prefixed with `simulator.`.

```bash
uv run examples/run_single_controller_simulator.py \
    --config <your-sc.yaml> \
    --simulator-config examples/configs/simulator/rollout_only.yaml \
    simulator.num_steps=20 simulator.num_nodes=4 \
    simulator.train_time_s=300 simulator.refit_time_s=20 \
    <recipe overrides...>
```

### Simulator config

| Key | Default | Meaning |
| --- | --- | --- |
| `mode` | `rollout_only` | Which side of the loop is real. `rollout_only` is the only value. |
| `train_backend` | `cpu` | The trainer. `cpu` is the no-op stand-in, which trains nothing and uses no GPU. It is the only value. |
| `train_time_s` | `0.0` | Seconds the controller holds each step where the optimizer step would run. |
| `refit_time_s` | `0.0` | Seconds the stand-in weight transfer holds inside the refit pause. |
| `num_nodes` | `null` | Inference nodes of the allocation. `null` keeps the recipe's generation node count. |
| `num_steps` | `null` | Steps to run. `null` keeps the recipe's `max_num_steps`. |

A key the schema does not know is rejected.

### What the launcher changes in the recipe

The recipe is used as-is and must be an SC recipe (`data_plane.enabled: true`, `async_rl:`, `grpo.async_grpo: null`). The launcher overrides only what a rollout-only run owns:

- `num_steps` sets `max_num_steps`, and widens `max_num_epochs` so a short dataset cannot clamp it.
- `num_nodes` sets the generation node count and `cluster.num_nodes`.
- `data.validation` is dropped.
- `vllm_cfg.load_format` defaults to `auto`, so the engine loads the real checkpoint instead of dummy weights plus a first refit. An explicit `dummy` is rejected.

Everything else about the engine is configured exactly as `run_grpo_single_controller.py` configures it, including the `skip_tokenizer_init` rule, so it does the same per-token work as an engine beside a trainer.

## What the Stand-Ins Do

Per step:

- The rollout pump, `RolloutManager`, `TQReplayBuffer` and the configured sampler run unchanged, so admission is gated exactly as in training: on the trainer version, or on buffer capacity for the `windowed` sampler.
- The train pump selects the step's groups and consumes them without training on them. The policy and reference logprob stages, training-mask filtering, the advantage stage and microbatch training are skipped, so the metrics they produce are absent: `train/reward`, `train/advantages/*`, `train/max_seq_mult_prob_error` and the loss metrics. The reward of the generated rollouts is reported as `train/unfiltered_reward`.
- After all chunks of the step are consumed, the controller holds `train_time_s` once, where the optimizer step would run. Generation keeps serving during the hold, but forward and backward work that overlaps later rollout chunks of the same step is not modeled. The hold ends early once no rollout is dispatched or in flight, which is polled every 50 ms.
- Cleanup of the consumed rows, the release of buffer capacity and the version advance are the code a training run uses.
- `_sync_weights` runs its full lifecycle (dispatch gate, `pause_generation_for_refit(clear_cache=recompute_kv_cache_after_weight_updates)`, resume) around a `NoOpWeightSynchronizer` that moves nothing and holds `refit_time_s`. The trainer version advances, which releases the sampler's next target.

Everything on the rollout side is available: `generation_router`, `generation_fleet_health` (probes and shard restarts), the stall watchdog, token capture, and the `NRL_NSYS_*` profiling hooks.

### Timing metrics

| Metric | Meaning |
| --- | --- |
| `train/simulated_train_time_s` | Seconds of the `train_time_s` hold that were held. `timing/train/policy_training` covers the same seconds. |
| `train/simulated_train_idle_skipped_s` | Seconds of the hold that were skipped because no rollout was in flight. Adding it to `timing/train/total_step_time` gives the step time of this coarse timing model. |
| `timing/train/weight_sync` | The refit pause, which includes the `refit_time_s` hold. |
| `timing/train/exposed_generation` | Time the train pump waited on generation, as in a training run. |

The hold models a step whose batch arrives whole. The timing of a step that streams in several chunks is not validated against a training run.

## Placement

There is no training cluster. The inference cluster is sized from the recipe's generation node count, or from `num_nodes`, and every GPU node of the allocation serves generation.

The controller and the generation router are placed as in a training run. The `SingleControllerActor` is pinned to the dedicated Ray head when the cluster has one (`DEDICATED_RAY_HEAD=1` in `ray.sub`, see [cluster setup](../cluster.md)); set `num_nodes` to the GPU nodes left after the head. The generation router, when enabled, is pinned to the driver's node.

Without a dedicated head, the Ray head node has GPUs and hosts a vLLM rank next to the driver and possibly the `SingleControllerActor`. On a training run that node may be a train node instead. Use a dedicated head, or expect the head's rollout rank to share CPUs with those processes, when comparing against a training run.

## Smoke Test

`tests/functional/grpo_async_gym_single_controller_rollout_only.sh` is the smoke for this path: the recipe and overrides of `tests/functional/grpo_async_gym_single_controller.sh` (`examples/nemo_gym/grpo_qwen3_30ba3b_instruct.yaml` on Qwen3-0.6B with a 1-GPU vLLM split) with the trainer stubbed out, gated on a scored rollout and on the stand-in's metrics.

## Limitations

`rollout_only` does not support the following. Setup rejects each before any engine starts.

- PPO.
- Generation backends other than vLLM. Megatron generation serves the trainer's own weights; TRT-LLM and SGLang are not validated under the SC refit lifecycle.
- Dynamo serving (`policy.generation.dynamo: true`), which is not validated under the simulator's refit lifecycle yet.
- Checkpointing and on-policy distillation.
- Trainer-side initial weights: `checkpointing.pretrained_checkpoint`, `policy.pretrained_checkpoint` and `megatron_cfg.peft.restore_from`. The engine serves `policy.model_name`, so point it at an HF-exported checkpoint.
- `async_rl.log_full_train_data` and `async_rl.refit_actor.enabled`.
- `token_capture.defer_routed_experts_to_policy` and `token_capture.forest_training`.
- Refittable fp8 KV caches and adaptive-ppo-clip.
- Online quantization (`precision: fp8`, `nvfp4_pertoken_rollout`, `quant_cfg`, `real_quant`): those quantize at refit, which this run has none of.
