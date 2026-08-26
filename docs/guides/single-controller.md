# Train with Single-Controller (Async GRPO and PPO)

:::{warning}
The Single-Controller path is a **beta feature** and still under active development. The API and configuration surface are not yet stable and may change without notice. Issues and feedback are welcome — please file them at [github.com/NVIDIA-NeMo/RL/issues](https://github.com/NVIDIA-NeMo/RL/issues).
:::

The Single-Controller (SC) path is an alternative async GRPO and PPO runtime that runs rollout generation and policy training as two independent *pumps* coordinated by a single Ray actor (`SingleControllerActor`) sitting over a shared TransferQueue (TQ) data plane. Compared to the legacy async GRPO in [async-grpo.md](./async-grpo.md), SC decouples per-prompt rollouts from the per-step batch boundary: producers push finished rollouts into `TQReplayBuffer` at group granularity, and a pluggable `StalenessSampler` decides which groups the trainer consumes on each step.

## Configure the Single-Controller Path

The SC path is launched via a dedicated entrypoint:

```bash
uv run examples/run_grpo_single_controller.py --config <your-sc.yaml>
```

`run_grpo_single_controller.py` mirrors `run_grpo.py` for config loading — the same YAML files apply — but requires a few settings the legacy path does not. The default exemplar lives at [examples/configs/grpo_math_1B_megatron_single_controller.yaml](../../examples/configs/grpo_math_1B_megatron_single_controller.yaml); the PPO one at [examples/configs/ppo_math_1B_megatron_single_controller.yaml](../../examples/configs/ppo_math_1B_megatron_single_controller.yaml).

### Mandatory settings

1. **Enable the TransferQueue data plane** (required — the entrypoint refuses to start otherwise):

    ```yaml
    data_plane:
      enabled: true
    ```

2. **Enable vLLM async engine** and **disable colocated inference** (SC drives rollout via `RolloutManager.generate_and_push`, which is only supported on the disaggregated async engine; setup rejects `colocated.enabled: true`):

    ```yaml
    policy:
      generation:
        backend: "vllm"
        vllm_cfg:
          async_engine: true
        colocated:
          enabled: false
          resources:
            num_nodes: 1
            gpus_per_node: 4  # inference GPUs; remainder go to training
    ```

3. **One RL step = one training batch.** The batch a step trains on is the whole step (see `validate_single_controller_config` in [nemo_rl/algorithms/single_controller_utils/config.py](../../nemo_rl/algorithms/single_controller_utils/config.py)). A GRPO step is also one optimizer step; a PPO step is `ppo.ppo_epochs` of them over that same batch.

    ```python
    num_prompts_per_step * num_generations_per_prompt == policy.train_global_batch_size
    num_prompts_per_step * num_generations_per_prompt == value.train_global_batch_size  # PPO
    ```

4. **Enable importance sampling correction** whenever the sampler admits off-policy data (any `max_staleness_versions > 0` on the `windowed`/`weight_fifo` samplers, or `max_lookahead_versions > 0` on `in_order`). The correction and its derivation are the same as for legacy async GRPO — see [Why Importance Sampling Correction Is Required for Async](./async-grpo.md#why-importance-sampling-correction-is-required-for-async):

    ```yaml
    loss_fn:
      use_importance_sampling_correction: true
    ```

5. **(PPO) Set `ppo:` instead of `grpo:`** — the two algorithm blocks are mutually exclusive, and SC reads every step setting from whichever one is present. A PPO run also needs `value:`, `value_loss_fn:` and `ppo.adv_estimator.name: gae` (same schemas as legacy PPO), a Megatron critic, and `policy.offload_optimizer_for_logprob: true`, which is what keeps the policy optimizer off the GPU while the critic runs. `ppo.policy_training_start_step: N` gives the usual critic warmup: for the first N steps the policy is neither trained nor refit, while the critic trains every step.

## Async-RL Knobs and Sampler Modes

All SC async-RL runtime knobs live under `async_rl:` in the master config. The most important choice is the `sampler`, which sets the staleness policy shared by the rollout pump (how far it may run ahead) and the train pump (which groups it may consume).

### Sampler modes

Pick one of four modes with `sampler.name`. Each mode takes its own knobs, listed below — a knob from one mode has no effect under another:

![Sampler modes: same buffer, four different training batches](../assets/sc-sampler-modes.png)

*Same buffer under trainer weight 2 (`num_prompts_per_step=2`, staleness window `[0, 2]`). `windowed` and `weight_fifo` select on `start_weight` (stamped at dispatch); `in_order` selects on `target_step` (stamped at admit). The three usually pick the same groups and diverge only when rollouts finish out of order, as drawn here.*


| `sampler.name` | Rollout gating                                                                                                          | Train selection                                                                                                   | Typical use                                                                                                  |
| -------------- | ----------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `in_order`     | Dispatch may lead the trainer by up to `max_lookahead_versions` batches. Each dispatch is stamped with a `target_step`. | Consume the group whose `target_step == current_train_weight`.                                                    | Sync mode (`max_lookahead_versions=0`) and legacy-async exact-batch semantics (`max_lookahead_versions>=1`). The only mode supported on a PPO run. |
| `weight_fifo`  | Same gate as `in_order` (`max_staleness_versions` of lookahead).                                                        | Drain the oldest in-window `start_weight` first, waiting for that weight's batch to fill.                         | Strict weight-version FIFO under a bounded lookahead.                                                        |
| `windowed`     | Ungated — rollout keeps producing until the buffer fills.                                                               | Take any ready group with `start_weight` in `[train - max_staleness_versions, train]`, optionally freshest-first. | Over-sampled streaming; aged groups outside the window are evicted (wasted compute).                         |
| `custom`       | Determined by the imported class.                                                                                       | Determined by the imported class.                                                                                 | `target: "module:ClassName"` — bring your own `PromptGroupSampler`.                                          |


### Config → behavior map

The shipped exemplars cover three of the four modes:


| Mode                             | `sampler.name` | Sampler knob                   | `min_groups_for_streaming_train` | `max_buffered_rollouts`                               | Exemplar |
| -------------------------------- | -------------- | ------------------------------ | -------------------------------- | ----------------------------------------------------- | -------- |
| Sync / on-policy                 | `in_order`     | `max_lookahead_versions: 0`    | `${grpo.num_prompts_per_step}`   | `num_prompts_per_step × 1`                            | [`grpo-qwen2.5-math-1.5b-instruct-1n8g-megatron-single-controller-sync.yaml`](../../examples/configs/recipes/llm/grpo-qwen2.5-math-1.5b-instruct-1n8g-megatron-single-controller-sync.yaml) |
| Async, exact batch→step matching | `in_order`     | `max_lookahead_versions: >= 1` | `x <= num_prompts_per_step`   | `num_prompts_per_step × (max_lookahead_versions + 1)` | [`grpo_math_1B_megatron_single_controller.yaml`](../../examples/configs/grpo_math_1B_megatron_single_controller.yaml) |
| Streaming, gated dispatch        | `weight_fifo`  | `max_staleness_versions: >= 1` | `x <= num_prompts_per_step`      | `num_prompts_per_step × (max_staleness_versions + 1)` | — (none shipped) |
| Streaming, over-sampled          | `windowed`     | `max_staleness_versions: >= 1` | `x <= num_prompts_per_step`      | Larger than the gated capacity (dispatch is ungated)  | [`grpo-llama3.1-8b-instruct-2n8g-async-1off-single-controller-streaming2.yaml`](../../examples/configs/recipes/llm/grpo-llama3.1-8b-instruct-2n8g-async-1off-single-controller-streaming2.yaml) |


Field definitions:

- `max_buffered_rollouts` — hard cap on unconsumed rollout groups buffered in the data plane. Validated at setup against the gated sampler's required capacity; a value too small deadlocks the rollout pump, so setup raises instead of silently blocking. Sized from the widest window the run ever uses, so `warmup_lookahead_versions` rather than `max_lookahead_versions` when it is set.
- `min_groups_for_streaming_train` — minimum ready groups the trainer waits for before dispatching a batch. Set to `num_prompts_per_step` for sync/legacy semantics; lower for streaming. (PPO) Must equal `num_prompts_per_step` — the critic has no split train API, so one `train_from_meta` call is one optimizer step, and streaming a step across chunks would step the critic once per chunk.
- `sampler.warmup_lookahead_versions` (PPO) — lookahead used while `ppo.policy_training_start_step` critic warmup is in progress, shrinking back to `max_lookahead_versions` afterwards. The SC equivalent of `ppo.async_ppo.warmup_generation_lead_steps`.

## Implementation Structure

The SC path splits the async-GRPO loop across a rollout pump and a train pump that share a `TQReplayBuffer` and are orchestrated by the `SingleControllerActor`.

![SC ownership and call model: driver, SingleControllerActor, and remote worker groups](../assets/sc-ownership.png)

*The driver builds every heavy object and cloudpickles it into `SingleControllerActor` (a CPU-only Ray actor). `TQReplayBuffer`, `RolloutManager`, and the sampler live inside that actor — reaching them is a direct call, not RPC. Only the generation worker group, `TQPolicy`, and the TransferQueue data plane are separate processes; the dashed arrows are the only hops that cross a process boundary.*

### Core components

#### 1. `SingleControllerActor` (`nemo_rl/algorithms/single_controller.py`)

- Single Ray actor that runs `_rollout_pump` and `_train_pump` concurrently as asyncio tasks.
- Receives a fully-constructed `SingleControllerActorArgs` (cloudpickled by the driver) — the actor does no construction work of its own, because running setup inside a nested Ray actor breaks `runtime_env` resolution. Exception: `Logger` is built inside the actor, because wandb/TB backends hold a `_thread.lock` that cloudpickle can't serialize.
- On startup, rebinds `self._rollout_manager._tq_buffer = self._buffer`. `rollout_manager` and `tq_buffer` are separate fields on the args dataclass, so Ray deserializes them as two independent buffers; without the rebind, the writer and the sampler would see different copies and the sampler would never observe committed groups.
- Pump crashes propagate to the driver; in-flight rollouts drain on exit.

#### 2. `TQReplayBuffer` (`nemo_rl/algorithms/async_utils/replay_buffer.py`)

- Group-granular replay buffer with reserve/commit slot accounting.
- `start_weight` is stamped on the slot at `reserve` (dispatch); `commit` tensorizes the group into N training-shaped rows, records `end_weight`, and flips the slot ready. Because slots are appended at reserve, buffer index order equals dispatch order and `start_weight` only ever increases down the buffer — which is why `windowed` and `weight_fifo` usually pick the same groups and diverge only when rollouts finish out of order.
- Tracks `target_step` per group when the sampler assigns one at admit time (used by `in_order`).

#### 3. `RolloutManager.generate_and_push` (`nemo_rl/experience/rollout_manager.py`)

- One entry point per prompt group: reserve a buffer slot, drive the rollout via `AsyncRolloutImpl` or `AsyncNemoGymRolloutImpl`, and commit with the observed weight versions.
- `env_handles` provide per-task environments to the rollout implementations.

#### 4. Samplers (`nemo_rl/algorithms/async_utils/staleness_sampler.py`)

- Filter-only prompt-group selector over `TQReplayBuffer`. The base `PromptGroupSampler` protocol defines `admit`, `select`, and `evict`.
- `WindowedSampler`, `WeightFifoSampler`, `InOrderSampler` are the built-in policies (one per row in the [Sampler modes](#sampler-modes) table). The `custom` mode (`CustomSamplerConfig.target`) makes `create_sampler` import a user-supplied class by FQN and type-check it against `PromptGroupSampler`.

#### 5. `_rollout_pump` and `_train_pump`

- `_rollout_pump`: pulls prompts from the dataloader, calls `sampler.admit`, dispatches `RolloutManager.generate_and_push`, and honours `max_inflight_prompts` as a backpressure cap.
- `_train_pump`: `sampler.evict → sampler.select → _value_stage (PPO only) → _advantage_stage → _value_train (PPO only) → TQPolicy split API (begin_train_step / train_microbatches_from_meta / finish_train_step) → dp_client.clear_samples`.

### Coordination Flow

1. **Driver setup**: `setup_single_controller` builds the worker groups, virtual cluster, dp client, dataloader, `TQReplayBuffer`, `RolloutManager`, and weight synchronizer, and packs them into a `SingleControllerActorArgs` that the entrypoint cloudpickles into the actor.
2. **Actor startup**: `SingleControllerActor` launches `_rollout_pump` and `_train_pump` concurrently as asyncio tasks; both share the same `TQReplayBuffer` and `StalenessSampler`.
3. **Rollout pump loop**: `sampler.admit` gates dispatch against the current trainer version (returning a `target_step` for `in_order`); the pump then reserves a buffer slot, drives `RolloutManager.generate_and_push`, and commits with the observed `start_weight` / `end_weight`.
4. **Train pump loop**: `sampler.evict` drops out-of-window groups, `sampler.select` picks the next batch, `_value_stage` and `_value_train` run the critic forward and its optimizer step on a PPO run, `_advantage_stage` computes advantages, and the TQPolicy split API runs one optimizer step per RL step on GRPO, or `ppo.ppo_epochs` of them on PPO.
5. **Weight sync**: after each optimizer step the pump bumps the trainer version, clears rollout permission, calls the weight synchronizer, and re-opens the rollout pump for the next version.

## Relation to Legacy Async GRPO

The [legacy async GRPO](./async-grpo.md) (`grpo.async_grpo.enabled: true` under `run_grpo.py`) and the SC path both target the same async training problem but partition responsibilities differently:


|                  | Legacy async GRPO                        | Single-Controller                                                                    |
| ---------------- | ---------------------------------------- | ------------------------------------------------------------------------------------ |
| Entrypoint       | `run_grpo.py`                            | `run_grpo_single_controller.py`                                                      |
| Data-plane       | Direct actor RPC                         | TransferQueue (`data_plane.enabled: true` required)                                  |
| Rollout batching | Full-batch `AsyncTrajectoryCollector`    | Per-prompt `RolloutManager.generate_and_push` into a group-granular `TQReplayBuffer` |
| Staleness policy | Single knob (`max_trajectory_age_steps`) | Pluggable `StalenessSampler` (`in_order` / `weight_fifo` / `windowed` / `custom`)    |
| Batch boundary   | Sampled by target weight                 | Sampler-defined; can decouple rollout dispatch from train batch (streaming)          |


### Migrating a legacy async config

SC reads its async knobs from `async_rl:` and **requires `grpo.async_grpo: null`** (or `ppo.async_ppo: null` on a PPO run) — `run_grpo_single_controller.py` raises if a legacy block is still present, so null it out when porting rather than leaving it in place.

| Legacy `grpo.async_grpo.*` / `ppo.async_ppo.*` | SC equivalent `async_rl.*` |
| -------------------------- | -------------------------- |
| `enabled: true` | Implicit — SC is always async; use `sampler.max_lookahead_versions: 0` for sync semantics, `>= 1` for async |
| `max_trajectory_age_steps: N` | `sampler.name: in_order` with `sampler.max_lookahead_versions: N` |
| `warmup_generation_lead_steps` (PPO) | `sampler.warmup_lookahead_versions` — the lookahead to use while critic warmup is in progress |
| `recompute_kv_cache_after_weight_updates` | `recompute_kv_cache_after_weight_updates` (same) |
| `in_flight_weight_updates` | Always effectively true; `false`-equivalent behavior is not yet supported (drain-gate tracked in [issue #2625](https://github.com/NVIDIA-NeMo/RL/issues/2625)) |
| *(no legacy equivalent — matches legacy full-batch train semantics)* | `min_groups_for_streaming_train: ${grpo.num_prompts_per_step}`, or `${ppo.num_prompts_per_step}` on a PPO run |
| *(no legacy equivalent — matches legacy `max_trajectory_age + 1` batches in flight)* | `max_inflight_prompts: num_prompts_per_step × (max_lookahead_versions + 1)` |
| *(no legacy equivalent — legacy sizes its buffer to `num_prompts_per_step × max_trajectory_age_steps × 2`)* | `max_buffered_rollouts: num_prompts_per_step × (max_lookahead_versions + 1)` (tight; see the [Config → behavior map](#config--behavior-map) for per-sampler values) |

## Known Missing Features

The SC path is still under active development. Feature gaps are tracked in [issue #2625](https://github.com/NVIDIA-NeMo/RL/issues/2625). Notable items:

- Train backend: only Megatron is supported and validated; the AutoModel training path has not been tested on SC.
- Generation backend: only vLLM is supported and validated; Megatron generation, SGLang, and TRT-LLM have not been tested on SC.
- Validation is not yet supported (setup raises on `val_period > 0`, `val_at_start`, or `val_at_end`).
- (PPO) Rollout drop budgets — `async_rl.rollout_failure.max_skipped_prompts` and `max_consecutive_dropped_prompts` must both be `0`. A drop shortens the step, and the critic shards it against the configured `value.train_global_batch_size` rather than its actual size, so setup rejects a non-zero budget. The resiliency layer stays available on GRPO.
- Reward shaping and sample filtering — `overlong_filtering`, `reward_shaping`, `reward_scaling`, and `use_dynamic_sampling` are implemented on neither algorithm block, so setup rejects them rather than silently skipping the shaping.
- The `windowed` sampler has no `over_sampling_ratio` cap — over-produced groups aged past the window are evicted, wasting rollout compute.
- The drain gate in refit is not yet supported.
