# Prefill/decode disaggregation under the SingleController: Megatron engines behind Dynamo

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run GRPO under the SingleController with a Dynamo frontend routing each rollout turn across a prefill pool and a decode pool of NeMo-RL Megatron inference engines, refit by the existing collective transport.

**Architecture:** The spec is the local design page "13. Design: prefill/decode disaggregation under the Single Controller" (docsify site in `~/dev/adlr/disagg-playground`, served at http://localhost:8765). In one paragraph: `DynamoGeneration` (the `backend: dynamo` adapter from the aggregated-Dynamo work) grows an `engine: megatron` mode. Behind it sit two new ports, `DynamoEnginePool` and `EngineRefitChannel`. The Megatron implementations build one `MegatronGeneration` per engine (its own `Policy` worker group, DP = 1, `skip_weight_load=True`) plus one CPU Ray actor per engine that runs Megatron-LM's own Dynamo adapter in external launch mode. The Megatron worker gains a small mixin that does what Megatron's `engine_service` does (Dynamo-flavoured engine class, `setup_kv_transfer(role)`, event reporter, `ready`). The controller gains a request-grain wait for idle rollouts (turn gate + outstanding counter) so no request is between the prefill and decode legs when the engines suspend; Megatron's suspend asserts that invariant. Rank offsets for the refit collective are cumulative over unequal engines and computed once in `DisaggRolePlan`.

**Tech Stack:** Python 3.12, Ray, pydantic v2, Megatron-LM (merged PR 6675: `megatron.inference.integrations.dynamo`, `StateHandoffDynamicInferenceEngine`), ai-dynamo 1.5.1 (frontend, etcd, NATS), NIXL, pytest with fakes (no GPU, no Dynamo, no Megatron import needed for the unit tests).

**Out of scope here (follow-ups):** bumping Megatron-Bridge to a main that carries the Megatron merge; the vLLM prefill/decode pool; `mcore` / `nccl_reshard` transports for two pools; NeMo-Gym through the frontend; fleet health for Dynamo pools.

**Local test command** (Mac, throwaway venv built for this branch):

```bash
cd /Users/laurad/dev/adlr/RL-pd-megatron-dynamo
export HF_HOME=/tmp/hf PYTHONPATH=$PWD
/tmp/venv-pd/bin/python -m pytest <test files> -o addopts="" -q
```

---

## File structure

| File | Responsibility |
|---|---|
| `nemo_rl/models/generation/dynamo/role_plan.py` (new) | `EngineSpec`, `EngineRecord`, `DisaggRolePlan`: engine order (decode first), per-engine world size, cumulative rank offsets, expected Dynamo components. The only place geometry is computed. |
| `nemo_rl/models/generation/dynamo/engine_pool.py` (new) | The two `Protocol`s: `DynamoEnginePool`, `EngineRefitChannel`. No implementation. |
| `nemo_rl/models/generation/dynamo/config.py` (modify) | `engine: Literal["vllm", "megatron"]`, `RolePoolConfig`, `DynamoDisaggregationConfig`, `vllm_cfg` optional for Megatron engines, Megatron-mode validation. |
| `nemo_rl/models/generation/dynamo/megatron_engine_pool.py` (new) | `DynamoMegatronAdapterActor` (Ray actor owning the `python -m megatron.inference.integrations.dynamo --engine-launch-mode external` subprocess), `build_megatron_adapter_argv`, `MegatronEnginePool`. |
| `nemo_rl/models/generation/dynamo/megatron_refit_channel.py` (new) | `MegatronRefitChannel`: per-engine fan-out of the Megatron non-colocated refit sequence with cumulative offsets and a pinned-block idle wait. |
| `nemo_rl/models/generation/dynamo/managed_runtime.py` (modify) | Accept an engine pool and expected components; split readiness out of `start()` so Megatron engines (which come up at the first refit) can be awaited later. |
| `nemo_rl/models/generation/dynamo/dynamo_generation.py` (modify) | `engine: megatron` branch; `requires_idle_rollouts_before_weight_update`; `idle_rollouts_timeout_s`; drain = `wait_idle` + `pause`; readiness after the initial refit. |
| `nemo_rl/models/generation/interfaces.py` (modify) | `requires_idle_rollouts_before_weight_update` property (default `False`). |
| `nemo_rl/experience/rollout_manager.py` (modify) | Turn gate + outstanding counter in `AsyncRolloutImpl._generate_response`; `RolloutManager.wait for idle rollouts()` / `release_turn_gate()`. |
| `nemo_rl/algorithms/single_controller.py` (modify) | `_sync_weights`: wait for idle rollouts before the drain when the generation asks for it; reopen the turn gate in `finally`. |
| `nemo_rl/models/generation/megatron/megatron_worker.py` (modify) | `attach_dynamo_adapter`, `pinned_handoff_block_count`, `dynamo_coordinator_address`; engine class + `setup_kv_transfer` + routable coordinator + event reporter when attached. |
| `nemo_rl/models/generation/megatron/megatron_generation.py` (modify) | One-line fan-outs for the three worker methods. |
| `nemo_rl/models/policy/workers/megatron_policy_worker.py` (modify) | `_generation_needs_logical_weights` also true for Dynamo-fronted Megatron engines. |
| `nemo_rl/algorithms/single_controller_utils/setup.py` (modify) | Pass the policy config and tokenizer into `DynamoGeneration` for Megatron engines; segment planning uses the largest engine; validation. |
| `examples/configs/grpo_math_1B_dynamo_pd_megatron_single_controller.yaml` (new) | Exemplar: 1 train GPU, 1 prefill engine, 1 decode engine. |
| `docs/guides/dynamo-generation.md` (modify) | Short section on the Megatron-engine prefill/decode mode. |
| Tests | `tests/unit/models/generation/test_dynamo_role_plan.py`, `test_dynamo_pd_config.py`, `test_megatron_refit_channel.py`, `test_dynamo_megatron_adapter_argv.py`; `tests/unit/experience/test_rollout_wait for idle rollouts.py`; new cases in `tests/unit/single_controller/test_single_controller_actor.py`. |

---

### Task 1: Role plan and engine records

**Files:** create `nemo_rl/models/generation/dynamo/role_plan.py`, test `tests/unit/models/generation/test_dynamo_role_plan.py`.

- [ ] Write failing tests: decode-first order; `rank_offsets(8)` for `[decode×2 @ E=2, prefill×1 @ E=4]` is `[8, 10, 12]`; `inference_world_size == 8`; `expected_components == {"backend": 2, "prefill": 1}`; mismatched GPU budget raises.
- [ ] Implement `EngineSpec(role, index, world_size, engine_cfg)`, `EngineRecord(role, world_size, instance_id, node_ip, control)`, `DisaggRolePlan(engines)` with `from_role_counts(decode_workers, decode_world_size, prefill_workers, prefill_world_size, decode_cfg, prefill_cfg)`, `rank_offsets`, `inference_world_size`, `expected_components`, `assert_fits(total_gpus)`.
- [ ] Run tests, commit.

### Task 2: Ports

**Files:** create `nemo_rl/models/generation/dynamo/engine_pool.py`.

- [ ] `DynamoEnginePool` Protocol: `start(plan) -> list[EngineRecord]`, `records`, `expected_endpoints() -> dict[str, set[str]]`, `is_alive()`, `validate(expected)`, `shutdown()`.
- [ ] `EngineRefitChannel` Protocol: `wait_idle(timeout_s)`, `pause()`, `resume()`, `flush_cache()`, `prepare(state_dict_info)`, `init_collective(ip, port, world_size, train_world_size)`, `update_weights(refit_timeout_s)`, `sender_spec`, `inference_world_size`, `client_copy()`.
- [ ] Commit.

### Task 3: Config

**Files:** modify `nemo_rl/models/generation/dynamo/config.py`; test `tests/unit/models/generation/test_dynamo_pd_config.py`.

- [ ] Tests: `engine: megatron` without `vllm_cfg` validates; `engine: vllm` without `vllm_cfg` rejects; `disaggregation` with both roles builds a plan with decode first; `prefill.megatron_cfg` may override `tensor_model_parallel_size` only from an allowlist; `context_parallel_size != 1` rejected; `kv_cache_management_mode != persist` rejected for Megatron P/D.
- [ ] Implement `RolePoolConfig`, `DynamoDisaggregationConfig`, `DynamoCfg.engine`, `DynamoCfg.disaggregation`, `DynamoCfg.megatron_root`, `DynamoCfg.nixl_transport` fields; make `vllm_cfg` optional when `engine == "megatron"`; `DynamoConfig.role_plan(policy_megatron_cfg)`.
- [ ] Run tests, commit.

### Task 4: Controller wait for idle rollouts

**Files:** modify `nemo_rl/models/generation/interfaces.py`, `nemo_rl/experience/rollout_manager.py`, `nemo_rl/algorithms/single_controller.py`; tests `tests/unit/experience/test_rollout_wait for idle rollouts.py`, `tests/unit/single_controller/test_single_controller_actor.py`.

- [ ] Tests: a turn blocks on `generate_async` → `wait for idle rollouts()` waits until it completes; new turns block while the gate is closed and proceed after `release_turn_gate()`; `wait for idle rollouts` times out and reopens the gate; `_sync_weights` calls `wait for idle rollouts` only when `requires_idle_rollouts_before_weight_update` is true, and before `drain_for_weight_update`; gate reopens even when the sync fails.
- [ ] Implement; run; commit.

### Task 5: Megatron refit channel

**Files:** create `nemo_rl/models/generation/dynamo/megatron_refit_channel.py`; test `tests/unit/models/generation/test_megatron_refit_channel.py`.

- [ ] Tests with fake per-engine generations: `init_collective` passes `train_world_size=offset_i`; `pause` orders decode then prefill and calls `suspend_for_refit` then `prepare_for_generation(tags=["weights"])`; `resume` calls `prepare_for_generation(tags=["kv_cache"])` then `resume_after_refit`, prefill first; `wait_idle` polls prefill `pinned_handoff_block_count` until 0 and raises on timeout; `update_weights` concatenates futures; `inference_world_size` sums.
- [ ] Implement; run; commit.

### Task 6: Megatron engine pool and adapter sidecar

**Files:** create `nemo_rl/models/generation/dynamo/megatron_engine_pool.py`; register the actor in `nemo_rl/distributed/actor_environments.py`; test `tests/unit/models/generation/test_dynamo_megatron_adapter_argv.py`.

- [ ] Tests: argv contains `--engine-launch-mode external`, role, component (`prefill` / `backend`), `--parent-event-host/port`, `--coordinator-host`, namespace, model; forbids a user `--engine-launch-mode local`.
- [ ] Implement `build_megatron_adapter_argv`, `DynamoMegatronAdapterActor`, `MegatronEnginePool` (per-engine `RayVirtualCluster` + `MegatronGeneration` + sidecar; `attach_dynamo_adapter` fan-out; `expected_endpoints`; `confirm_terminated` on decode death during shutdown).
- [ ] Commit.

### Task 7: Runtime and generation adapter

**Files:** modify `managed_runtime.py`, `dynamo_generation.py`, `dynamo/__init__.py`; extend `tests/unit/models/generation/test_dynamo_generation.py`.

- [ ] Tests: existing vLLM tests unchanged; with `engine: megatron` the generation builds the Megatron pool and channel, `requires_idle_rollouts_before_weight_update` is true, `drain_for_weight_update` calls `wait_idle` then `pause`, frontend readiness is awaited at `resume_after_weight_update` of the first refit, `__getstate__` keeps the channel.
- [ ] Implement; run; commit.

### Task 8: Worker hooks

**Files:** modify `megatron_worker.py`, `megatron_generation.py`, `megatron_policy_worker.py`.

- [ ] Add `attach_dynamo_adapter(parent_event_address, role, nixl_backend)`, `pinned_handoff_block_count()`, `dynamo_coordinator_address()`; in `_initialize_inference_engine` select `DynamoDynamicInferenceEngine` and call `setup_kv_transfer` when attached; in `_start_inference_coordinator` bind on the node IP when attached; after the coordinator starts, start the reporter and send `ready` from rank 0. `_generation_needs_logical_weights` covers Dynamo-fronted Megatron engines.
- [ ] No Mac-runnable test (needs `megatron.core`); covered by the functional test in Task 10. Commit.

### Task 9: Setup wiring

**Files:** modify `single_controller_utils/setup.py`.

- [ ] `_build_generation`: pass `policy_config=master_config.policy`, `tokenizer` for Megatron engines. `_build_clusters`: largest engine width for segment planning. Validation: Megatron engines require the Megatron trainer.
- [ ] Run `tests/unit/single_controller/test_setup.py` (Mac recipe); commit.

### Task 10: Exemplar, docs, functional test

**Files:** create the exemplar YAML, extend `docs/guides/dynamo-generation.md`, add `tests/functional/grpo_dynamo_pd_megatron_single_controller.sh` (GPU, not run here).

- [ ] Commit.

---

## Self-review

Spec coverage: 13c (flow) → Tasks 4, 6, 7; 13e (wait for idle rollouts) → Task 4; 13f (placement, one worker group per engine, sidecars) → Task 6; 13g (geometry) → Tasks 1, 5; 13h (config) → Task 3; 13i (startup, worker hook) → Tasks 7, 8; 13j (failures: wait for idle rollouts timeout, pin wait, decode death) → Tasks 4, 5, 6; 13o (seam) → Task 2; 13u (class map) → names used verbatim. Not covered on purpose: Megatron bump, vLLM pool, layout-aware transports.
