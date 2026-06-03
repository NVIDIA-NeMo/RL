# Scope: multi-reward NeMo Gym environment + bridge extension for GDPO

Goal: let GDPO train on a **NeMo Gym** environment (e.g. tool-calling) by surfacing
per-component rewards (`reward1, reward2, ...`) through the Gym→NeMo-RL bridge, the
same way the native `math_multi_reward` environment already does. This unlocks the
paper's tool-calling/coding results, not just math.

## Status

- **Done (RL side, this branch):** the bridge helper `extract_reward_components`
  (`nemo_rl/environments/nemo_gym.py`), the multi-reward assembly in
  `run_async_nemo_gym_rollout` (`nemo_rl/experience/rollouts.py`), the template config
  `examples/nemo_gym/gdpo_multireward.yaml`, and the helper unit test
  (`tests/unit/environments/test_nemo_gym.py::test_extract_reward_components`).
- **Pending (Gym side):** Part A below — the Gym verifier must emit `reward_components`
  (and the `BaseVerifyResponse` field in the Gym repo). Until that lands, the RL-side
  code is inert (single-reward results return `None` and fall back to the scalar path).

## Current state (verified)

- **Native path already supports multi-reward.** `run_multi_turn_rollout`
  (`nemo_rl/experience/rollouts.py`) infers `number_of_rewards` from
  `env_output.rewards.ndim >= 2` (~L475–489), accumulates a `[B, N]` tensor, and the
  result builder (~L974–989) stacks `reward1..rewardN` keys that
  `get_gdpo_reward_component_keys` discovers. GDPO consumes these directly.
- **Gym path is single-scalar only.** The NeMo Gym bridge
  (`nemo_rl/environments/nemo_gym.py`) is an integration wrapper: Gym owns the
  environment/agent/verifier and returns a result whose
  `_postprocess_nemo_gym_to_nemo_rl_result` passes through as `full_result`. The Gym
  rollout `run_async_nemo_gym_rollout` then sets the reward at **`rollouts.py:1181`**:
  `"total_reward": r["full_result"]["reward"]` — one scalar. No `reward1..N`, so GDPO
  has nothing to decouple and would raise `ValueError` ("requires multiple reward
  components").

## Work breakdown

### A. Gym side (NVIDIA-NeMo/Gym repo) — emit multiple scores
The verifier currently produces one aggregate `reward`. Make it also emit the
component scores it already computes internally, e.g. add to the Gym result payload:

```jsonc
"reward": 0.5,                       // keep the existing aggregate (back-compat)
"reward_components": {               // NEW: named, ordered components
  "correctness": 1.0, "schema_valid": 0.0, "format": 1.0
}
```

Implement in the environment's resource server / verifier (`resources_servers/<env>/`).
A natural first target is a tool-calling task scored on: (1) task correctness,
(2) tool-call schema validity, (3) output format. Ship a small `data/*.jsonl` and a
`configs/*.yaml` so it runs via `ng_run` / `ng_collect_rollouts`.

### B. Bridge side — `nemo_rl/environments/nemo_gym.py`
`_postprocess_nemo_gym_to_nemo_rl_result` already forwards the raw result as
`full_result`, so `reward_components` is available downstream without change. Optional
but cleaner: extract and normalize the components here into a fixed-order list so the
rollout layer doesn't have to know component names.

### C. Gym rollout — `nemo_rl/experience/rollouts.py` (`run_async_nemo_gym_rollout`)
This is the core change. At the reward-assembly site (currently `:1181`):

1. If `full_result` carries `reward_components`, write per-sample `reward1..rewardN`
   onto each sample state using the exact key convention
   `get_gdpo_reward_component_keys` matches (regex `reward\d+$`), in a **stable order**
   (sort component names once, map to indices). Otherwise fall back to scalar-only
   (preserves current single-reward behavior).
2. Set `total_reward` = the Gym aggregate (or the component sum — pick one and document
   it; the native path uses the sum).
3. Ensure the Gym path's batch assembly (~`:1215`) runs the same `reward_component_keys`
   stacking the native builder does at ~L974–989 — factor that block into a shared
   helper and call it from both paths to avoid drift.

### D. Config + example
Add `examples/nemo_gym/gdpo_<task>.yaml` mirroring an existing nemo_gym recipe but with:

```yaml
grpo:
  adv_estimator:
    name: "gdpo"
    normalize_rewards: true
    use_leave_one_out_baseline: false
```

Run via `examples/nemo_gym/run_grpo_nemo_gym.py`; smoke-test with
`examples/nemo_gym/run_nemo_gym_single_node_sanity_tests.sh`.

### E. Tests
- Unit: feed a fake `nemo_gym_result` with `reward_components` through the rollout
  assembly; assert `reward1..rewardN` appear with correct shape and that `total_reward`
  matches the chosen aggregation. Add a single-reward fake to assert the fallback path
  is unchanged.
- Reuse the collapse assertion from `examples/run_gdpo_grpo_advantage_demo.py` as the
  algorithm-level regression.

## Risks / decisions to confirm
- **Component ordering must be deterministic** across samples and steps, or `reward1`
  won't consistently mean the same signal. Sort component names once at the env/bridge
  boundary.
- **Missing components.** Different Gym envs in one batch may expose different
  components; the native builder already pads absent components with `0.0` (L989) —
  match that, and confirm 0.0 is a sane neutral baseline per component.
- **Aggregate vs sum for `total_reward`.** GRPO control runs read `total_reward`; keep
  it consistent with the native path (sum of components) so GDPO-vs-GRPO stays
  apples-to-apples.
- **Cross-repo change.** Part A lives in NVIDIA-NeMo/Gym; B–E live in NeMo-RL. Land the
  Gym change first (or behind a flag) so the bridge can rely on the field.
