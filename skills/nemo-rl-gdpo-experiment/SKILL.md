---
name: nemo-rl-gdpo-experiment
license: Apache-2.0
description: "Playbook for running and comparing multi-reward RL in NeMo-RL with GDPO (Group reward-Decoupled Normalization Policy Optimization) versus GRPO. Covers when multi-reward training is appropriate, wiring an environment that emits per-component rewards (reward1, reward2, ...), configuring the gdpo advantage estimator, launching a matched GDPO-vs-GRPO comparison, and diagnosing reward advantage collapse. Do NOT use for: single-reward GRPO runs, bug fixes, code review, refactoring, or generic experiment campaigns (use nemo-rl-auto-research for open-ended campaigns)."
when_to_use: run GDPO; multi-reward RL; compare GDPO vs GRPO; advantage collapse; per-reward normalization; train with several reward signals; reproduce GDPO paper result; set up math_multi_reward.
allowed-tools: Bash Read Grep Glob Edit Write
---

# nemo-rl-gdpo-experiment — multi-reward RL with GDPO

Use this when a task has **more than one reward signal** (e.g. correctness + format + structure) and you want each signal to keep its own learning signal instead of being summed into one scalar. GDPO ("Group reward-Decoupled Normalization Policy Optimization", https://arxiv.org/abs/2601.05242) is a drop-in replacement for the GRPO advantage estimator for this case.

**The problem GDPO solves.** Standard GRPO sums the reward components into `total_reward` and normalizes that sum within each prompt group. Two responses with the same total but different composition then get an *identical* advantage — "advantage collapse" — so the model gets no signal to prefer one over the other. GDPO normalizes each component independently, aggregates, then renormalizes per batch, preserving the distinctions.

See `examples/run_gdpo_grpo_advantage_demo.py` for a 10-second CPU demonstration: responses `(1,0)` and `(0,1)` share total reward `1.0`; GRPO gives both `-0.500`, GDPO gives `-0.398` vs `-0.690`.

**Safety:** This skill writes config/example files and launches training jobs that consume GPU resources. Confirm the run plan with the user before launching. Do not force-push or run destructive git operations. For long autonomous campaigns, compose with `nemo-rl-auto-research` and `nemo-rl-session-memory`; for cluster launches use `launch-nemo-rl`.

## 1. Preconditions — you need a multi-reward environment

GDPO only has an effect with **two or more reward components**. The advantage estimator (`GDPOAdvantageEstimator` in `nemo_rl/algorithms/advantage_estimator.py`) reads batch keys `reward1, reward2, ...` (discovered by `get_gdpo_reward_component_keys`); it raises `ValueError` if fewer than two are present. The rollout exposes these keys only when the environment's `step` returns a reward tensor of shape `[batch_size, num_components]` (see `nemo_rl/experience/rollouts.py` and `EnvironmentReturn.rewards` in `nemo_rl/environments/interfaces.py`).

The reference multi-reward env is `MathMultiRewardEnvironment` / `HFMultiRewardVerifyWorker` in `nemo_rl/environments/math_environment.py`, which returns three components: `reward1`=correctness, `reward2`=integer-format, `reward3`=structural-format. Confirm any candidate env emits a `[B, N]` reward tensor before proceeding.

## 2. Configure the gdpo estimator

Set the advantage estimator in the `grpo` config block. The paper's configuration:

```yaml
grpo:
  adv_estimator:
    name: "gdpo"
    normalize_rewards: true        # per-component std normalization
    use_leave_one_out_baseline: false
```

`examples/configs/gdpo_math_1B.yaml` is the ready-made recipe (Qwen2.5-1.5B + GSM8K via the `math_multi_reward` env). Read it and its parent `grpo_math_1B.yaml` before editing.

## 3. Run a matched GDPO-vs-GRPO comparison

The only legitimate way to claim GDPO helps is an apples-to-apples comparison: identical everything, estimator name the sole difference. `examples/configs/grpo_math_1B_multireward.yaml` is exactly that control (inherits `gdpo_math_1B.yaml`, flips `adv_estimator.name` to `grpo`).

```bash
# GDPO
uv run examples/run_grpo.py --config examples/configs/gdpo_math_1B.yaml
# GRPO control on the SAME multi-reward env
uv run examples/run_grpo.py --config examples/configs/grpo_math_1B_multireward.yaml
```

Before committing to a long run, sanity-check the collapse mechanism on CPU:

```bash
uv run examples/run_gdpo_grpo_advantage_demo.py
```

## 4. What to measure

The headline metric is **per-reward convergence**, not just aggregate reward. Log each component (`reward1`, `reward2`, ...) separately and compare GDPO vs GRPO curves:

- Under GRPO, expect components to move together or for low-variance components to stall (their signal is swamped by the dominant one in the sum).
- Under GDPO, expect each component to improve on its own schedule.
- Watch the per-prompt advantage spread: if distinct reward combinations produce near-identical advantages under GRPO, that is the collapse GDPO is meant to fix.

Use the `math_multi_reward` env's accuracy plus the individual `correctness/int/format` rates. For autonomous tracking, hand off to `nemo-rl-auto-research` (git + TSV ledger).

## 5. Authoring a new multi-reward environment

To run GDPO on a new task, the environment must:

1. Compute each reward signal separately in its worker (mirror `HFMultiRewardVerifyWorker`, which keeps a list-of-lists, one inner list per component).
2. Return `rewards` from `step` with shape `[B, N]` (stack components as columns; `total_reward` is derived as the row sum by the rollout layer).
3. Register the env name in `nemo_rl/distributed/ray_actor_environment_registry.py` and reference it via `data.default.env_name` in the config.

See `docs/guides/environments.md` ("Multi-reward support (GDPO)") for the contract.

## 6. Gotchas

- **Single reward → error, by design.** If you point GDPO at a single-reward env it raises; switch `adv_estimator.name` to `grpo`.
- **Per-reward weights are optional.** Aggregation is `sum_n w_n A_n`; set `grpo.adv_estimator.reward_weights` (one entry per component, ordered to match `reward1, reward2, ...`) to weight them, e.g. `[1.0, 0.5, 0.25]`. Omit for equal weighting (all 1.0). A wrong-length list raises `ValueError`. Note that any *uniform* weighting is a no-op because the final per-batch normalization cancels a global scale.
- **Reward scaling applies per component.** `reward_scaling` in the config rescales each `rewardN` as well as `total_reward` (see `grpo.py`); keep components on comparable scales or rely on GDPO's per-component normalization.
- **Final batch normalization always runs** in `GDPOAdvantageEstimator` regardless of `normalize_rewards` (which only gates the per-component std division). Account for this when reasoning about advantage magnitudes.

## 7. Where things live

- Estimator: `nemo_rl/algorithms/advantage_estimator.py` (`GDPOAdvantageEstimator`, `GRPOAdvantageEstimator`).
- Component-key helper: `get_gdpo_reward_component_keys` in `nemo_rl/algorithms/utils.py`.
- Multi-reward rollout plumbing: `nemo_rl/experience/rollouts.py`, `nemo_rl/experience/sync_rollout_actor.py`, `nemo_rl/algorithms/grpo_sync.py`.
- Reference env: `nemo_rl/environments/math_environment.py` (`MathMultiRewardEnvironment`).
- Configs: `examples/configs/gdpo_math_1B.yaml`, `examples/configs/grpo_math_1B_multireward.yaml`.
- Demo / regression: `examples/run_gdpo_grpo_advantage_demo.py`.
- Docs: `docs/guides/grpo.md` (GDPO section), `docs/guides/environments.md`.
