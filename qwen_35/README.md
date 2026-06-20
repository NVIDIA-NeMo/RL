# Qwen 3.5 runtime overlay

This directory contains the Qwen 3.5-specific recipe and runtime overlay for
running `Qwen/Qwen3.5-397B-A17B` on top of the Carlos/grpo-studies
`mlperf-training` branch.

The goal is to keep Qwen 3.5 support isolated. Non-Qwen/Nemotron runs should use
the baked container code and normal recipes unless this directory is explicitly
selected.

## Why this exists

Qwen 3.5 support currently needs more than YAML values:

- Megatron Bridge must recognize the HF architecture/model type
  `Qwen3_5MoeForConditionalGeneration` / `qwen3_5_moe`.
- The Megatron setup path needs Qwen 3.5 compatibility for the GatedDeltaNet
  code path used by the current container stack.
- The vLLM async worker needs Qwen 3.5 prompt/prefix repair and parser handling
  for multi-turn tool-use trajectories.
- NeMo-Gym needs Qwen-specific tolerance for token-contiguity issues caused by
  reasoning/tool-call retokenization.
- The SWE reward path should not apply Nemotron/Nano-specific scalar response
  penalties that can zero valid Qwen trajectories.

Hydra config can select model-family values, but it cannot create Pyxis/container
file mounts. Since these compatibility files must override files inside the baked
container at `/opt/nemo-rl`, a small launcher hook is required.

## How selection works

Use the normal NeMo-Gym launcher and select the Qwen 3.5 recipe:

```bash
export RECIPE=qwen_35/configs/grpo_qwen35_397b_swe_openhands_async.yaml
examples/nemo_gym/launch_nemo_gym_multinode_training.sh <optional Hydra overrides>
```

When `RECIPE` points under `qwen_35/`,
`examples/nemo_gym/launch_nemo_gym_multinode_training.sh` automatically mounts:

- `qwen_35/configs` to `/opt/nemo-rl/qwen_35/configs`
- every file under `qwen_35/overrides` to its matching path under `/opt/nemo-rl`

For example:

```text
qwen_35/overrides/nemo_rl/models/megatron/community_import.py
  -> /opt/nemo-rl/nemo_rl/models/megatron/community_import.py
```

This lets the selected config and only that config bring the Qwen 3.5 runtime
patches into the container.

## What the config changes

`qwen_35/configs/grpo_qwen35_397b_swe_openhands_async.yaml` inherits:

```text
examples/nemo_gym/grpo_qwen3_235b_swe_openhands_async.yaml
```

It intentionally carries only model-family-specific settings:

- Qwen 3.5 vLLM tool/reasoning parser defaults.
- Qwen 3.5 special token IDs used by token-aware checks.
- Qwen-safe response-penalty defaults.
- Narrow Megatron/vLLM compatibility values that should travel with Qwen 3.5.

It does **not** bake in cluster size, data paths, checkpoint paths, experiment
names, walltime, or node allocation. Those remain normal launcher environment
variables and Hydra overrides.

## What the overlay changes

The current overlay files are:

- `nemo_rl/models/megatron/qwen35_bridge_patch.py`
  - Registers Qwen 3.5 MoE with Megatron Bridge.
- `nemo_rl/models/megatron/community_import.py`
  - Imports the Qwen 3.5 bridge patch during Megatron setup.
- `nemo_rl/models/megatron/setup.py`
  - Adds Qwen 3.5 Megatron setup compatibility, including GatedDeltaNet handling.
- `nemo_rl/models/policy/workers/megatron_policy_worker.py`
  - Carries the policy-worker compatibility used by the Qwen 3.5 runs.
- `nemo_rl/models/generation/vllm/__init__.py`
  - Ensures the patched vLLM generation path is selected.
- `nemo_rl/models/generation/vllm/vllm_worker_async.py`
  - Handles Qwen 3.5 parser, prefix repair, prompt truncation, and related
    multi-turn generation behavior.
- `nemo_rl/environments/nemo_gym.py`
  - Handles Qwen-specific token-contiguity behavior for NeMo-Gym rollouts.

These are intentionally under `qwen_35/overrides` instead of directly modifying
the base tree. That makes it clear which files are Qwen 3.5-specific and keeps
base grpo-studies behavior visible.

## Escape hatches

- `QWEN35_OVERLAY=0`: disable automatic Qwen overlay mounting.
- `QWEN35_OVERLAY=1`: force overlay mounting even if `RECIPE` is not under
  `qwen_35/`.
- `QWEN35_OVERLAY_DIR=/path/to/overrides`: use a different overlay directory.
- `QWEN35_CONFIG_DIR=/path/to/configs`: use a different config directory.
- `NEMO_RL_QWEN35_TRUNCATE_PROMPT_TOKENS=<N|none>`: controls the prompt
  truncation fallback used by the Qwen vLLM async worker patch.
- `NEMO_RL_QWEN35_FORCE_TORCH_GDN=1`: forces the torch GatedDeltaNet fallback.

## Safety boundary

Non-Qwen runs are unaffected unless `QWEN35_OVERLAY=1` is set explicitly. The
default behavior is:

- recipes under `qwen_35/`: Qwen config and Qwen overlay are mounted.
- all other recipes: no Qwen overlay is mounted.

This is the cleanest current compromise: selecting the Qwen 3.5 config is enough
for users, while the only base-tree code change is the small launcher hook needed
to make container mounts possible.
