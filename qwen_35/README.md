# Qwen 3.5 runtime overlay

This directory contains the Qwen 3.5-specific recipe and file overlay for
running `Qwen/Qwen3.5-397B-A17B` on top of the Carlos/grpo-studies branch.

## Usage

Use the normal NeMo-Gym launcher and select the Qwen 3.5 recipe:

```bash
export RECIPE=qwen_35/configs/grpo_qwen35_397b_swe_openhands_async.yaml
examples/nemo_gym/launch_nemo_gym_multinode_training.sh <optional Hydra overrides>
```

When `RECIPE` points under `qwen_35/`, the launcher automatically mounts:

- `qwen_35/configs` to `/opt/nemo-rl/qwen_35/configs`
- `qwen_35/overrides` over the corresponding files under `/opt/nemo-rl`

The config carries only Qwen 3.5-family settings: vLLM parser selection,
Qwen token IDs, response-penalty defaults, and narrow Megatron/vLLM compatibility
settings. Job shape, data paths, checkpoint paths, and node counts remain normal
launcher inputs/overrides.

## Escape hatches

- `QWEN35_OVERLAY=0`: disable automatic Qwen overlay mounting.
- `QWEN35_OVERLAY=1`: force overlay mounting even if `RECIPE` is elsewhere.
- `QWEN35_OVERLAY_DIR=/path/to/overrides`: use a different overlay directory.
- `QWEN35_CONFIG_DIR=/path/to/configs`: use a different config directory.
- `NEMO_RL_QWEN35_TRUNCATE_PROMPT_TOKENS=<N|none>`: controls the prompt
  truncation fallback used by the Qwen vLLM async worker patch.
- `NEMO_RL_QWEN35_FORCE_TORCH_GDN=1`: forces the torch GatedDeltaNet fallback.

Non-Qwen runs are unaffected unless `QWEN35_OVERLAY=1` is set explicitly.
