# RL-v2 image-tools port

This branch keeps `RL-v2-rebase`'s native multi-turn and multimodal rollout
stack and adds the Gym image-tools agent plus the PivotRL verifier.

## Tools

The agent supports `image_zoom_in_tool`, `image_crop_tool`,
`image_rotate_tool`, `image_flip_tool`, `image_diff_tool`,
`image_side_by_side_tool`, `image_overlay_tool`, `count_objects_tool`,
`find_color_tool`, and `color_at_tool`.

## Local executor smoke

```bash
PYTHONPATH=. python tools/smoke_image_tools_gym.py \
  examples/data/image_tools_smoke.jsonl
```

The Python environment must contain Pillow, NumPy, and SciPy.

## One-step GRPO sample

The sample launcher creates a deterministic image and 16 JSONL rows, then
uses the Super Omni launcher for one synchronous GRPO update:

```bash
DRY_RUN=true \
  examples/nemo_gym/nemotron-3-super-omni/image_tools_sample_launch.sh
```

Inspect the printed command, then omit `DRY_RUN=true` to submit. Override
`MODEL_PATH`, `CONTAINER`, `SANDBOX_CONTAINER`, `PERSISTENT_CACHE`, or Slurm
variables when the defaults do not apply.

## Super Omni PivotRL

The ignored data files must exist at:

```text
3rdparty/Gym-workspace/Gym/resources_servers/image_tools/data/train.jsonl
3rdparty/Gym-workspace/Gym/resources_servers/image_tools/data/validation.jsonl
```

Validate them and inspect the launch:

```bash
python tools/preflight_pivot_config.py \
  3rdparty/Gym-workspace/Gym/resources_servers/image_tools/configs/image_tools_pivot.yaml

DRY_RUN=true \
  examples/nemo_gym/nemotron-3-super-omni/image_tools_pivot_launch.sh
```

The PivotRL recipe runs the `tool_simulation_agent` for one decision, without
executing the tool. The verifier scores tool identity, image target, and the
primary argument for the tool family. Zoom and crop are treated as equivalent
region-inspection decisions by default.
