# Nemotron 3 Super Omni Image MOPD

This recipe distills a non-colocated Nemotron 3 Super Omni teacher into a
Super Omni policy over multimodal NeMo Gym trajectories. It extends the
MTP-disabled Super Omni GRPO recipe with OPD advantages, teacher resources,
and image-aware teacher log-probability computation.

## Data

Generate deterministic circle-count examples from the pinned NeMo Gym
submodule:

```bash
uv run python \
  examples/nemo_gym/nemotron-3-super-omni/prepare_circle_count_mopd_data.py \
  --out /shared/data/circle_count_train.jsonl \
  --num-samples 512
```

Each row contains one structured `input_image` data URL and an `agent_ref`
routing it to `circle_count_simple_agent`. The verifier metadata remains
outside `responses_create_params` and is not included in the model prompt.

## Launch

The production recipe uses ten nodes with eight GPUs per node:

- one vLLM generation node;
- one non-colocated teacher node;
- eight Megatron policy nodes using TP8, EP16, and CP2.

Set the paths and Slurm values required by the shared Super Omni launcher:

```bash
MODEL_PATH=/shared/models/super-omni-hf \
TEACHER_MODEL_PATH=/shared/models/super-omni-teacher-hf \
TRAIN_PATH=/shared/data/circle_count_train.jsonl \
CONTAINER=/shared/containers/nemo-rl.sqsh \
SANDBOX_CONTAINER=/shared/containers/nemo-skills-sandbox.sqsh \
PERSISTENT_CACHE=/shared/cache/nemo-rl-super-omni \
EXTRA_MOUNTS=/shared:/shared \
SLURM_ACCOUNT=<account> \
SLURM_PARTITION=<partition> \
WANDB_API_KEY=<key> \
examples/nemo_gym/nemotron-3-super-omni/run_mopd_circle_count.sh
```

`TEACHER_MODEL_PATH` is optional. When omitted, the recipe uses
`MODEL_PATH` for self-distillation. A self-distillation run should have a
near-zero mean OPD advantage while retaining non-zero token-level spread.

The recipe disables in-flight weight updates and enables vLLM encoder-cache
invalidation. This orders each encoder-cache reset after refit and before the
next image request when the vision tower is trainable.

## Three-step smoke

Use the four-node smoke before a production run:

```bash
CONFIG_PATH=examples/configs/recipes/vlm/mopd-nemotron-super-omni-120ba12b-4n8g-smoke.v1.yaml \
EXP_NAME=mopd-super-omni-circle-count-smoke \
examples/nemo_gym/nemotron-3-super-omni/run_mopd_circle_count.sh
```

The smoke runs three optimizer/refit steps. With one-step asynchronous
trajectory staleness, the third step uses trajectories generated after the
first weight update.
