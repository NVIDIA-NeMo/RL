# Async GRPO on SWE-bench (Qwen3-30B-A3B)

Launcher and recipe for agentic RL on SWE-bench via NeMo-Gym/OpenHands:
16 nodes by default (8 training + 8 generation, non-colocated), async GRPO
with staleness window 1.

| File | Purpose |
|---|---|
| `run_grpo_qwen3_30b_async_swe.sh` | SLURM launcher (submits `ray.sub`) |
| `grpo_qwen3_30b_async_swe.yaml` | Recipe config |

The launcher supports two entrypoints:

- `SC_MODE=1` (default): `examples/run_grpo_single_controller.py` —
  single-controller with the TransferQueue data plane.
- `SC_MODE=0`: `examples/nemo_gym/run_grpo_nemo_gym.py` — classic async GRPO
  (async behavior comes from the yaml's `grpo.async_grpo` block).

## Prerequisites

1. A NeMo-RL checkout (or code snapshot) containing `ray.sub` at its root —
   the launcher submits from the root it lives under.
2. An enroot container image of a recent NeMo-RL nightly.
3. The SWE train dataset (`.jsonl`).
4. The SWE sandbox images (apptainer/singularity `.sif` files for the
   swe-bench / sweap instances). **Edit the `container_formatter` lists in the
   yaml** to point at your local copies (they ship as `/path/to/...`
   placeholders).
5. A HF checkpoint to train from.

## Quick start

Run from the repo root:

```bash
ACCOUNT=<slurm_account> \
CONTAINER=/path/to/nemo-rl-nightly.sqsh \
MODEL_PATH=/path/to/hf_checkpoint \
TRAIN_DATA_PATH=/path/to/swe_train.jsonl \
EXTRA_MOUNTS=/path/to/shared_fs:/path/to/shared_fs \
bash examples/swe_bench/run_grpo_qwen3_30b_async_swe.sh
```

`EXTRA_MOUNTS` must make the model / data / `.sif` locations visible inside
the container (the default mounts only cover the repo tree and the gym
source). Add `DRY_RUN=1` to print the sbatch command and config without
submitting.

Secrets (`WANDB_API_KEY`, `HUGGINGFACE_TOKEN`, ...) are read from the calling
environment and never stored in this directory. The recommended pattern is a
small personal wrapper script that exports your site-specific paths, cluster
account, and secrets, then calls this launcher.

## Common variations

```bash
# Classic (non-single-controller) entrypoint
SC_MODE=0 bash examples/swe_bench/run_grpo_qwen3_30b_async_swe.sh

# Different training tensor parallelism
TP=2 bash examples/swe_bench/run_grpo_qwen3_30b_async_swe.sh

# streaming-2 dispatch semantics (over-generation + out-of-order consumption;
# default is streaming-1: strict repro of classic async_grpo age-1 dispatch)
OVER_SAMPLING=true FORCE_IN_ORDER=false \
bash examples/swe_bench/run_grpo_qwen3_30b_async_swe.sh

# Finer intra-step dispatch: start training once 2 prompt groups are ready
# instead of waiting for the full batch (SC only; PPS=8 by default)
MIN_PROMPT_GROUPS=2 bash examples/swe_bench/run_grpo_qwen3_30b_async_swe.sh

# Smaller / shorter run
NUM_NODES=8 NUM_GEN_NODES=4 TIME=2:0:0 MAX_NUM_STEPS=5 \
bash examples/swe_bench/run_grpo_qwen3_30b_async_swe.sh
```

The full knob list (parallelism, batch sizes, staleness, agent turn limits,
caches, ...) is documented in the launcher's header comment.

## Monitoring

The submit banner prints the experiment name, W&B target, and log location.
The driver log lands at `<run_root>/logs/slurm/<jobid>-logs/ray-driver.log`;
training progress appears there as `train step ...` / `step_metrics=...`
lines and in W&B under `train/reward` and `timing/train/*`.
