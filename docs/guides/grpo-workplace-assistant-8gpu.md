# Direct TransferQueue Rollouts with NeMo Gym

This guide explains the experimental vLLM-to-TransferQueue rollout writer and
shows how to run controlled legacy, shadow, and direct comparisons with the
eight-GPU Workplace Assistant recipe. It intentionally contains no benchmark
results.

## What changes

The normal synchronous TransferQueue path returns complete token-bearing
trajectories through Gym. The rollout actor flattens them and publishes the
canonical training rows:

```text
vLLM -> Gym -> rollout actor -> TransferQueue train partition
```

With `data_plane.rollout_writer.enabled=true`, NeMo RL attaches a signed sample
identity to each Gym request. The vLLM model-owner worker reserves the next
turn, verifies the historical token prefix, and writes the new token, mask,
and sampling-log-probability delta to a staging partition. The rollout actor
then verifies and assembles the canonical row:

```text
                           +-> rollout_staging --+
Gym -> vLLM model owner ---+                      +-> finalizer -> train
                           +-> cursor registry --+
```

The cursor rejects concurrent turns, stale leases, duplicate completed
requests, prefix drift, and weight-version mismatches. The finalizer verifies
turn order, identity, hashes, tensor shapes, masks, and finite log
probabilities. A missing or corrupt trajectory becomes a masked placeholder
and is excluded from reward baselines and advantage normalization.

The main implementation is in:

- `nemo_rl/experience/rollout_writer.py`: signed context, cursor state machine,
  staging verification, canonical assembly, and manifests.
- `nemo_rl/models/generation/vllm/vllm_worker_async.py`: request reservation and
  per-turn TransferQueue writes from model-owner workers.
- `nemo_rl/experience/sync_rollout_actor.py`: finalization and canonical publish.
- `nemo_rl/algorithms/grpo_sync.py`: setup guards, writer lifecycle, and invalid
  trajectory masking.
- `3rdparty/Gym-workspace/Gym/nemo_gym/`: token-only response conversion needed
  to continue multi-turn tool calls without returning sampling log probabilities.

## Supported modes

| Mode | Overrides | Purpose |
| --- | --- | --- |
| Legacy TQ | `rollout_writer.enabled=false` | Control path; rollout actor publishes rows |
| Shadow | `enabled=true`, `mode=shadow` | Stage direct rows and require tensor equality with legacy rows |
| Direct | `enabled=true`, `mode=direct` | Train from verified staged tensors |

The writer is opt-in and currently requires:

- synchronous NeMo Gym GRPO;
- the async vLLM HTTP backend;
- signed rollout contexts;
- a GRPO or GDPO advantage estimator; and
- router replay and asynchronous GRPO to be disabled.

The initial writer performs synchronous staging writes. The
`max_pending_writes_per_worker` setting is reserved for a future asynchronous
writer.

## Prerequisites

Run from the NeMo RL repository root in an interactive allocation with eight
visible GPUs.

```bash
git submodule update --init --recursive
nvidia-smi
```

Prepare the Workplace Assistant data if the recipe paths do not exist:

```bash
cd 3rdparty/Gym-workspace/Gym
printf 'hf_token: %s\n' "${HF_TOKEN:?Set HF_TOKEN first}" > env.yaml

uv run ng_prepare_data \
  "+config_paths=[resources_servers/workplace_assistant/configs/workplace_assistant.yaml]" \
  +output_dirpath=data/workplace_assistant \
  +mode=train_preparation \
  +should_download=true \
  +data_source=huggingface

cd -
```

Keep `env.yaml` local. Never commit credentials.

For Nemotron Nano v2, use copies whose system prompts end in `/no_think` so
multi-turn reconstruction keeps a stable historical prefix. Set the launch
variables to those prepared files:

```bash
export TRAIN_DATA=results/runtime/workplace_assistant_no_think/train.jsonl
export VAL_DATA=results/runtime/workplace_assistant_no_think/validation.jsonl
test -s "$TRAIN_DATA"
test -s "$VAL_DATA"
```

## Launch a comparison

The helper below fixes the workload and changes only the rollout-writer mode.
Every attempt gets a new result directory.

```bash
export EXPERIMENT_ROOT=results/tq-rollout-writer-comparison
mkdir -p "$EXPERIMENT_ROOT"
set -o pipefail

run_variant() {
  mode=$1
  output_dir=$2
  test ! -e "$output_dir"

  case "$mode" in
    legacy)
      writer_overrides=(data_plane.rollout_writer.enabled=false)
      ;;
    shadow|direct)
      writer_overrides=(
        data_plane.rollout_writer.enabled=true
        "data_plane.rollout_writer.mode=$mode"
      )
      ;;
    *)
      echo "unsupported mode: $mode" >&2
      return 2
      ;;
  esac

  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    uv run examples/nemo_gym/run_grpo_nemo_gym.py \
      --config examples/nemo_gym/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml \
      ++cluster.num_nodes=1 \
      "data.train.data_path=$TRAIN_DATA" \
      "data.validation.data_path=$VAL_DATA" \
      data.shuffle=false \
      data_plane.enabled=true \
      data_plane.backend=simple \
      data_plane.observability.enabled=true \
      "${writer_overrides[@]}" \
      grpo.seed=42 \
      grpo.max_num_steps=8 \
      grpo.num_prompts_per_step=2 \
      grpo.num_generations_per_prompt=2 \
      grpo.val_period=0 \
      grpo.val_at_start=false \
      grpo.val_at_end=false \
      policy.train_global_batch_size=4 \
      checkpointing.enabled=false \
      env.should_log_nemo_gym_responses=false \
      logger.wandb_enabled=false \
      logger.tensorboard_enabled=false \
      "logger.log_dir=$output_dir" \
      2>&1 | tee "$output_dir.console.log"
}
```

If cached worker environments are stale, set
`NRL_FORCE_REBUILD_VENVS=true` for one setup run, then unset it before a
comparison. Do not mix environment rebuild time into measured repetitions.

Run correctness gates first:

```bash
run_variant legacy "$EXPERIMENT_ROOT/gate-legacy"
run_variant shadow "$EXPERIMENT_ROOT/gate-shadow"
run_variant direct "$EXPERIMENT_ROOT/gate-direct"
```

Continue only when all jobs exit successfully, shadow mode has no tensor
mismatch, direct mode has no rejected manifest rows, an optimizer step
completes with finite loss and KL values, and all GPUs are released after each
job.

For repeated measurements, use a counterbalanced sequential order:

```bash
run_variant legacy "$EXPERIMENT_ROOT/legacy-1"
run_variant direct "$EXPERIMENT_ROOT/direct-1"
run_variant direct "$EXPERIMENT_ROOT/direct-2"
run_variant legacy "$EXPERIMENT_ROOT/legacy-2"
run_variant legacy "$EXPERIMENT_ROOT/legacy-3"
run_variant direct "$EXPERIMENT_ROOT/direct-3"
```

Discard the first step of each repetition as warm-up. Interpret timing or
transport metrics only when configuration matches, direct mode has zero
rejections, and token, turn, and HTTP-exchange distributions are comparable.

## Artifacts and diagnosis

Each run writes:

- `rollout_perf_metrics.jsonl`: per-step HTTP, Gym, TransferQueue, latency,
  CPU, and memory instrumentation;
- `rollout_writer_manifest.jsonl`: per-sample finalization or rejection status
  for shadow and direct modes; and
- the normal NeMo RL console and logger output.

Use the manifest before any performance analysis. Common rejection reasons
identify prefix mismatch, missing staging rows, invalid turn order, identity or
weight-version mismatch, hash mismatch, invalid masks, or non-finite log
probabilities. Do not treat masked rejected rows as equivalent workload.
