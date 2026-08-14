# Context compaction

Context compaction bounds the visual context of a multimodal, multi-turn
rollout without changing its logical reward or advantage semantics. The current
implementation compacts only at model-turn boundaries: every `K` completed
actions it starts a new chunk and materializes the prompt with only the most
recent `N` image groups. Text and reasoning history remain present.

This feature is work in progress. It has been validated with a deterministic
multimodal test agent, including 100-turn generation and synchronous and
asynchronous GRPO training, and with OSWorld running real Ubuntu desktop VMs
through the Cell 2 OpenSandbox SDK.

## Code and validated runtime

The public draft PRs are:

- NeMo RL: [NVIDIA-NeMo/RL#3642](https://github.com/NVIDIA-NeMo/RL/pull/3642)
- NeMo Gym: [NVIDIA-NeMo/Gym#2555](https://github.com/NVIDIA-NeMo/Gym/pull/2555)

### Provenance

The generic Context Compaction implementation comes from Ali Roshan Ghias's
original [NeMo RL branch](https://gitlab-master.nvidia.com/aroshanghias/nemo-rl/-/tree/aroshanghias/context-compaction-v2-clean)
and [matching NeMo Gym branch](https://gitlab-master.nvidia.com/aroshanghias/Gym/-/tree/aroshanghias/context-compaction-v2-clean-gym).
His signed implementation is preserved through
[commit `42a65427`](https://github.com/NVIDIA-NeMo/RL/commit/42a65427dce038f57f7fd8eed6a24f6a8ce72c2b).

The initial OSWorld environment and Cell 2 OpenSandbox integration was
developed by Terry Kong in
[commit `275f0ae9`](https://github.com/NVIDIA-NeMo/Gym/commit/275f0ae94c98c1a484658a5c995b97dce1bb1b4b).

This draft connects those components for OSWorld GRPO, hardens the training and
evaluation pipeline, and validates Context Compaction in a real computer-use
environment. It does not claim Context Compaction or the initial OSWorld
environment as new contributions.

The NeMo RL draft pins the matching Gym draft as a submodule. Check out and
initialize the publication branch with:

```bash
git clone https://github.com/jinglinglingling/RL.git nemo-rl-osworld
cd nemo-rl-osworld
git checkout feature/osworld-grpo-training-eval-signed
git submodule sync --recursive
git submodule update --init --recursive
```

The validated internal runtime is:

| Item | Value |
|---|---|
| Slurm account | `coreai_dlalgo_nemorl` |
| Slurm partition | `batch` |
| Container | `/lustre/fs1/portfolios/coreai/users/aroshanghias/omni-main-migration/containers/cuda-dl-base-26.03-cuda13.2-devel-ubuntu24.04.sqsh` |
| Checkpoint | `/lustre/fs1/portfolios/coreai/users/aroshanghias/checkpoints/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` |
| Public model ID | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` |
| NeMo Gym submodule | `d17601d9d841a1b3f576cb4180d81ff791b29253` |
| W&B project | `nvidia/nemo-rl-context-compaction` |

Use writable cache and venv directories belonging to your user. Do not share
another run's mutable venv directory.

## Run OSWorld GRPO

The main entry points are:

| Purpose | File |
|---|---|
| Training entry point | `examples/nemo_gym/run_grpo_nemo_gym.py` |
| OSWorld GRPO recipe | `examples/nemo_gym/grpo_nemotron_omni_30ba3b_osworld_cc.yaml` |
| Convert an OSWorld JSONL | `examples/nemo_gym/prepare_osworld_context_compaction_data.py` |
| Build the Stable251 split | `examples/nemo_gym/prepare_osworld_stable_cc_split.py` |
| Independent checkpoint evaluation | `examples/nemo_gym/submit_osworld_cc_eval.sh` |
| Gym OSWorld agent config | `3rdparty/Gym-workspace/Gym/responses_api_agents/nemotron_osworld/configs/nemotron_osworld_cc.yaml` |
| Gym OSWorld environment config | `3rdparty/Gym-workspace/Gym/resources_servers/osworld/configs/osworld.yaml` |
| Cell 2 OpenSandbox config | `3rdparty/Gym-workspace/Gym/resources_servers/osworld/configs/opensandbox_osworld.yaml` |

### Prepare data

To convert any NeMo Gym OSWorld task manifest:

```bash
python examples/nemo_gym/prepare_osworld_context_compaction_data.py \
  --input <osworld-tasks.jsonl> \
  --output <osworld-cc.jsonl> \
  --num-repeats 1
```

For the larger stability-filtered split:

```bash
python examples/nemo_gym/prepare_osworld_stable_cc_split.py \
  --train-input <osworld-train-tasks.jsonl> \
  --eval-input <osworld-eval-tasks.jsonl> \
  --output-dir <stable251-output>
```

`Tiny32` is a fixed 32-task development subset used for inexpensive
end-to-end debugging and learning-signal checks. It is not a generalization
benchmark. `Stable251` contains 251 training tasks that do not require a proxy
and have a low likelihood of environment changes. It is paired with a
non-overlapping 71-task held-out split that excludes tasks with a high
likelihood of environment changes.

### Configure and submit training

The recipe is configured through environment variables so the same file can be
used for a smoke test or a larger run:

```bash
export OPENSANDBOX_DOMAIN=<cell-2-opensandbox-domain>
export OPENSANDBOX_API_KEY=<cell-2-opensandbox-api-key>
export OSWORLD_POOL_REF=osworld-kvm

export NANO_OMNI_MODEL_NAME=<path-to-Nemotron-3-Nano-Omni-checkpoint>
export OSWORLD_GRPO_TRAIN_DATA=<train.jsonl>
export OSWORLD_GRPO_VAL_DATA=<heldout.jsonl>
export OSWORLD_RESULTS_DIR=<results-directory>
export CHECKPOINT_DIR=${OSWORLD_RESULTS_DIR}/checkpoints
export NEMO_GYM_EXTRA_ROOTS=${PWD}/3rdparty/Gym-workspace/Gym
export HF_HOME=${OSWORLD_RESULTS_DIR}/hf-home

export NUM_NODES=2
export OSWORLD_MAX_STEPS=15
export OSWORLD_MAX_MODEL_LEN=16384
export OSWORLD_NUM_PROMPTS_PER_STEP=8
export OSWORLD_NUM_GENERATIONS=8
export OSWORLD_NEMO_GYM_NUM_WORKERS=64
export OSWORLD_MAX_PARALLEL_ROLLOUTS=64
export OSWORLD_CC_MAX_TOTAL_TOKENS=16384
export GRPO_MAX_NUM_STEPS=32
export CHECKPOINTING_ENABLED=true
export WANDB_ENABLED=true

export CONTAINER=<path-to-pyxis-container.sqsh>
export MOUNTS=/lustre:/lustre
export BASE_LOG_DIR=${OSWORLD_RESULTS_DIR}/slurm
mkdir -p "${BASE_LOG_DIR}"

export COMMAND="cd ${PWD} && uv run --locked \
  examples/nemo_gym/run_grpo_nemo_gym.py \
  --config examples/nemo_gym/grpo_nemotron_omni_30ba3b_osworld_cc.yaml"

sbatch --parsable \
  --nodes="${NUM_NODES}" --gres=gpu:8 \
  --account=<slurm-account> --partition=batch --time=04:00:00 \
  --job-name=osworld-grpo \
  --output="${BASE_LOG_DIR}/slurm-%j.out" \
  --export=ALL \
  ray.sub
```

Start with fewer workers and a short `OSWORLD_MAX_STEPS` when validating a new
OpenSandbox deployment. `OSWORLD_CC_KEEP_LAST_IMAGE_GROUPS` controls `N`, and
`OSWORLD_CC_ACTIONS_PER_CHUNK` controls `K`.

### Evaluate a checkpoint independently

The evaluation wrapper runs the same OSWorld environment with optimizer steps
disabled:

```bash
export EVAL_NAME=step-32
export EVAL_CHECKPOINT_PATH=<checkpoint-directory>
export OSWORLD_GRPO_VAL_DATA=<heldout.jsonl>
export EVAL_MAX_STEPS=15
export EVAL_NUM_WORKERS=32
export RESULTS_DIR=<evaluation-results-directory>
export SBATCH_ACCOUNT=<slurm-account>
export SBATCH_PARTITION=batch

bash examples/nemo_gym/submit_osworld_cc_eval.sh
```

The wrapper also requires the OpenSandbox, model, container, mount, and Slurm
environment variables shown above. Set `EVAL_CHECKPOINT_PATH` to an empty
string to evaluate the base model.

## Deterministic smoke configuration

The runnable recipe is
[`examples/nemo_gym/grpo_nemotron_omni_30ba3b_scripted_multiturn_cc.yaml`](../../examples/nemo_gym/grpo_nemotron_omni_30ba3b_scripted_multiturn_cc.yaml).
It loads the generic vLLM Responses model and the deterministic scripted
multimodal agent from the pinned Gym checkout.

The compaction policy is configured on the Gym agent:

```yaml
visual_history:
  enabled: true
  shadow_only: false
  policy:
    type: recency
    config:
      protect_initial_context: true
      keep_last_image_groups: 1  # N
      keep_all_text: true
      image_omission_marker: "[Earlier image omitted]"
  schedule:
    type: turn_chunked_recency
    actions_per_chunk: 2         # K
```

The supplied configuration is
`responses_api_agents/scripted_multimodal_agent/configs/scripted_multimodal_agent_recency_k2_nonzero_advantage.yaml`.
It uses `K=2` and `N=1`. For an environment that emits one screenshot per
turn, one image group is one screenshot. If an observation contains several
images that must remain atomic, they may form one image group.

The first `K` actions run without compaction. Compaction occurs before the next
model call, never inside a model turn. The materialized view is then frozen for
the rest of that chunk unless a configured token, image, or vision-token guard
closes the chunk early. Guard-triggered closes are recorded in the trace.

Training must explicitly opt in:

```yaml
grpo:
  context_compaction_training:
    enabled: true
```

One logical rollout may become several physical training traces. All physical
traces retain the rollout's advantage, while loss normalization, scheduler
progress, and the optimizer boundary continue to use logical-rollout
semantics.

## Dummy data

Set `CC_SMOKE_DATA_PATH` to a NeMo Gym JSONL manifest. A minimal row is:

```json
{"responses_create_params":{"input":[{"role":"user","content":"For every turn, inspect the supplied screen and output ACK."}],"max_output_tokens":128},"agent_ref":{"type":"responses_api_agents","name":"scripted_multimodal_agent"},"context_compaction_contract_version":2,"context_compaction_group_id":"cc-smoke-0000","context_compaction_task_id":"cc-smoke-task","context_compaction_rollout_index":0,"context_compaction_attempt_index":0}
```

The manifest used for the existing internal smokes is:

```text
/lustre/fs1/portfolios/coreai/users/aroshanghias/context-compaction-v2-runs/data/scripted_multimodal_ack_128_repeated_100.jsonl
```

Each generated rollout receives a caller-owned group, task, rollout, and
attempt identity. Real environments should preserve the same stable identity
fields so retries can be deduplicated safely.

## Run a synchronous smoke

From the repository root on a Slurm login node:

```bash
export SBATCH_ACCOUNT=coreai_dlalgo_nemorl
export CONTAINER=/lustre/fs1/portfolios/coreai/users/aroshanghias/omni-main-migration/containers/cuda-dl-base-26.03-cuda13.2-devel-ubuntu24.04.sqsh
export MOUNTS=/lustre:/lustre
export NANO_OMNI_MODEL_NAME=/lustre/fs1/portfolios/coreai/users/aroshanghias/checkpoints/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
export CC_SMOKE_DATA_PATH=/lustre/fs1/portfolios/coreai/users/aroshanghias/context-compaction-v2-runs/data/scripted_multimodal_ack_128_repeated_100.jsonl

export RUN_ROOT=/lustre/fs1/portfolios/coreai/users/${USER}/nemo-rl-cc
export UV_CACHE_DIR_OVERRIDE=${RUN_ROOT}/uv-cache
export UV_PYTHON_INSTALL_DIR=${RUN_ROOT}/uv-python
export NEMO_RL_VENV_DIR=${RUN_ROOT}/nemo-rl-venvs
export NEMO_GYM_VENV_DIR=${RUN_ROOT}/nemo-gym-venvs
export HF_HOME=${RUN_ROOT}/hf-home
export BASE_LOG_DIR=${RUN_ROOT}/slurm-logs
mkdir -p "${RUN_ROOT}" "${BASE_LOG_DIR}"

# Set WANDB_API_KEY in the environment; do not put it in the command or YAML.
export COMMAND="cd ${PWD}
uv run --locked \
  examples/nemo_gym/run_grpo_nemo_gym.py \
  --config examples/nemo_gym/grpo_nemotron_omni_30ba3b_scripted_multiturn_cc.yaml \
  cluster.num_nodes=2 \
  logger.wandb_enabled=true \
  logger.wandb.entity=nvidia \
  logger.wandb.project=nemo-rl-context-compaction \
  logger.wandb.group=shared-cc-smoke \
  logger.wandb.name=cc-sync-k2-n1"

sbatch --parsable \
  --nodes=2 --exclusive --gres=gpu:8 \
  --account=${SBATCH_ACCOUNT} --partition=batch --time=02:00:00 \
  --job-name=cc-sync-k2-n1 \
  --output=${BASE_LOG_DIR}/slurm-%j.out \
  ray.sub
```

The default topology is Megatron TP=8, EP=16, expert-TP=1, CP=1, PP=1 and
vLLM TP=2. Both engines are colocated across two 8-GPU nodes. The recipe keeps
`overlap_grad_reduce=false` and `overlap_param_gather=false`; enabling
parameter-gather overlap caused TMPE drift after optimizer/refit cycles in the
validated setup.

The recipe also uses the real checkpoint (`load_format` is not `dummy`), router
replay, raw vLLM logprobs, string-formatted chat content, matching
`enable_thinking=true` and `truncate_history_thinking=false` template kwargs,
FP32 Mamba SSM cache state, and disabled chunked prefill. These settings are
part of the validated Nemotron Omni setup; preserve them when first integrating
a new environment.

## Run an asynchronous smoke

Use a third node for non-colocated vLLM and add these overrides to `COMMAND`:

```text
grpo.async_grpo.enabled=true
grpo.async_grpo.max_trajectory_age_steps=1
policy.generation.colocated.enabled=false
cluster.num_nodes=3
logger.wandb.name=cc-async-k2-n1
```

Submit `ray.sub` with `--nodes=3`. Async training retains complete logical
comparison groups in replay, validates policy-version provenance, and
deduplicates identical retries before physical trace materialization.

## Generation-only trace inspection

Before training against a new environment, run generation only:

```text
env.nemo_gym.is_trajectory_collection=true
checkpointing.enabled=false
```

The experiment directory will contain `trajectory_collection.jsonl`. Useful
inspection commands are:

```bash
jq . logs/<run>/exp_*/trajectory_collection.jsonl | less -R
jq '{reward, response: .response.context_compaction_contract, bundle: .nemo_rl_trace_bundle}' \
  logs/<run>/exp_*/trajectory_collection.jsonl | less -R
```

For training runs, inspect `train_data_step*.jsonl` and the W&B metrics
`context_compaction/logical_rollouts`, `physical_traces`, `physical_rows`,
`padding_rows`, and `eligible_action_tokens`. Async runs additionally report
retry, replay-buffer, scheduler-increment, and optimizer-step metrics.

Existing dummy-environment results are available at
<https://wandb.ai/nvidia/nemo-rl-context-compaction/workspace>.

## Current limitations

- The built-in non-identity policy compacts images while retaining all text;
  `keep_all_text=false` is not implemented.
- Compaction inside a model turn is unsupported.
- Standard token-level GRPO is supported. Sequence-level importance ratios,
  sequence-level loss reduction, sequence-level TIS, KL-in-reward rewriting,
  and sequence-level logprob-error masking fail closed when CC training is
  enabled.
- The initial training implementation requires the Megatron policy backend and
  does not support the data-plane/TQ training path.
- The deterministic environment validates correctness, not task-learning
  quality. OSWorld validation provides a real computer-use learning signal,
  but larger held-out generalization experiments are still in progress.
