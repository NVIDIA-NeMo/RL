# SGLang Generation and Weight Refit

NeMo RL GRPO can train with Megatron while SGLang serves rollouts, then stream
each new policy version directly into the SGLang engines. The supported paths
do not write an intermediate checkpoint:

| Topology | Policy backend | Transfer |
|---|---|---|
| Colocated | Megatron or DTensor | CUDA IPC through the SGLang weight-update API |
| Non-colocated | Megatron | NCCL broadcast from trainer rank 0 |

Non-colocated DTensor-to-SGLang refit is not supported. The non-colocated path
is currently wired through GRPO. SGLang refit also requires
`policy.generation.refit_transport: null`; the vLLM sparse and checkpoint-engine
transports are separate implementations.

Initial engine process launch, HTTP health/cache-flush checks, router
registration, and the driver wait share the end-to-end
`engine_startup_timeout_s` deadline. The `refit_timeout_s` deadline and forced
engine quarantine described below apply to the non-colocated NCCL broadcast
path used by both reproducer recipes. Colocated refit also quarantines engines
after a reported partial IPC update, but its trainer-side gather and per-bucket
IPC waits are not yet deadline-bounded; terminate a run that stalls there.

## Reproducible Source and Image

Build from a clean clone of a public HTTPS remote. Do not build the shareable
image from a linked worktree or a checkout whose root or submodule remotes point
to local paths. Before sharing a reproducer, both the NeMo RL commit and its
pinned Gym commit must be fetchable from their configured public remotes.

```bash
export NEMO_RL_REPO="https://github.com/<public-owner>/RL.git"
export NEMO_RL_SHA="<40-character-public-commit>"

git clone --no-checkout "$NEMO_RL_REPO" nemo-rl-sglang-refit
cd nemo-rl-sglang-refit
git checkout --detach "$NEMO_RL_SHA"
git submodule sync --recursive
git submodule update --init --recursive

export NEMO_GYM_SHA=$(git rev-parse HEAD:3rdparty/Gym-workspace/Gym)
test "$(git -C 3rdparty/Gym-workspace/Gym rev-parse HEAD)" = "$NEMO_GYM_SHA"
test "$(git -C 3rdparty/Gym-workspace/Gym remote get-url origin)" = \
  "https://github.com/NVIDIA-NeMo/Gym.git"
test -z "$(git status --porcelain --ignore-submodules=none)"
```

The project container builds separate Megatron and SGLang environments from the
locked dependency groups. The local `nemo-rl` build context is required;
otherwise the Dockerfile fetches NVIDIA NeMo RL `main`. The `release` target is
also required because the Dockerfile's final target is a cache-export artifact,
not the runtime image.

Use a registry visible to every Ray node and an immutable commit tag:

```bash
export IMAGE_REPOSITORY="<registry>/<namespace>/nemo-rl-sglang-refit"
export IMAGE_REF="${IMAGE_REPOSITORY}:${NEMO_RL_SHA}"

docker buildx build \
  --target release \
  --build-context nemo-rl=. \
  --build-arg NEMO_RL_COMMIT="$NEMO_RL_SHA" \
  --build-arg NEMO_GYM_PREFETCH_CONFIGS=examples/nemo_gym/prefetch_sglang_swe1.yaml \
  --tag "$IMAGE_REF" \
  --push \
  -f docker/Dockerfile .

export IMAGE_DIGEST=$(
  docker buildx imagetools inspect "$IMAGE_REF" |
    awk '$1 == "Digest:" {print $2; exit}'
)
test -n "$IMAGE_DIGEST"
export NRL_IMAGE_REF="${IMAGE_REF}@${IMAGE_DIGEST}"
printf '%s\n' "$NRL_IMAGE_REF"
```

Use `NRL_IMAGE_REF`, including its digest, on every node. For NeMo-Gym rollouts,
the pinned Gym submodule must include its native SGLang model adapter. The
adapter sends OpenAI Responses requests through SGLang's `/generate` endpoint
while preserving the exact sampled token prefix between turns.

## Cluster and Storage Preflight

The reproducer drivers run **inside an existing Ray allocation**. They do not
allocate nodes or start Ray. Their CPU-only preflight attaches with
`address="auto"` and fails if an existing cluster cannot be found; this prevents
NeMo RL from silently starting a one-node local Ray instance.

| Run | Alive GPU nodes | GPUs per node | Total GPUs | Expected refit group |
|---|---:|---:|---:|---|
| Two-node smoke | 2 | 4 | 8 | `world_size=5 engines=4` |
| Async Gym integration | 16 | 8 | 128 | `world_size=65 engines=32` |

Before running either driver, verify:

1. Every node uses the same `NRL_IMAGE_REF` digest and the checked-out source is
   the commit stored in the image's `NEMO_RL_COMMIT`.
2. `HF_HOME`, `HF_DATASETS_CACHE`, and `NRL_MEGATRON_CHECKPOINT_DIR` are writable
   shared paths mounted at the same location on every node. Use a fresh
   Megatron-conversion cache directory for each model revision and NeMo RL
   commit. This cache is only for the initial Hugging Face-to-Megatron
   conversion; policy refits remain in memory.
3. Model and dataset snapshots are available from every node.
4. The nodes have routable cross-node TCP and NCCL interfaces.
5. `ray status` reports the full allocation before the driver starts.

For example:

```bash
export HF_HOME=/shared/path/huggingface
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export NRL_MEGATRON_CHECKPOINT_DIR="/shared/path/megatron-conversion/${NEMO_RL_SHA}"
export NRL_RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
export NEMO_GYM_COMMIT="$NEMO_GYM_SHA"
ray status
```

The preflight validates alive Ray nodes, per-node GPU capacity, and currently
available aggregate GPUs without scheduling GPU work. Image identity, shared
mounts, and network routing remain operator checks. See the
[test-suite launcher](../../tests/test_suites/README.md) for the supported Slurm
allocation flow, or
[`nrl-k8s`](../../infra/nrl_k8s/README.md) for Kubernetes Ray clusters.

## Non-Colocated Configuration

The key distinction is the dedicated generation pool. This example creates one
TP1 SGLang engine per generation GPU:

```yaml
policy:
  generation:
    backend: sglang
    refit_transport: null
    vllm_cfg: null
    sglang_cfg:
      model_path: ${policy.model_name}
      dtype: ${policy.precision}
      context_length: ${policy.max_total_sequence_length}
      tp_size: 1
      pp_size: 1
      dp_size: 1
      ep_size: 1
      random_seed: 42
      skip_server_warmup: true
      engine_startup_timeout_s: 1800
      refit_timeout_s: 1800
      quantization:
        scheme: bf16
      sglang_server_config:
        num_gpus: 4
        num_gpus_per_engine: 1
        needs_offload: false
        cpu_weight_backup: false
        pause_generation_mode: retract
        sglang_server_concurrency: 64
        weight_transfer_mode: broadcast
      sglang_router_config:
        use_external_router: false
    colocated:
      enabled: false
      resources:
        gpus_per_node: 4
        num_nodes: 1
```

`num_gpus` must equal the GPUs reserved for generation, and
`num_gpus_per_engine` must match `tp_size` while pipeline parallelism is 1. A
non-colocated generation pool should set `needs_offload: false` and
`cpu_weight_backup: false`; those memory-saver settings are for engines that
share GPUs with training.

The broadcast communicator contains trainer rank 0 plus every GPU rank in the
SGLang engines. Only trainer rank 0 broadcasts because Megatron restores full
Hugging Face tensors there; the other trainer ranks participate in the
Megatron-side tensor restoration but do not join the SGLang communicator.

## Refit Smoke Test

The two-node smoke recipe runs three synchronous GRPO steps on a public model.
It uses one 4-GPU training node and one 4-GPU generation node, producing a
five-rank SGLang refit communicator. It is manual-only because the generic
nightly launcher does not inject the immutable model, submodule, image, and
conversion-cache identities that this evidence producer requires.

Resolve the model to the public revision below and place that immutable
snapshot on the shared `HF_HOME` mount:

```bash
export NRL_MODEL_REVISION=aafeb0fc6f22cbf0eaeed126eff8be45b0360a35
export NRL_MODEL_PATH="$HF_HOME/model_snapshots/qwen2.5-math-1.5b/$NRL_MODEL_REVISION"
uvx --from huggingface-hub hf download \
  Qwen/Qwen2.5-Math-1.5B-Instruct \
  --revision "$NRL_MODEL_REVISION" \
  --local-dir "$NRL_MODEL_PATH"
test -s "$NRL_MODEL_PATH/config.json"

export NRL_IMAGE_REF="${IMAGE_REF}@${IMAGE_DIGEST}"
export NRL_RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
bash tests/test_suites/llm/grpo-qwen2.5-math-1.5b-instruct-2n4g-megatron-sglang-noncolocated-quick.sh
```

Run the script on the head of the allocated two-node Ray cluster. The driver
creates a new `runs/${NRL_RUN_ID}` evidence directory and refuses to reuse an
existing directory. It requires all of the following:

1. `train/loss` reaches step 3.
2. `NRL_SGLANG_REFIT_GROUP_READY world_size=5 engines=4` appears, with no
   conflicting topology marker.
3. At least two `NRL_SGLANG_REFIT_SUCCESS` markers appear.
4. No `NRL_SGLANG_REFIT_FAILURE` marker appears.

The recipe is intentionally synchronous. A fresh SGLang run performs an initial
baseline refit before its first generation. Each trained policy version that is
followed by another generation is then refit before that generation resumes.
The final training step is not streamed unless a later generation or validation
phase needs it, so do not interpret the marker count as one post-training refit
per completed step.

With the Slurm test launcher, pass the immutable image and shared cache into the
job environment:

```bash
EXTRA_ENV="NRL_IMAGE_REF=$NRL_IMAGE_REF NRL_RUN_ID=$NRL_RUN_ID NEMO_GYM_COMMIT=$NEMO_GYM_COMMIT NRL_MODEL_REVISION=$NRL_MODEL_REVISION NRL_MODEL_PATH=$NRL_MODEL_PATH NRL_MEGATRON_CHECKPOINT_DIR=$NRL_MEGATRON_CHECKPOINT_DIR" \
HF_HOME="$HF_HOME" \
HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
CONTAINER="$NRL_IMAGE_REF" \
MOUNTS="<shared-root>:<shared-root>" \
ACCOUNT="<account>" \
PARTITION="<partition>" \
./tools/launch \
  tests/test_suites/llm/grpo-qwen2.5-math-1.5b-instruct-2n4g-megatron-sglang-noncolocated-quick.sh
```

## Async GRPO with NeMo-Gym

The public SWE1 integration recipe exercises the complete async path. It is a
manual 16-node run and is intentionally excluded from nightly CI.

Prepare immutable model and dataset snapshots at the exact shared paths used by
the recipe:

```bash
export NRL_MODEL_REVISION=144afc2f379b542fdd4e85a1fcd5e1f79112d95d
export NRL_MODEL_PATH="$HF_HOME/model_snapshots/qwen3-30ba3b-thinking-2507/$NRL_MODEL_REVISION"
uvx --from huggingface-hub hf download \
  Qwen/Qwen3-30B-A3B-Thinking-2507 \
  --revision "$NRL_MODEL_REVISION" \
  --local-dir "$NRL_MODEL_PATH"
test -s "$NRL_MODEL_PATH/config.json"

export NRL_DATASET_REVISION=b90f74f1d0bafeec6d1f1321173f6775ba5bda2e
export SUPER_BLEND_REVISION="$NRL_DATASET_REVISION"
export SUPER_BLEND_SOURCE="$HF_HOME/superv3_source/$SUPER_BLEND_REVISION"
export SUPER_BLEND_FILLED="$HF_HOME/superv3_filled/$SUPER_BLEND_REVISION"
export SWE1_DATA_DIR="$HF_HOME/superv3_data/swe1"

uvx --from huggingface-hub hf download \
  nvidia/Nemotron-RL-Super-Training-Blends \
  --repo-type dataset \
  --revision "$SUPER_BLEND_REVISION" \
  --local-dir "$SUPER_BLEND_SOURCE"
chmod +x "$SUPER_BLEND_SOURCE/fill_placeholders.py"
"$SUPER_BLEND_SOURCE/fill_placeholders.py" \
  --input-dir "$SUPER_BLEND_SOURCE" \
  --output-dir "$SUPER_BLEND_FILLED"
mkdir -p "$SWE1_DATA_DIR"
head -n -100 "$SUPER_BLEND_FILLED/swe1.jsonl" \
  > "$SWE1_DATA_DIR/train-split.jsonl"
tail -n 100 "$SUPER_BLEND_FILLED/swe1.jsonl" \
  > "$SWE1_DATA_DIR/val-split.jsonl"
test -s "$SWE1_DATA_DIR/train-split.jsonl"
test -s "$SWE1_DATA_DIR/val-split.jsonl"
```

Then run:

```bash
export NRL_RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
bash tests/test_suites/llm/grpo-qwen3-30ba3b-thinking-swe1-16n8g-megatron-async-gym-sglang.sh
```

With the Slurm test launcher, propagate the immutable source, model, dataset,
and cache identities into the 16-node job:

```bash
EXTRA_ENV="NRL_IMAGE_REF=$NRL_IMAGE_REF NRL_RUN_ID=$NRL_RUN_ID NEMO_GYM_COMMIT=$NEMO_GYM_COMMIT NRL_MODEL_REVISION=$NRL_MODEL_REVISION NRL_MODEL_PATH=$NRL_MODEL_PATH NRL_DATASET_REVISION=$NRL_DATASET_REVISION NRL_MEGATRON_CHECKPOINT_DIR=$NRL_MEGATRON_CHECKPOINT_DIR" \
HF_HOME="$HF_HOME" \
HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
CONTAINER="$NRL_IMAGE_REF" \
MOUNTS="<shared-root>:<shared-root>" \
ACCOUNT="<account>" \
PARTITION="<partition>" \
./tools/launch \
  tests/test_suites/llm/grpo-qwen3-30ba3b-thinking-swe1-16n8g-megatron-async-gym-sglang.sh
```

The recipe uses eight training nodes and eight generation nodes. It is a
three-step integration check, not a convergence run. Its evidence validator
requires the exact `world_size=65 engines=32` topology and at least one
non-empty NeMo-Gym `train_data_step*.jsonl` artifact.

In both launcher examples, replace `<shared-root>` with a host path that
contains `HF_HOME` and `NRL_MEGATRON_CHECKPOINT_DIR`, mounted at the same path
inside the container. Omit `MOUNTS` only when the cluster runtime already
provides those identical mounts on every node.

The SGLang async configuration deliberately sets:

- `grpo.async_grpo.in_flight_weight_updates: false`. Replay collection is
  pipelined, but SGLang refit pauses generation and remains a synchronization
  barrier.
- `policy.router_replay.enabled: false`. The NeMo-Gym SGLang adapter does not
  return routed-expert traces.
- `env.nemo_gym.truncate_noncontiguous_episodes: false`. SWE1 is single-turn;
  token-prefix mismatches should fail rather than be hidden.
- A non-null SGLang `context_length` and the same
  `context_length` in the Gym adapter, so an unlimited
  Responses request consumes the remaining model context instead of SGLang's
  short default generation cap.

## Evidence Bundle

Each driver prints its fresh evidence directory. Retain the whole directory:

- `provenance.txt`: NeMo RL commit, Gym commit, immutable image reference, run
  ID, and shell-escaped driver arguments.
- `preflight.json`: required and observed Ray node/GPU capacity.
- `run.log`: complete driver log.
- `logs/`: TensorBoard events and, for Gym, rollout JSONL files.
- `metrics.json`: extracted TensorBoard metrics.
- `validation.json`: exact topology, refit-success count, terminal training
  step, and optional Gym-artifact checks.

The provenance file records the model and dataset revisions supplied to the
driver. Also retain the scheduler allocation record and the `NRL_IMAGE_REF`
digest. Runtime logs can contain hostnames, scheduler IDs, or shared-storage
paths; review and sanitize the bundle before attaching any part of it to a
public issue or pull request.

## Reading the Markers

`NRL_SGLANG_REFIT_GROUP_READY` means both sides completed communicator
construction. `"Connected all rings"` is an intermediate NCCL bootstrap trace,
not proof that the bootstrap all-gather or a weight broadcast completed.

`NRL_SGLANG_REFIT_SUCCESS` is emitted only after the weight transfer,
post-processing, and generation resume complete. Treat these as fatal:

- `NRL_SGLANG_REFIT_FAILURE`
- KV-cache invalidation failure
- NCCL system or remote error
- collective watchdog timeout
- `Failed to recv`
- connection reset

There is no disk-refit fallback. Diagnose and repair the failed in-memory path
before retrying.

## Troubleshooting

- **Failure before `GROUP_READY`:** inspect the trainer-rank-0 and engine-leader
  logs together. The rendezvous store can be healthy even when NCCL
  communicator finalization fails.
- **Timeout during a bucket:** stop the run and preserve the complete evidence
  directory. Do not resume generation from engines that may contain a partial
  weight update; restart clean engine processes before retrying.
- **Colocated IPC stall:** the colocated trainer gather and IPC waits are not
  covered by the non-colocated `refit_timeout_s` deadline. Terminate the run;
  do not assume the engines contain one complete policy version.
- **Generation fails after a successful refit:** verify dedicated engines use
  `needs_offload: false`; otherwise released weights may not have been restored
  before generation resumes.
- **Gym token-prefix assertion:** verify the Gym SGLang adapter is active
  (`responses_api_models.sglang_model`), the context limits match, and the
  configured tool format matches the model template.

See [Weight Refit](refit.md) for the transport matrix and
[Async GRPO](async-grpo.md) for replay-age and startup-barrier semantics.
