# Nemotron 3 Ultra: Proof RL for IMO 2026

This guide describes the proof-generation and proof-verification reinforcement
learning stages used in our research effort toward gold-medal-level performance
at the International Mathematical Olympiad (IMO) 2026. It builds on the
[Nemotron 3 Ultra post-training guide](nemotron-3-ultra.md) and uses asynchronous
GRPO with NeMo Gym proof environments.

The release is intentionally proof-only. The recipe loads the shared policy
model adapter and exactly three NeMo Gym resources:

- `proof_judge` for generating, self-evaluating, and externally judging proofs
- `proof_genselect` for selecting between candidate proofs
- `proof_verification` for training a verifier against labelled judgements

No unrelated Gym environment is required by this recipe.

## Pipeline

The two published stages use the same proof config and launcher but different
models, data, context lengths, and cluster shapes:

```text
Ultra SFT checkpoint
        |
        v
  proof generation  -- run_proof_v1.sh
        |
        | exported Hugging Face checkpoint
        v
 proof verification -- run_verif_v1.sh
```

| Stage | Entry point | Ray nodes | Training / rollout nodes | Context | External judges |
|---|---|---:|---:|---:|---:|
| Proof generation | `run_proof_v1.sh` | 256 | 128 / 128 | 131,072 | 8 × 2-node services |
| Proof verification | `run_verif_v1.sh` | 112 | 64 / 48 | 65,536 | 6 × 2-node services |

The defaults assume four GPUs per GB200 node. Every value is an environment
variable and can be overridden for another compatible SLURM cluster.

## Released artifacts

- [`run_proof_v1.sh`](../../run_proof_v1.sh): proof-generation stage defaults
- [`run_verif_v1.sh`](../../run_verif_v1.sh): proof-verification stage
  defaults, including overlong reward shaping
- [`launch_ultra_proofs.sh`](../../launch_ultra_proofs.sh): proof-specific SLURM and
  heterogeneous-judge orchestration
- [`grpo_proof_rl_64n.yaml`](../../examples/configs/grpo_proof_rl_64n.yaml):
  shared proof-only GRPO configuration
- [`ray_het.sub`](../../ray_het.sub): dedicated heterogeneous Ray launcher for
  the proof-policy workers and external judge services

The proof resource implementations and preparation utilities live under
`3rdparty/Gym-workspace/Gym/resources_servers/proof_*`.

## Prepare the code and containers

Clone recursively so the NeMo Gym resource configs are available:

```bash
git clone --recursive https://github.com/NVIDIA-NeMo/RL.git
cd RL
```

Build the NeMo RL image and prepare the Ultra checkpoint as described in the
[base Ultra guide](nemotron-3-ultra.md#container). The proof launcher expects
the NeMo RL tree at `/opt/nemo-rl` inside that image. Set `USE_WORKTREE=1` only
when developing against a checkout whose proof-specific files must overlay the
image.

The external judge components use SGLang v0.5.8. A compatible Docker image is
available as `lmsysorg/sglang:v0.5.8`; set `HET_SERVER_CONTAINER` to this image
or to an equivalent cluster-accessible image or squashfs. It must provide the
dependencies required to run:

```bash
python3 -m sglang.launch_server --help
```

The policy and judge containers may be the same image if it contains both
dependency sets. Models, datasets, and container images are not redistributed
by this recipe; review their licenses and terms before use or redistribution.

The launcher binds each judge to `0.0.0.0` so the Ray workers can reach it on
the cluster network. Keep these endpoints on a trusted, access-controlled
cluster network; do not route or expose the judge ports to the public internet.

Prepare the starting Ultra SFT checkpoint using the
[Ultra v5-to-v4 conversion instructions](nemotron-3-ultra.md#prepare-the-starting-checkpoint).
Set `MODEL_PATH` to the converted output; neither the config nor launcher
contains a private or machine-specific model path.

## Prepare proof data

The proof RL data is published separately in
[Nemotron-Math-Proofs-v3](https://huggingface.co/datasets/nvidia/Nemotron-Math-Proofs-v3):

- use the `rl_proof` subset for proof generation
- use the `rl_verification` subset for proof verification

Export the required splits to JSONL for the launch scripts. Confirm that the
dataset terms permit your intended use and redistribution.

### Proof generation

The raw proof-generation JSONL needs a `problem` field. Convert it to the Gym
schema with the released prompt template:

```bash
uv run python \
  3rdparty/Gym-workspace/Gym/resources_servers/proof_judge/prepare_data.py \
  --input /path/to/raw_problems.jsonl \
  --output /path/to/proof_generation.train.jsonl
```

Prepare a disjoint validation file the same way. Each converted row contains a
`proof_simple_agent` reference, an OpenAI Responses-style input, and the
original problem.

The optional generation-selection resource accepts `problem`, `proof_1`,
`proof_2`, and `correct_index`:

```bash
uv run python \
  3rdparty/Gym-workspace/Gym/resources_servers/proof_genselect/prepare_data.py \
  --input /path/to/raw_pairs.jsonl \
  --output /path/to/proof_genselect.train.jsonl
```

### Proof verification

Each raw verification row must provide:

- `problem`
- `proof`
- `ground_truth_judgement`
- `ground_truth_verify_score`, with a value in `{0, 0.5, 1}`

Convert training and validation files separately:

```bash
uv run python \
  3rdparty/Gym-workspace/Gym/resources_servers/proof_verification/prepare_data.py \
  --input /path/to/raw_verification.jsonl \
  --output /path/to/proof_verification.train.jsonl
```

## Configure the cluster

Both scripts use the same public variables:

| Variable | Purpose |
|---|---|
| `CONTAINER` | NeMo RL image URI or squashfs path |
| `HET_SERVER_CONTAINER` | SGLang judge image URI or squashfs path |
| `MODEL_PATH` | Hugging Face model ID or mounted checkpoint path |
| `TRAIN_PATH`, `VAL_PATH` | Converted Gym JSONL files |
| `PERSISTENT_CACHE` | Shared cache directory visible to all Ray nodes |
| `SLURM_ACCOUNT`, `SLURM_PARTITION` | Values for your SLURM cluster |
| `EXTRA_MOUNTS` | Comma-separated `host:container` mounts for Ray-side data and models |
| `HET_SERVER_MOUNTS` | Optional mounts for a judge model stored outside the shared Hugging Face cache |
| `PROOF_JUDGE_MODEL` | Judge model ID or mounted path; defaults to `deepseek-ai/DeepSeek-Math-V2` |
| `HF_HOME`, `HF_TOKEN` | Shared Hugging Face cache override and optional token |
| `WANDB_API_KEY`, `WANDB_PROJ`, `WANDB_ENTITY` | Optional W&B logging; disabled when the API key is absent |

Published GB200 jobs use `SEGMENT_SIZE=16`, which sets NeMo RL's topology
segment and adds `--segment=16` to the Ray component. Set `SEGMENT_SIZE=` to
set `cluster.segment_size=null` and omit the SLURM option on clusters that do
not support topology segments. The published 256- and 112-node Ray shapes and
their training-node subsets are divisible by 16.

The launcher automatically mounts its results and persistent-cache directories
at the same paths inside the Ray containers. Judge containers receive only the
shared Hugging Face cache by default. Use `EXTRA_MOUNTS` for Ray-side model and
data paths, and `HET_SERVER_MOUNTS` when a judge checkpoint is a local path
outside `HF_HOME`. For example, if all policy inputs live under one shared root:

```bash
export SHARED_ROOT=/path/to/shared/storage
export EXTRA_MOUNTS="${SHARED_ROOT}:${SHARED_ROOT}"
```

Credentials are never required merely to invoke the launcher. `HF_TOKEN` is
needed only when the selected model requires authentication, and W&B logging is
optional.

## Run proof generation

Set paths for your cluster and inspect the fully resolved command first:

```bash
export CONTAINER=/path/to/nemo-rl.sqsh
export HET_SERVER_CONTAINER=/path/to/sglang.sqsh
export MODEL_PATH=/path/to/ultra_sft_checkpoint_v4
export TRAIN_PATH=/path/to/proof_generation.train.jsonl
export VAL_PATH=/path/to/proof_generation.validation.jsonl
export PERSISTENT_CACHE=/path/to/shared/cache/nemotron-3-ultra-imo
export EXTRA_MOUNTS=/path/to/shared:/path/to/shared
export SLURM_ACCOUNT=your_account
export SLURM_PARTITION=your_partition

DRY_RUN=1 ./run_proof_v1.sh
```

When the printed command and allocation match your cluster, submit it:

```bash
./run_proof_v1.sh
```

Results default to `results/nemotron-3-ultra-imo-proof-v1`. Set `RESULTS_DIR`
or `EXP_SUFFIX` to choose another shared location or run name.

The stage enables MTP speculative decoding by default. Set
`ENABLE_MTP_INFERENCE=0` if your vLLM build does not include the Ultra MTP
support used by the base recipe.

## Run proof verification

Proof generation uses the Megatron backend with consolidated saves disabled,
so each retained step contains distributed weights under
`policy/weights/iter_*`; it does not produce a ready-to-load `policy/hf`
directory. Verification therefore has an explicit Hugging Face export handoff.
This release does not claim an Ultra-specific, end-to-end conversion workflow;
use the general
[Megatron checkpoint conversion instructions](../design-docs/checkpointing.md#converting-megatron-checkpoints-to-hugging-face-format)
as a starting point, select the retained step's actual `iter_*` directory, and
validate the exported model before launching verification.

Point the verification stage at that exported model:

```bash
export MODEL_PATH=/path/to/exported-proof-policy
export TRAIN_PATH=/path/to/proof_verification.train.jsonl
export VAL_PATH=/path/to/proof_verification.validation.jsonl

DRY_RUN=1 ./run_verif_v1.sh
./run_verif_v1.sh
```

The verification wrapper reduces the context to 65,536 tokens, uses expert
parallel size 32, caps simplified-sampling concurrency at 128 trajectories,
enables an 8,192-token overlong reward-shaping buffer, and force-keeps every
fifth step in addition to the normal checkpoint policy.

## Algorithm details

The common YAML enables fully asynchronous GRPO and the simplified sampling
path. Its proof-specific stability settings include:

- asymmetric PPO ratio clipping (`0.2`, `0.28`)
- adaptive-position minimum-probability masking (`min_p_mask_type: ada-pos`)
- a truncated importance-sampling ratio of 5
- sequence packing for long proof trajectories
- optional overlong reward shaping, enabled by the verification wrapper

Judge services run as heterogeneous SLURM components. Component 0 is always the
Ray cluster; components 1 onward are independent SGLang services. `ray_het.sub`
exports each service's first hostname to the proof resources, which wait for
the corresponding endpoint before scoring trajectories.

For a small configuration check without allocating nodes, override the shapes
and keep `DRY_RUN=1`. A dry run creates output directories but does not call
`sbatch`:

```bash
NUM_ACTOR_NODES=16 \
GENERATION_NUM_NODES=8 \
HET_SERVER_COUNT=1 \
SEGMENT_SIZE=8 \
NRL_MAX_STEPS=1 \
DRY_RUN=1 \
./run_proof_v1.sh
```

Actual training remains a large-model, multi-node workload; reducing a dry-run
shape does not imply that the model will fit or perform correctly on it.

## Logs and troubleshooting

The default results directory contains:

- `checkpoints/` for training and fine-tuning checkpoints
- `logs/nemo_gym/` for Gym service logs
- `logs/proof_judge.jsonl`, `proof_verification.jsonl`, and
  `proof_genselect.jsonl` for reward diagnostics
- `ray_logs/` for Ray head and worker logs
- `slurm/` for the launcher-selected SLURM output location

Common startup failures are missing shared mounts, a judge image without
SGLang, an inaccessible judge model, or a cluster that does not implement
`--segment`. Start with `DRY_RUN=1`, confirm every path is visible inside the
appropriate image, and use `SEGMENT_SIZE=` when topology segments are not a
supported SLURM option.
