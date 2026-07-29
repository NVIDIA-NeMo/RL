# Managed Dynamo on Slurm

The Dynamo backend uses a derived image so its Python 3.12 dependency set does
not change NeMo RL's Ray or vLLM dependencies. Build the standard NeMo RL
release image and derive the Dynamo image in one command:

```bash
IMAGE=registry.example.com/nemo-rl-dynamo:dev \
PLATFORM=linux/amd64 \
bash docker/dynamo/build.sh
```

The derived image contains:

- `ai-dynamo[vllm]==1.3.0.post1` under `/opt/dynamo_venv`;
- vLLM 0.23.0 with the vLLM PR #44814 backport;
- etcd v3.5.21;
- `nats-server` v2.11.6 with JetStream support.

Convert or import that OCI image to the squashfs format used by your Slurm
cluster. Then submit the one-node, two-GPU smoke:

```bash
CONTAINER=/shared/images/nemo-rl-dynamo.sqsh \
SLURM_ACCOUNT=my-account \
SLURM_PARTITION=my-partition \
bash examples/slurm/dynamo/launch.sh
```

The smoke assigns one GPU to the DTensor policy and one to a TP1 managed
Dynamo worker. It runs two GRPO steps, including NCCL weight refit, KV-cache
invalidation, and deterministic service shutdown.

The larger SWE acceptance recipe uses four 4-GPU training nodes and two TP4
Dynamo inference nodes. All site-specific paths are supplied by the launcher:

```bash
CONTAINER=/shared/images/nemo-rl-dynamo.sqsh \
MODEL_PATH=/shared/models/nemotron-nano-v3.5 \
TRAIN_PATH=/shared/data/swe-train.jsonl \
VAL_PATH=/shared/data/swe-validation.jsonl \
SIF_FORMATTERS='["/shared/swe/{instance_id}.sif"]' \
SANDBOX_CONTAINER=/shared/images/nemo-skills-sandbox.sqsh \
SLURM_ACCOUNT=my-account \
SLURM_PARTITION=my-partition \
WANDB_API_KEY=... \
EXTRA_MOUNTS=/shared:/shared \
bash examples/swe_bench/run_grpo_nano_v3_5_swe_dynamo_hsg_r2_wandb.sh
```

Set `DRY_RUN=1` to print the `sbatch` invocation without submitting. The
launcher never places `WANDB_API_KEY` in the generated command or dry-run
output.
