# Nano3 QAOPD with TransferQueue

This experimental example trains an NVFP4 Nemotron 3 Nano student against the
matching BF16 teacher. Student generation produces NeMo-Gym trajectories, the
teacher writes top-k logits directly to TransferQueue, and the Megatron policy
worker consumes the queued fields during training.

The supplied recipe preserves the development topology:

- 8 nodes with 4 GPUs per node
- Megatron TP=4, CP=4, EP=8
- vLLM TP=4 and EP=4
- 128 prompts with 4 generations per prompt
- global training batch size 512
- teacher top-k 300, BF16 logits, and int32 indices
- `mooncake_cpu` storage with microbatch-streamed top-k writeback

Treat these as starting values. Memory capacity and parallelism must be checked
again when the GPU type, sequence length, or batch shape changes.

## Build an Enroot image

Enroot imports OCI/Docker images; it does not build a Dockerfile itself. Build
the NeMo RL release image first, then convert it to a squashfs image.

Initialize the repository and build on the same CPU architecture as the target
GPU cluster whenever possible:

```bash
git submodule update --init --recursive

docker buildx build \
  --platform linux/arm64 \
  --target release \
  --build-context nemo-rl=. \
  --file docker/Dockerfile \
  --tag nemo-rl:nano3-tq \
  --load \
  .

enroot import \
  --output nemo-rl-nano3-tq.sqsh \
  dockerd://nemo-rl:nano3-tq
```

Use `linux/amd64` for an x86_64 target. Cross-architecture builds require a
Buildx builder with binfmt/QEMU support and are slower and less reliable for
CUDA extension builds. A native build on the target architecture is preferred.

If Docker and Enroot are on different machines, push to a registry and import
the image there. The `#` separates the registry from the image path in Enroot's
Docker URI syntax:

```bash
docker buildx build \
  --platform linux/arm64 \
  --target release \
  --build-context nemo-rl=. \
  --file docker/Dockerfile \
  --tag registry.example.com/team/nemo-rl:nano3-tq \
  --push \
  .

enroot import \
  --arch arm64 \
  --output nemo-rl-nano3-tq.sqsh \
  docker://registry.example.com#team/nemo-rl:nano3-tq
```

The resulting `.sqsh` file can be passed directly to Pyxis through
`--container-image` or the repository's `ray.sub` launcher.

Do not put registry credentials, Hugging Face tokens, W&B keys, or other
secrets in the image, scripts, Git configuration, or recipe. Authenticate with
the registry before import and inject runtime credentials through your
cluster's secret-management mechanism.

## Prepare data

The launcher expects two files on storage shared by every Ray node:

1. NeMo-Gym JSONL training data accepted by `NemoGymDataset`.
2. JSONL calibration data used to initialize the NVFP4 quantizers.

The model may be a shared local directory or the public Hugging Face model ID
used by the recipe. For gated models, configure Hugging Face authentication in
the runtime environment without adding credentials to this repository.

## Run on an existing Ray cluster

From the repository root inside the container:

```bash
bash examples/modelopt/nano3_qaopd_tq/run_training.sh \
  /shared/data/train.jsonl \
  /shared/data/quant-calibration.jsonl \
  /shared/results/nano3-qaopd-tq \
  nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

W&B is disabled in the example recipe. Checkpoints, JSON logs, and TensorBoard
files are written below the supplied output directory.

## Submit through Slurm, Pyxis, and Ray

The submission wrapper takes all cluster-specific values as arguments. No
account, partition, filesystem root, or credential is stored in the script.

```bash
bash examples/modelopt/nano3_qaopd_tq/submit_slurm.sh \
  /shared/containers/nemo-rl-nano3-tq.sqsh \
  SLURM_ACCOUNT \
  SLURM_PARTITION \
  /shared \
  /shared/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  /shared/data/train.jsonl \
  /shared/data/quant-calibration.jsonl \
  /shared/results/nano3-qaopd-tq
```

The wrapper mounts only the specified shared root and the source checkout. The
underlying `ray.sub` invocation uses exclusive nodes and disables automatic
home-directory mounting inside the container.

For a smoke test, copy the recipe, reduce only the number of steps and prompt
count, and keep sequence length, model parallelism, quantization, top-k, and
TQ dtype settings unchanged when validating memory behavior.
