# NeMo RL v2 + OSWorld

This directory is the portable launcher for the validated OSWorld training path:

- NeMo RL v2 `SingleController`
- `TransferQueue` data plane
- 4 nodes x 8 GPUs
- B=8 prompt groups, N=8 generations per prompt, and K=8 groups in flight
- Molt-aligned REINFORCE group-mean baseline
- rewards clipped to `[0, 1]`
- token-level importance sampling and sequence-mask TIS `[0.99, 1.01]`
- windowed sampling with maximum staleness 1
- full step-boundary policy, optimizer, dataloader, sampler, and TransferQueue
  checkpoint/resume

The exact recipe is
`examples/nemo_gym/grpo_nemotron_omni_30ba3b_osworld_v2_cc_molt_b8k8_checkpoint.yaml`.
Mid-turn VM snapshots are not part of this release; recovery is at a completed
optimizer-step boundary.

## 1. Prerequisites

The validated launcher requires:

- Linux x86_64, Slurm, and Pyxis/Enroot.
- Four homogeneous nodes with eight supported NVIDIA GPUs each.
- A filesystem visible at the same absolute path from the submit host and every
  compute node.
- The qualified immutable `.sqsh` image described below.
- A local Nemotron-Omni SFT checkpoint in Hugging Face layout, including
  `config.json` and `chat_template.jinja`.
- An OpenSandbox deployment with a warm KVM pool named `osworld-kvm` (or set
  `OSWORLD_POOL_REF`) and guest OSWorld control API on port 5000.
- Access to GitHub/Python package indexes the first time NeMo Gym builds its
  shared component environment. The OSWorld dependency is pinned to
  `terrykong/OSWorld@5d8c93592be588ab8e9909bed82f3b480fd4f533`.

## 2. Clone the pinned code

```bash
git clone --recurse-submodules \
  --branch osworld-v2-cc-integration \
  https://github.com/NVIDIA-NeMo/RL.git nemo-rl-osworld-v2
cd nemo-rl-osworld-v2
git submodule update --init --recursive
```

The RL branch pins the matching Gym fork commit through the submodule, so no
second manual checkout is required. Record the exact source state with:

```bash
git rev-parse HEAD
git submodule status
```

## 3. Copy the qualified container

The image is intentionally not stored in git. Copy
`rl-osworld-v2-qualified-69764640.sqsh` to a shared path on the destination
cluster. It must have this SHA-256 digest:

```text
1b4edcdeac017210e6abe25885372e025ac111c45aeb038bda334659436eb74e  rl-osworld-v2-qualified-69764640.sqsh
```

Verify it once:

```bash
sha256sum /shared/containers/rl-osworld-v2-qualified-69764640.sqsh
```

`submit.sh` verifies the digest again by default. Do not replace this image with
a moving nightly tag; that was the source of earlier environment drift.

## 4. Prepare the 361-task training dataset

The Gym submodule includes the same 361 no-Google-Drive OSWorld tasks used by
the v1-parity run. Build the 5-repeat, 1,805-row training file:

```bash
python examples/nemo_gym/prepare_osworld_context_compaction_data.py \
  --input 3rdparty/Gym-workspace/Gym/resources_servers/osworld/data/test_nogdrive.jsonl \
  --output /shared/data/osworld/train-5x.jsonl \
  --num-repeats 5
wc -l /shared/data/osworld/train-5x.jsonl
# expected: 1805
```

The preparation script removes framework-owned rollout IDs and adds the
`nemotron_osworld` Gym route. SingleController creates fresh logical owner and
attempt IDs at runtime.

## 5. Configure the cluster

Copy the template to a private location, fill in every required path and
credential, and restrict its permissions:

```bash
cp examples/nemo_gym/osworld_v2/env.example /shared/private/osworld-v2.env
chmod 600 /shared/private/osworld-v2.env
${EDITOR:-vi} /shared/private/osworld-v2.env
```

Required values are:

- `CONTAINER`
- `NANO_OMNI_MODEL_NAME`
- `NANO_OMNI_CHAT_TEMPLATE`
- `OSWORLD_GRPO_TRAIN_DATA`
- `OPENSANDBOX_DOMAIN` and, when enabled by the service,
  `OPENSANDBOX_API_KEY`
- `SLURM_ACCOUNT` and `SLURM_PARTITION`
- `OSWORLD_RESULTS_ROOT`

The default 4-hour segment saves a clean checkpoint by 3h40m. If the cluster
uses a different time limit, change both `OSWORLD_JOB_TIME` and
`OSWORLD_CHECKPOINT_MUST_SAVE_BY`, leaving enough time for the synchronous
full-model checkpoint.

## 6. Submit

An optional host-side dry run performs every check and builds the mount list
without calling `sbatch`:

```bash
OSWORLD_SUBMIT_DRY_RUN=1 \
  bash examples/nemo_gym/osworld_v2/submit.sh /shared/private/osworld-v2.env
```

Submit the training job:

```bash
bash examples/nemo_gym/osworld_v2/submit.sh \
  /shared/private/osworld-v2.env
```

Before calling `sbatch`, the launcher:

1. validates all required files and the container checksum;
2. prepares the four shared runtime libraries needed by the qualified image;
3. mounts the checkout at `/opt/nemo-rl` and mounts model, data, cache, and
   result paths at identical paths inside the container;
4. syntax-checks the shell launchers; and
5. configures the Ray head and workers to use the qualified driver venv.

Each node then checks Python 3.13.14, Ray 2.56.1, RL/Gym import provenance, and
the fully resolved SingleController configuration before model loading begins.

The command prints `series_root`, `checkpoint_dir`, and the submitted Job ID.
Useful checks are:

```bash
squeue -j JOB_ID -o '%.18i %.12T %.30j %.10M %.20R'
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed
cat SERIES_ROOT/submissions.tsv
less SERIES_ROOT/segments/segment-001/slurm-JOB_ID.out
less SERIES_ROOT/segments/segment-001/JOB_ID-logs/ray-driver.log
```

## 7. Continue or chain segments

Checkpoints are written to `SERIES_ROOT/checkpoints/step_N`. To continue an
existing series, set this in the private environment file and submit again:

```bash
OSWORLD_SERIES_ROOT=/shared/results/nemo-rl-v2-osworld/existing-series
```

The launcher chooses the next unused segment directory and the trainer restores
the latest complete `step_N` automatically. To queue several segments at once,
set `OSWORLD_CHAIN_LENGTH`. The default dependency is `afterok`; use
`OSWORLD_CHAIN_DEPENDENCY=afterany` only when the site's expected time-limit
termination is nonzero and the checkpoint has been independently verified.

## Evaluation recipes

The repository also includes the v1-parity 361-task x 4-rollout evaluator:

- SFT or directly loadable policy:
  `grpo_nemotron_omni_30ba3b_osworld_v2_inference_v1_parity.yaml`
- restored NeMo RL checkpoint:
  `grpo_nemotron_omni_30ba3b_osworld_v2_checkpoint_inference_v1_parity.yaml`

These intentionally use the legacy validation runner because SingleController
does not own the evaluation loop. Training itself remains entirely on
SingleController + TransferQueue.

## Troubleshooting

- A checksum mismatch means the container is not the validated image. Do not
  suppress it unless the image was copied through a system that independently
  verified content integrity.
- A failure in `osworld-v2-node-preflight-ok` is an image, mount, submodule, or
  Ray-version problem; training has not started.
- OpenSandbox setup failures should be diagnosed from the Gym server logs and
  guest control API before resubmitting. Confirm that `OSWORLD_POOL_REF` exists
  and that all compute nodes can resolve `OPENSANDBOX_DOMAIN`.
- NeMo Gym environments are not relocatable. Keep `OSWORLD_GYM_VENV_DIR` on the
  shared destination cluster and let this image build it there.
