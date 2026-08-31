# HSG Codex handoff: Super model video RL with reasoning profile band

Date: 2026-08-28

## Mission

Continue this work on the HSG Slurm cluster. Fetch the branch below, add the
user's Super-model recipe, and run a 16-node async video GRPO smoke test with
reasoning-length profile-band reward shaping. Verify that video loading no
longer fails with the temporal-patch placeholder/TMPE issue.

Do not create a pull request. Do not push additional changes unless the user
explicitly asks. Do not submit more than one active 16-node allocation unless
the user explicitly requests a scheduler race.

## Repository and branch

- GitHub repository: `git@github.com:NVIDIA-NeMo/RL.git`
- Working branch: `ehsan/super35-video-rpb-tmpe-fix`
- Target branch: `super-v3.5-posttraining`
- Target/base commit when the work started:
  `20ae0bb867163cc4fb1d22b5ded3fa86a4144ba1`
- First branch commit:
  `4543694d3cdf725553e28af51387cfc38241b1f9`
- DFW checkout used to prepare the branch:
  `/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_mlops/users/ehosseiniasl/github_repos/nemorl/RL-super-v3.5-posttraining`

The working branch was created directly on top of official
`origin/super-v3.5-posttraining`, not on top of a PR-only head.

Verify this on HSG:

```bash
git fetch origin
git switch ehsan/super35-video-rpb-tmpe-fix
git submodule update --init --recursive
git status --short --branch
git log --oneline --decorate -6
git merge-base --is-ancestor origin/super-v3.5-posttraining HEAD
```

The final command must exit with code `0`.

## Important model distinction

The committed reference experiment is for Nemotron Omni 30B-A3B. It is not a
120B-A12B Super-model recipe:

```text
examples/configs/recipes/vlm/vlm_grpo-nemotron-omni-30ba3b-16n8g-megatron-tp2ep16-async-gym-video-adi-profile-band.v1.yaml
```

The user will add the Super recipe on HSG. Use the reference recipe only for
the known-good video and profile-band settings. Start the new Super recipe
from the official Super Omni recipe when appropriate:

```text
examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-16n8g-megatron-tp8ep16cp2-async-gym.v1.yaml
```

Do not copy the 30B TP2/EP16 topology blindly into the 120B Super recipe.
Preserve the Super checkpoint's required TP/EP/CP topology.

## Confirmed root cause of the prior video failure

The failed Super run used this unsupported legacy key:

```yaml
policy:
  generation:
    vllm_cfg:
      video_loader:
```

Current Super code reads only:

```yaml
policy:
  generation:
    vllm_cfg:
      video:
        sampling_style: nemotron_vl
        num_frames: 64
        temporal_patch_size: 2
```

The wrong key meant the video sampling values never reached the loader and
led to the temporal-patch placeholder/TMPE failure. The branch includes a
fail-fast guard that rejects `video_loader` and tells the user to use `video`.

This was a recipe/configuration bug, not a missing video port from main:

- Video PR commit `7571a7d5` is already an ancestor of Super.
- Super and main use the same relevant video implementation.
- MBridge pin is `8c46dc4259080c510b7455f43e836fdff222c5d3`.
- Nested Megatron-Core pin is
  `14346b65a2d0790e451919858f7771078105c5f0`.

## Profile-band settings to preserve

Use reasoning-only token length:

```yaml
grpo:
  length_penalty:
    default:
      enabled: true
      length_type: tokens
      profile_band_total: false
      profile_band_reasoning: true
      profile_band_answer: false
    profile_band:
      enabled: true
      defaults:
        reasoning:
          a: 1024
          b: 4096
          f: 0.95
```

Behavior for a correct rollout with reasoning length `r`:

```text
r <= 1024              multiplier 1.00
1024 < r < 4096        linear decrease from 1.00 to 0.95
r >= 4096              multiplier remains fixed at 0.95
```

The multiplier does not keep decreasing after `b`. Global defaults work for
rows without profile-band metadata; row and agent overrides take precedence.
Prompt-history tokens must not count toward the reasoning length.

Keep generation log probabilities in raw mode:

```yaml
policy:
  generation:
    vllm_cfg:
      logprobs_mode: raw_logprobs
```

All prior successful main video RL runs used `raw_logprobs`. The earlier
failed Super run also used raw log probabilities, so logprob mode was not the
cause of that failure.

## HSG container

Use the image already present on HSG:

```bash
export CONTAINER='/lustre/fsw/portfolios/coreai/users/yifuw/enroot-images/gitlab-master.nvidia.com/dl/joc/nemo-ci/main-mirror/rl-gym:pipe.64391373.squashfs'
test -r "${CONTAINER}"
ls -lh "${CONTAINER}"
unsquashfs -s "${CONTAINER}"
```

Do not wait for or copy the DFW `.sqsh` back to HSG. The source image above is
already HSG-local. This newer image is useful for matching the current Super
Python/Gym/vLLM environment, but it is not itself the TMPE fix.

Validate these interpreters before allocating 16 nodes:

```text
/opt/nemo_rl_venv/bin/python
/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python
/opt/gym_venvs/responses_api_models/vllm_model/.venv/bin/python
```

## HSG checkout

Recommended HSG location:

```bash
export HSG_USER_ROOT=/lustre/fsw/portfolios/nemotron/users/ehosseiniasl
export HSG_REPO_ROOT=${HSG_USER_ROOT}/github_repos/nemorl
export HSG_REPO=${HSG_REPO_ROOT}/RL-super-v3.5-posttraining

mkdir -p "${HSG_REPO_ROOT}"
git clone --branch ehsan/super35-video-rpb-tmpe-fix \
  git@github.com:NVIDIA-NeMo/RL.git "${HSG_REPO}"
cd "${HSG_REPO}"
git submodule update --init --recursive
```

If the checkout already exists, fetch and switch instead of cloning again.
Preserve any unrelated user changes in an existing checkout.

## Inputs the HSG run still needs

The HSG Codex must obtain or be given HSG-visible paths for:

```bash
export MODEL_PATH=/lustre/fsw/portfolios/nemotron/<owner>/<super-hf-checkpoint>
export TRAIN_PATH=/lustre/fsw/portfolios/nemotron/<owner>/<video-train-data>.jsonl
export VAL_PATH=${TRAIN_PATH}
export PERSISTENT_CACHE=/lustre/fsw/portfolios/nemotron/users/ehosseiniasl/cache/nemo-rl-super-omni
export SANDBOX_CONTAINER=/lustre/fsw/portfolios/nemotron/<owner>/<nemo-skills-sandbox>.sqsh
export CONFIG_PATH=examples/configs/recipes/vlm/<new-super-video-profile-band-recipe>.yaml
```

Do not silently reuse the DFW Arushi paths; `/portfolios/llmservice` may not be
mounted on HSG. For reference only, the 30B DFW experiment used:

```text
Checkpoint:
/lustre/fsw/portfolios/llmservice/users/arushig/workspace/output/generalist-49k-video-mpo-20260812-113202/checkpoints/tp_1_hf/iter_0000125/mcore_to_hf

Dataset:
/lustre/fsw/portfolios/llmservice/users/arushig/nemo_gym_rl_video_0803/nemo-rl/results/video_frame_cache_caprl_passrate_n5_easy_to_hard_lt60s_20260806_f64/stable_split_95_5/train_exclude_line6215.jsonl
```

Validate HSG inputs before submission:

```bash
test -f "${MODEL_PATH}/config.json"
test -s "${TRAIN_PATH}"
test -s "${VAL_PATH}"
test -f "${CONFIG_PATH}"
test -r "${CONTAINER}"
test -r "${SANDBOX_CONTAINER}"
mkdir -p "${PERSISTENT_CACHE}"
```

## Slurm accounts and partitions

Do not copy the DFW accounts `nemotron_edge_omni` or
`nemotron_omni_vision` into the HSG launch without checking HSG associations.
Discover the actual HSG values:

```bash
sacctmgr -nP show assoc where user="${USER}" format=Account,Partition
sinfo -o '%P %a %l %D %G'
```

Then set exactly one initial candidate:

```bash
export SLURM_ACCOUNT=<hsg-account>
export SLURM_PARTITION=<hsg-gpu-partition>
```

## W&B

Use:

```bash
export WANDB_ENTITY=adlr
export WANDB_PROJ=Nemotron-omni-RL-debug
export EXP_NAME=s35-super-video-rpb-hsg
```

Load `WANDB_API_KEY` from the user's HSG credentials environment. Never write
the key into YAML, scripts, logs, this handoff, or Git history. Keep the W&B
run name short and include only the important identifiers: Super, video,
profile band, HSG, and optionally the checkpoint step.

## Resolve and inspect the new Super recipe

Before submission:

```bash
cd "${HSG_REPO}"
uv run --no-sync python tools/config_cli.py expand "${CONFIG_PATH}" \
  > /tmp/s35-super-video-rpb-resolved.yaml

rg -n 'num_nodes:|gpus_per_node:|logprobs_mode:|video:|video_loader:|sampling_style:|num_frames:|temporal_patch_size:|length_penalty:|profile_band:|reasoning:|a:|b:|f:' \
  /tmp/s35-super-video-rpb-resolved.yaml
```

Confirm all of the following:

1. No `video_loader` key appears.
2. `vllm_cfg.video.sampling_style` is `nemotron_vl`.
3. `num_frames` is 64 and `temporal_patch_size` is 2.
4. `logprobs_mode` is `raw_logprobs`.
5. Profile band is reasoning-only with `a=1024`, `b=4096`, `f=0.95`.
6. Model parallelism matches the Super checkpoint.
7. The full cluster shape is intentional. The reference Super launcher uses
   16 nodes with 8 GPUs per node and 8 generation nodes.
8. `allowed_local_media_path` covers the HSG video/cache location.
9. The Responses API reasoning parser and chat template match the Super
   checkpoint; do not blindly copy the 30B parser.

## Dry-run and launch

Use the official Super Omni Slurm launcher unless the new recipe requires a
small, reviewed extension:

```bash
cd "${HSG_REPO}"
export EXTRA_MOUNTS=/lustre:/lustre
export SLURM_TIME_LIMIT=4:0:0

DRY_RUN=true \
  bash examples/nemo_gym/nemotron-3-super-omni/super_omni_launch.sh \
  | tee /tmp/${EXP_NAME}-dry-run.txt
```

Review the rendered command and `sbatch` request. It must show the intended
HSG container, recipe, Super checkpoint, data, account, partition, node count,
and W&B project.

Only after the dry run is correct:

```bash
bash examples/nemo_gym/nemotron-3-super-omni/super_omni_launch.sh
```

Capture the job ID and monitor it:

```bash
export JOB_ID=<job-id>
squeue -j "${JOB_ID}" -o '%.18i %.12a %.20P %.30j %.2t %.10M %.6D %R'
scontrol show job "${JOB_ID}"
```

## Success criteria

Do not report success merely because Slurm says `RUNNING`. Verify:

1. The driver resolves the intended Super recipe and checkpoint.
2. All expected Ray nodes and workers start.
3. The video loader initializes without a temporal-patch/TMPE error.
4. Generation uses raw log probabilities.
5. At least one prompt batch finishes generation.
6. At least one optimizer step completes.
7. W&B appears under `adlr/Nemotron-omni-RL-debug`.
8. Reasoning-length/profile-band logs or metrics show the expected multiplier.

Useful searches:

```bash
find logs results -type f -newermt '-30 minutes' -print | sort

rg -n 'Traceback|ERROR|TMPE|temporal|video_loader|Version mismatch|OutOfMemory|CUDA out of memory' \
  logs results -g '*.out' -g '*.err' -g '*.log'

rg -n 'wandb.ai|Processed prompts|optimizer|profile.band|reasoning_len|length_penalty' \
  logs results -g '*.out' -g '*.err' -g '*.log'
```

If the run is clearly misconfigured, release the allocation immediately:

```bash
scancel "${JOB_ID}"
sacct -j "${JOB_ID}" --format=JobID,State,Elapsed,ExitCode -X
```

## Test status inherited from DFW

The focused unit test for the new legacy-key guard could not complete on DFW
because the test process attached to a stale external Ray cluster running Ray
2.48/Python 3.10 while the process used Ray 2.56/Python 3.13. This was an
environment/setup failure before the relevant test executed, not a failure of
the guard assertion.

On HSG, run the focused test in a clean environment with no inherited Ray
address or stale cluster:

```bash
unset RAY_ADDRESS
uv run --no-sync pytest -q \
  tests/unit/models/generation/test_vllm_video_utils.py \
  -k 'video_config or legacy_video_loader' \
  --maxfail=1
```

Run the test in an isolated Slurm allocation/container so Ray cannot discover
a cluster owned by another user or job. Do not stop a shared Ray cluster.
