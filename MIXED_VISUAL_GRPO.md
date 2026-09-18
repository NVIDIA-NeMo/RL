# Experimental Gym-V + VisGym + image-tools training

Working branches for internal experimentation, not a merged or generally
qualified release. Both repositories use `aroshanghias/mixed-visual-grpo`.
This RL branch is based on `a562db3077ea9e4dc65aaf4902ed810a42c94940`
from `super-v3.5-posttraining`; it pins Gym commit
`7ee38197c0c31f7644c34de6a9f38bfbc8d01acf` exactly.

## Included

- Separate Gym-V and VisGym agents/resources, combined with image-tools and its
  string/math/MCQA graders. Gym PRs #2700 and #2730 remain unmerged; their
  selectively ported implementations are included in our Gym branch.
- Mixed manifest preparation/auditing, agent routing, per-game rewards and compact
  rollout diagnostics; image-tool execution provenance and bounded context/errors.
- Mixed compact/legacy image-row collation fix, hybrid MTP-disable handling,
  engine input encoding synchronization, RL runtime callback fix (PR #4116),
  and the nullable checkpoint-retention schema correction.
- Recipes, HSG preparation/launch helpers and focused regression tests.

No full training datasets, checkpoint weights, credentials, experiment logs or prebuilt VisGym
wheel are included. Historical mixed execution reached 220 updates in job7100810;
that establishes an earlier integration baseline, not numerical correctness or a
new GPU qualification of these sharing commits. TMPE remains under investigation.

## Checkout

The Gym branch is hosted in `aroshanghias-nvd/Gym`; the RL branch is hosted in
`NVIDIA-NeMo/RL`. The submodule URL selects the fork automatically:

```bash
git clone --branch aroshanghias/mixed-visual-grpo https://github.com/NVIDIA-NeMo/RL.git mixed-visual-grpo
cd mixed-visual-grpo
git submodule update --init --recursive
```

Gym is published first so the pinned commit is available. For an existing checkout,
run `git submodule sync -- 3rdparty/Gym-workspace/Gym` before updating submodules.
Do not update the Gym submodule to main. Gym provenance and the open PR links are
in `3rdparty/Gym-workspace/Gym/environments/visual_games_image_tools/SHARING.md`.

## Lightweight checks

A separate Python3.13 environment with `pytest pyyaml hydra-core omegaconf pydantic
pillow` is sufficient for the focused tests below; it is not a training environment.

```bash
PYTHON=/path/to/test-venv/bin/python bash tools/test_mixed_visual_lightweight.sh
```

These check manifest contracts, executed-call provenance, metrics, processor
asset export, recipe inheritance, launcher helpers, encoder locking and compact
diagnostics. They do not run model rollouts, CUDA, distributed optimizer updates,
full MasterConfig imports, or checkpoint reload. See `SHARING_VALIDATION.md`.

## Runtime and data preparation

Use the pinned Super3.5 container/dependencies. The historical HSG image was
`rl-gym.67009223-gym_ln_fix.sqsh`; obtain an accessible copy and set `CONTAINER`.
The HSG geometry is 40 four-GPU nodes: 32 learner GPUs and128 generation GPUs.
Other hardware layouts need their own qualification.

1. Build the local VisGym dependency from Gym's pinned build script:
   `resources_servers/visgym/scripts/build_visgym_wheel.sh`, run from the Gym root.
   Preserve separate Gym-V/VisGym venvs; see each server's README and requirements.
2. Stage game assets and image-tool images under paths accessible to every worker.
   Obtain the existing manifests from their owner or generate them with the server
   tools. `tools/prepare_image_tools_grpo.py` audits/splits supplied image-tool rows;
   `tools/stream_image_tools_bundle.py` supports asset transfer. Neither grants
   dataset access nor downloads our complete private training mixture.
3. Combine already prepared game and image-tool training rows:

```bash
python -m tools.prepare_visual_image_tools_mix --games /data/games-train.jsonl \
  --images /data/image-tools-train.jsonl --validation /data/games-validation.jsonl \
  --output /data/mixed-train.jsonl
```

The reference mixture is5280 Gym-V +2880 VisGym +2237 image-tool rows, without
oversampling. Validation is64 rows from eight held-out Gym-V games only.
The builder checks all22 Gym-V and12 VisGym training games and512-token row caps.

## Training

The reference recipe is
`examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-visual-games-image-tools-40n4g-megatron-tp8ep16cp2-async.v1.yaml`.
It uses8 prompts×16 responses/GBS128,512-token turns, async age1, and retains20-step
recovery checkpoints. Later32×8/GBS256,2028-token/cosine runs are separate
experiments; those settings are not silently made the shared default.

`tools/launch_image_tools_train_hsg.sh` is the existing qualified-runtime HSG
workflow. Set `IMAGE_TOOLS_SUITE=visual-games`, `HSG_ROOT`, `SLURM_ACCOUNT`,
`CONTAINER`, `PROJECT_ROOT`, `RUN_DIR`, `OVERLAY_DIR`, `MODEL_CHECKPOINT`,
`TRAIN_MANIFEST`, `EVAL_MANIFEST`, and `WANDB_RUN_ID`; the mixed mode additionally
requires `GAMES_TRAIN_MANIFEST`, `IMAGE_TRAIN_MANIFEST`, `GAMES_VENV_ROOT`,
`VISGYM_ASSET_ARCHIVE`, and the source/runtime qualification receipts listed at
its entrypoint. `SLURM_PARTITION`/`SLURM_QOS` default to batch/normal.
Credentials go in an ignored, mode0600 `.env`; never commit them.
This helper retains the historical nvidia/games-rlvr-nemotron-super logging target;
adapt it and its explicit startup assertions if using another entity/project.

The legacy callback qualification additionally checks Bridge's independent
callback site. Its exact patch is included rather than hidden in a dirty
submodule or published as a third branch. Apply it to the pinned Bridge checkout:

```bash
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge apply --check ../../../patches/megatron-bridge-runtime-callbacks.patch
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge apply ../../../patches/megatron-bridge-runtime-callbacks.patch
python tools/callback_test_runtime.py --download
```

This intentionally leaves that Bridge checkout locally modified; the RL gitlink
and patch together specify the dependency. The separate RL callback fix is
already committed. After runtime qualification, dry-run the launcher with
`DRY_RUN=1`; actual submission uses `DRY_RUN=0`. Do not bypass its receipt checks
or point it at an existing results directory. Existing cluster-specific helper
scripts retain historical path assumptions; this is a research handoff, not a
one-command clean-cluster installer.
