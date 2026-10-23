# bf16_009 with prefix-rollout checkpoints

Reference: [joc/bits/u7lao4f0, bf16_009](https://forge.coreweave.com/wandb/joc/bits/runs/u7lao4f0).
The resolved W&B configuration, rather than the FP32/MTP-disabled smoke recipe,
was used for the comparison.

## Launch and resume

From the NeMo-RL checkout root (no pipeline repository required):

```bash
EXP_NAME=nano35-bf16-009-prefix-100step \
  bash examples/nemo_gym/nemotron-3.5-nano/scripts/launch_nano35_bf16_009_prefix_checkpoint.sh
```

Prefix with `DRY_RUN=1` to validate paths and inspect the generated submission
without allocating nodes. The launcher uses the local `scripts/.env` if present;
otherwise it uses the secret-free `scripts/env_site_oci.sh` asset map for the
validated cluster. Export credentials such as `WANDB_API_KEY` and `HF_TOKEN`
separately. A pipeline `.env` is not automatically loaded and may contain
incompatible root assumptions; use the native site template instead.

Use a **new experiment name** for the first launch. Do not point this recipe at
the original `bf16_009` checkpoints or the FP32/MTP-disabled smoke checkpoints.
Subsequent launches with the same `EXP_NAME` and results directory restore the
latest compatible state and continue toward **100 total optimizer steps**.
The launcher does not automatically requeue itself.

Each allocation is 90 minutes, with a stop/save deadline at 75 minutes. Reaching
100 steps may require multiple submissions. If overriding `WALLTIME`, also pass
an appropriate `checkpointing.checkpoint_must_save_by` Hydra override. Changing
the allocation duration does not change the 100-step training target.

## Preserved reference settings

- 86 nodes: 32 training, 32 generation, 2 Gym, 20 external services.
- Same Nano 3.5 SFT policy, original unfiltered amplified-dolphin dataset,
  GenRM, NL2Bash, safety model and parsers from the existing site configuration.
- BF16 training/generation; no FP32-LM-head override; HF MTP architecture intact.
- Five repeated MTP layers, detached heads, auxiliary loss scaling 0.3.
- Ready-first, staleness 1; 992 in-flight prompt groups, 16 samples per prompt,
  512 prompts / 8192 samples per optimizer step; seed 42.
- Same optimizer, learning rate, warmup, loss and generation parameters,
  including `max_num_seqs=768`. Router replay remains disabled.
- All 33 reference Gym config paths are inherited. `ns_tools` has no rows in
  this blend; if another dataset uses it, its live Python sessions are
  restart-only, not resumable mid-session.
- Duplicated-reasoning, empty-final-answer and unwanted-token reward penalties
  remain enabled, with the reference token IDs.
- Reference trainer retention: periodic every 10 steps, rolling every step,
  optimizer and data-plane state included. Connector limit is pinned to the
  reference's 16384, not the newer base recipe's 65536.

## Changes relative to the reference

| Area | Reference | This recipe |
|---|---|---|
| Malformed-thinking reward penalty | Enabled | Disabled, as requested |
| Invalid-tool-call advantage override | `-5.0` | `null`: capture does not emit the required mask |
| Token capture | Disabled | Enabled, node-local `/tmp/nemo_rl/token_capture/${SLURM_JOB_ID}` |
| Recovery target | Sibling | Prefix; supported in-progress model calls and Gym participants can resume |
| Periodic rollout snapshots | Disabled | Attempt every 120 s, retain latest two, restore latest compatible |
| Control timeout | 60 s | 120 s; Gym prepare gets a separate 300 s budget |
| Cut-control executor | Not used | Two guarded control threads per generation worker |
| Walltime / save deadline | Long allocation / `00:23:35:00` | 90 min / `00:01:15:00` |
| Software | Original image and older RL/Gym | Image `rl-gym.75221028.sqsh` plus mounted current RL/Gym checkpoint stack |
| Observability and outputs | Original run | Checkpoint telemetry, new experiment/output paths and W&B project |

The invalid-tool-call advantage change is **also an objective difference**; it
is separate from malformed-thinking reward penalization. Simply retaining
`-5.0` would make the current token-capture configuration fail validation.
`grpo.malformed_thinking_advantage` remains `null`, as in the reference.

The current base also supplies the newer runtime controls
`offload_policy_before_refit=false` and `offload_optimizer_for_refit=true`;
they were absent from the old reference schema. No new MTP/precision/refit
offload experiment is introduced here.

This is not a bit-for-bit reproduction: objective differences, newer software,
ready-first scheduling and snapshot pauses can affect training. Compare metrics
at the same optimizer step / token count, not only by elapsed time. For a clean
checkpoint-only A/B, launch a **separate** experiment with the same recipe and
`rollout_checkpointing.snapshot_attempt_interval_s=null`; keep token capture and
the two objective changes the same in both runs. Do not change a running or
already checkpointed experiment's reward settings to create the baseline.

Files and config composition can be checked on the login node; MTP-enabled
prefix save/restore and convergence still require an actual GPU run.

## Moving to another cluster

This directory contains the entire recipe: all three inherited YAMLs, launcher,
shared submission logic and site profiles. External service orchestration comes
from this checkout's `tools/external_gym_vllm/`; training uses its `ray.sub`,
entrypoints and mounted `nemo_rl/` source. No outer `RLVR/` files are read.

1. Commit and push this bundle **and the required runtime changes** on the
   checkpoint-capable NeMo-RL integration branch. This recipe does not run on
   unmodified main. Commit/push the matching Gym changes too, and update the Gym
   submodule pointer so the destination can reproduce the actual source.
   A gitlink alone does not transfer dirty or untracked source.
2. Clone that NeMo-RL branch at the destination and initialize its submodules:
   `git submodule update --init --recursive`. Retain a compatible container
   and sandbox image, plus the GenRM reasoning parser, alongside the models
   and dataset. Record their versions/checksums.
3. Copy `scripts/env_site_template.sh` to `scripts/site.local.sh` and edit
   asset paths, writable output/cache paths, and scheduler settings. Keep the
   checkout, results and mounted assets under the configured shared mount.
   Local site files and `scripts/.env` are ignored; do not commit secrets.
4. Validate and then submit from the NeMo-RL root:

   ```bash
   SITE_OVERLAY="$PWD/examples/nemo_gym/nemotron-3.5-nano/scripts/site.local.sh" \
   EXP_NAME=nano35-bf16-009-prefix-100step DRY_RUN=1 \
     bash examples/nemo_gym/nemotron-3.5-nano/scripts/launch_nano35_bf16_009_prefix_checkpoint.sh
   ```

   Remove `DRY_RUN=1` to allocate nodes. Dry-run writes local run metadata,
   stages helper scripts and creates cache directories, but does not call
   Slurm or start Ray. Its command/provenance output can contain environment
   values; do not publish it unredacted.

The validated topology is Grace-Blackwell/AArch64 with four GPUs per node,
Slurm heterogeneous allocations, the site's `--segment` support, and
Pyxis/Enroot. The shared launcher mounts six host X11 libraries for `ether0`.
A cluster with different scheduler flags, CPU architecture, or GPUs per node
needs an explicit topology/launcher adaptation; changing asset paths alone is
not sufficient.

By default, both RL and Gym use mounted checkout sources with the image's
prebuilt environments (`USE_IMAGE_GYM=0`, `GYM_VENV_DIR=/opt/gym_venvs`,
`UV_NO_SYNC=1`, `NRL_FORCE_REBUILD_VENVS=false`). Ignoring a fingerprint mismatch
does not make dependencies compatible. Keep a tested image/source pairing;
missing/incompatible environments need a suitable image or an explicit rebuild.
Native caches default to `<nemo-rl>/.cache/nano35`, so the first native run may
have a cold cache; set `PERSISTENT_CACHE` to a compatible existing cache to reuse it.

The original pipeline files are retained for existing jobs. This migration
changes recipe location, site/path resolution and provenance, not the composed
training configuration. A new native lineage defaults to `<nemo-rl>/results`;
to resume an existing compatible lineage, explicitly preserve its
`RESULTS_ROOT`/`CHECKPOINT_DIR` and `EXP_NAME`.

## CPU checks

From the NeMo-RL root, in an environment with OmegaConf:

```bash
uv run python -m unittest discover -s tests/unit -p test_nano35_prefix_recipe.py -v
bash -n examples/nemo_gym/nemotron-3.5-nano/scripts/*.sh
```

Tests exercise config composition and a complete dry-run from a standalone
temporary NeMo-RL layout, without a pipeline checkout, scheduler or GPU.
