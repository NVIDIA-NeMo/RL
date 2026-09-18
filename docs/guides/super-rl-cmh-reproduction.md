# SCI2-118: reproduce the proven CMH full-scale runtime

## What this branch means

This is a reproduction branch, separate from draft PR #4136.
Its training implementation is based on `bf3a59e483a52c8d0f3d76d9abda8678de8fc5fd`,
with the exact strict-verdict helper committed in
`b41e457589ff9c9040aec739121fa0ba199de4c8`.
Both descend from PR3941's fixed
`ca06137460b7e2edcaf6f1fd67ddbda5ddd6b8e2`.
The helper SHA256 is
`b43bac079b55cc9c22d21eadb3d866f0fdce69e6a5c421ab836f9d707ecfa2cf`.

**Proven:** the original CMH runtime completed rollout/reward, optimizer update,
refit, distributed checkpoint save and cross-allocation restore. Job 3818655
saved steps 40–46 before preemption; continuation 3831731 loaded 46, updated
and saved 47. W&B contains step-47 loss/reward, not merely GPU telemetry:
https://wandb.ai/aiapps/super-rl-baseline/runs/s120-cot-cmh-s100-3791850

**New packaging:** the user-parameterized launcher below replaces private run
paths, creates a fresh Gym/helper tree, performs CPU native preflight and uses
a fresh writer root. Its validation status is recorded in the reproduction PR.
It is not the exact original sbatch and must not be described as GPU-certified
merely because the original training implementation ran. PR4136's independent
128-GPU smoke 3824740 is not evidence for this wrapper.

This recipe covers AWS-CMH ARM/GB300 and Math/equivalence/SciCode only.
It is not an HSG/H100, CCC/TIR or Kimi reward-shaping certification.
Those capabilities remain in the broader codebase; unsupported data routes
are rejected, never silently dropped or rewritten.

## Files to share

Local package checks: **27 passed** (wrapper/config/staging and strict-verdict
tests), with exact pinned Gym present; shell syntax and Ruff checks passed.
Native preflight of this new wrapper and a GPU run remain separate gates.

| Purpose | Repository path |
| --- | --- |
| User configuration, no secret values | `training_configs/super_rl/cmh_repro.user.example.env` |
| Submission entrypoint, default test-only | `tools/super_rl/cmh_repro/submit.sh` |
| CPU preflight and 256-GPU Slurm entry | `tools/super_rl/cmh_repro/job.sbatch` |
| Fixed scientific configuration | `training_configs/super_rl/experiments/regular_s120_cmh_fullscale.yaml` |
| Judge model/provider configuration | `training_configs/judges/nvidia_deepseek_v4_flash.yaml` |
| ARM runtime flags and node setup | `tools/super_rl/cmh_repro/profile.env`, `node.sh` |
| Fresh/resume driver and checkpoint gate | `driver.sh`, `validate_resume.py` |

Do not share the old private run directory as a reusable launcher. Do not copy
another user's API key, W&B identity, caches or checkpoint writer root.

## Dependencies: required before launching

A Git clone alone does **not** supply these artifacts.

1. An immutable clean checkout of this reproduction branch, on CMH's physical
   `/scratch` filesystem, with these exact initialized submodules:
   - Gym `749432dc5de23b8eeb3d044c80350a7c0ae9a03f`
   - Megatron-Bridge `3961f399ef181bca689de8e984110b85c4df00fe`
   - Megatron-LM `f2f0f7bfd88fcb1243df55275988d6af52daea35`
   - Automodel `24b47e856263d313b942f0ed666c63fff83306b4`
2. ARM training SquashFS, historically named
   `nemo-rl:super35_20260902_prefetched_venvs_arm64.squashfs`,
   SHA256 `c39942d83407b08e0b0ad73cdc2334ddee744ab420cd9c5945a03fe3a0abef41`.
   CPU preflight hashes the image; a similarly named image is not sufficient.
3. s120 HF policy/tokenizer (64 nonempty safetensor shards, MTP1), the original
   reasoning parser `ultra_v3_reasoning_parser.py`, SciCode's
   `combined_test_data.h5`, and the offline OpenAI 2.25.0 wheel.
4. Ordered JSONL training data with at least 25,600 rows for 100 steps. Supported
   agent refs: `math_with_judge_simple_agent`,
   `equivalence_llm_judge_simple_agent`, `scicode_agent`.
   The observed dataset has 37,435 rows, SHA256
   `8cb86085bbb5518fe9e85402bd48ade21b29485a28894b40f1debc4f94e5dd1d`.
   Replacing data is supported as an input, not a claim of equivalent scientific
   results. Supply its actual digest. Prompts and reasoning stay unchanged.
5. Your CMH Slurm account, current batch_long/normal and cpu/cpu-short access,
   a writable parent directory, enough quota for 1.5-TB checkpoints, and a
   private mode-600 file exporting `NVIDIA_API_KEY` and `WANDB_API_KEY`.
   Set up credentials without shell tracing; never put values in commands,
   Git, chat, or the user configuration.

**Current dependency-access blocker:** GitHub's NVIDIA-NeMo/Gym endpoint could
not fetch the fixed 749432 commit during this audit. Exact objects exist in the
CMH reference closure and were used by the successful native preflight. Obtain
an authorized team-readable mirror or dependency checkout; do not substitute a
moving Gym branch or assume `git clone --recursive` works. Private objects,
weights and credentials are not republished by this PR. Users lacking access
to that closure or the image cannot run yet.

The image supplies driver Python `/opt/nemo_rl_venv/bin/python`, exact worker
Pythons under `/opt/ray_venvs` and Gym environments under `/opt/gym_venvs`.
Do not copy a user's activated login-node environment. Preflight stages Gym
from the fixed Git object and compiles helpers using the exact ARM worker
Python into the new run directory, then verifies the helper mount read-only.

## Fresh run

Clone the reproduction branch and pin the full commit reported by its PR.
Initialize the exact dependencies using your authorized source. On the CMH
login node, copy the user template outside the checkout and fill every field.
Use a **new, nonexistent** run root with an existing writable parent, and your
own unused W&B run ID. Keep the source checkout clean.

```bash
cp training_configs/super_rl/cmh_repro.user.example.env /scratch/YOUR_WORK/my-run.env
# Edit paths/account and W&B identity; secret values belong in the private file.
bash tools/super_rl/cmh_repro/submit.sh /scratch/YOUR_WORK/my-run.env fresh
bash tools/super_rl/cmh_repro/submit.sh /scratch/YOUR_WORK/my-run.env fresh --submit
```

The first call only checks inputs and Slurm acceptance. The second creates one
private run root, submits a CPU preflight and a 256-GPU training job depending
on successful preflight. A failed dependency cancels the queued training job;
it does not spend GPU time debugging failed imports or a missing judge key.
If either submission's response is lost, inspect Slurm before retrying.
A partial preparation is retained for diagnosis; choose a new root for retry.

CPU preflight checks image identity, builds/stages runtime dependencies, verifies
native component imports and schema, scans data routes/digest/horizon and model
shards, and makes reasoning-on positive/negative hosted-judge requests. This is
not a GPU smoke. For a new image/model/topology or unvalidated adapter, run a
separate topology-preserving smoke before full-scale use.

## Effective full-scale contract

| Setting | Value |
| --- | --- |
| GPU shape | 64 nodes × 4 GB300; 64 learner + 192 rollout |
| Slurm | batch_long/normal, segment16, 140 CPUs/node, exclusive |
| Allocation/deadline | 48h / safe-save at 47h; preemption can occur much earlier |
| Training | TP4 / CP4 / EP16 / PP1, compact Router Replay / Ray |
| Batches | 256 prompts × 16 generations = GBS4096, age2 |
| Policy | reasoning ON; 102400 output, 131072 context/packing |
| Serving | max_num_seqs256, max_num_batched_tokens32768, MTP1 |
| Judge | hosted DeepSeek V4 Flash, medium reasoning, T1/top_p1, 8192 cap |
| Verdicts | 3 total attempts (2 retries), fail closed; valid negatives accepted |
| Schedule | 100 absolute optimizer steps, ordered data |
| Checkpoints | permanent every10; FT every1, keep latest1 |
| W&B | own entity/project/run ID, online; same ID only for own resumes |

The recipe retains the original full config, including historical unconsumed
`fully_parallel_*` and `distributed_timeout_minutes` keys. Do not interpret
these as verified effective parallel-save or timeout overrides in this code
revision. Correcting them would be a separately reviewed behavior boundary,
not an undocumented reproduction change.

## Resume after preemption

Wait until the previous allocation and its writers have fully exited. Keep
the same source, user file, checkpoint root and W&B identity.

```bash
bash tools/super_rl/cmh_repro/submit.sh /scratch/YOUR_WORK/my-run.env resume
bash tools/super_rl/cmh_repro/submit.sh /scratch/YOUR_WORK/my-run.env resume --submit
```

The launcher refuses a still-live previous job. A writer lock prevents two
allocations writing the same root concurrently. The driver checks the newest
finalized `step_N`, matching latest-status/training metadata, 64 nonempty shards,
optimizer, dataloader and rollout-frontier state. It restores model/optimizer/
dataloader, sets `WANDB_RESUME=must`, and **does not load serialized Ray replay**.
Interrupted rollout work is regenerated; target 100 remains absolute.

The launcher's native restore receipt reports structural validation, not proof
of full distributed loading. Verify the subsequent successfully-loaded message,
first resumed optimizer update, finalized checkpoint and matching W&B metrics.

## Outputs and sharing

- `RUN_ROOT/slurm/`: allocation logs
- `RUN_ROOT/ray-logs/JOBID-logs/ray-driver.log`: driver
- `RUN_ROOT/attempts/JOBID/`: per-allocation logs and restore receipt
- `RUN_ROOT/checkpoints/step_N/`: DCP training checkpoint, not an HF export
- `RUN_ROOT/manifests/`: preflight, import and judge receipts

Raw Gym logs/configs may contain resolved credentials. The run root is private
by design. Share only sanitized receipts and explicitly approved checkpoint
artifacts after final writers stop and ancestor/descendant permissions are
verified; never recursively make the raw run public.
