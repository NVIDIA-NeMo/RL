# Qwen3-8B cadence screening preparation report

## Outcome

The 13-arm, 200-step Qwen3-8B DFlash/DSpark K5 screen is prepared as an
experiment-only harness. Nothing was pushed or submitted. The harness rejects
the current Task 7 helper-only product base before scheduling because the real
synchronous controller and receipt producers are not complete.

## Frozen experiment contract

- Shared no-spec baseline.
- DFlash and DSpark: static, every-step, fixed 5/10/20, and adaptive.
- Static requires a successful initial version-0 draft refit and no scheduled
  draft update or refit in policy steps 1-200.
- Adaptive: min interval 5, max interval 20, EWMA alpha 0.1, degradation 0.02,
  recovery 0.01, 20 minimum observations, and at most 10 burst updates.
- DAPOMath17K, seed 42, OSL 1024, max input 2048, max model length 4096,
  GBS 8, two prompts per step, four generations per prompt.
- One OCI-HSG node with four GPUs, TP2/PP1/CP1, no sequence packing or sequence
  parallelism. This follows the later verified DFlash/DSpark OCI runs; only the
  earlier PARD-2 experiment's OSL1024/GBS8 workload scale is reused.
- Draft-only Megatron optimizer: LR 5e-6, minimum LR 5e-7, weight decay 0.01.
- PIECEWISE CUDA Graph mode with the explicit 22-bucket coverage pinned in the
  manifest.
- W&B entity/project `nvidia/sna-specdec`, group
  `qwen3-8b-dflash-dspark-cadence-200step-v1`, and unique names for all arms.
- Durable checkpoints at steps 50, 100, 150, and 200. The cadence runtime root
  is the arm result root so product checkpoint receipts and the training
  checkpoint tree share the same identity.
- Ray and generic temporary storage are pinned to `/tmp`; SLURM output is kept
  below the durable experiment result root.

## Pinned identities

| Artifact | Identity |
|---|---|
| Target | `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218` |
| DFlash | `z-lab/Qwen3-8B-DFlash-b16@9b41424b7109f9c5413454f481b09a82b85333f4` |
| DSpark | `deepseek-ai/dspark_qwen3_8b_block7@03326e5043815da1f81b109078b2889737c26017` |
| Container | `nemo_rl_nightly_20260818_20260818_6296116.sqsh` |
| Container SHA256 | `6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44` |

The exact target/drafter snapshot directories, each snapshot's `config.json`,
the adjacent immutable container metadata digest, and a fresh SHA256 of the
container bytes are mandatory remote preflight checks. A missing or different
identity fails closed.

## Evidence gates

Before even invoking `sbatch`, the launcher requires the exact signed product
SHA, a clean source tree and recursive submodules, the pinned model snapshots,
the pinned container digest, and a product capability check. Resume additionally
binds the arm and product identity to the latest durable checkpoint and its
digest-bound contiguous decision-ledger prefix.

After training, the job itself invokes the terminal receipt gate. Reporting
requires terminal 200/200 success, 200 contiguous decisions for each spec arm,
target refit success every step, draft-selective update/refit outcomes, exact
deterministic cadence counts, adaptive acceptance/count binding and interval
semantics, continuous serving versions, initial version-0 refit evidence,
checkpoint receipts, CUDA Graph coverage, and Step 1/Step 2 completion. The
baseline requires a neutral empty schedule ledger.

The report consumes steps 21-200 inclusive. It uses canonical logged generation
and E2E throughput, never reconstructed throughput, and reports generation and
step time, cumulative/mean refit-path time, count-weighted acceptance, window
requests, full-run updates/refits, forced/skipped decisions, and per-reason
schedule counters.

## Genuine RED evidence

The implementation was developed test-first. Observed RED states included:

- missing matrix and report modules during initial test collection;
- missing launcher and resume/container validators;
- a corrupted terminal decision-reason count being accepted;
- unprefixed acceptance keys not matching canonical `train/vllm/...` W&B keys;
- failed initial draft refit evidence being accepted;
- no explicit draft optimizer overrides;
- cadence runtime and checkpoint roots being different;
- the resume and terminal gates not being invoked by the run script;
- scheduler logs being routed outside the experiment result directory;
- adaptive observed acceptance not being bound to accepted/drafted counts;
- an adaptive update before 20 observations being accepted without replaying
  the pinned state machine;
- a tampered checkpoint decision-ledger prefix being accepted for resume;
- stale container metadata being trusted without hashing the image bytes;
- an omitted four-GPU Ray environment value and missing SLURM working directory;
- product capability checks running only after allocation instead of before
  `sbatch`;
- ambiguous K5 naming on the no-spec baseline; and
- absent flattened full-run schedule counters in CSV output.

Each RED state failed for the intended reason before the corresponding harness
change was added.

## Local GREEN evidence

The final local verification commands are:

```text
python3 -m unittest discover -s research/qwen3_8b_draft_cadence_200step/tests -v
uvx ruff check research/qwen3_8b_draft_cadence_200step
uvx ruff format --check research/qwen3_8b_draft_cadence_200step
bash -n research/qwen3_8b_draft_cadence_200step/run_arm.sh
shellcheck research/qwen3_8b_draft_cadence_200step/run_arm.sh
git diff --cached --check
```

The dependency-independent suite contains 29 contract tests. The exact final
command outputs are recorded in the local review snapshot handoff.

## Submission blockers

1. The current base `e8464ccfa66a04d22a99b5ad17385b7effa98213` explicitly
   rejects adaptive cadence and calls the receipt requirement with absent
   worker/apply receipts before training.
2. `grpo_train_sync` does not yet call the count-weighted prepared decision and
   scheduled refit finalizer in the real loop.
3. The weight-sync interface and TQ policy do not yet supply/advertise the
   required digest-bound draft apply receipt.
4. The final integrated product's exact terminal output adapter must map its
   native cadence checkpoint/schedule artifacts to the normalized terminal,
   decision-ledger, and runtime-evidence contract used by this harness. This is
   intentionally not guessed against a helper-only base.
5. Remote checkpoint presence, container digest, FairShare account eligibility,
   and `sbatch --test-only` remain cluster-side gates. They were not claimed
   from this local preparation.

No W&B run, SLURM job ID, throughput result, or speedup exists for this prepared
screen yet.

## 2026-08-23 first-array failure and recovery contract

Product Linux gate `6476034` completed successfully, and all 13 scheduler
test-only checks passed. Actual array `6476732` then failed all 13 tasks before
training with the same Pyxis error: the `/home` checkout selected as
`SLURM_SUBMIT_DIR` was not mounted because `ray.sub` uses
`--no-container-mount-home`. No W&B or cadence evidence was created.

The recovery contract stages one SHA256-bound source archive on Lustre and
extracts one private source tree per array task to node-local `/raid/scratch`.
The recovered runner explicitly mounts that source path, keeps all durable
outputs on Lustre, and permits full arm submission only after an identical
preflight-only startup canary terminates successfully. Recovery uses a new
result root and W&B run generation; none of the failed array's manifests or
ledgers may be overwritten.

Startup canary `6479002` then exposed a separate staging assertion before Ray:
the repository tracks `ray.sub` as mode `100644`, but the first wrapper required
it to be executable even though it invokes the file through `bash`. The recovery
wrapper now requires a regular file instead, records failing shell commands with
an ERR trap, and routes scheduler stderr to a durable per-task `.err` file.

Recovery canary `6479237` subsequently completed successfully, proving the
signed source archive, node-local extraction, explicit Pyxis mount, Ray startup,
and artifact preflight. The following array `6479331` reached the real GRPO
entrypoint on every arm, then failed before training because the harness used a
plain Hydra override for keys absent from the structured base config. The first
rejected key was `policy.generation.vllm_kwargs.max_num_seqs` on all 13 arms.
All rendered fields now use Hydra's `++` force-add-or-override form, and the
startup canary performs real `MasterConfig` composition for every arm before a
new array may be submitted.
