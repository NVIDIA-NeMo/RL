# Qwen3-8B DFlash/DSpark cadence screening

This harness prepares the user-approved 200-step matched screen:

- one shared no-spec baseline;
- DFlash K5 and DSpark K5 with static, every-step, fixed 5/10/20, and adaptive cadence;
- DAPOMath17K, seed 42, OSL 1024, GBS 8, PPS 2, GPS 4, TP2/CP1 on one four-GPU OCI-HSG node;
- W&B project `sna-specdec` with one unambiguous run identity per arm;
- closed analysis window steps 21-200.

The DFlash and DSpark revisions are the exact public snapshots used by the prior
verified OCI-HSG online/fixed experiments. The target revision and immutable
2026-08-18 container digest are pinned as well. The older OSL1024/GBS8 result
was PARD-2; only its workload scale is reused. Its drafter and results are not
used as DFlash/DSpark evidence. The four-GPU TP2/DP2 topology follows the later
verified DFlash/DSpark online runs because OCI-HSG nodes expose four GB200 GPUs.
The draft-only optimizer is explicitly pinned to the recipe's inherited
Megatron values: LR `5e-6`, minimum LR `5e-7`, and weight decay `0.01`.

`static` performs the normal fresh-start draft installation as version 0 and
uses a sparse-update interval of 201, so no draft training/refit is allowed in
the 200 policy steps. `always` updates each step. Fixed arms update at exact
5/10/20-step intervals. Adaptive uses min 5, max 20, EWMA 0.1, degradation
0.02, recovery 0.01, 20 minimum observations, and a maximum burst of 10.

The harness intentionally fails before SLURM submission until the integrated
product head supplies all of the following: real worker update receipts, real
weight-apply receipts, selected-rollout count/version provenance, synchronous
controller calls to the cadence decision/finalizer, and adaptive runtime
enablement. The current base `e8464ccfa` contains Task 7 helpers but is not a
runnable cadence product head. The exact final product artifacts must also be
adapted without loss into the normalized `runtime-evidence.json`,
`terminal.json`, and `decision-ledger.jsonl` contract before submission.

## Local contracts

```bash
python3 -m unittest discover \
  -s research/qwen3_8b_draft_cadence_200step/tests -v
bash -n research/qwen3_8b_draft_cadence_200step/run_arm.sh
```

## Preparation only

After the final signed product head is integrated, create one immutable
manifest outside Git. This does not submit a job:

```bash
python3 - <<'PY'
from pathlib import Path
import os
from research.qwen3_8b_draft_cadence_200step.launch import materialize_manifest

print(materialize_manifest(
    result_root=Path(os.environ["RESULT_ROOT"]),
    product_head=os.environ["PRODUCT_HEAD"],
    harness_head=os.environ["HARNESS_HEAD"],
))
PY
```

For each arm, construct the test-only submission with `build_submission(...)`
and execute it only through `run_submission(...)`; that function performs the
signed-source, product-capability, checkpoint, and byte-level container gates
before invoking `sbatch`. The caller must first select the highest eligible
current FairShare account. Only after all 13 test-only calls pass should the
caller rebuild each command with `test_only=False`, submit exactly once, and
monitor the filtered job set at 60-second cadence for at least five minutes.

## Terminal gates and report

Every spec arm requires 200 contiguous decision rows, target synchronization
on all 200 steps, selective draft update/refit accounting, accepted/drafted
token counts, the successful version-0 initial draft refit, serving-version
provenance, and durable checkpoint receipts at steps 50/100/150/200. Missing or
inconsistent evidence is a failed arm.

The run script invokes both the checkpoint resume gate and the terminal gate;
a process exit of zero is not accepted as experiment success by itself.

Place the canonical W&B history export at
`RESULT_ROOT/ARM/wandb-history.jsonl`, then run:

```bash
python3 -m research.qwen3_8b_draft_cadence_200step.report \
  --result-root RESULT_ROOT --output-root REPORT_ROOT
```

The report uses logged generation and E2E throughput instead of reconstructing
throughput from averaged token counts and times.

## Compute-visible source recovery

The first actual 13-arm array, parent job `6476732`, failed before Python,
Ray, W&B, or training. Every task inherited `/home/sna/RL-q8-cadence-d9f3a89ac`
as the Slurm submit directory while `ray.sub` intentionally passed
`--no-container-mount-home`; Pyxis therefore could not enter the requested
container working directory. The failed result root and its exactly-once
ledger are immutable and must not be reused.

Recovery uses `staged_launch.py`: the exact signed recursive-clean checkout is
archived and SHA256-bound on Lustre, then each array task extracts an isolated
copy under
`/raid/scratch/q8c200-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}-r${SLURM_RESTART_COUNT:-0}/source`.
That exact path is explicitly mounted into the nested Pyxis container. This
also prevents the 13 concurrent MCore jobs from racing on generated helper
artifacts in one shared checkout. A preflight-only task-0 canary using the same
archive, renderer, mount, and `ray.sub` path is mandatory before the recovered
`0-12` array is submitted.
