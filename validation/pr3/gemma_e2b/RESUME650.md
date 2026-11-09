# Authorized continuation of the same Gemma streaming LA0 run to650

SUPERSEDED: the user extended this SAME experiment to900 on2026-11-09.
Read [RESUME900.md](RESUME900.md) first. The runtime state/lock/HOLD paths retain
their historical slurm/resume650 names; target_steps900 is now authoritative.
Older650 monitor messages do not override900 authorization. History follows.

On2026-11-08 the user requested more than600 steps. Target650 supersedes the
previous150-step stopping point. This remains the same experiment and W&B run:
https://wandb.ai/yuekaiz/nemo-rl-automodel-sc/runs/yuekaiz5090gemmala0s42.

Initial allocation21546953 completed150 and the first150 comparison with driver
exit0, then released its idle node normally at23:31PST on2026-11-08. Saved state
audit found trainer/sampler/optimizer counters150, scheduler last_epoch150,
LR2e-6 and4800 consumed prompts. The supervisor waited for its terminal state
and removal from squeue before requesting continuation allocation21557906.
That allocation completed150->340 on pool0-00941 with driver exit0 and was
released normally after3h45. Preflight and the same W&B resume passed. First recovered update151 committed
with trainer_v151,lag0,two dispatches,nonzero advantages/gradient and LR2e-6;
the same W&B run's151 metrics independently matched the native receipt exactly.
See results/resume650-first-recovery-21557906.json.
Checkpoint340's four optimizer shards/all505 counters340, scheduler last_epoch340,
LR2e-6 and10880 consumed prompts were audited before release. The stored optimizer
metadata hash still matches the audit. W&B finished this segment at340 with metrics
matching the final native receipt; a finished segment is not completion of650.
The supervisor then acquired21568255 on pool0-00671, starting03:23:49PSTNov9.
Its container is ready and direct attachment has started. Checkpoint340 preflight
passed with unchanged model/config/numerical source and the same W&B identity;
first recovered341 committed with trainer_v341,lag0,two dispatches,512 valid
samples,nonzero advantages/gradient and LR2e-6. The same W&B run resumed and its
341 selected metrics matched the native receipt exactly. See
results/resume650-first-recovery-21568255.json. The recovery log contained64
unfinished rollout groups next to0 canonical groups and redispatched unfinished
work before reading new prompts. This observation does not independently prove
all-group recovery uniqueness or exercise completed canonical replay groups.
Read the live state.json/native log before acting; this snapshot
does not establish completion of650.

resume650.yaml inherits streaming.yaml and changes ONLY max_num_steps650 and
the W&B display name. Same local Gemma4 E2B IT,32x16=batch512,4training/4rollout,
FSDP2/EP1/TP1/CP1,LA0/threshold16,buffer32,seed42,LR2e-6/FusedAdam,
warmup10+ConstantLR,seq4096/gen2048,checkpoint period10/keep2 and image1030.
Do not reset the optimizer/scheduler or data cursor. max_num_epochs remains1:
the original runtime loaded1791700 training rows, enough for650x32=20800 prompts.
No change to the data construction or numerical worker. Preflight compares the
whole original resolved config except target/name and checks the saved counters,
source hashes and model/data settings before each continuation.

Slurm interactive has MaxTime4h. Continuations therefore use sequential4h debug
allocations, with the existing3h45 checkpoint deadline. They run Ray idle and then
directly attach run_resume650.sh; no batch training job/dependency chain. Only
one node is allocated at a time. Normally three further segments should suffice;
the supervisor has a hard maximum of four additional allocations, not unlimited
retries. On an actual failure it records the cause, notifies this same conversation
and stops for diagnosis; it never blindly retries a failed driver.

Launch the login-node supervisor once, after inspecting existing state:

```bash
uv run --no-project --python /usr/bin/python3 python \
  validation/pr3/gemma_e2b/resume650_supervisor.py --thread "$CODEX_THREAD_ID"
```

The deployed process is detached with nohup/setsid. It holds an exclusive lock.
Runtime state is under slurm/resume650/: state.json, supervisor.pid,
snapshots.jsonl and notifications.jsonl. It monitors every30s and records10min
snapshots/milestones into the SAME existing conversation; no new thread or agent.
The old150-only watcher is stopped after the new supervisor is verified healthy.
An old queued150 notice cannot override this newer650 authorization.

Each allocation has slurm/resume650-driver-JOB.log, resume650-exit-JOB and
resume650-launcher-JOB.log. Logs are preserved across allocations. Preflight writes
results/resume650-provenance-JOB.json with the restored step and source/config
hashes. Successful segments audit their durable optimizer/scheduler scalar state
before the supervisor releases ONLY their recorded allocation. Normal cleanup
shows Slurm CANCELLED; actual driver exit0, durable progress and native receipts
are the training evidence. Verify the first recovered step/version/lag and W&B
continuity after each allocation. Any real recovery rollback must remain explicit
in the logs; do not silently deduplicate repeated optimizer-step receipts.

To prevent new allocations/launches during a user-requested hold or cancellation,
create slurm/resume650/HOLD first and inspect the currently running allocation.
The flag alone does not kill active model work. Stop/checkpoint/release that work
according to the user's instruction. The supervisor checks HOLD before every
new allocation and launch. Do not remove it without a later user resume request.

At650, plot.py --max-step650 merges original/continuation committed receipts,
checks contiguous versions and lag0, audits checkpoint650 (expected20800
consumed prompts) and generates the same8panels. The default remains150 for the
original first-segment post-hook. Both actual baseline histories
are retained: V174860be5 ends157, fullSC96779513 ends164. Shared statistics stop
at157 if those histories remain unchanged; absent baseline values are not
extrapolated and the unshared region is shaded. No new baseline training and no
fabricated validation accuracy. After computation, verify W&B finished650,
inspect plots and publish the updated results on the existing branch.

This is exploratory evidence. Large Gemma fixed-input streaming parity is NOT
ASSESSED; the original Qwen BF16 1% gate remains FAILED. No convergence,
same-input numerical or controlled speedup claim follows from reaching650.
Keep Qwen600/old900/MCore/sibling KL investigations separate. Later user hold or
cancellation takes precedence over this continuation authorization.
