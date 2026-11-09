# Same Gemma streaming LA0 experiment extended to 900

COMPLETED: the same experiment reached 900 on 2026-11-09. All owned allocations
were released and the supervisor exited. See [RESULTS900.md](RESULTS900.md) and
[final status](results/final900-status.json). Stale monitor notifications must not
start more work. Gemma fixed-input parity remains unassessed; the earlier Qwen
BF16 1% gate remains failed. The historical procedure follows.

On2026-11-09 the user explicitly requested900 steps. This supersedes650 and150,
including queued older monitor messages. Continue the SAME W&B run:
https://wandb.ai/yuekaiz/nemo-rl-automodel-sc/runs/yuekaiz5090gemmala0s42.

Allocation21568255 completed340->524 normally with its already-loaded650 target
and unchanged run_resume650.sh/launch_resume650.sh/config. Driver exit0 and
checkpoint524 scalar audit passed before its idle node was released: Slurm
CANCELLED0:0 after3h45m37s is expected cleanup. All505 saved optimizer counters,
scheduler/trainer/sampler counters524,LR2e-6 and16768 consumed prompts match;
the saved optimizer metadata SHA matches the audit on readback.

The900 supervisor then acquired ONLY stage3 interactive allocation21579059,
pool0-01905,start07:13:19PSTNov9,end11:13:19. Its target is900 and driver_prefix
resume900. The container/direct attachment started and actual checkpoint524
preflight passed: only target/display name changed, saved counters and frozen
model/config/numerical source matched. The driver restored the saved data
loader and weights, then first525 committed with trainer_v525,lag0,two dispatches,
512 valid samples,nonzero advantages/gradient and LR2e-6. The SAME W&B run resumed
with the900 display name and525 selected metrics matched the native receipt
exactly. See results/resume900-first-recovery-21579059.json. The log restored64
unfinished groups next to0 canonical groups; this observation is not an
independent all-group recovery uniqueness or completed canonical replay test.
Inspect live state/logs before acting; do not duplicate this job. W&B temporarily
finished524 with metrics matching the native final receipt during the transition;
this is not completion of900. See results/checkpoint524-state.json,
resume900-progress-524.json and resume900-provenance-21579059.json.

Stage3 subsequently completed524->709 with driver exit0 and a durable709 scalar
audit, then released its idle node normally. All505 optimizer counters and
scheduler/trainer/sampler counters709,LR2e-6 and22688 consumed prompts match;
the stored optimizer metadata SHA matches the audit. W&B finished709 and the
selected final metrics agree with the native receipt. An ignored W&B atexit
BrokenPipeError occurred while closing its service connection; saved state and
the final online step were independently checked, so this was not a failed
training update. The teardown warning is retained in the original log.

The supervisor then acquired ONLY stage4 allocation21594351,pool0-00826,
start11:02:06PSTNov9,end15:02:06,target900,resume_from709. At this snapshot its
container/direct attachment started and actual checkpoint709 preflight passed
with unchanged model/config/numerical source. The first recovered710 committed
with trainer_v710,lag0,two dispatches,512 valid samples,nonzero advantages/gradient
and LR2e-6. Unsampled W&B history709,710,711 exists and selected metrics match
native receipts exactly on this SAME run. Checkpoint710 independently has all505
optimizer counters710,scheduler/trainer/sampler710,LR2e-6 and22720 consumed prompts.
See results/resume900-first-recovery-21594351.json,checkpoint710-state.json and
resume900-provenance-21594351.json. The log explicitly redispatched64 unfinished
groups before new dataloader work, next to0 canonical groups; this is not an
independent all-group consumption/uniqueness or raw optimizer parity oracle.
Inspect live state and per-job logs before acting; do not duplicate this allocation.

resume900.yaml inherits streaming.yaml and changes ONLY max_num_steps900 and
the W&B display name. Keep Gemma4 E2B IT,32prompts x16responses=batch512,
4training+4rollout H100,FSDP2/EP1/TP1/CP1,LA0/in_order/readiness16,buffer32,
seed42,train/logprob MBS1,seq4096/gen2048,LR2e-6/FusedAdamBF16states,
saved warmup10+ConstantLR,checkpoint every10/keep2/deadline3h45,image1030.
max_num_epochs remains1; actual original loader1791700rows covers900x32=28800
consumed prompts. Restore weights, optimizer, scheduler, dataloader and TQ/recovery;
never restart warmup or reset the cursor. Allocated-root configuration validation
compares the entire original config except target/display name and checks frozen
worker/harness hashes. Each future preflight also checks its saved stage/counters.

For continuity, runtime state, global lock and HOLD flag KEEP their historical
paths under slurm/resume650/. The executable remains resume650_supervisor.py,
but its current TARGET is900. Do not launch another supervisor or create a second
runtime directory. Read state.json before acting. The old650 supervisor was
stopped without stopping the independently-sessioned training launcher; the new
process adopts that job and preserves all completed stage records. Stages record
their driver_prefix and target_steps explicitly: old stages resume650/650,
new stages resume900/900. New job names are yuekaiz-gemma900-debug-N, comment
yuekaiz5090gemmala0s42-resume900, and this repository workdir is mandatory.

The bounded allocation budget is now six continuation allocations TOTAL,
including the two already recorded, to allow the longer900 target. This is a
maximum, not six immediate requests or blind retries. One node at a time;
no batch training, dependency chain or new baseline. On failure, persist cause
and notify this SAME thread; diagnose preserved logs before any repair/restart.

New-stage files: launch_resume900.sh/run_resume900.sh/preflight_resume900.py;
slurm/resume900-driver-JOB.log,resume900-exit-JOB,resume900-launcher-JOB.log;
results/resume900-provenance-JOB.json. Old650 logs and proof files stay intact.
Monitoring polls30s and records10minute snapshots/milestones/failures in the same
conversation. Check first recovered step=checkpoint+1, trainer version,lag0,
nonzero advantages/gradients,LR2e-6 and exact selected W&B/native metrics.

On user hold/cancel, create slurm/resume650/HOLD FIRST to prevent new allocation
or launch, then handle the active owned work as requested. The flag alone does
not kill active training. Do not clear it without user resume authorization.

At900, require saved optimizer/scheduler/trainer/sampler counters900 and28800
consumed prompts, driver exit0, final same-run W&B state/metrics and eight plots.
plot.py --max-step900 merges original+650+900 native logs with strict contiguous,
unique receipts/versions/lag checks. Historical V1/fullSC baselines end157/164;
common statistics stop157 and plots shade absent coverage. No extrapolated
baselines, extra baseline training or fabricated validation accuracy. Release
only the owned node after successful audit/plot; inspect the figure/report and
push the existing yuekaiz/pr5090-automodel-streaming branch.

This remains exploratory. Gemma fixed-input streaming parity NOT ASSESSED;
original Qwen BF16 1% gate FAILED. Policy KL spikes and gradual generation KL
drift remain unresolved; completion does not establish numerical acceptance,
convergence equivalence or controlled speedup. Do not touch Qwen600,old900,MCore
or sibling RL-pr4963 experiments. Later user hold/cancel takes precedence.
