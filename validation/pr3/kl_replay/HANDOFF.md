# Streaming KL/performance investigation handoff

User authorized completing same-checkpoint/same-rollout KL, numerical state and
fixed-workload/actual-overlap diagnosis. Keep the separate300-step run running.

- Implementation/code commit:76c5a10dc on yuekaiz/pr5090-automodel-streaming,
  pushed to yuekaizhang/RL. Product worker/parity harness remain unchanged at
  SHA2567f598242.../ccb26bda...; their source guards protect original continuation.
- Full diagnostic job21373254:3nodes24H100,2h,batch,image1030. Submitted22:30PST
  Nov5; pendingPriority at22:34, scheduler estimate unstable. No duplicate job.
  Entry run.sh; source hashes checked again on completion. Read actual Slurm/logs.
- Protected source: results/pinned_checkpoint/step_210 via hardlinks, read only.
  Original training may prune its own step210; the diagnostic pin stays until
  diagnosis is done. No physical extra copy. Do not remove before retries finish.
- Large protocol: independent stream/full native resumes210->212, disabledW&B/
  checkpointwrites/newoutputs; compare first token hash multisets before paired
  timing. Warm second iteration profiled separately with no gate/delay. Replay
  FIRST captured32responses at version210 on16train GPUs from the same full model/
  raw optimizer/scheduler. Seven controlled cases, common outerpadding; native
  captures include actualpadding. Require nonzeroadvantages. Preserve failed
  original1% BF16 gate; no tolerance change or qualification pass inferred.
- Debug21370799 manually released22:30 after successful capture/replay/profile
  plumbing. Its CANCELLED+0:0 is intentional resource cleanup. Six-case smoke
  measured genKL full .0006826675102 vs split .0006826675110 (~8e-13 difference).
  Backend timing full4.580s vs split6.564s. Cold/warm ungated traces both0overlap;
  warm first training kernel1.945s after lastgeneration, everygroupmax256.
  BUT shortenedQwen2.5-1.5B rollout had uniform rewards/zero gradients: this is
  plumbing/metrics/timing evidence only, not nonzero-update/largeMoE proof.
  Seventh full_recompute is a composition added after the six-case smoke;
  production source is frozen/hashed. Staticruff/format/bash/diffchecks passed.
- Native capture now occurs AFTER backward before TQcleanup, avoiding disturbance
  of first-training admission. Captures grouped by step_<rollout_weight_version>.
  Source resume with load_replay_bufferfalse uses saved loader cursor but fresh
  rollouts, not original210ledger. LPdelta writes do not expandcaller metadata;
  explicitly fetch actual training columns. Localadapterallowsoneput/partition,
  so replay newLPs injected into materializedinput. Reshard FSDP publicly before
  restoring snapshots after post-updateLP (inference may leave root unsharded).
- Other independent RL-pr4963/validation/kl_diagnosis investigates A_log FP32refit;
  do not modify its code/jobs/600checkpoint. Old900 jobs remainheld. No agents.
- Original300run:21331885 active, >232/300 by22:31;21331930afterokcontinuation.
  Read-onlywatch3727915 at10min now includes diagnostic ID for status. A dedicated
  metadata watcher in session/20261105_pr3_design/monitor_replay.py will enqueue
  terminal notices into THIS existing conversation via codexqueue--resume,
  target verified against this conversation; local runtime target/PID metadata
  lives in slurm/watch-target.json (ignored, not published).
  Localdaemon confirmed running by codex app-server daemon version. Do not create
  a new conversation/subagent, and respect any newer userhold/cancel instruction.

Next: inspect results/ep16-21373254/{stream.log,full.log,profile.log,replay/node-*.log,
summary.json}. If failure, recordactualphase andfix diagnosis without touching the
live worker/harness; before retry ensure no duplicateactiveallocation. Verify all
rank exact flags, fixedLPmetricaggregation, full-vs-numerator gradientrounding,
productionplacement/recomputedLP effects, postupdatecommon-layoutLPs, nonzero
advantages, tokenhashcomparability and warmGPUoverlap. Report backend timings
separately from native queue/setup/capture/coldstep effects. Never call completion
of this diagnostic full numerical acceptance. Push final evidence to same branch.
