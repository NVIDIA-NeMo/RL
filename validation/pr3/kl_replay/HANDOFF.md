# Streaming KL/performance investigation handoff

User authorized completing same-checkpoint/same-rollout KL, numerical state and
fixed-workload/actual-overlap diagnosis. The original300-step run completed;
on2026-11-08 the user explicitly authorized continuing it to600. Preserve the
protected300snapshot; the earlier no-continuation instruction is superseded.

Latest2026-11-08 06:46PST: continue ORIGINALstreamingLA0 to600, not another
diagnostic. Read ../RESUME600.md and session_state.md latest section. Same
model/LA0/24H100/image1030/W&B id yuekaiz5090explorela0s42, numericalworker
unchanged. Resumejobs21509383->21509384->21509385->21509386; firstRUNNING on
pool0-[00139,00585,01164], othersDependency. ActualYAMLtarget600 and dataloader/
weightrestorefromstep300verified; first301commitstillpending. Protected300
hardlinks at ../results/resume600-source/step_300; originalpin210unchanged.
Monitor uv2448041/python2448158 polls10min and resumesTHISthread; source
session/20261105_pr3_design/monitor_resume600.py. Runtime filesunder
../slurm/resume600-monitor, readlateststatus/logs. Code+recipecommit2a9ec1e77
pushed; docs/600plot-loader updatefollow. DoNOTstartold900/MCore/siblingtasks
or repeat completedKLdiagnostics. OriginalBF161%gateFAILED remainsunchanged;
continuationisexploratory. Historicalstatusbelowdoesnotoverridenewuserrequest.

Latest verified2026-11-06 07:34PST (supersedes all historical status below):

- Original300,8plots,21373254,21391011 andCPU21392643 ALLCOMPLETED0:0.
  No activePR3training/diagnosticjob and no continuation needed. Preserve300.
- Followup21391011 source756c5eb77,22m08s,all16sourcehashes matched;112rank
  files/all16counter210/inputIDs+validcolumns/dtypes/layouts verified. Initial
  rawmomentsBF16/scalespresent, LR1e-6. Exactfullrepeat; fixedsplit matches
  independentnumeratorgrad/model/public+rawoptimizer onALL16ranks.
- MatchingproductionDPmicrobatches reduces30.6616%stored-LPgradientgap to
  1.18555%, and20.2634%recomputedpipelinegap to1.13169%. Original1%BF16
  qualification STAYSFAILED; don't relabelpass. This supportslayout-dependent
  execution plusnormalizerrounding; exactoperator and300stepcausalKLgapunknown.
- Newfull/recomputedproductionevalLPmaxdiff0. CPU21392643(3m07s)checksold
  productionLPvsnewunifiedfull: ALL38413validtokens bitexact acrossallocations.
  Oldfullvsunifiedfullmax.676,p99.15555. Postupdatenumerator/fixedsplit/prodsplit
  savedLPsbitexact too; directraw-statecomparisonprod-vs-numeratorNOTrecorded.
  DoNOTinferthatmissingcomparisonwasexecuted. Normalization-onlypostupdate
  maxLPdiff4.2069,p99.2161,3tokens>1; thisusesoldrollout, notnextliveLA0genKL.
- Unifiedlayoutfixedworkerfull5.630319s/split8.732814s median3pairs (excludes
  rollout/networkRayTQ), noendtoendspeedupestablished. Earlierwarmungated26.77ms
  traceintersectionremainsprovisionalwithoutcapture-timehostclockoffsetbound.
- Archivedproduction-layout-summary.json, -rank-manifest.json, -source-hashes.txt,
  -lp-comparison.json. EP16_FINDINGS.md andRESULTS.md explainfindings/limits.
  CPUcomparisonpy/sh onlypostprocesssavedtensors; nosubagents/newthreads,
  no900/MCore/siblingexperimentchanges. Preservepin210/rawcapturesforfollowup.
- Dedicatedwatch1876913/1876993endednormallyafterterminalnotice. Oldread-only
  watch3727915wasverifiedasTHISrepo'sandstoppedafterallPR3jobscompleted; logsretained.
  No autorerun needed; anyfuturedebugmustreuseevidence/explicitremainingquestion.

Historical status2026-11-06 06:55PST (superseded):

- Original300 and refreshed8plots COMPLETE, pushed5dd1a2735; no continuation.
- Diagnostic21373254 COMPLETED0:0,53m32s; all13source hashes matched at end.
  Sourcec7ddf1035. Seven cases/112rank files verified, archived ep16-summary.json,
  ep16-native-capture-check.json,ep16-rank-manifest.json,ep16-source-hashes.txt.
  Read EP16_FINDINGS.md. Original1%BF16 gate staysFAILED.
- Stored-input genKL full/split differs5.5e-13, independentK3error3.8e-9.
  Fullrepeat exactall16; fixedshardstream matches independentnumerator reference
  exactly in grads/model/public+rawoptimizer. Originalnormalizergradgap1.20855%.
  NEW production DP placement30.6616%global gradientdifference; recomputedLP
  full/split20.2634%. Needslayoutcontrol, doNOTdismissas1%rounding/declarebugfixed.
- Fixedworker medianfull5.452882s/split8.692046s (3alternatingpairs). No rollout/
  Ray/TQnetwork included. Nativefull/streamtokens differ despite sameprompts/IDs.
  Warmungated traces intersect26.77363ms/8108intervals, buthistoricalcrossnode
  clock-offsetboundmissing: provisional; notthroughput. Profileexportadds220s.
- Followup21391011 RUNNING onpool0-[00148,01102],2nodes16H100,interactive4hmax,
  image1030, auto-exit/release. No newrollout/300run. Reuses SAME saved32responses
  and pinned210. Entrypoint run_layout.sh->distributed.py --replay-script
  production_layout.py. Code committed/PUSHED756c5eb77. Worker,oldparityharness
  AND replay.py unchanged. Wrappermakes full-inputcontrols useproductionDP
  microbatches too; validatesfetchedvalid-tokenfields/dtypes, recordsinitial
  optimizer/schedulercounters. Exactmatchproductionstream-vs-numeratorwould
  narrow30%findingtoexecutionlayout; mismatchrequiresstateinvestigation.
- New10minwatcher uvPID1876913 (--result-prefix production-layout) resumesTHIS
  same thread on start/terminal/stall. Logwatch-layout.log; main300notice already
  seededhandled. Olddedicatedwatcherendednormallyafter21373254COMPLETED.
  CheckcurrentlivePIDs/logs. No agents/newthreads/duplicateallocations.
- Freeze diagnosticpy/sh/yaml afterfollowupsource-hashes capture. Keep pin210
  and originalcaptures untilfollowup/retriesfinish. Productsourcehashes still
  7f598242.../ccb26bda...; noold900/MCore/siblingRL-pr4963changes.

Latest verified 2026-11-06 06:04 PST: original streaming LA0 reached300,
checkpoint300 durable; final21331930 COMPLETED0:0. Do not launch any continuation.
Curve refresh CPU-only job21389289 submitted once. Diagnostic21373254 RUNNING
since05:40 on pool0-[01901,01903,01908]; stream and full native210->212 phases
finished, warm profiling in progress, distributed replay still pending.
First stream update has nonzero grad_norm0.5121751 and genKL0.001129127546;
native full first-update genKL0.001208236162 is NOT yet a same-token comparison.
Source HEADc7ddf1035 and frozen worker/harness hashes verified. Preserve pinned210.

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
