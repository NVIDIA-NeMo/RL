# AGENTS_LOG

## 2026-09-04 — Production resumed after output-permission failure; GitHub port prepared

- Continuation job `6889718` restored checkpoint 70 and completed training
  steps 70 and 71 with finite TROPD advantages, but diagnostic output then
  failed with `PermissionError`: the shared `results/tropd-super35/runs`
  directory had mode `0666` and could not be traversed. Restored owner
  traversal with `chmod u+x`; checkpoint 70 remained intact.
- Submitted singleton continuation `6896017` under `nemotron_n4_post`
  with the same namespace, data, W&B run name, caches, and checkpoint path,
  excluding `nvl72130-T14,nvl72131-T02`. It restored step 70, tolerated one
  30-minute long-tail trajectory, and had advanced through training step 84
  with durable checkpoints 75 and 80 at the latest inspection.
- Ported the complete branch as a squash onto the GitHub `RL` clone's current
  `super-v3.5-posttraining` head `cdcbcd8f81b0572cbe21e80605adfe9d6613a53c`
  on local branch `kbhardwaj/super-mopd`. Resolved the sole
  `lm_policy.py` overlap by preserving the newer
  `fetch_returned_only=True` behavior alongside the new top-k padding schema.
- Validation in the target clone: 23 focused TROPD/diagnostics tests passed,
  Ruff lint passed, Ruff format check passed, shell syntax and
  `git diff --check` passed. Broader local collection remains constrained by
  missing optional audio dependencies, and Pyrefly cannot resolve the
  uninitialized Megatron submodule in this clone; the completed HSG smoke and
  live production checkpoints provide end-to-end runtime coverage.


## 2026-09-03 — Production continued from step 70

- Continuation job `6842961` restored checkpoint 60, trained successfully
  through step 70, and committed checkpoints 65 and 70. At step 71 it waited
  for one long-tail trajectory with 31/32 target groups ready until the cluster
  scheduler account `svc-hwinf-cs-sched` cancelled the allocation after 1h52m.
  Slurm again recorded no reason/comment, and no application exception was
  present; checkpoint `step_70` is intact.
- Verified that no job remained active, then submitted exactly one n4
  continuation, job `6889718`, with the same namespace, data, W&B name, cache,
  and checkpoint directory. It is pending on `batch_long` and will restore
  `step_70`. The fixed launcher passed
  `--exclude=nvl72130-T14,nvl72131-T02` in the actual `sbatch` invocation.

## 2026-09-02 — Production continued from step 60 under n3 account

- Continuation job `6834915` restored the step-55 checkpoint successfully,
  advanced through training step 64, and committed checkpoint `step_60`.
  It was then cancelled after 1h42m by the same cluster scheduler account,
  `svc-hwinf-cs-sched` (UID 146504), again with no Slurm reason/comment and no
  application exception in the driver log.
- Verified that no production job remained active, then submitted exactly one
  continuation, job `6842961`, using the user-approved `nemotron_n3_post`
  account instead of `nemotron_n4_post`. It preserves the production namespace,
  data, W&B name, cache, and checkpoint directory and is pending on
  `batch_long`; it will resume from `step_60`.
- Found that the target `super_launch.sh` did not translate the wrapper's
  `EXCLUDE_NODES` variable into an `sbatch --exclude` argument. Applied
  `ExcNodeList=nvl72130-T14,nvl72131-T02` directly to pending job `6842961`
  and added the missing launcher propagation so future submissions preserve
  registered exclusions. The two `nvl72d...` names remain filtered out on
  this cluster because they are not registered in its Slurm inventory.
- At the user's request, moved the still-pending singleton job `6842961` from
  `nemotron_n3_post` to `nemotron_n4_post` in place. This preserved its job ID,
  queue age, checkpoint path, and bad-node exclusions and did not create a
  duplicate submission.

## 2026-09-02 — Production resumed from step 55 after scheduler cancellation

- Production job `6805299` completed checkpoints through step 40 on its first
  allocation. Slurm terminated that allocation with signal 143 and
  automatically requeued the same job ID; the replacement allocation restored
  `step_40`, including the replay-buffer frontier at ordinal 1,280, and
  continued successfully through checkpoints 45, 50, and 55.
- The replacement allocation reached training step 56 but was cancelled at
  09:22 UTC by cluster scheduler account `svc-hwinf-cs-sched` (UID 146504).
  Slurm recorded no reason or comment. This was an external SIGTERM rather
  than an application exception; the newest committed checkpoint is `step_55`.
- Submitted exactly one continuation job, `6834915`, with the same production
  namespace, data, W&B name, cache, and checkpoint directory. It is pending on
  `batch_long` under `nemotron_n4_post` for resources and will resume from
  `step_55`. No duplicate production job exists.

## 2026-09-02 — Production bootstrap retry with bad-node exclusions

- Production job `6804959` failed before Ray, W&B, model initialization, or
  checkpoint activity. On batch host `nvl72130-T14`, the sandbox nginx process
  rejected its generated upstream list because allocated host
  `nvl72131-T02` did not resolve (`host not found in upstream`). The global
  sandbox step then terminated the other 31 healthy sidecars. This was an
  infrastructure/bootstrap failure and did not consume a training step.
- Resubmitted the same singleton namespace and stable checkpoint/cache paths as
  job `6805299`. It remains the only production job. While it was pending,
  applied `ExcNodeList=nvl72130-T14,nvl72131-T02` so neither node from the
  failed bootstrap can be allocated to the retry.
- Added the shared bad-node candidates `nvl72d183-T06`, `nvl72d042-T11`,
  `nvl72130-T14`, and `nvl72131-T02` to the HSG launcher. The launcher filters
  defaults against the current Slurm inventory because `nvl72d183-T06` and
  `nvl72d042-T11` are not registered on this cluster and Slurm rejects the
  entire exclusion request when any listed name is unknown. If they appear on
  another HSG inventory, they will be excluded automatically.

## 2026-09-02 — Final SSH push authenticated but project denied writes

- Retried `git push --set-upstream origin kbhardwaj/super-mopd` at local HEAD
  `f933733eb`. GitLab accepted SSH authentication (there was no key/public-key
  failure) but returned `You are not allowed to push code to this project` for
  `terryk/nemo-rl-internal`. The registered SSH key is working; publishing now
  requires a role with write access to that project or a writable fork remote.

## 2026-09-02 — Full 100-step production run submitted

- Submitted exactly one production job after smoke acceptance: Slurm
  `6804959`, namespace
  `tropd-super35-v24-to-oraclev1-alpha0p2-20260902T023410Z`, from commit
  `e975385ae`. It is pending on `batch_long` for priority under
  `nemotron_n4_post`; no duplicate production job exists.
- The job requests 32 nodes/128 GPUs for 23:59:59 and uses 16 student, 8
  generation, 4 Oracle-teacher, and 4 Gym nodes. It uses the requested mixed
  `train.jsonl` once, Super v24 student, Oracle step-75 teacher, MTP disabled,
  alpha 0.2 with global-baseline subtraction, 100 steps, and regular/fault-
  tolerance checkpoints every 5 steps. Output root:
  `results/tropd-super35/runs/tropd-super35-v24-to-oraclev1-alpha0p2-20260902T023410Z`.

## 2026-09-02 — Compact mixed-data smoke passed end to end

- Smoke job `6803725` completed successfully (`COMPLETED`, exit `0:0`) on 8
  `batch` nodes in 22m42s. It used the requested 93,252-row mixed OpenCode and
  13-environment dataset, the Super v24 student, Oracle step-75 teacher, and
  MTP disabled. All Gym/OpenCode services, two TP4/EP4 vLLM replicas, the
  16-rank TP4/CP4/EP8 student, and TP4/CP1/EP4 teacher initialized.
- One OpenCode prompt reached the smoke-only 65,536-token limit; the async
  collector tolerated that batch failure and gap-filled 32 target-step
  trajectories from later rows. Student/teacher logprobs completed with zero
  sequence-error masks, finite TROPD advantages, and no invalid tool calls.
- Optimizer step 1 completed with loss `0.0009`. The reduced one-buffer refit
  configuration fixed the prior OOM: the critical post-step student-to-vLLM
  refit completed in 8.03s. The regular/fault-tolerance `step_1` checkpoint and
  replay-buffer state were committed successfully.
- Validated `opd_sample_stats_step1.jsonl` (32 parseable rows) and
  `opd_token_stats_step1.pt` (20 nonempty finite tensors, including 22,335 raw
  teacher/student token gaps). W&B run `36s37hmd` completed under
  `nvidia/kbhardwaj-tropd-super`.
- Production dry-run passed with the exact requested data/student/teacher,
  32 Slurm nodes (`28` NeMo-RL + `4` Gym), `batch_long`, 23:59:59, MTP off,
  validation disabled, and `data.train.repeat=1`. The one-pass override avoids
  triplicating a dataset that already exceeds the 51,200 samples required for
  100 steps.

## 2026-09-02 — Compact smoke passed rollouts/TROPD; post-step refit OOM

- Smoke job `6800965` ran on 8 `batch` nodes. All 29 Gym processes, both
  TP4/EP4 vLLM replicas, the 16-worker student, and the TP4 teacher initialized.
  Both `ns_tools` rows (task indices 28 and 58) completed and buffered, proving
  the restored local `math_with_judge` verifier fixed the prior deterministic
  failure without starting a model judge. The replay buffer selected 32/32
  target-step trajectories, student and teacher logprobs completed, zero
  sequences were masked for logprob error, and TROPD advantages were computed.
- The optimizer computation reached `Training policy`, but the subsequent
  student-to-vLLM collective refit failed on rank 0. The default packed transfer
  allocated a 3.69 GiB CUDA buffer after optimizer state had left only 2.28 GiB
  free (`torch.OutOfMemoryError` in `packed_broadcast_producer`). No completed
  step checkpoint or final OPD artifacts were accepted from this attempt.
- Set the supported non-colocated refit controls to one staging buffer at
  `NRL_REFIT_BUFFER_MEMORY_RATIO=0.004` (about 0.74 GiB on these GPUs) and
  `NRL_REFIT_NUM_BUFFERS=1`. The retry must pass the post-step refit,
  checkpoint, and diagnostic gates.
- The requested replacement training JSONL has 93,252 rows and SHA-256
  `f943c2bd364878fba25c7b4c107c1cf232399bea6d6c4c2ff22be7ac515cf2b1`:
  91,332 `terminal_multi_harness_opencode_agent` rows plus the known 1,920-row
  13-agent environment distribution. Added its deterministic terminal-harness
  Gym config and strict shared teacher alias. Explicit `TRAIN_PATH` inputs now
  bypass proxy construction, preventing mutation of supplied datasets. The
  next smoke will use this replacement data before production is submitted.

## 2026-09-02 — Compact smoke reached rollouts; restored ns_tools verifier dependency

- Smoke job `6799081` ran on 8 `batch` nodes for 30m41s. All 32 Ray workers,
  28 intended Gym agent/resource processes, both TP4/EP4 generation replicas,
  the 16-worker student, and the TP4 teacher initialized successfully. The
  prior Mamba cache failure was resolved: vLLM accepted
  `max_num_batched_tokens=8480`, completed MoE autotuning/CUDA graph capture,
  and served both generation endpoints. Student-to-generation refit also
  completed.
- The async collector generated and teacher-scored 61 trajectories, but step 0
  remained one eligible target-weight-0 trajectory short because the
  `ns_tools` verifier delegates to the `math_with_judge` resource server. The
  direct env override had removed that local dependency, causing
  `ConfigKeyError: Missing key math_with_judge` for the same pending prompt on
  every retry. Cancelled the deterministically blocked smoke; it performed no
  optimizer step and wrote no final checkpoint or OPD diagnostics.
- Restored only the required `math_with_judge` Gym config and explicitly set
  `should_use_judge: false`, matching the historical 13-agent run config. This
  supplies ns_tools' deterministic verification service without allocating a
  model judge or enabling ORM, validation, or reference KL.

## 2026-09-01 — Compact smoke reached model init; fixed two resolved-config blockers

- Smoke job `6796863` ran on 8 `batch` nodes for 8m35s. All sandbox
  sidecars became ready, all 32 Ray GPU workers registered, the driver and W&B
  started, the 4-node student initialized its 16 workers, and both TP4/EP4
  generation replicas reached KV-cache setup. The job failed before rollouts or
  checkpoint/diagnostic output because the Super Mamba attention block size is
  2,080 tokens while vLLM inherited `max_num_batched_tokens=2048`.
- Set `policy.generation.vllm_kwargs.max_num_batched_tokens=8480`, matching the
  target's existing Super recipes and satisfying the Mamba cache constraint.
- Found that the nested `env.nemo_gym._override_` marker was ineffective because
  this config loader only handles `_override_` on direct child sections. The
  resolved env therefore still contained inherited DP8 safety judge, nl2bash
  judge, GenRM, and other unused GPU services. Moved `_override_: true` to the
  top-level `env` section so only the 13 requested agent/tool configs and policy
  endpoint remain.
- OmegaConf resolution passed with no inherited judge/GenRM keys, 14 intended
  Gym config paths, `max_num_batched_tokens=8480`, and the compact 4+2+1 actor
  arithmetic. No training step, checkpoint, or OPD diagnostic artifact was
  produced by job `6796863`.

## 2026-09-01 — Added an 8-node smoke-only topology

- Added smoke-only launcher overrides for an 8-node allocation: 4 student
  nodes, 2 generation nodes, 1 teacher node, and 1 unclaimed Gym/service node.
  The smoke uses 65,536 tokens, student TP4/CP4/EP8, generation TP4/EP4,
  teacher TP4/CP1/EP4, PPS32/GPP1/GBS32, one optimizer step, and a two-hour
  walltime on `batch`. The production YAML and production dry-run remain on
  `batch_long` at 32 nodes, 196,608 tokens, PPS32/GPP16/GBS512, and 23:59:59.
- Added `SUBMIT_DRY_RUN=true` so the smoke command can be rendered without
  submitting. The credential-safe dry-run verified the scheduler request,
  role counts, sharding, checkpoint/log paths, and absence of validation-path
  injection. OmegaConf resolution and arithmetic checks passed, including
  student/teacher world-size divisibility and PPS * GPP = GBS.

## 2026-09-01 — TROPD smoke attempt 6 queued with selectable HSG account

- Submitted only the one-step, full-topology smoke as Slurm job `6795019`,
  namespace
  `tropd-super35-v24-to-oraclev1-alpha0p2-20260901T214415Z-smoke`, from commit
  `42201c490`. The run uses generation TP4/EP4 and a smoke-only two-hour time
  limit; no production run was submitted.
- Preserved `nemotron_n4_post` as the launcher default while allowing an
  explicit `SLURM_ACCOUNT` override. Moved the same pending job to
  `nemotron_n3_post` after the user approved either account, retaining the
  existing queue position and avoiding a duplicate submission.
- The job remains pending without allocated nodes while the Slurm controller
  reports reason `None`. Continue monitoring this singleton job through step 1,
  checkpoint creation, and OPD sample/token artifact validation.

## 2026-09-01 — Super generation expert-parallel correction

- Changed policy-generation vLLM from TP4/EP1 to TP4/EP4. EP now spans the TP
  group, preventing the unsupported 672-wide per-expert tensor split that
  caused smoke `6766734` to fail in the FlashInfer TRTLLM refit path.
- Resolved-config validation passed with generation TP4/EP4, EP divisible by
  TP, and the expected Ray distributed executor backend. Shell syntax and diff
  checks passed. Full host-side MasterConfig import was not repeated because
  the lightweight login-node environment lacks unrelated optional `mlflow`;
  the previous full config validation passed and the next container smoke is
  the authoritative runtime gate.

## 2026-09-01 — TROPD smoke attempt 5 failed during vLLM model initialization

- Smoke job `6766734` allocated the full 32-node topology and ran for 7m50s.
  The compatible container resolved the prior Transformers import failure;
  Ray/Gym reached the driver gate, W&B initialized, and generation workers
  began loading the Super model.
- vLLM model initialization then failed on every generation replica because
  `policy.generation.vllm_cfg` uses TP4 with EP1. The resulting non-gated MoE
  partition size is 672, which the FlashInfer TRTLLM kernel pads to 768; the
  NeMo-RL refit patch explicitly rejects that padded layout. The runtime error
  recommends enabling expert parallelism to avoid the MoE TP split or selecting
  a backend that does not expand the weights.
- No training step, checkpoint, sample JSONL, or token PT artifact was created.
  The branch is not end-to-end ready until the generation parallelism/backend
  is corrected and another one-step smoke passes. Two untracked core files from
  the failed vLLM workers were preserved for debugging.

## 2026-09-01 — TROPD smoke attempt 5 submitted; branch push blocked

- Committed the compatible-image fix as `a0d479b83` and attempted to push
  `kbhardwaj/super-mopd`. GitLab rejected writes to the configured
  `terryk/nemo-rl-internal` origin. The expected
  `kbhardwaj/nemo-rl-internal` fork was not reachable, and the supplied
  `GITLAB_PAT` returned HTTP 401 to the membership API. A credential-safe
  retry against the suggested HTTPS origin also reached GitLab but failed
  HTTP Basic authentication, indicating that the token is invalid, expired,
  or lacks Git write scope. No remote branch was created; updated GitLab
  access or a writable remote is required.
- Submitted only the requested one-step, full-topology smoke as Slurm job
  `6766734`, namespace
  `tropd-super35-v24-to-oraclev1-alpha0p2-20260901T085716Z-smoke`. Reduced
  only this job's walltime to 02:00:00 for backfill; the production launcher
  remains at 23:59:59. Slurm accepted the 32-node request and currently holds
  it pending on priority with an estimated 09:54 UTC start.
- No production or other full run was submitted.

## 2026-09-01 — Super 3.5 training-container compatibility fix

- Replaced the stale `rl.49470403.sqsh` default with the readable training
  image used by the current colleague Super 3.5 MOPD launcher:
  `rl-gym:pipe.64391373.squashfs`.
- Credential-safe launcher dry-run passed with the new image and unchanged
  32-node topology, paths, W&B namespace, and production scheduler request.
- Import-only container preflight `6766135` reported `transformers=5.5.4`,
  successfully imported
  `MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS`, and completed
  `import nemo_rl.models.policy` (`policy_import=OK`). The preflight was
  cancelled after emitting its successful result to release its one allocated
  node; it performed no training.

## 2026-09-01 — TROPD full-topology smoke attempt 4 failed at driver import

- Smoke job `6760398` allocated all 32 nodes and ran for 2m30s. Ray reached
  128/128 worker units, all 32 Gym sandbox instances became ready, and the
  driver command started successfully.
- The driver then failed before configuration/model initialization because the
  historical `rl.49470403.sqsh` training container has a `transformers` version
  incompatible with this Super 3.5 checkout. Importing
  `nemo_rl.models.policy` could not find
  `MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS`; the checkout requires
  `transformers>=5.5.0,<5.9.0`.
- No training step, checkpoint, sample JSONL, or token PT artifact was created.
  The current colleague Super 3.5 MOPD image
  `rl-gym:pipe.64391373.squashfs` exists and is readable, but has not yet been
  substituted or smoke-tested in this branch.

## 2026-09-01 — TROPD full-topology smoke attempt 4 queued

- Submitted smoke job `6760398` with run namespace
  `tropd-super35-v24-to-oraclev1-alpha0p2-20260901T060320Z-smoke` after
  simplifying the Gym command. Slurm accepted the requested 32-node topology;
  the job remains pending on scheduler priority and has not consumed nodes.
- Shortened only this one-step job's reservation from 23:59:59 to 02:00:00
  with `scontrol`, matching the current colleague 32-node MOPD launcher and
  improving backfill eligibility. The committed production launcher remains at
  the authoritative 23:59:59 walltime.

## 2026-09-01 — TROPD full-topology smoke attempt 3

- Submitted smoke job `6760034`; it allocated all 32 nodes and mounted the
  current MOPD sandbox image, but failed before Ray initialization. The retry
  command added around Gym's entrypoint was malformed by nested shell parameter
  expansion, producing an unmatched quote inside the sidecar `bash -c` script.
- Reduced `SANDBOX_COMMAND` to the proven `/start-with-nginx.sh` entrypoint.
  The existing launcher already monitors every sandbox process and fails fast
  if an instance exits or does not open its port, so no wrapper is required.

## 2026-09-01 — TROPD full-topology smoke attempt 2

- Submitted replacement smoke job `6759476` with the requested 32-node
  topology. It allocated all nodes but failed before Ray initialization and
  model startup because the generic Nemo Skills image did not contain Gym's
  `/start-with-nginx.sh` entrypoint. No training step, checkpoint, or OPD
  diagnostic artifact was created.
- Replaced that generic image with the readable sandbox image and retrying Gym
  startup command used by the current 32-node Super 3.5 MOPD launcher:
  `nemo-skills-sandbox-no-sync.sqsh`. This resolves the failed entrypoint
  contract while retaining an environment override for future image updates.

## 2026-09-01 — TROPD full-topology smoke attempt 1

- Submitted smoke job `6759195` with the requested 32-node topology. It was
  allocated immediately but failed in 23 seconds before Ray initialization.
- Root cause: the two container paths copied from the historical runbook no
  longer existed on HSG. Pyxis reported `No such file or directory` for both
  the training and sandbox `.sqsh` files; no model, Gym service, training step,
  checkpoint, or diagnostic artifact was created.
- Located the same training build at its current high-stripe path
  `rl.49470403.sqsh` and an accessible current Nemo Skills sandbox image
  `nemo-skills-dc43f3e.sqsh`. Updated the HSG wrapper to use those exact paths,
  permit explicit environment overrides, and fail locally unless both images
  are readable before consuming a Slurm allocation.

## 2026-09-01 — Super 3.5 TROPD branch-readiness implementation

- Prepared local branch `kbhardwaj/super-mopd` from pinned Super 3.5 commit
  `00295c5b373993124da8b5ce63d122e9940d118c`; no remote push was performed.
- Synchronized and initialized recursive submodules. Verified pins:
  Automodel `24b47e856263d313b942f0ed666c63fff83306b4`, Gym
  `2251ef7f7fcbe60a352b790b0bc14a4f0d522f01`, and Megatron-Bridge
  `8c46dc4259080c510b7455f43e836fdff222c5d3`.
- Ported TROPD into the typed OPD configuration and advantage estimator:
  probability-space teacher/student interpolation, legacy-compatible alpha
  `1.0`, validated alpha range `(0, 1]`, and masked global-baseline subtraction.
  Raw teacher/student gaps remain in diagnostics while advantage metrics use
  the actual TROPD training signal.
- Ported the legacy sample, token, deferred top-k, and online top-k diagnostic
  payloads into the Super async loop. Added optional full-vocabulary logsumexp
  and fused student-logprob/top-k capture to the Megatron policy path while
  retaining CP and sequence-packing handling.
- Added the Super-to-Super run config, byte-preserving proxy builder, and HSG
  launcher under `scripts/kbhardwaj-super-super/`. The launcher requires
  `WANDB_API_KEY` and `GITLAB_PAT` without printing either and makes validation
  injection conditional.
- Proxy verification passed: 1,920 rows, the expected 13-agent distribution,
  and SHA-256
  `1fa08931c7321a149172bd5fb87ba024b68d1e880859792875b5d53a84c0548d`.
- `MasterConfig` resolution passed. Student and teacher paths exist and their
  architecture contracts match. All 13 strict aliases and Gym config paths
  exist. Topology resolves to 16 training + 8 generation + 4 teacher + 4 Gym
  nodes, with four GPUs per node.
- Verification passed:
  - OPD/TROPD/diagnostic tests: 54 passed.
  - Structured 3-D/4-D batch aggregation tests: 2 passed.
  - Full-vocabulary logsumexp matched a stable fp32 baseline in full and
    chunked modes. Multi-GPU TP cases were collected but skipped on the
    CPU-only host.
  - Ruff format/lint, Pyrefly for the new diagnostics module, `compileall`,
    `bash -n`, and `git diff --check` passed.
  - Credential-safe HSG dry-run passed with account `nemotron_n4_post`,
    partition `batch_long`, QoS `normal`, two-node segments, 32 nodes, four
    GPUs per node, 144 CPUs per worker, 23:59:59 walltime, `/lustre:/lustre`,
    the intended images, W&B namespace, and stable checkpoint paths.
- During readiness review, Ruff exposed top-k state initialized in the sync
  loop but consumed in the async loop. Moved it into the async step and reran
  the focused suite successfully. Also made top-k worker result aggregation
  preserve/pad variable sequence lengths.
- Full Megatron unit import is unavailable in the host environment because
  Transformer Engine is absent; this is supplied by the configured training
  container. The required full-topology one-step smoke and production job were
  intentionally not submitted during this branch-readiness pass. The branch is
  ready for that smoke gate, but is not yet production-proven.
- Preserved pre-existing untracked `scripts/conversion/`, the older unrelated
  files in `scripts/kbhardwaj-super-super/`, and
  `3rdparty/Megatron-LM-workspace/`; none are included in this work.
