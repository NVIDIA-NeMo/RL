# PR3 implementation and validation record

Updated 2026-11-06. Base PR1: `047d31a7f75dda062e82e29561a7e8460dc24b7c`;
development branch: `yuekaiz/pr5090-automodel-streaming`. Main did not contain PR1.
Image: `nemo_rl.1030.sqsh`; actual pinned AutoModel/PyTorch versions are recorded
in source-evidence.json and dependency-runtime.json. Product dependency pins unchanged.

| Check | Result and evidence |
| --- | --- |
| Shared controller/config/wrapper regressions | 368 passed (`implementation-regression.log`), including PPO setup, sampler, checkpoint races and live TQ lifecycle. Earlier suites overlap; counts are not additive. |
| TQ resume schema / cancellation races | 44 passed (`recovery-regression.log`). Includes live integer TQ rows on client reattachment and repeated cancellation while an actual thread is blocked. |
| Existing full-batch dense and native EP2 | Exact gradient/model/optimizer/scheduler regression passed, rtol=atol=0. Rechecked with raw optimizer-state comparison in `raw-optimizer-tests.log`. |
| Dense FP32 streaming | 2 and 3 dispatches, token/sequence objectives, 3 updates each passed. Preclip gradient errors approximately 1e-8. |
| BF16 dense/native EP2 | Four cases passed the original overall gradient gates and additional independent optimizer-update oracle (`optimizer-oracle-tests.log`). **Original v1 per-parameter and optimizer gates failed**; see BF16_NUMERICS.md. |
| EP2 activation checkpointing | Three dispatches, both objectives, three updates passed the BF16 gradient/oracle checks (`activation-checkpoint-tests.log`). |
| Real Gate bias and abort | Actual AutoModel Gate with nonzero bias factor, checkpointing on/off: load retained across chunks/LP, recompute does not duplicate load, bias updates once at commit, abort restores state. Worker-state suite: 33 passed (`final-worker-state-tests.log`). |
| Fully masked DP rank | Passed (`masked-rank-tests.log`): rank1 entirely masked, token/sequence losses, three updates, three dispatches. |
| FusedAdam compressed state | Native BF16 EP2, activation checkpointing, three dispatches, token/sequence objectives: gradient and independent optimizer oracle passed (`raw-optimizer-tests.log`). Actual parameters BF16; raw moments BF16 and master remainders int16; scales compared too. |
| Real vLLM streaming / resume | Audit v3 committed updates through 6 after recovery from step2. Durable snapshot: 4 ready groups plus 4 in-flight groups. Receipt checker passed group/sample uniqueness, ready-group reuse, lag0 and resumed steps2–5. |
| Actual overlap | **Passed with controlled tail admission**: CUDA traces show 13,111 intersecting kernel intervals, 76.116ms total overlap (`results/overlap-gated/kernel-overlap.json`). Initial ungated/delayed-decode traces showed no overlap; synchronized-region receipts alone were insufficient. Private gating/profiling changes timing, so this proves overlap and does not establish throughput. |
| Stronger cancellation | Audit v4 passed: four dispatches on full steps, cancel after two completed partial dispatches at version2, drain then abort. Durable snapshot: 3 ready and 5 in-flight groups; restored and completed through step6. Receipt checks passed (`integration-audit-check-v4.log`). |
| Qwen3.5 EP16 + recipe FusedAdam | **FAILED** job21330640 on the first token-level preclip gradient comparison: observed rank relativeL2 about1.1–1.5%, exceeding1%; some cosine values also below.9999. Original limits unchanged. See ep16-failure.json; raw node logs retained. This was synthetic-input replay, not saved real-GRPO-input replay. |
| Qwen3.5 strict qualification-gated LA0 300 steps | **NOT STARTED.** Jobs21330687–21330690 cancelled on failed qualification dependency. No extra full-batch convergence run. The distinct exploratory run below does not change the failed gate. |
| Normalizer-placement diagnostic | **PASSED** tiny EP2 and Qwen3.5 EP16 job21331026, both objectives, one update each. All16 ranks match the independent full-input numerator reference exactly in preclip gradients/model/public+raw optimizer/scales/scheduler. Global original-normalizer differences: token1.343%, sequence1.322%; original gate remains failed. See normalizer-control-ep16-summary.json. |
| Additional convergence evidence | **300/300 committed**, checkpoint300 saved. Final segment21331930 COMPLETED0:0 at2026-11-06 06:02PST; all four segments completed normally. Same exploratory LA0 run,2dispatches/step,lag0. Original1% qualification stays failed; completion does not establish numerical or convergence equivalence. |
| Historical baseline curve comparison | All three series now cover1..300, eight panels: reward, generation length, gen_kl_error, policy_kl_error, loss, entropy, gradient norm and step time. V1 synchronous2a121400, full-batch LA0c926b37d and exploratory streaming LA0; raw plus trailing10 means. CPU-only refresh21389289 COMPLETED0:0, snapshot2026-11-06T14:07:01UTC,900CSV rows, no missing steps/metrics. See CURVES.md and results/curve-comparison/. No duplicate baseline experiment. |
| Fixed-checkpoint KL and timing investigation | Diagnostic21373254 COMPLETED0:0,53m32s. Actual version210,32responses,nonzero advantages;112rank reports verified. Same stored LPs give full/split genKL agreement within5.5e-13, independent K3 error3.8e-9. Fixed-shard split matches the numerator reference exactly on all16ranks, but production DP placement gives30.66%global gradient difference; **unresolved numerical finding, not qualification**. Fixed-workload backend medians5.45s full/8.69s split. Warm ungated trace intersection26.77ms is provisional without an independent clock-offset bound. See kl_replay/EP16_FINDINGS.md and archived JSON evidence. |
| Production-layout follow-up | Diagnostic21391011 COMPLETED0:0,22m08s. Same checkpoint/capture; initial optimizer/scheduler210, actual valid input fields/layout and112rank reports verified. Giving full references the production microbatches reduces stored/recomputed-LP gradient gaps to1.18555%/1.13169%; pre-update eval LPs agree exactly. Fixed split matches independent numerator gradients/model/public+raw optimizer exactly on all16ranks. **Original1%BF16 qualification still FAILED**. Backend medians5.63s full/8.73s split; no throughput gain established. See production-layout-*.json and EP16_FINDINGS.md. |

Numerical snapshots include preclip normalized gradient shards, every parameter
and registered buffer, complete optimizer state_dict, scheduler state and exact
scalar counters. The recipe-optimizer lane preserves FusedAdam's compressed
moments/master/remainder state and private scaling tensors instead of substituting
AdamW. TE's public state_dict dequantizes moments, so the harness separately saves
and restores raw buffers/scales exactly before replay. Each passing harness
report explicitly distinguishes original full-batch parity gates from the BF16
gradient-plus-execution-oracle contract. BF16 convergence and EP16 qualification
cannot be inferred from tiny-model passes.

Research commands and raw evidence live under validation/pr3; large logs, traces,
model/optimizer dumps and checkpoints are local artifacts and are not committed.
The image's optional NIXL extension emits an ABI warning during vLLM import;
the exercised vLLM/HybridEP paths still initialize and run. No dependency was
silently upgraded to suppress that warning.

During v3 resume the old partition warmup tried to write float placeholders over
restored int64 fields. Training completed but TQ logged a schema error. The minimal
fix and live regression were ported from the prior experiment's commit72d218264;
v4 completed recovery without the schema mismatch. No changes to the held900-step jobs.

Supported first-stage scope: text GRPO/per-prompt advantages, full-parameter FSDP2,
fixed batches, TP=CP1, in_order LA0, disaggregated vLLM, zero MoE auxiliary loss and
positive-example NLL. LA1, CPU offload, HSDP, LoRA, packing, dynamic batching, VLM,
PPO and distillation remain rejected. Online validation/acc remains unavailable.

Submitted jobs and the intended W&B run identity are recorded in jobs.json.
The default training launcher requires original qualification and matching source
hashes; Qwen3.5 has not passed that contract. The explicit exploratory-train mode
checks the completed exact numerator control plus source hashes and uses a distinct
run identity to collect additional authorized convergence evidence. The deterministic
watch_10min.sh process records queue/accounting/progress every ten minutes in
slurm/watch.log and marks observed terminal failures in slurm/FAILURE_DETECTED.
It performs status collection only; it does not retry experiments or send chat notifications.
Expired completed job IDs in Slurm's live controller do not stop monitoring;
the collector falls back to accounting and records query failures explicitly.

First qualification attempt21329914 failed at the launcher before model initialization:
Ray services started before run.sh's exports, so remote tasks lacked PR3_PYTHON.
Fixad5582855 explicitly propagates the required non-secret interpreter/cache/model
environment on remote tasks. Its blocked training chain was cancelled and replaced;
raw failure logs remain in slurm/21329914-logs. This was not a model numerical result.

The normalizer control runs ordinary V1, ordinary V1 with only its denominator
moved from backward to clip entry, and SC streaming from identical state/RNG.
The latter two must match exactly. This isolates normalization rounding from
dispatch/LP state; it records original-gradient error separately and writes
normalizer-control.json with releases_training:false, never a qualifying parity.json.
Run it with --normalizer-control --steps 1; its EP16 launcher uses a separate output folder.
The diagnostic started from an uncommitted worktree at HEADad5582855; tested source
hashes match the committed files in2cc30b16a. The numerical worker is unchanged.

Exploratory run: https://wandb.ai/yuekaiz/nemo-rl-automodel-sc/runs/yuekaiz5090explorela0s42
The first update had all samples truncated and reward0; the second had reward.25
but no within-group gradient signal. A nonzero gradient appears on update3,
reward.21875. Initial step times include cold initialization/JIT and are not
steady throughput. The exploratory run is now complete; convergence equivalence
is not established. In the final20 updates, V1/full-batch LA0/streaming LA0 means
are genKL0.000811697/0.000845502/0.001570812, reward0.628125/0.621875/0.709375,
and generated length1243.67/1211.59/1059.80. Over steps11..300, mean step times
are138.58/133.59/133.30seconds, including checkpoint and resumed-first-step
overheads. Streaming is approximately the same time as full-batch LA0 while
generating shorter responses; this historical comparison does not demonstrate
a streaming throughput gain. The visible KL gap still requires the same-input
diagnostic and cannot be assigned to policy lag (observed lag remains0).
