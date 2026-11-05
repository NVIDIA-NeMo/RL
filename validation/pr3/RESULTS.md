# PR3 implementation and validation record

Updated 2026-11-05. Base PR1: `047d31a7f75dda062e82e29561a7e8460dc24b7c`;
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
| Qwen3.5 streaming LA0 300 steps | **NOT STARTED.** Jobs21330687–21330690 cancelled on failed qualification dependency. No extra full-batch convergence run. |
| Normalizer-placement diagnostic | **PASSED** tiny EP2 and Qwen3.5 EP16 job21331026, both objectives, one update each. All16 ranks match the independent full-input numerator reference exactly in preclip gradients/model/public+raw optimizer/scales/scheduler. Global original-normalizer differences: token1.343%, sequence1.322%; original gate remains failed. See normalizer-control-ep16-summary.json. |
| Additional convergence evidence | Separate exploratory LA0 recipe prepared with identical training configuration and separate output/W&B identity. This research lane uses the exact numerator control and records original qualification failure; it does not redefine the1% gate or establish completed acceptance. |

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
