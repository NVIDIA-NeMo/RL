# Gemma4 E2B IT streaming LA0 comparison

The active user target is900, superseding650/150. Training has continued beyond
650 on allocation21579059,pool0-01905; latest independently checked W&B/native
snapshot656 had trainer_v656,lag0,two dispatches and LR2e-6. See
[RESUME900.md](RESUME900.md), jobs.json and live slurm/resume650/state.json.
The historical650 directory name does not change target_steps900.

The [eight-panel comparison](results/curve-comparison/comparison.png) is updated
through650 at2026-11-09T17:59:52UTC. It shows reward, generation length,
generation/policy KL diagnostics, policy loss, approximate entropy, preclip
gradient norm and measured step time. Raw curves and trailing10-step means are
shown. This is an intermediate snapshot; final900 plots remain pending.

Both baselines are the original PR author's EXISTING batch512 runs, not new
experiments launched here:

- V1 synchronous: https://wandb.ai/ys_fishcool-nvidia/nemo-rl-automodel-sc/runs/74860be5
  (157 observed steps).
- Full-batch SC LA0: https://wandb.ai/ys_fishcool-nvidia/nemo-rl-automodel-sc/runs/96779513
  (164 observed steps).
- Streaming LA0: https://wandb.ai/yuekaiz/nemo-rl-automodel-sc/runs/yuekaiz5090gemmala0s42
  (same run throughout all continuations).

The [summary](results/curve-comparison/comparison-summary.json) and
[CSV](results/curve-comparison/comparison.csv) use actual endpoint coverage.
Shared comparison statistics below stop at157. The plot shades unshared coverage,
retains the full SC baseline's observed tail to164 and extrapolates nothing.

| Metric, common steps1..157 | V1 synchronous | Full-batch SC LA0 | Streaming LA0 |
| --- | ---: | ---: | ---: |
| Mean reward |0.41849|0.41843|0.41544|
| Mean generation tokens/sample |1429.90|1424.12|1384.05|
| Mean gen_kl_error |0.000392377|0.000386466|0.000405584|
| Mean step time, steps11..157 (s) |72.83|71.86|69.78|

Streaming tail statistics have a different window: steps631..650 gen_kl_error
mean0.000578418 and approximate entropy0.231685. The generation KL trace rises
around600 while entropy also rises; this is an observation, not a causal result.
The largest generation KL value through650 remains392=0.000959170. Policy KL
has isolated large outliers including18=2.28353,93=2.41500,538=1.81858.
Preclip gradient spikes include571=13.7712 and subsequently return lower.
The actual AutoModel max_grad_norm is1.0; the plotted metric is before clipping.
Causes of these diagnostic differences remain unresolved. Shorter generated
outputs, independent rollouts, one seed and historical runs prevent an isolated
streaming speedup or convergence-equivalence claim.

All first650 committed metrics are finite and lag0; the full audit through656
also found nonzero advantages/gradients and unique, contiguous step/version
receipts.649 of the first650 updates used two dispatches; step12 selected all32
already-ready groups in one dispatch. Readiness16 is a threshold, not a fixed
chunk cap or a forced overlap gate. Actual warmed GPU overlap was not measured
for this Gemma long run. No online validation/accuracy is recorded.

The [checkpoint650 audit](results/checkpoint650-state.json) checked four saved
optimizer shards, all505 counters650, scheduler/trainer/sampler650,LR2e-6 and
20800 consumed prompts. The [progress audit](results/resume900-progress-650.json)
records the slightly later656 snapshot. These are saved scalar/metric checks,
not a live all-rank raw optimizer or per-prompt read-ahead cursor oracle.
Initial150 scalar/metric evidence remains in checkpoint150-state.json and
first150-metrics-audit.json; saved source/model provenance is retained separately.

Recovery evidence is retained for [150->151](results/resume650-first-recovery-21557906.json),
[340->341](results/resume650-first-recovery-21568255.json) and
[524->525](results/resume900-first-recovery-21579059.json). Each first recovered
receipt had the expected version,lag0,nonzero signal,LR continuity and exact
selected W&B/native metrics. Later recoveries observed64 unfinished groups next
to0 canonical groups. Independent all-group uniqueness and completed canonical
replay recovery were not established by these observations. The older idle
allocations were released only after driver exit0,durable progress and audit;
Slurm CANCELLED0:0 records normal cleanup.

The unchanged recipe uses local google/gemma-4-E2B-it,one8H100 debug node with
4training/FSDP2/EP1 and4async-vLLM rollout GPUs,TP=CP1,32prompts x16responses
=batch512,seed42,train/logprob MBS1,seq4096/gen2048,LR2e-6/FusedAdamBF16states,
warmup10 then ConstantLR,save every10/keep2/deadline3h45,image1030.
LA0/in_order/readiness16/buffer32; max_num_epochs1 with the original1791700-row
loader. Preflight matched the original model/optimizer/scheduler/data/sampling/
loss/resources against both baseline configs; only target/display name changed
for continuation. The numerical worker remains7f598242... and parity harness
ccb26bda... from this PR3 branch (base PR1 revision047d31a7). Newer PR1 revision
097524718 was inspected but not rebased into this active experiment.

The900-aware resume650_supervisor.py retains the historical runtime/lock/HOLD
paths under slurm/resume650. It adopts existing stages, uses new900-prefixed
launch/log files for later stages, polls30s and records10minute snapshots and
milestone/failure notifications in this SAME conversation. One owned node at a
time, direct attach on sequential interactive allocations, no batch training
chain or additional baseline. On failure inspect preserved per-job logs before
repair; on user hold/cancel create HOLD first and handle active work as requested.
Do not clear it without user resume authorization. See RESUME900.md for details.

This remains exploratory. Gemma fixed-input streaming parity is NOT ASSESSED;
the original Qwen BF16 1% gate remains FAILED. Finite training, scalar recovery
checks and visual curves do not constitute numerical acceptance. Keep Qwen600,
old900,MCore and sibling RL-pr4963 investigations separate; do not act on their
jobs. At900 verify final saved state and W&B, regenerate/review these eight plots,
release this experiment's node and publish on the existing branch.
