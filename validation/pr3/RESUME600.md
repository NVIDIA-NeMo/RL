# Streaming LA0 continuation from300 to600

The user explicitly authorized this continuation on2026-11-08. This supersedes
the earlier instruction in session/handoff notes to leave the completed300-step
run stopped. It does not resume the separate held900/MCore/sibling experiments.
The original1%BF16 numerical qualification remains failed; this is continuation
of the same exploratory run, not a new qualification result.

Completed on 2026-11-08 around17:49PST: segment21509385 COMPLETED0:0 after
3h36m36s, committing504..600 and saving checkpoint600. Native receipt and
read-only W&B API both confirm step600/trainer_v600/lag0/two dispatches/LR1e-6.
Final genKL0.00160220171357, policyKL0.00178326417351, reward0.75.
W&B reports finished and the native SDK reports successful sync. Checkpoint600
has total_steps/trainer_version/sampler_dispatch_index600, consumed_samples4800,
epoch0. Segment21509386 completed in25s after the pre-container guard detected
600; it did not load another model. The ten-minute watcher exited normally.
No training allocation remains. See results/resume600/final-wandb.json.

CPU-only finalization job21539655 COMPLETED0:0 in3m20s oncpu1-00099 using
image1030. All600 optimizer-step receipts are unique and contiguous, match their
trainer versions, and have lag0, two dispatches and32 valid sequences. All logged
metrics are finite. All300 continuation updates use LR1e-6. Checkpoint600 has16
optimizer shards, all693 saved optimizer step entries600, all693 learning rates
1e-6, scheduler last_epoch600 and active ConstantLR last_epoch590. This checks
saved scalar state, not live per-rank raw optimizer-buffer equality.

| Continuation window | gen_kl_error mean |
| --- | --- |
| 301..400 | 0.001488790 |
| 401..500 | 0.001509982 |
| 501..600 | 0.001613112 |
| 581..600 | 0.001720197 |

GenKL rises slowly in the later windows; no explosive growth was observed.
The last100 maximum is0.002385499. This does not resolve the earlier baseline
gap, identify its remaining cause, or change the failed numerical gate. Evidence:
results/resume600/final-summary.json, segment3-summary.json and
checkpoint600-state.json. Reproduce on an allocated node with
`python validation/pr3/summarize_resume600.py` followed by
`python validation/pr3/plot_comparison.py --max-step 600` in image1030.

The existing eight panels were refreshed at2026-11-09T01:58:02UTC. Both baseline
runs have500 recorded steps and streaming has600; all eight metrics are complete.
Shared statistics stop at500 and the plots shade501..600. No baseline was rerun,
and no validation accuracy was invented. See CURVES.md and
results/curve-comparison/comparison.png.

Historical handoff on 2026-11-08 around 14:23 PST: segment21509384 COMPLETED0:0
after3h44m16s, committing402..503 and saving checkpoint503 at its configured
checkpoint deadline. Segment21509385 ran onpool0-[00707,00711,00731].
All16 workers loaded503 weights, and the saved data-loader/native TQ state was
restored. Checkpoint503 has total_steps/trainer_version/sampler_dispatch_index503,
consumed_samples4024 and epoch0; weights and optimizer shards are present.
The existing W&B run has resumed online. Step504 committed with trainer_v504,
lag0, two dispatches and LR1e-6; genKL0.00149343815 and policyKL0.00155217709.
Read-only W&B API confirmed the same run has received504 with matching metrics
(results/resume600/segment3-wandb-startup.json). Segment21509386 was dependent.

The checkpoint's16 optimizer shards are present. Reading only DCP scalar state
inside the allocated container verified all693 saved optimizer step entries503,
all693 learning rates1e-6, scheduler last_epoch503 and active ConstantLR
last_epoch493. See results/resume600/checkpoint503-state.json. These are saved
counters; this monitoring audit does not compare live optimizer buffers per rank.

All102 updates402..503 have finite metrics,lag0,LR1e-6 and two dispatches.
First20 versus last20 genKL means0.00150175/0.00151579; segment mean0.00151009,
range0.00110983..0.00198676. Last20 reward mean0.7125, generated length959.80,
measured step time127.272s. PolicyKL briefly reached0.0582654 at501 and returned
to0.00179282/0.00124514 at502/503. No gen-KL explosion observed through503;
these are observational results, not numerical qualification or an isolated
throughput claim. See results/resume600/segment2-summary.json. The preceding
301..401 segment had first/last20 genKL means0.00153708/0.00154503; its evidence
is in segment1-summary.json. No additional training submission or numerical
code/configuration change was needed to reach600.

- Recipe: streaming_qwen35_exploratory_600.yaml inherits the original exploratory
  recipe. Only grpo.max_num_steps changes to600 and the display name says600.
- Same Qwen3.5-35B-A3B-Base, image nemo_rl.1030.sqsh,3nodes/24H100:
  16training/EP16 plus8rollout/TP8, LA0/in_order,8prompts x4responses,
  train/logprob MBS1, streaming threshold4 (two dispatches per update), seed42,
  total sequence4096/new tokens2048. No validation configuration change.
- Same optimizer/scheduler, checkpoint and data-loader state. LR1e-6 with the
  saved scheduler; do not restart warmup. Checkpoint300 metadata has total_steps
  and trainer_version300, consumed_samples2400, current_epoch0.
- Same checkpoint/log directory under
  results/yuekaiz-qwen35-streaming-la0-300-exploratory/; retain the old directory
  name so automatic recovery finds the existing state. Checkpoint300 is also
  protected by hardlinks at results/resume600-source/step_300 (no physical copy).
- Same W&B run: https://wandb.ai/yuekaiz/nemo-rl-automodel-sc/runs/yuekaiz5090explorela0s42
  using the existing runtime credential file; no credential is stored in docs.
- No numerical worker change. SHA256worker7f59824258aafb13f5540810776dfa18c86a623856c9c89e4c6e26881259f74d;
  original parity harnessccb26bda2ad5a1408cbbd125cf2aec5a119d232ecaa1777bebbce0071c0dc399.

Submit entrypoint: bash validation/pr3/submit_resume600.sh. It refuses to submit
again if slurm/resume600-submitted.tsv already exists; inspect existing jobs
before any retry. Four dependent4hsegments use the original3h45checkpoint
deadline. They all target600 and restore the latest checkpoint automatically.
The pre-container check requires an existing checkpoint>=300 and skips startup
once600 is durable. The in-container guard verifies the actual YAML target and
preserves the original numerical-source checks. No throwaway smoke run is used.

Submitted on2026-11-08 06:41PST, code2a9ec1e77:

| Segment | Slurm job | Dependency |
| --- | --- | --- |
| 1 | 21509383 | None |
| 2 | 21509384 | afterok:21509383 |
| 3 | 21509385 | afterok:21509384 |
| 4 | 21509386 | afterok:21509385 |

At06:54PST first job is RUNNING onpool0-[00139,00585,01164]; others wait on
dependencies. Updates301and302 committed with trainer versions301/302,lag0,
two dispatches each and LR1e-6. Step301 genKL0.0015835980; step302 genKL0.0014097517.
W&B read-only API confirmed the existing run is running and has received301,
matching native metrics. Step301 has a policy-KL spike5.04649 while gradient/loss
remain finite; step302 returns to its ordinary range. Record and monitor subsequent
steps rather than treating that spike as a gen-KL explosion. Latest durable
checkpoint remains300 until the next periodic save. Always inspect current
Slurm/logs/latest_checkpoint_status.json rather than treating these notes as live.

Every10minutes, session/20261105_pr3_design/monitor_resume600.py reads only this
job chain, committed step receipts and checkpoint metadata. It queues start,
failure/stall, first-resumed-step, segment handoff and400/500/600 milestones into
THIS existing conversation with --resume. Runtime PID/thread/status/notification
files are ignored under slurm/resume600-monitor/ and resume600-watch.*. No new
conversation, subagent or external message is created. The watcher never submits,
cancels or restarts jobs. Respect any later user cancellation/hold.

On callback, inspect source/config hashes, checkpoint restore, trainer version,
lag0 and W&B continuity. If a segment fails, identify the actual failure before
repair/retry; do not duplicate a live allocation. Keep the same experiment identity.
Once600 is durable, verify job/log/W&B completion and release redundant pending
segments if needed; stop the completed watcher. Refresh the existing8panels
with plot_comparison.sh --max-step 600, comparing baselines only where actual
records exist (previously500). The loader includes /300 and /600 continuation
receipts and shades steps outside shared coverage. Do not fabricate
validation/acc or submit additional baseline experiments.
