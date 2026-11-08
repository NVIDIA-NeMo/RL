# Streaming LA0 continuation from300 to600

The user explicitly authorized this continuation on2026-11-08. This supersedes
the earlier instruction in session/handoff notes to leave the completed300-step
run stopped. It does not resume the separate held900/MCore/sibling experiments.
The original1%BF16 numerical qualification remains failed; this is continuation
of the same exploratory run, not a new qualification result.

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
