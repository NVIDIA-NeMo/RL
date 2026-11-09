# Gemma4 E2B IT streaming LA0 comparison

Latest user target is650 (requested2026-11-08), superseding150. The original
150-step process finishes normally, then resumes the same experiment on
sequential debug allocations. See [RESUME650.md](RESUME650.md) and the live
slurm/resume650/state.json. Historical150 instructions below describe that first
segment, not the new stopping point.

User authorized a direct debug-node run on2026-11-08. Target150 updates, reusing
the existing batch512 baselines; no additional full-batch experiment. Numerical
worker unchanged from8b9fae1fd (PR3 based on PR1 revision047d31a7). Latest PR1
097524718 was inspected, not rebased into this active experiment. Image1030.

At20:36PST, updates1and2 committed with trainer versions1/2,lag0 and two
dispatches each (16+16 groups). All512 samples valid; nonzero advantages and
preclip gradients observed. GenKL0.000402249/0.000381952. The existing W&B
API confirmed step2 and all selected metrics match native records. Learning rate
follows the saved recipe's initial warmup (2e-14 then2.00000018e-7), not a changed
peak LR. No convergence/speedup conclusion from these startup steps.
See results/startup.json. Always inspect current logs rather than treating this
snapshot as live progress.

At20:47PST checkpoint10 was durable (trainer/sampler versions10,320 consumed
prompts). Inside the allocated container, its four optimizer shards and all505
saved optimizer step entries10 were checked; saved learning rates2e-6 and
scheduler last_epoch10 match the end of warmup. All first10 updates have finite
plotted metrics, lag0, two dispatches,512 valid sequences and nonzero advantages
and gradients. GenKL mean0.000385428, range0.000342696..0.000424718. These are
early observations, not convergence qualification or a checkpoint/resume test.
See results/checkpoint10-state.json and results/early-10-summary.json.

Step12 selected all32 already-ready groups in one dispatch; first11 used two.
This follows the unchanged sampler readiness contract. The comparison summary
records dispatch-count frequencies and the number of multiple-dispatch updates;
the experiment does not force two equal chunks or claim measured GPU overlap.

- Model: local google/gemma-4-E2B-it, same path as both baselines.
- One interactive debug allocation21546953, pool0-00343,8H100,4h walltime.
  Four training GPUs/FSDP2/EP1 and four async-vLLM rollout GPUs; TP=CP1.
  The allocation only starts Ray; launch.sh attaches and directly runs run.sh.
  No batch training job or dependency chain was submitted.
- Batch512=32 prompts x16 responses, train/logprob MBS1, seed42, total4096,
  generation2048, LR2e-6, warmup10 then ConstantLR, recipe FusedAdam BF16 states.
  Checkpoint every10 updates, optimizer/data-plane retained, deadline3h45.
- LA0/in_order, min_groups_for_streaming_train16, buffer/inflight32. This is a
  readiness threshold rather than a fixed chunk cap; inspect actual dispatches.
- Same model/optimizer/scheduler/sampling/data/loss/resource/checkpoint-frequency
  settings as both baselines checked before startup. Dataset dict-versus-list
  representation normalized with the repository's update_single_dataset_config;
  no training input change. Two failed preflight checks are retained under slurm;
  no model training or W&B run was started by those checks.
- Own W&B: https://wandb.ai/yuekaiz/nemo-rl-automodel-sc/runs/yuekaiz5090gemmala0s42
- V1 baseline: https://wandb.ai/ys_fishcool-nvidia/nemo-rl-automodel-sc/runs/74860be5
  (157 updates). Full-batch SC LA0 baseline:
  https://wandb.ai/ys_fishcool-nvidia/nemo-rl-automodel-sc/runs/96779513 (164).
  Compare their existing first150 updates using unsampled history exports.

This is exploratory training evidence. Fixed-input streaming parity for this
Gemma batch512 recipe has not been assessed. The previous Qwen BF16 1% gate
remains FAILED; completion/finite metrics do not change numerical qualification.
Independent rollouts, one seed and historical baselines do not isolate speedup
or prove convergence equivalence. No online validation/accuracy is recorded.

Runtime: slurm/driver.log, launcher.pid, driver-exit-code, monitor/status.json.
Resolved config/source/model-config hashes are in results/provenance.json and
results/resolved-config.json. Inspect existing job/driver before any restart;
archive old native logs before resuming so all committed steps remain available.
On failure repair in this allocation and resume the same checkpoint/W&B identity.
Do not duplicate a live driver or act on other user jobs.

Read-only session/20261105_pr3_design/monitor_gemma.py checks every10minutes and
queues milestone/failure/stall/completion notices into the SAME current thread.
It does not submit, cancel or restart anything. Respect later user cancellation.

After150, run.sh generates eight panels (reward, length, generation/policy KL,
loss, entropy, preclip gradient norm, measured step time) with plot.py inside the
allocated container. It validates committed-step/version/lag/finite plot coverage.
Native histories/CSV/PNG/SVG and common1..150 statistics go under
results/curve-comparison/. Before plotting, plot.py checks checkpoint150's saved
optimizer/scheduler scalar state and data cursor, writing checkpoint150-state.json;
this does not compare live per-rank raw optimizer buffers. launch.sh releases ONLY this debug allocation after
successful training+plotting and durable checkpoint150. The allocation will then
show CANCELLED because it was an idle interactive Ray allocation, not a failed
training job. Confirm driver exit0,checkpoint150 and W&B finished independently.

Keep the completed Qwen600 results and old900/MCore/sibling KL investigations
separate; do not cancel/restart their jobs. Their live state may differ from old
session notes (old900 first segment was already RUNNING when this task began).
