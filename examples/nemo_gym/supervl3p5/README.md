# CapRL video prefix-checkpoint smoke

This recipe ports the recorded settings from
`nvidia/rohit-unified-teacher-supervl3p5/supervl3p5-caprl-v2-ready-first-s1-20261019`
into the current SingleController schema. It does not depend on a pipeline
checkout or the reference author's private source checkout.

The allocation is 32 nodes with 4 GPUs per node: 16 learner nodes and 16
generation nodes, with segment size 8. Gym runs CPU-only; MCQA grading needs
no external judge. The full dataset has 2,816 prompts with 64 cached video
frames each. R3 stays disabled. Batch size, optimizer/scheduler, loss,
precision, model, and video preprocessing follow the reference.

## Intentional differences

- Stop at 5 total training steps, not 50.
- Disable `grpo.deduplicate_multimodal_data`, required by media token capture.
- Enable prefix recovery and attempt rollout snapshots every 120 seconds.
- Save a trainer anchor every step, not every 10; retain three trainer
  checkpoints and three snapshots per anchor. Disable the deadline save/stop.
- Restore the replay buffer and allow fallback to a previous matching
  trainer/snapshot pair.
- Use node-local capture ledgers and per-allocation Gym/log directories.
- Load only policy-model and MCQA Gym configs; unused services are removed.
- Explicitly disable engine-level video pruning, required by capture.

The six bad-word restrictions are preserved. The worker's single-token-ban
fix must be present. Video source MP4s are not needed: the current native-video
adapter reads the cached PNG frames. Each finished sibling stages its own
processed media tensors; prefix chunks contain media hashes, not pixels.

## Launch

From the NeMo-RL root, first check the command without submitting:

```bash
EXP_NAME=supervl35-caprl-prefix-smoke-001 DRY_RUN=1 \
  bash examples/nemo_gym/supervl3p5/scripts/launch_caprl_prefix_smoke.sh
```

Then submit:

```bash
EXP_NAME=supervl35-caprl-prefix-smoke-001 \
  bash examples/nemo_gym/supervl3p5/scripts/launch_caprl_prefix_smoke.sh
```

Defaults are account `nemotron_sw_post`, partition `batch`, QoS `normal`, and
walltime `1:00:00`; export `SLURM_ACCOUNT`, `SLURM_PARTITION`, `SLURM_QOS`, or
`WALLTIME` to override. `CAPRL_MODEL_PATH`, `CAPRL_TRAIN_PATH`, `CONTAINER`,
`CAPRL_CONVERSION_CACHE`, and `RESULTS_ROOT` are site asset overrides. The
conversion cache is mounted **read-only**, and its model/override key must
match. All shared paths must be visible under `/lustre` on every node.

The launcher reuses the image's prebuilt RL and Gym environments while
mounting the current checkout. It bypasses fingerprint mismatches explicitly;
it does not rebuild environments. Container-side preflight checks actual
imports, config schema, bad-word tokenization, and the first real 64-frame
video through the dataloader's processor before training. It checks native-video
routing, frame counts, and that the prompt was not filtered. This is a
compatibility check, not a completed GPU save/restore validation.

Export `WANDB_API_KEY` separately for online logging to
`nvidia/supervl3p5-caprl-prefix-smoke`. Without it, logging defaults to offline.
No credentials are stored in the recipe or launcher.

## Save/restore test

Wait for a committed rollout snapshot containing active prefixes beneath
`results/<EXP_NAME>/checkpoints/step_*/rollout_snapshots/`. Do not interrupt
until the snapshot commit succeeds and its matching trainer anchor exists.
End the first allocation before the five-step target is reached, then run:

```bash
EXP_NAME=supervl35-caprl-prefix-smoke-001 RESUME=1 \
  bash examples/nemo_gym/supervl3p5/scripts/launch_caprl_prefix_smoke.sh
```

Reuse the same configuration, checkpoint directory, and total step target.
To queue a continuation before the previous allocation ends, also set
`SLURM_DEPENDENCY=afterany:<previous-job-id>`; the scheduler prevents concurrent
writers to the same checkpoint directory, including when the parent times out.
Confirm that the second run reports restored unfinished rollouts and
generation prefixes, passes processed-media fingerprint checks, and completes
another training step. A snapshot that contains only completed rows, or a run
that regenerates every active call, does not establish prefix recovery.

Logs are under `results/<EXP_NAME>/<JOBID>-logs/ray-driver.log`; trainer and
rollout checkpoints remain in the shared `checkpoints` directory. Changing
`grpo.max_num_steps` or other training settings on resume can change the
snapshot fingerprint; choose the desired target before the first submission.

## Snapshot/no-snapshot comparison

Submit two independent fresh 32-node runs (no dependency between them):

```bash
DRY_RUN=1 bash examples/nemo_gym/supervl3p5/scripts/launch_caprl_snapshot_ablation.sh
bash examples/nemo_gym/supervl3p5/scripts/launch_caprl_snapshot_ablation.sh
```

The paired launcher targets 30 steps with a 3-hour normal-QoS allocation per
arm. Both use the same seed, reference training settings, token capture,
trainer/TQ saves every step, and router replay disabled. Batch attribution is
enabled, vLLM metrics are sampled every 10 seconds, and independent rollout
throughput/barrier telemetry is sampled every 30 seconds. Full token dumps
and per-request OpenTelemetry tracing remain disabled to limit overhead.

The `baseline-nosnapshot` arm sets sibling recovery and a null rollout
snapshot interval. It still saves trainer and data-plane checkpoints; it
does not freeze Gym or capture generation prefixes. The `prefix600` arm
enables prefix recovery with a 600-second interval. Trainer saves also wake
the rollout snapshot pump, so 600 seconds is not a strict minimum interval.
Both start in new directories; neither restores the previous convergence run.

Set `ABLATION_NAME` to choose the shared experiment-name prefix. Override
`WALLTIME` or `SLURM_QOS` for site scheduling. Compare at matching optimizer
steps, not W&B history-row indices. Group lifecycle/selection records live
under each allocation's logger directory in `batch_attribution/groups-*.jsonl`;
per-task batch statistics and existing training/finalization timing metrics
are sent to W&B. Leave the mounted code unchanged while the pair runs.
