# Rollout Checkpointing in the Single Controller

The single controller (`nemo_rl/algorithms/single_controller.py`) runs GRPO and PPO as a
rollout pump and a train pump. Rollout checkpointing saves what those pumps have in
flight: completed groups in the replay buffer, unfinished prompt groups in the recovery
ledger, spare prompts, and, at turn-level recovery, the environment's own rollouts. A
resumed run restores all of it and dispatches the unfinished work before new prompts.

This page describes where that code lives, how the controller calls it, and the rules
for changing it. For configuring recovery, the checkpoint layout and resuming a run, see
[Checkpointing and Replay Recovery](../guides/single-controller.md#checkpointing-and-replay-recovery)
in the Single-Controller guide.

## Layout

```text
nemo_rl/algorithms/rollout_checkpointing/
    state.py        RolloutRuntimeState, CapturedRolloutState, ResumeSource,
                    TrainingProgress, and RolloutParts: the controller parts both
                    halves read
    inventory.py    TQ inventory checks that saving and restoring both run
    snapshots.py    RolloutSnapshotter: save lock, optimizer-commit window, capture,
                    snapshot pump and publication
    restore.py      RolloutRestorer: restore chain, prompt rehydration and redispatch
    manifest.py     snapshot manifests, bootstrap anchors, atomic publication and
                    retention

nemo_rl/environments/checkpointing/
    interfaces.py   EnvironmentCheckpoint and EnvironmentSnapshot (Protocols),
                    NoEnvironmentCheckpoint, EnvironmentEpisode
    nemo_gym.py     GymEnvironmentCheckpoint
    __init__.py     build_environment_checkpoint: picks the backend from config
```

The recovery ledger stays in `nemo_rl/experience/rollout_recovery.py`, because the
rollout manager updates it as rollouts run.

## How the Controller Uses It

The controller builds one `RolloutParts` and hands it to both halves:

```python
rollout_parts = RolloutParts(master_config=..., dp_client=..., buffer=..., dataloader=...,
                             sampler=..., rollout_manager=..., data_plane_barrier=...,
                             environment_checkpoint=..., runtime_state=self._rollout_runtime,
                             read_progress=self._training_progress,
                             log_metrics=self._log_telemetry_metrics)
self._rollout_snapshots = RolloutSnapshotter(rollout_parts, checkpointer=..., timeout=...,
                                             bootstrap_identity=..., teacher_coordinator=...,
                                             records_prompt_groups=...)
self._rollout_restorer = RolloutRestorer(rollout_parts, algo_cfg=..., buffer_capacity=...,
                                         resume=ResumeSource(...))
```

The two halves share only these parts, of which only `runtime_state` changes, and never
call each other. The controller calls only their public methods; see the
`RolloutSnapshotter` and `RolloutRestorer` docstrings.

Neither half reads controller fields. They read the training counters through the
`read_progress` callback, which returns the controller's live values on every call, and
the only controller state they change is the shared `RolloutRuntimeState`.

## Shared State

`RolloutRuntimeState` holds the rollout state the pumps change and every checkpoint
saves: the spare-prompt pool, each stamped step's shortfall, whether the sampler stamps
target steps, and the finalizer results kept until a group is trained. The controller
owns the one instance. Checkpoints read it under the exclusive data-plane cut, and
restore fills it before the pumps start.

`ResumeSource` is what setup restored from: the checkpoint path, whether it is a rollout
snapshot, its native TQ metadata and the load timings. Its defaults describe a fresh run.

## Saving

Trainer checkpoints and rollout snapshots save rollout state through one
`capture(cut, path, kind)` and one `write_controller_sidecars`. The kind keeps the real
differences:

| | Trainer checkpoint | Rollout snapshot |
|---|---|---|
| Model state | Saved by the controller | Referenced: anchored to a durable trainer checkpoint |
| TQ and replay metadata | Only with `checkpointing.save_data_plane`, and replay metadata only for a sampler that restores buffered groups | Always |
| Rows an unfinished streamed step owns | Refused | Re-indexed into the saved replay view |
| Recovery ledger | Only while dispatch records prompt groups | Always |
| Environment state | Never; the environment keeps running | Captured when the backend holds rollout state |

One lock, held by `trainer_checkpoint()` for the whole trainer save, by the publish of a
deferred save, and by each snapshot attempt, keeps the two from publishing at the same
time. A trainer checkpoint holds no environment state, so once one is published,
immediately or at a later step boundary with `defer_async_checkpoint_finalize`,
`trainer_checkpoint_published()` wakes the snapshot pump to capture the environment
anchored to it.

A rollout snapshot attempt runs in this order:

1. Take the save lock. Skip if a train step's optimizer commit is in progress, or if no
   data-plane mutation happened since the last snapshot (unless forced).
2. Resolve the anchor: the bootstrap anchor before the first train step, otherwise
   `step_<N>`. Skip if that trainer checkpoint is not durable yet.
3. Enter the environment phase. For Gym this closes new `/run` dispatch, pauses rollout
   deadlines, retires abandoned attempts, then prepares, commits and drains. If a train
   step starts committing meanwhile, the attempt stops and the snapshot is discarded.
4. Take the exclusive data-plane cut, re-check that the trainer has not moved, check the
   environment's export against the cut, and `capture` the rollout state.
5. Leave the cut, then the environment phase. The environment resumes and dispatch
   reopens, so new rollouts do not wait for the snapshot's disk writes. Once the
   environment's commit succeeded, a resume that still fails is only a warning: Gym's
   checkpoint lease reopens that deployment on its own, and the snapshot stays valid.
6. Publish: write the sidecars, config and manifest, and commit the snapshot atomically
   under retention.

Every attempt logs its outcome under `rollout/checkpoint_outcome`, and every saved
snapshot its timings under `timing/rollout_checkpoint`.

## Restoring

`restore()` runs once, after the weights are in sync and before the pumps start. It
restores, in order:

1. The replay buffer, from the native TQ checkpoint, taking one replay-capacity permit
   per restored group. Before it, and before the ledger, the checkpoint's forest-training
   settings must match the run's.
2. The recovery ledger, from its sidecar, after checking the sidecar's checksum, schema
   and group count against the TQ metadata (`read_recovery_sidecar`). Turn-level groups
   from a rollout snapshot are restored inside the environment's restore scope, which
   drops what it restored if anything later fails. With token capture, the staging
   inventory is checked against the ledger before restart normalization, so nothing is
   discarded on a mismatch, and again after it, clearing rows nothing references.
3. The spare-prompt pool. Checkpoints with rollout-recovery schema 5 or later always
   carry one, even if empty; older ones may not.
4. A check that the sampler cursor covers the restored work.

It returns whether it queued groups that only ledger-recorded dispatch can launch, such
as replay groups it discarded on a replay-free resume. The rollout pump then calls
`redispatch_restored` to launch the restored groups before reading the dataloader.

## Dispatch With and Without the Ledger

When the run saves recovery state, or resumes from a checkpoint that carried some, the
rollout pump records every prompt group in the recovery ledger. Otherwise it records
nothing, and the rollout manager mints group IDs as it runs them.

The pump has one dataloader loop and one spare-pool drain for both modes. Four small
controller methods hold the difference:

- `_dispatch_reservation(kind)` holds a data-plane mutation cut, counted as `kind`, only
  with ledger dispatch;
- `_reserve_prompt_group` and `_discard_prompt_group` record in the ledger under that cut,
  or do nothing;
- `_admit_step` admits a step through the ledger's transactional admission or the
  sampler's own `admit()`.

A run without ledger dispatch never takes a mutation slot to dispatch, so it never waits
on a checkpoint to start new work.

## The Environment Interface

An environment backend that keeps rollout state between turns implements the
`EnvironmentCheckpoint` protocol in
[`interfaces.py`](../../nemo_rl/environments/checkpointing/interfaces.py); its method
docstrings describe each member, and
[`GymEnvironmentCheckpoint`](../../nemo_rl/environments/checkpointing/nemo_gym.py) is the
Gym implementation.

`NoEnvironmentCheckpoint` is the backend that keeps nothing. `build_environment_checkpoint`
picks Gym when `rollout_recovery.target_level` is `turn`. A new backend implements the
interface in its own module and adds one branch to that builder.

The recovery ledger names the environment's rollouts with backend-neutral types:
`EnvironmentEpisode` and the `environment_*` queries. Its on-disk keys keep their
original names (`gym_instance_id`, `gym_attempt`), so older checkpoints restore.
`EnvironmentEpisode`'s row key format (`<rollout_id>-a<attempt>`) is Gym's, and a new
backend must use it.

## Tracing What a Restore Cost

The metrics above are averages and counts. To check a stop and restore rollout by
rollout, set `rollout_checkpointing.trace: true`. The controller then appends one
JSON line per event to `rollout_trace.jsonl` in the log directory; a run that is
killed loses only the last few milliseconds of events. Each line has the
wall-clock time `t`, a `run` ID that is new for each process, the `event` name
and that event's fields. The events are listed in
`nemo_rl/algorithms/rollout_checkpointing/trace.py`. Per-rollout events need
ledger dispatch, so they appear only with rollout recovery.

Rows match across runs by `prompt_id` (the dataset index) and `generation_index`;
group and attempt IDs are new in every run. A stop's thrown-away time is, for
each attempt abandoned with `reason: restart`, the time from its last
`attempt_dispatched` to the restored snapshot's `cut_time`; its lost progress is
how much later the groups running at the stop reached `group_sealed` than the
same prompts in runs that never stopped. A model call redone after a turn-level
resume is not visible here: that needs the time the environment parked each
rollout, which it does not report yet.

Batch age, the token-weighted number of policy updates between generating a
token and training on it, is logged every step as `batch_age` when token capture
is on, and each `group_published` event carries the same per-version token
counts, so the trace gives the same number per step:

```bash
python -m nemo_rl.algorithms.rollout_checkpointing.trace <log_dir>/rollout_trace.jsonl
```

vLLM keeps running calls across a refit, so a call's tokens count under the
version it began on until a refit paused it, and under the version that refit
installed from there on. At each refit the controller reads each running
captured call's token count from vLLM's output-processor request state, which
has no public API; a backend that does not report it counts every token of a
call under the version the call began on.

## Rules for New Code

- Keep backend-specific code out of `single_controller.py` and `rollout_recovery.py`. Code
  that names an environment backend's concepts belongs in that backend's module, behind
  `EnvironmentCheckpoint`.
- New rollout-checkpointing behavior goes in this package and is reached through the
  snapshotter's and restorer's public methods, not through new controller methods or new
  branches in the pumps.
- State that the pumps change and checkpoints save belongs in `RolloutRuntimeState`.
- Changing what a checkpoint saves changes its format: bump the relevant schema version
  and keep reading the previous one.

## Tests

- `tests/unit/environments/test_gym_environment_checkpoint.py` builds the Gym checkpoint
  directly: retire, forget, drain, cut validation, startup checks and the builder.
- `tests/unit/single_controller/test_checkpointing.py` and
  `test_checkpoint_dispatch_races.py` cover saving, restoring and their races with the
  pumps. Tests that build a controller without `__init__` attach the snapshotter and
  restorer with `tests/unit/single_controller/_rollout_checkpoints.py`.
- `tests/unit/single_controller/test_rollout_checkpoint.py` covers manifests, anchors,
  publication and retention.
