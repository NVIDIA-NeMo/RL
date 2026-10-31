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

## Why It Is Separate

Rollout checkpointing grew inside the controller one recovery level at a time. By the
time turn-level recovery with NeMo Gym landed, the controller held 31 checkpoint and
recovery methods, about 1,700 lines, and:

- trainer checkpoints and rollout snapshots each built, validated, serialized and saved
  the same rollout state in their own copy;
- the controller built the Gym checkpoint coordinator, retired and forgot Gym episodes,
  parsed Gym's manifest, and branched on whether Gym was present;
- the rollout pump kept two copies of its dataloader loop and spare-pool drain, one
  recording prompt groups in the ledger and one not;
- tests had to build a controller to exercise any of it.

The controller now calls this code through a small set of objects, and its checkpoint
code names the environment backend only for Gym's task index (see Open Items).

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

The controller builds one `RolloutParts` in `__init__` and hands it to both halves:

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
call each other. Every call the controller makes:

| Where | Call |
|---|---|
| `_run_pumps`, before the pumps start | `environment_checkpoint.check_startup()`, the weight sync, then `_rollout_restorer.restore()`; if it returns `True`, the controller turns on ledger-recorded dispatch |
| `_run_pumps` | `_rollout_snapshots.run()` as a background task when `snapshot_attempt_interval_s` is set, and `stop_requested` when it returns |
| Rollout pump, before the dataloader | `_rollout_restorer.redispatch_restored(launch, admit)` when dispatch records prompt groups |
| Train pump | `begin_optimizer_commit(drain_snapshot=...)` where the model update lands, `end_optimizer_commit()` once the step's TQ cleanup and counters publish |
| `_save_checkpoint` | `async with _rollout_snapshots.trainer_checkpoint():` around the whole trainer save, after publishing any save still pending |
| `_save_checkpoint_impl`, under the exclusive cut | `_rollout_snapshots.capture(cut, path, "trainer_checkpoint")`, then `write_controller_sidecars(path, captured)` before `_publish_when_policy_write_done` |
| `_finish_pending_save` | `async with _rollout_snapshots.trainer_checkpoint():` around publishing a save that `defer_async_checkpoint_finalize` left pending |
| `_publish_when_policy_write_done`, `_finish_pending_save` | `_rollout_snapshots.trainer_checkpoint_published()` once a trainer checkpoint is published |

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

An environment backend that keeps rollout state between turns implements
`EnvironmentCheckpoint`:

| Member | Purpose |
|---|---|
| `holds_rollout_state` | Whether rollout snapshots must capture this backend. |
| `check_startup()` | Refuse, or warn about, a deployment that cannot recover as configured. Gym: restart-only servers. |
| `snapshot(path)` | Pause the backend, save its rollouts and yield an `EnvironmentSnapshot`; resume on exit. |
| `EnvironmentSnapshot.check_against_cut(cut)` | Under the TQ cut, require the backend's rollouts to match the ledger. |
| `EnvironmentSnapshot.metrics(paused_seconds)` | Backend metrics logged with the snapshot. |
| `restored(path)` | Restore the backend's rollouts and yield the attempts it continued and the staging rows they reference; drop them if the rest of restore fails. |
| `is_unavailable(error)` | Whether a failed snapshot only met a briefly unreachable backend, so the next attempt can retry. |

`NoEnvironmentCheckpoint` is the backend that keeps nothing. `build_environment_checkpoint`
picks Gym when `rollout_recovery.target_level` is `turn`. A new backend implements the
interface in its own module and adds one branch to that builder.

The recovery ledger names the environment's rollouts with backend-neutral types:
`EnvironmentEpisode` and the `environment_*` queries. Its on-disk keys keep their
original names (`gym_instance_id`, `gym_attempt`), so older checkpoints restore.

## Rules for New Code

- Keep backend-specific code out of `single_controller.py` and `rollout_recovery.py`. Code
  that names an environment backend's concepts belongs in that backend's module, behind
  `EnvironmentCheckpoint`.
- New rollout-checkpointing behavior goes in this package and is reached through the
  calls listed above, not through new controller methods or new branches in the pumps.
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

## Open Items

- The Gym rollout path in `nemo_rl/experience/rollout_manager.py` still threads its
  recovery-tracking arguments through several layers; it is not part of this package.
- The ledger still keeps Gym's retire and forget queues. They could move into the Gym
  backend, with the ledger reporting only which attempts it dropped or continued.
- `EnvironmentEpisode`'s row key format (`<rollout_id>-a<attempt>`) is Gym's; a second
  backend would have to adopt it.
- Whether rollout state is saved, whether dispatch is recorded, and which environment
  backend is used are still decided from several config keys in separate places.
- Captured state, the snapshot manifest and the trainer save state carry Gym's next
  task index under its Gym name (`next_nemo_gym_task_index`), and the controller sets
  it on the rollout manager at startup.
