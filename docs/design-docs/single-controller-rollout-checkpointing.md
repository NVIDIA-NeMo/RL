# Rollout Checkpointing as One Component of the Single Controller

**Status:** proposal. The move happens in five steps, detailed under
[Migration Plan](#migration-plan):

1. Move the Gym checkpoint code behind an `EnvironmentCheckpoint` interface. **Done in this PR.**
2. Give the ledger and rollout manager backend-neutral names. *Proposed.*
3. Move the snapshot writer into one `RolloutCheckpointing` component. *Proposed.*
4. Move the restore chain into that component. *Proposed.*
5. Collapse the rollout pump's two dispatch loops into one. *Proposed.*

## The Problem

`SingleControllerActor` (`nemo_rl/algorithms/single_controller.py`) runs GRPO and PPO as a
train pump and a rollout pump. Partial rollout checkpointing grew inside it, one recovery
level at a time:

| PR | What it added |
|---|---|
| #3480 | Replay-buffer restore from native TQ checkpoints |
| #3923 | Sibling-level rollout checkpointing at the train boundary |
| #3924 | Periodic rollout snapshots |
| #3925 | Rollout checkpoint telemetry |
| #4500 | Turn-level recovery with NeMo Gym (+642 / −53 lines in the controller) |

At #4500 the file is 6,446 lines. About 1,800 of them are 32 checkpoint and recovery
methods on the controller, and about 250 more are hooks inside the loop itself
(`__init__`, `_run_pumps`, `_rollout_pump`, `_train_pump`, `_save_checkpoint_impl`,
`_finalize_with_actor`, `_abort_stale_inflight`). That causes four problems:

1. **There is no single off switch.** Five independent settings control the feature.
   With `checkpointing.enabled` and a replay-capable sampler, setup forces
   `save_data_plane=true`, which in turn forces ledger-tracked dispatch on.
2. **The loop has two copies of dispatch.** `_rollout_pump` and
   `_drain_reserve_into_steps` each have a recovery path and a plain path. Ten
   `if self._rollout_recovery_enabled:` branches choose between them.
3. **Two save paths repeat each other.** `_save_checkpoint_impl` rebuilds the recovery
   state, saves TQ and writes the same sidecar files that `_capture_rollout_checkpoint_cut`
   and `_write_rollout_checkpoint_sidecars` already handle for rollout snapshots.
4. **The controller knows it is talking to NeMo Gym.** It built the
   `GymCheckpointCoordinator`, owned the Gym dispatch gate, retired and forgot Gym
   episodes, parsed `gym_checkpoint.json`, and branched on whether a Gym coordinator
   existed. A run on another environment backend would carry all of it.

## The Goal

The controller is the train pump and the rollout pump. It calls one rollout-checkpointing
component through a small API and never names the environment backend. Turning recovery
off means handing the controller a component whose methods do nothing.

```text
SingleControllerActor            train pump + rollout pump
  └─ RolloutCheckpointing        one object, one small API          (steps 3-5)
       ├─ data-plane barrier, save lock, optimizer-commit window
       ├─ snapshot writer         TQ cut, sidecars, manifest, retention
       ├─ restorer                replay buffer, ledger, reserve, cursor checks
       ├─ snapshot pump           periodic attempts, outcome telemetry
       └─ EnvironmentCheckpoint   backend plug-in                     (step 1, done)
            ├─ GymEnvironmentCheckpoint   nemo_rl/environments/
            └─ NoEnvironmentCheckpoint    backends that keep no rollout state
  └─ NoRolloutCheckpointing      every method a no-op, when recovery is off
```

### The Controller-Facing API (Target)

```python
class RolloutCheckpointing(Protocol):
    # Startup.
    async def check_startup(self) -> None: ...
    async def restore(self, state: ControllerRolloutState) -> RestoreReport: ...
    def background_tasks(self) -> list[Coroutine[Any, Any, None]]: ...  # snapshot pump
    @property
    def stop_requested(self) -> bool: ...  # pre-step deadline snapshot asked to stop

    # Rollout pump. Replaces `if enabled: async with barrier.mutation(...)`.
    def reservation(self, kind: str) -> AsyncContextManager[Reservation]: ...
    async def redispatch_restored(self, launch: LaunchFn) -> None: ...

    # Train pump. Replaces the save lock, _begin_optimizer_commit and the
    # _optimizer_commit_in_progress flag.
    def optimizer_commit(self) -> AsyncContextManager[None]: ...

    # Trainer checkpoint. Replaces the duplicated block in _save_checkpoint_impl.
    async def save_with_trainer_checkpoint(
        self, checkpoint_path: Path, state: ControllerRolloutState
    ) -> None: ...
```

`ControllerRolloutState` is the controller's own state that a snapshot must persist:
dataloader state, sampler dispatch index, replacement reserve, batch shortfall,
`sampler_stamps_target_steps`, finalizer metrics and rewards, and the step counters. The
controller builds it when the component asks, under the data-plane cut. The component never
reads controller fields directly, which is what the tests do today: they set about 40
private controller fields by hand.

`NoRolloutCheckpointing` has a reservation that mints group IDs without a ledger, an
`optimizer_commit` that just yields, a `restore` that restores nothing, and no background
tasks. With it, the rollout pump runs a single dispatch loop, and the plain path is the
same code with nothing recorded.

### The Environment API (Implemented in Step 1)

`nemo_rl/experience/environment_checkpoint.py`:

| Member | Purpose |
|---|---|
| `holds_rollout_state` | Whether rollout snapshots must capture this backend. If so, a trainer checkpoint does not stand in for a snapshot. |
| `check_startup()` | Refuse or warn about a deployment that cannot recover as configured. Gym: restart-only servers. |
| `snapshot(path)` | Context manager. Pause the backend, save its rollouts and yield an `EnvironmentSnapshot`, or `None` when there is nothing to save. The backend resumes on exit. |
| `EnvironmentSnapshot.check_against_cut(cut)` | Under the TQ cut, require the backend's rollouts to match RL's ledger. |
| `EnvironmentSnapshot.metrics(paused_seconds)` | Backend metrics logged with the snapshot. Gym: `gym_parked_seconds`, `gym_restarted_episodes`. |
| `restored(path)` | Context manager. Restore the backend's rollouts and yield `RestoredEnvironmentRollouts`: the attempts it continued, plus the staging rows they reference. A failure drops them before the backend reopens. |
| `is_unavailable(error)` | Whether a failed snapshot only met a briefly unreachable backend, so the next attempt can retry. |

A new backend implements these in its own package and adds one branch to
`build_environment_checkpoint` (`nemo_rl/environments/checkpointing.py`). The controller
does not change.

## Migration Plan

Each step is one PR that keeps behavior the same. Each one moves code and the tests for it.

1. **Environment seam (done).**
   - All Gym-specific checkpoint code leaves the controller for `GymEnvironmentCheckpoint`: eight methods (inventory, queued retirements, retire, forget, restart-only check, drain, cut validation, prepared commit), plus the inline Gym restore block and the Gym metrics.
   - The controller goes from 6,446 to 6,032 lines. Its only remaining checkpoint-related Gym name is the ledger argument `restored_gym_episodes`.
2. **Neutral names below the controller.**
   - The ledger's Gym API gets backend-neutral names: `gym_checkpoint_inventory` and the retire/forget queues become `environment_*`; `restored_gym_episodes` becomes `continued_attempts`; the record fields `gym_instance_id` and `gym_attempt` become `environment_*` (needs a schema bump or a read alias).
   - Rollout-ID formats tied to Gym's `EpisodeId` move behind the backend.
   - `_summarize_gym_episodes`, which #4501 adds to the controller, moves to the Gym module.
3. **Snapshot writer out.** These move into `RolloutCheckpointing`, about 800 lines:
   - `_save_rollout_checkpoint`, `_capture_rollout_checkpoint_cut`, `_write_rollout_checkpoint_sidecars`
   - `_save_data_plane_checkpoint`, `_validate_replay_inventory`, `_validate_rollout_recovery_inventory`
   - `_rollout_checkpoint_pump`, outcome logging, and the warnings
   - the save lock, the commit window and the wakeup event

   `_save_checkpoint_impl` then calls `save_with_trainer_checkpoint`, which removes its duplicated block.
4. **Restorer out.** These become `restore()`, about 600 lines:
   - the `_maybe_restore_*` chain
   - `_load_recovery_prompt`, `_rehydrate_rollout_recovery_prompts`, `_queue_restored_replay_groups_for_regeneration`
   - `_validate_restored_sampler_cursor`, `_log_rollout_restore_metrics`
5. **One dispatch loop.** The rollout pump always reserves through `reservation()`. This removes the second loop copies and the ten `_rollout_recovery_enabled` branches.

**End state:**
- The controller loses about 1,800 lines.
- Rollout checkpointing lives in one package with its own tests.
- Turning it off is one setting that hands the controller `NoRolloutCheckpointing`.

## Rules for New Code

- **Keep backend-specific code out of `single_controller.py`.** Code that names or parses an environment backend's concepts (episodes, participants, rollout-ID formats) does not go there or in `rollout_recovery.py`. It belongs in the backend's package, behind `EnvironmentCheckpoint`.
- **New recovery behavior goes in the component.** It is reached through its API, not through new controller methods or new `if` branches in the pumps.
