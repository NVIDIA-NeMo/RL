# Rollout Checkpointing as One Component of the Single Controller

**Status:** all five steps are done in this stack. They are detailed under
[Migration Plan](#migration-plan):

1. Move the Gym checkpoint code behind an `EnvironmentCheckpoint` interface. **Done in this stack.**
2. Give the ledger and rollout manager backend-neutral names, and make the interfaces ABCs. **Done in this stack.**
3. Move the snapshot writer into one `RolloutCheckpointing` component. **Done in this stack.**
4. Move the restore chain into that component. **Done in this stack.**
5. Collapse the rollout pump's two dispatch loops into one. **Done in this stack.**

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
       │                         nemo_rl/algorithms/rollout_checkpointing/
       ├─ save lock, optimizer-commit window, wakeup and stop events  (step 3, done)
       ├─ snapshot writer         TQ cut, sidecars, manifest, retention (step 3, done)
       ├─ snapshot pump           periodic attempts, outcome telemetry  (step 3, done)
       ├─ restorer                replay buffer, ledger, reserve, cursor checks (step 4, done)
       ├─ prompt dispatch         ledger or unrecorded reservations   (step 5, done)
       └─ EnvironmentCheckpoint   backend plug-in                     (step 1, done)
            ├─ GymEnvironmentCheckpoint   nemo_rl/environments/
            └─ NoEnvironmentCheckpoint    backends that keep no rollout state
  └─ NoRolloutCheckpointing      no pump, no lock, no TQ, when nothing is saved;
                                 still restores through the same restorer
```

### The Controller-Facing API

What steps 3 to 5 built is in `nemo_rl/algorithms/rollout_checkpointing/interface.py`.

```python
class RolloutCheckpointing(abc.ABC):
    # Every member is an @abc.abstractmethod; an implementation subclasses this.

    # Startup (step 3).
    def background_tasks(self) -> list[Coroutine[Any, Any, None]]: ...  # snapshot pump
    @property
    def stop_requested(self) -> bool: ...  # pre-step deadline snapshot asked to stop

    # Train pump (step 3). Replaces the save lock, _begin_optimizer_commit and the
    # _optimizer_commit_in_progress flag. drain_snapshot=True first waits for a
    # snapshot that is still saving the old trainer identity (PPO's early refit).
    def optimizer_commit(
        self, *, drain_snapshot: bool
    ) -> AbstractAsyncContextManager[None]: ...

    # Trainer checkpoint (step 3). trainer_checkpoint() holds the save lock for the
    # whole save and wakes the snapshot pump after it when the environment keeps
    # rollout state. save_with_trainer_checkpoint replaces the duplicated block in
    # _save_checkpoint_impl: it runs under the exclusive cut, saves TQ and returns
    # the files to write before the checkpoint is finalized.
    def trainer_checkpoint(self) -> AbstractAsyncContextManager[None]: ...
    async def save_with_trainer_checkpoint(
        self,
        cut: DataPlaneMutationCut,
        checkpoint_path: PathLike,
        state: ControllerRolloutState,
    ) -> ControllerSidecars: ...

    # Startup (step 4). Called once in _run_pumps, after the weight sync and before
    # the pumps start. The controller applies the report to its own fields.
    async def restore(self) -> RestoreReport: ...

    # Rollout pump (step 4). Launches the unfinished groups restore loaded, before
    # the pump reads the dataloader. `admit` is the pump's sampler admission for
    # groups saved before their batch was admitted.
    async def redispatch_restored(self, launch: LaunchFn, admit: AdmitFn) -> None: ...

    # Rollout pump (step 5). Replaces `if enabled: async with barrier.mutation(...)`
    # and the plain-path copies of the dispatch loop. With ledger dispatch the
    # reservation holds a data-plane mutation slot counted as `kind`; without it,
    # it holds nothing.
    @property
    def records_prompt_groups(self) -> bool: ...
    def reservation(
        self, kind: CheckpointMutationKind
    ) -> AbstractAsyncContextManager[PromptReservation]: ...


class PromptReservation(abc.ABC):
    # What the rollout pump changes in one step. LedgerReservation records each
    # call in the recovery ledger under the reservation's cut; UnrecordedReservation
    # does nothing, and reserve_prompt_group returns None so the rollout manager
    # mints the group ID, as the plain path always did.
    def reserve_prompt_group(
        self, prompt, *, target_step, admitted=True, admission_id=None
    ) -> Optional[str]: ...
    async def admit(self, sampler, *, trainer_version_fn) -> Optional[int]: ...
    def mark_prompt_group_admitted(self, group_id, *, target_step) -> None: ...
    def discard_prompt_group(self, group_id) -> None: ...
    async def discard_recovery_group(self, group_id: str) -> None: ...


class RolloutCheckpointingHost(abc.ABC):
    # What the component may ask the controller. The controller passes a small
    # adapter, _ControllerRolloutHost, so the component never reads its fields.
    @property
    def train_steps(self) -> int: ...
    @property
    def trainer_version(self) -> int: ...
    @property
    def current_epoch(self) -> int: ...
    def rollout_state(self) -> ControllerRolloutState: ...  # under the data-plane cut
    def log_telemetry_metrics(
        self, metrics: dict[str, float], *, step: int, prefix: str
    ) -> None: ...
```

The train pump keeps one `contextlib.AsyncExitStack` per step. Each place that used to
call `_begin_optimizer_commit` enters `optimizer_commit(...)` on it, and the stack closes
right after the step's counters publish, where the flag used to be cleared. A step that
fails leaves the window open, as before.

`ControllerRolloutState` (`controller_state.py`) is the controller's own state that a
checkpoint must persist: the step counters, dataloader state, sampler dispatch index,
whether the sampler can restore buffered groups, replacement reserve, batch shortfall,
`sampler_stamps_target_steps`, finalizer metrics and rewards, and whether dispatch records
unfinished groups in the ledger. The controller builds it when asked, under the data-plane
cut: the snapshot writer asks through the host, and the trainer checkpoint builds it in
`_save_checkpoint_impl`, which also needs it for `GRPOSaveState`. Its dictionaries are the
controller's live objects, so what is serialized is the same as before the move.

Trainer checkpoints and rollout snapshots write the same controller files through one
function, `write_controller_sidecars`. The checks that the TQ rows match the replay index
and the ledger (`validate_replay_inventory`, `validate_rollout_recovery_inventory`) and
`capture_reward_settings` are plain functions in `inventory.py`, not component members:
restore runs them too, and turning rollout checkpointing off must not turn off a restore's
checks.

The controller gets `NoRolloutCheckpointing` when it takes no rollout snapshots
(`snapshot_attempt_interval_s` unset) and its trainer checkpoints skip the data plane
(checkpointing off, or `save_data_plane` off). It has no background tasks, no lock and no
commit window. Its `save_with_trainer_checkpoint` returns only the dataloader state and
spare prompts, which every trainer checkpoint needs. With unrecorded dispatch, its
`reservation` takes no mutation slot and records nothing, and the rollout manager mints
group IDs as before. So the rollout pump runs a single dispatch loop, and the plain path
is the same code with nothing recorded.

Whether dispatch records prompt groups in the ledger does not follow from which
implementation was built. Both hold one `PromptDispatch` (`dispatch.py`).
`_build_rollout_checkpointing` turns it on when the run saves recovery state
(`checkpointing.enabled`, `save_data_plane` and a sampler that restores buffered groups)
or when the resumed checkpoint carried a recovery payload. `restore()` also turns it on
when the report says restore queued groups only ledger dispatch can launch, such as a
replay-free resume. A run with `NoRolloutCheckpointing` can therefore still dispatch
through the ledger. `redispatch_restored` does nothing while ledger dispatch is off,
because then there is nothing restored to launch.

Restore is the one thing `NoRolloutCheckpointing` does not skip. The checkpoint a run
resumes from can hold buffered rollouts, unfinished groups and spare prompts even when the
resumed run saves none, and the restore-time checks must still run. So both
implementations hand `restore` and `redispatch_restored` to one `RolloutRestorer`
(`restore.py`), which the controller builds with the same parts. Its input is a
`CheckpointToRestore` (path, whether it is a rollout snapshot, the native TQ metadata and
the setup timers), built in `__init__` from what setup restored. Its output is a
`RestoreReport` the controller applies in `_apply_restore_report`: replay-capacity permits
for the restored groups, spare prompts, batch shortfall, finalizer metrics and rewards,
`sampler_stamps_target_steps`, and whether restore queued groups that only ledger-tracked
dispatch can launch. A `None` field means the checkpoint had nothing for it. The restorer
never writes controller fields.

The environment's `check_startup()` is not a component member: the controller calls it
on `EnvironmentCheckpoint` directly at the top of `_run_pumps`, as before. It runs for
every run, saving or not, and needs nothing from the component.

### The Environment API (Implemented in Steps 1 and 2)

`nemo_rl/experience/environment_checkpoint.py` defines `EnvironmentCheckpoint` and
`EnvironmentSnapshot` as `abc.ABC` classes. A backend subclasses them, and every member is an
`@abc.abstractmethod`, so a backend that misses one fails when it is built, not mid-run.

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
2. **Neutral names below the controller (done).**
   - The ledger's Gym API has backend-neutral names: `environment_inventory`, `environment_retirements` / `mark_environment_retired`, `environment_forgets` / `mark_environment_forgotten`; `prepare_for_restart` takes `continued_attempts`; the record fields are `environment_instance_id`, `environment_attempt` and `holds_environment_state`; `environment_episode()` returns an `EnvironmentEpisode`, which Gym's `GymCheckpointEpisode` now aliases.
   - The on-disk ledger keys keep their old names (`gym_instance_id`, `gym_attempt`) and the schema version is unchanged, so older checkpoints still restore.
   - The rollout manager's checkpoint-owner argument is `environment_instance_id`.
   - `EnvironmentCheckpoint` and `EnvironmentSnapshot` are `abc.ABC` classes instead of `Protocol`s.
   - The controller has no Gym checkpoint names left.
   - Still open: the attempt-qualified key format (`<rollout_id>-a<attempt>`) lives in `EnvironmentEpisode`, not behind the backend. `_summarize_gym_episodes`, which #4501 adds to the controller, moves to the Gym module once #4501 lands.
3. **Snapshot writer out (done).**
   - These moved into `DataPlaneRolloutCheckpointing` (`nemo_rl/algorithms/rollout_checkpointing/data_plane.py`): `_save_rollout_checkpoint`, `_capture_rollout_checkpoint_cut`, `_write_rollout_checkpoint_sidecars`, `_save_data_plane_checkpoint`, `_enter_environment_phase_unless_a_step_commits`, `_begin_optimizer_commit`, `_rollout_checkpoint_pump`, outcome logging, the two warnings, the save lock, the commit window, the wakeup and stop events, and the snapshot counters.
   - `_validate_replay_inventory`, `_validate_rollout_recovery_inventory` and `_capture_reward_settings` became plain functions in `inventory.py`, because restore calls them too.
   - `_save_checkpoint_impl` calls `save_with_trainer_checkpoint` and `write_controller_sidecars` instead of its own TQ capture and sidecar writes. It went from 223 to 130 lines.
   - The controller goes from 6,032 to 5,270 lines. The package is 1,454 lines in five files.
   - Still open: the controller still owns the data-plane barrier and opens the trainer checkpoint's exclusive cut itself. Step 5 keeps it there; see below.
4. **Restorer out (done).**
   - These moved into `RolloutRestorer` (`nemo_rl/algorithms/rollout_checkpointing/restore.py`): `_maybe_restore_replay_buffer`, `_maybe_restore_rollout_recovery`, `_maybe_restore_replacement_reserve`, `_queue_restored_replay_groups_for_regeneration`, `_load_recovery_prompt`, `_rehydrate_rollout_recovery_prompts`, `_validate_restored_sampler_cursor`, `_log_rollout_restore_metrics` and `_redispatch_restored_rollouts`. `_load_recovery_prompt` had no caller outside restore, so it moved too.
   - `restore()` runs them in the old order and returns a `RestoreReport`. `redispatch_restored(launch, admit)` replaces `_redispatch_restored_rollouts`; the rollout pump passes its `_launch` and `_admit_reserved_prompt_groups`.
   - The four restore-only controller fields (`_last_checkpoint_path`, `_restored_from_rollout_snapshot`, `_data_plane_checkpoint_metadata`, `_rollout_checkpoint_load_metrics`) and `_restored_replay_groups_to_regenerate` are gone. `__init__` builds a `CheckpointToRestore` instead, and `_run_pumps` calls `restore()` and `_apply_restore_report()` in place of the five-call chain.
   - The replay-capacity permits for restored groups are now taken in `_apply_restore_report`, right after `restore()` returns, instead of inside the replay restore. Nothing reads the semaphore before the pumps start, so the order change has no effect.
   - The controller goes from 5,270 to 4,656 lines. The package is 2,394 lines in six files.
   - Still in the controller, for step 5: `_admit_reserved_prompt_groups` and `_require_unoccupied_target_step`. Both are live dispatch: the rollout pump and `_drain_reserve_into_steps` admit every new batch through them. The sampler cursor restore and the `restoring_rollout_recovery` check also stay in `__init__`: the first is part of building the sampler, and the second decides the dispatch mode, which step 5 reworks.
5. **One dispatch loop (done).**
   - The rollout pump, `_drain_reserve_into_steps`, `_abort_stale_inflight` and the finalizer's drop path reserve, admit and discard through `reservation(kind)`. `PromptDispatch` (`dispatch.py`) hands out a `LedgerReservation` (one mutation cut, every call recorded in the ledger) or an `UnrecordedReservation` (no cut, nothing recorded).
   - The plain-path copies of the dataloader loop and the spare-pool drain are gone, and so are the `if self._rollout_recovery_enabled:` branches: nine in the pump, drain and abort, plus the one in `_apply_restore_report` that turned the flag on. One check is left: `_launch` refuses a launch without a group ID while `records_prompt_groups` is on.
   - The dispatch-mode decision moved from `__init__` into `_build_rollout_checkpointing`, and the restore-time switch moved from `_apply_restore_report` into the component's `restore()`. `ControllerRolloutState.rollout_recovery_enabled` now reads `records_prompt_groups`.
   - `_admit_reserved_prompt_groups` and `_require_unoccupied_target_step` stay in the controller. They are the rollout pump's sampler admission in both modes, and they read and write controller state (the sampler, `_sampler_stamps_target_steps`, the trainer version, the replay buffer). The ledger part of admission (the cut, the sampler's `commit_admission`, `mark_prompt_group_admitted`) is in the reservation.
   - The data-plane barrier stays in the controller. Every run uses it, including runs with `NoRolloutCheckpointing`: the replay buffer and rollout manager hold it, finalizer commits, known-outcome cleanup, consumed-row cleanup and advantage writeback take mutation slots, and the trainer checkpoint's exclusive cut also covers the controller's own `GRPOSaveState`. The component receives it; moving it would only make the component hand it back.
   - Small differences on the plain path, none of which changes what is trained: `_drain_reserve_into_steps` now sets `_sampler_stamps_target_steps` when the sampler stamps, as the ledger path did; a transactional sampler's gate is checked once before the reservation and again (immediately) inside `admit`; and the "substituting a spare prompt" log names the step the group was dropped from in both modes (the ledger path used to print the lender step).
   - The controller goes from 4,656 to 4,593 lines. The package is 2,717 lines in seven files.

**End state:**
- The controller went from 6,446 lines at #4500's head to 4,593 lines.
- Rollout checkpointing lives in one package, `nemo_rl/algorithms/rollout_checkpointing/` (2,717 lines in seven files), with its own tests.
- Turning it off is one setting that hands the controller `NoRolloutCheckpointing`. The rollout pump runs the same dispatch loop either way.

## Rules for New Code

- **Keep backend-specific code out of `single_controller.py`.** Code that names or parses an environment backend's concepts (episodes, participants, rollout-ID formats) does not go there or in `rollout_recovery.py`. It belongs in the backend's package, behind `EnvironmentCheckpoint`.
- **New recovery behavior goes in the component.** It is reached through its API, not through new controller methods or new `if` branches in the pumps.
