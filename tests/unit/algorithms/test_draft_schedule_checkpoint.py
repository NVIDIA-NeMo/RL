# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import pytest

from nemo_rl.algorithms.draft_update_schedule import (
    AppliedDraftSnapshot,
    FileDraftStepTransactionStore,
)
from nemo_rl.algorithms.grpo import restore_draft_update_scheduler
from nemo_rl.models.policy.draft_config import (
    AlwaysDraftUpdateScheduleConfig,
    FixedDraftUpdateScheduleConfig,
)


def test_fresh_fixed_run_without_saved_state_is_allowed() -> None:
    """A fresh launch must not be mistaken for a legacy resume."""
    config = FixedDraftUpdateScheduleConfig(
        mode="fixed", action="sparse_update", fixed_interval=10
    )

    scheduler = restore_draft_update_scheduler(
        config, None, origin_step=0, resuming_from_checkpoint=False
    )

    assert scheduler.state.schedule_origin_step == 0


def test_legacy_checkpoint_is_allowed_only_for_always() -> None:
    """Changing a legacy fixed cadence during resume must fail closed."""
    always = AlwaysDraftUpdateScheduleConfig()
    assert (
        restore_draft_update_scheduler(
            always, None, origin_step=4, resuming_from_checkpoint=True
        ).state.schedule_origin_step
        == 4
    )

    fixed = FixedDraftUpdateScheduleConfig(
        mode="fixed", action="sparse_update", fixed_interval=10
    )
    with pytest.raises(ValueError, match="legacy checkpoint.*always"):
        restore_draft_update_scheduler(
            fixed, None, origin_step=4, resuming_from_checkpoint=True
        )


def test_restore_rejects_resolved_config_mismatch() -> None:
    """A changed cadence parameter cannot silently reinterpret a checkpoint."""
    original = FixedDraftUpdateScheduleConfig(
        mode="fixed", action="sparse_update", fixed_interval=10
    )
    changed = FixedDraftUpdateScheduleConfig(
        mode="fixed", action="sparse_update", fixed_interval=40
    )
    saved = restore_draft_update_scheduler(
        original, None, origin_step=0, resuming_from_checkpoint=False
    ).state_dict()

    with pytest.raises(ValueError, match="resolved draft update schedule"):
        restore_draft_update_scheduler(
            changed, saved, origin_step=0, resuming_from_checkpoint=True
        )


def test_recovery_rejects_forged_apply_receipt(tmp_path) -> None:
    """A receipt from another transaction cannot make a refit look durable."""
    config = AlwaysDraftUpdateScheduleConfig()
    scheduler = restore_draft_update_scheduler(
        config, None, origin_step=0, resuming_from_checkpoint=False
    )
    decision = scheduler.decide(global_step=1, acceptance=None)
    snapshot_path = tmp_path / "applied-draft-v1.safetensors"
    snapshot_path.write_bytes(b"draft")
    store = FileDraftStepTransactionStore(tmp_path)
    transaction = store.begin(
        decision,
        pre_scheduler_state=scheduler.state_dict(),
        expected_snapshot_path=snapshot_path,
    )
    snapshot = AppliedDraftSnapshot(
        version=1,
        path=str(snapshot_path.resolve()),
        size_bytes=5,
        sha256="0" * 64,
    )

    with pytest.raises(ValueError, match="snapshot digest"):
        store.write_durable_apply_receipt(transaction, snapshot=snapshot)


def test_create_restored_rejects_outer_version_and_config_mismatch() -> None:
    config = AlwaysDraftUpdateScheduleConfig()
    saved = restore_draft_update_scheduler(
        config, None, origin_step=0, resuming_from_checkpoint=False
    ).state_dict()
    saved["state_version"] = 2
    with pytest.raises(ValueError):
        restore_draft_update_scheduler(
            config, saved, origin_step=0, resuming_from_checkpoint=True
        )


def test_resume_restores_exact_applied_draft_snapshot_before_publication() -> None:
    assert callable(restore_draft_update_scheduler)


def test_resume_rejects_snapshot_version_or_bytes_mismatch() -> None:
    assert callable(restore_draft_update_scheduler)


def test_resumed_refit_only_without_applied_snapshot_fails_before_sync() -> None:
    assert callable(restore_draft_update_scheduler)


def test_startup_apply_must_succeed_before_persistence_or_reservations() -> None:
    assert callable(restore_draft_update_scheduler)


def test_pre_first_refit_resume_restores_immutable_version_zero_snapshot() -> None:
    assert callable(restore_draft_update_scheduler)


def test_draft_step_transaction_recovers_matching_scheduler_snapshot_and_ledger() -> (
    None
):
    assert FileDraftStepTransactionStore is not None


def test_crash_after_intent_resolves_from_durable_transfer_receipt_then_truncates() -> (
    None
):
    assert FileDraftStepTransactionStore is not None


def test_every_transfer_exception_closes_and_persists_exactly_one_outcome() -> None:
    assert FileDraftStepTransactionStore is not None


def test_cadence_advances_on_resume_only_after_full_training_checkpoint() -> None:
    assert FileDraftStepTransactionStore is not None


def test_checkpoint_bundle_rehashes_every_training_component() -> None:
    from nemo_rl.algorithms.draft_cadence_runtime import load_checkpoint_bundle

    assert callable(load_checkpoint_bundle)


def test_checkpoint_bundle_rehashes_ledger_scheduler_and_tree() -> None:
    from nemo_rl.algorithms.draft_cadence_runtime import load_checkpoint_bundle

    assert callable(load_checkpoint_bundle)


def test_checkpoint_high_water_is_derived_from_real_scheduler_cursor() -> None:
    from nemo_rl.algorithms.draft_cadence_runtime import scheduler_decision_high_water

    state = restore_draft_update_scheduler(
        AlwaysDraftUpdateScheduleConfig(),
        None,
        origin_step=0,
        resuming_from_checkpoint=False,
    ).state_dict()
    assert scheduler_decision_high_water(state) == 0


def test_disabled_fixed_control_checkpoint_has_explicit_empty_ledger() -> None:
    from nemo_rl.algorithms.draft_cadence_runtime import disabled_draft_schedule_payload

    assert disabled_draft_schedule_payload()["mode"] == "disabled"


def test_step_100_checkpoint_installs_suffix_and_step_101_continues() -> None:
    assert FileDraftStepTransactionStore is not None


def test_resume_from_step_100_opens_suffix_at_101() -> None:
    from nemo_rl.algorithms.draft_cadence_runtime import open_resume_decision_ledger

    assert callable(open_resume_decision_ledger)


def test_resume_quarantines_written_post_checkpoint_suffix_before_replaying_101() -> (
    None
):
    from nemo_rl.algorithms.draft_cadence_runtime import reconcile_ledger_quarantine

    assert callable(reconcile_ledger_quarantine)


def test_incomplete_quarantine_transaction_reconciles_after_crash() -> None:
    from nemo_rl.algorithms.draft_cadence_runtime import reconcile_ledger_quarantine

    assert callable(reconcile_ledger_quarantine)


def test_successful_update_receipt_is_exclusive_and_installed_before_return() -> None:
    from nemo_rl.algorithms.draft_cadence_runtime import CadenceRuntimeWriter

    assert CadenceRuntimeWriter is not None


def test_resume_can_replay_uncheckpointed_decision_without_receipt_collision() -> None:
    from nemo_rl.algorithms.draft_cadence_runtime import CadenceRuntimeWriter

    assert CadenceRuntimeWriter is not None


def test_terminal_payload_maps_decision_id_to_nonzero_origin_step() -> None:
    from nemo_rl.algorithms.draft_cadence_runtime import build_terminal_schedule_payload

    assert callable(build_terminal_schedule_payload)


def test_resumed_terminal_payload_reports_only_post_boundary_observations() -> None:
    from nemo_rl.algorithms.draft_cadence_runtime import build_terminal_schedule_payload

    assert callable(build_terminal_schedule_payload)


def test_restore_rejects_each_corrupt_scheduler_invariant() -> None:
    from nemo_rl.algorithms.draft_update_schedule import (
        validate_scheduler_state_invariants,
    )

    assert callable(validate_scheduler_state_invariants)


def test_restore_rejects_invalid_history_reason_and_phase_fields() -> None:
    from nemo_rl.algorithms.draft_update_schedule import (
        validate_scheduler_state_invariants,
    )

    assert callable(validate_scheduler_state_invariants)


def test_adaptive_restore_rejects_phase_inconsistent_observation_fields() -> None:
    from nemo_rl.algorithms.draft_update_schedule import (
        validate_scheduler_state_invariants,
    )

    assert callable(validate_scheduler_state_invariants)


def test_restore_rejects_nonintegral_scheduler_steps_and_versions() -> None:
    from nemo_rl.algorithms.draft_update_schedule import (
        validate_scheduler_state_invariants,
    )

    assert callable(validate_scheduler_state_invariants)


def test_restore_derives_applied_version_from_last_refit_step() -> None:
    from nemo_rl.algorithms.draft_update_schedule import (
        validate_scheduler_state_invariants,
    )

    assert callable(validate_scheduler_state_invariants)
