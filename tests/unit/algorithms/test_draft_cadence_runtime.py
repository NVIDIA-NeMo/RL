# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_rl.algorithms.draft_cadence_runtime import (
    CadenceRuntimeConfig,
    CadenceRuntimeWriter,
    CadenceTerminalEvidence,
    initialize_cadence_scheduler,
    initialize_or_recover_cadence_resume,
    load_checkpoint_bundle,
    preflight_cadence_receipt_capability,
    produce_cadence_decision,
    require_cadence_step_receipts,
    write_draft_apply_identity,
)
from nemo_rl.algorithms.draft_update_schedule import (
    DraftDecisionLedger,
    DraftUpdateScheduler,
    FileDraftStepTransactionStore,
    decision_outcome_payload,
)
from nemo_rl.models.policy.draft_config import (
    AdaptiveDraftUpdateScheduleConfig,
    AlwaysDraftUpdateScheduleConfig,
    DFlashDraftConfig,
    DraftUpdateScheduleConfig,
    FixedDraftUpdateScheduleConfig,
)


def _dflash_config(
    *, enabled: bool, schedule: DraftUpdateScheduleConfig | None = None
) -> DFlashDraftConfig:
    return DFlashDraftConfig(
        enabled=enabled,
        gamma=5,
        anchors_per_sample=1,
        mask_token_id=0,
        target_hidden_state_layer_ids=[1],
        update_schedule=schedule or AlwaysDraftUpdateScheduleConfig(),
    )


def test_controller_scheduler_produces_one_immutable_always_decision() -> None:
    scheduler = initialize_cadence_scheduler(
        _dflash_config(enabled=True),
        None,
        origin_step=7,
        resuming_from_checkpoint=False,
    )

    decision = produce_cadence_decision(scheduler, global_step=8)

    assert decision is not None
    assert decision.global_step == 8
    assert decision.decision_id == 1
    assert decision.update_requested is True
    assert decision.draft_refit_requested is True
    assert decision.applied_draft_version == 0


def test_disabled_or_fixed_drafter_control_has_no_cadence_decision() -> None:
    disabled = initialize_cadence_scheduler(
        _dflash_config(enabled=False),
        None,
        origin_step=0,
        resuming_from_checkpoint=False,
    )

    assert disabled is None
    assert produce_cadence_decision(disabled, global_step=1) is None
    assert produce_cadence_decision(None, global_step=1) is None


def test_legacy_resume_compatibility_is_limited_to_always_schedule() -> None:
    always = initialize_cadence_scheduler(
        _dflash_config(enabled=True),
        None,
        origin_step=4,
        resuming_from_checkpoint=True,
    )
    assert always is not None
    assert always.state.schedule_origin_step == 4

    with pytest.raises(ValueError, match="legacy checkpoint.*always"):
        initialize_cadence_scheduler(
            _dflash_config(
                enabled=True,
                schedule=FixedDraftUpdateScheduleConfig(
                    mode="fixed", action="sparse_update", fixed_interval=10
                ),
            ),
            None,
            origin_step=4,
            resuming_from_checkpoint=True,
        )


def test_adaptive_schedule_initializes_for_task7_selected_rollout_provenance() -> None:
    scheduler = initialize_cadence_scheduler(
        _dflash_config(enabled=True, schedule=AdaptiveDraftUpdateScheduleConfig()),
        None,
        origin_step=0,
        resuming_from_checkpoint=False,
    )

    assert scheduler is not None
    assert scheduler.config.mode == "adaptive"


def test_selected_draft_identity_binds_world_roots_exclusively(
    tmp_path: Path,
) -> None:
    scheduler = DraftUpdateScheduler.create(
        AlwaysDraftUpdateScheduleConfig(), origin_step=0
    )
    decision = scheduler.decide(global_step=1, acceptance=None)
    receipt = {
        "successful": True,
        "decision_id": 1,
        "global_step": 1,
        "draft_model_sha256": "a" * 64,
        "draft_optimizer_sha256": "b" * 64,
    }

    request = write_draft_apply_identity(tmp_path, decision, receipt)

    assert request.version == 1
    assert request.receipt()["sha256"] == request.sha256
    assert (
        b"nemo-rl-draft-apply-identity-v1" in Path(request.snapshot_path).read_bytes()
    )
    retry_request = write_draft_apply_identity(tmp_path, decision, receipt)
    assert retry_request.snapshot_path != request.snapshot_path


def test_missing_task4_or_task5_receipt_fails_before_cadence_apply() -> None:
    scheduler = DraftUpdateScheduler.create(
        AlwaysDraftUpdateScheduleConfig(), origin_step=0
    )
    decision = scheduler.decide(global_step=1, acceptance=None)

    with pytest.raises(RuntimeError, match="update receipt"):
        require_cadence_step_receipts(decision, worker_receipt=None, apply_receipt=None)
    with pytest.raises(RuntimeError, match="apply receipt"):
        require_cadence_step_receipts(
            decision,
            worker_receipt={
                "successful": True,
                "decision_id": 1,
                "global_step": 1,
            },
            apply_receipt=None,
        )


@pytest.mark.parametrize(
    ("update_receipts_supported", "apply_receipts_supported", "message"),
    [
        (False, False, "update receipt capability"),
        (True, False, "apply receipt capability"),
    ],
)
def test_receipt_capability_preflight_does_not_mutate_scheduler(
    update_receipts_supported: bool,
    apply_receipts_supported: bool,
    message: str,
) -> None:
    scheduler = DraftUpdateScheduler.create(
        AlwaysDraftUpdateScheduleConfig(), origin_step=0
    )
    before = scheduler.state_dict()

    with pytest.raises(RuntimeError, match=message):
        preflight_cadence_receipt_capability(
            scheduler,
            update_receipts_supported=update_receipts_supported,
            apply_receipts_supported=apply_receipts_supported,
        )

    assert scheduler.state_dict() == before
    assert scheduler.decide(global_step=1, acceptance=None).decision_id == 1


def test_legacy_always_resume_initializes_before_ledger_receipt_open(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "legacy-checkpoint" / "step_4"
    checkpoint.mkdir(parents=True)
    result_root = tmp_path / "cadence"
    ledger = DraftDecisionLedger(result_root / "legacy-live.jsonl")
    store = FileDraftStepTransactionStore(result_root, base_checkpoint_id="step_4")
    save_state = SimpleNamespace(
        draft_update_schedule=None,
        applied_draft_snapshot=None,
        draft_terminal_evidence=None,
        draft_decision_ledger_prefixes=[],
    )

    resumed = initialize_or_recover_cadence_resume(
        _dflash_config(enabled=True),
        saved=None,
        origin_step=4,
        checkpoint_path=checkpoint,
        result_root=result_root,
        transaction_store=store,
        decision_ledger=ledger,
        save_state=save_state,
    )

    assert resumed.scheduler is not None
    assert resumed.scheduler.state.schedule_origin_step == 4
    assert resumed.ledger is ledger
    assert resumed.quarantine_receipt_path is None
    assert save_state.draft_update_schedule == resumed.scheduler.state_dict()
    assert store.pending_intents() == ()


def test_saved_schedule_resume_requires_cadence_receipt(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint" / "step_4"
    checkpoint.mkdir(parents=True)
    result_root = tmp_path / "cadence"
    ledger = DraftDecisionLedger(result_root / "live.jsonl")
    store = FileDraftStepTransactionStore(result_root, base_checkpoint_id="step_4")
    scheduler = DraftUpdateScheduler.create(
        AlwaysDraftUpdateScheduleConfig(), origin_step=0
    )
    for step in range(1, 5):
        decision = scheduler.decide(global_step=step, acceptance=None)
        scheduler.record_outcome(
            decision,
            update_attempted=True,
            update_successful=True,
            draft_refit_attempted=True,
            draft_refit_successful=True,
        )
    save_state = SimpleNamespace(
        draft_update_schedule=scheduler.state_dict(),
        applied_draft_snapshot=None,
        draft_terminal_evidence=None,
        draft_decision_ledger_prefixes=[],
    )

    with pytest.raises(ValueError, match="saved cadence state requires.*receipt"):
        initialize_or_recover_cadence_resume(
            _dflash_config(enabled=True),
            saved=scheduler.state_dict(),
            origin_step=4,
            checkpoint_path=checkpoint,
            result_root=result_root,
            transaction_store=store,
            decision_ledger=ledger,
            save_state=save_state,
        )

    assert ledger.next_decision_id == 1
    assert store.pending_intents() == ()


def test_checkpoint_receipt_binds_all_training_components(tmp_path: Path) -> None:
    """Changing a checkpoint member must make it unusable as a resume authority."""
    root = tmp_path / "cadence"
    checkpoint = root / "checkpoints" / "step_1"
    model = checkpoint / "policy" / "weights"
    optimizer = checkpoint / "policy" / "optimizer"
    dataloader = checkpoint / "train_dataloader.pt"
    snapshot = root / "applied-draft-v1.safetensors"
    for path, contents in (
        (model, b"model"),
        (optimizer, b"optimizer"),
        (dataloader, b"rng"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    snapshot.write_bytes(b"draft")

    scheduler = DraftUpdateScheduler.create(
        AlwaysDraftUpdateScheduleConfig(), origin_step=0
    )
    decision = scheduler.decide(global_step=1, acceptance=None)
    scheduler.record_outcome(
        decision,
        update_attempted=True,
        update_successful=True,
        draft_refit_attempted=True,
        draft_refit_successful=True,
    )
    ledger = DraftDecisionLedger(root / "draft-decision-ledger-after-step_0.jsonl")
    ledger.append_closed(
        decision,
        decision_outcome_payload(
            decision,
            update_attempted=True,
            update_successful=True,
            draft_refit_attempted=True,
            draft_refit_successful=True,
        ),
    )
    state = SimpleNamespace(
        draft_update_schedule=scheduler.state_dict(),
        applied_draft_snapshot={
            "version": 1,
            "path": str(snapshot.resolve()),
            "size_bytes": len(b"draft"),
            "sha256": hashlib.sha256(b"draft").hexdigest(),
        },
        draft_terminal_evidence=None,
        draft_decision_ledger_prefixes=[],
    )
    writer = CadenceRuntimeWriter(
        CadenceRuntimeConfig(enabled=True, result_dir=str(root))
    )

    writer.checkpoint_closed(
        current_step=1,
        checkpoint_path=checkpoint,
        save_state=state,
        component_paths={
            "model": model,
            "optimizer": optimizer,
            "dataloader_rng": dataloader,
        },
        decision_ledger=ledger,
        terminal_evidence=CadenceTerminalEvidence({}, {}),
    )

    bundle = load_checkpoint_bundle(checkpoint)
    assert bundle["checkpoint_id"] == "step_1"
    model.write_bytes(b"corrupt")
    try:
        load_checkpoint_bundle(checkpoint)
    except ValueError as error:
        assert "model checkpoint digest" in str(error)
    else:  # pragma: no cover - assertion failure produces the useful test error
        raise AssertionError("corrupted checkpoint must not be accepted")
