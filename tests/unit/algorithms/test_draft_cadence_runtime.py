# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import hashlib
from pathlib import Path
from types import SimpleNamespace

from nemo_rl.algorithms.draft_cadence_runtime import (
    CadenceRuntimeConfig,
    CadenceRuntimeWriter,
    CadenceTerminalEvidence,
    load_checkpoint_bundle,
)
from nemo_rl.algorithms.draft_update_schedule import (
    DraftDecisionLedger,
    DraftUpdateScheduler,
    decision_outcome_payload,
)
from nemo_rl.models.policy.draft_config import AlwaysDraftUpdateScheduleConfig


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
