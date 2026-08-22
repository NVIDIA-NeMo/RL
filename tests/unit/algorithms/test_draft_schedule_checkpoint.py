# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nemo_rl.algorithms import draft_cadence_runtime as runtime_module
from nemo_rl.algorithms import draft_update_schedule as schedule_module
from nemo_rl.algorithms.draft_update_schedule import (
    AppliedDraftSnapshot,
    DraftDecisionLedger,
    DraftStepTransaction,
    DraftUpdateDecision,
    DraftUpdateScheduler,
    FileDraftStepTransactionStore,
    close_draft_step_transaction,
    decision_outcome_payload,
    durably_install_startup_snapshot,
    validate_applied_draft_snapshot,
    validate_scheduler_state_invariants,
)
from nemo_rl.algorithms.draft_cadence_runtime import (
    CadenceRuntimeConfig,
    CadenceRuntimeWriter,
    CadenceTerminalEvidence,
    build_terminal_schedule_payload,
    disabled_draft_schedule_payload,
    load_checkpoint_bundle,
    open_resume_decision_ledger,
    record_terminal_post_refit_observation,
    recover_draft_step_transactions,
    scheduler_decision_high_water,
)
from nemo_rl.algorithms.grpo import (
    restore_draft_update_scheduler,
    restore_serving_draft_after_startup_sync,
)
from nemo_rl.models.policy.draft_config import (
    AdaptiveDraftUpdateScheduleConfig,
    AlwaysDraftUpdateScheduleConfig,
    FixedDraftUpdateScheduleConfig,
)


class CadenceTestRun:
    """Small real-file cadence run used to exercise crash and resume behavior."""

    def __init__(
        self,
        root: Path,
        *,
        store: FileDraftStepTransactionStore | None = None,
        origin_step: int = 0,
    ) -> None:
        self.root = root.resolve()
        self.writer = CadenceRuntimeWriter(
            CadenceRuntimeConfig(enabled=True, result_dir=str(self.root))
        )
        self.config = AlwaysDraftUpdateScheduleConfig()
        self.scheduler = DraftUpdateScheduler.create(
            self.config, origin_step=origin_step
        )
        self.ledger = DraftDecisionLedger(
            self.root / f"draft-decision-ledger-after-step_{origin_step}.jsonl"
        )
        self.transaction_store = store or FileDraftStepTransactionStore(
            self.root, base_checkpoint_id=f"step_{origin_step}"
        )
        self.terminal_evidence = CadenceTerminalEvidence({}, {})
        snapshot_path = self.root / "applied-draft-v0.safetensors"
        snapshot_path.write_bytes(b"serving-draft-version-0")
        raw = snapshot_path.read_bytes()
        self.save_state = SimpleNamespace(
            draft_update_schedule=self.scheduler.state_dict(),
            applied_draft_snapshot=asdict(
                AppliedDraftSnapshot(
                    version=0,
                    path=str(snapshot_path.resolve()),
                    size_bytes=len(raw),
                    sha256=hashlib.sha256(raw).hexdigest(),
                )
            ),
            draft_terminal_evidence=self.terminal_evidence.state_dict(),
            draft_decision_ledger_prefixes=[],
        )
        self.last_full_checkpoint_id = f"step_{origin_step}"
        self.resumed_from: str | None = None

    def _rows(self) -> list[dict[str, object]]:
        paths = [Path(item.path) for item in self.ledger.sealed_prefixes]
        if self.ledger.path.is_file():
            paths.append(self.ledger.path)
        return [
            json.loads(line) for path in paths for line in path.read_text().splitlines()
        ]

    def entry(self, decision_id: int) -> dict[str, object]:
        matches = [row for row in self._rows() if row["decision_id"] == decision_id]
        assert len(matches) == 1
        return matches[0]

    def entries_for_decision(self, decision_id: int) -> int:
        return sum(row["decision_id"] == decision_id for row in self._rows())

    def entries_after(self, decision_id: int) -> list[dict[str, object]]:
        return [row for row in self._rows() if int(row["decision_id"]) > decision_id]

    def decision_for_step(self, step: int) -> DraftUpdateDecision:
        return self.scheduler.decide(global_step=step, acceptance=None)

    def success_outcome(self, decision: DraftUpdateDecision) -> dict[str, bool]:
        return decision_outcome_payload(
            decision,
            update_attempted=decision.update_requested,
            update_successful=decision.update_requested,
            draft_refit_attempted=decision.draft_refit_requested,
            draft_refit_successful=decision.draft_refit_requested,
        )

    def open_step_intent(self, step: int) -> DraftStepTransaction:
        pre_scheduler_state = self.scheduler.state_dict()
        decision = self.decision_for_step(step)
        snapshot_path = self.root / f"applied-draft-v{decision.decision_id}.safetensors"
        return self.transaction_store.begin(
            decision,
            pre_scheduler_state=pre_scheduler_state,
            expected_snapshot_path=snapshot_path,
        )

    def persist_durable_apply_receipt(
        self, transaction: DraftStepTransaction
    ) -> AppliedDraftSnapshot:
        path = Path(str(transaction.expected_snapshot_path))
        path.write_bytes(
            f"serving-draft-version-{transaction.decision.decision_id}".encode()
        )
        raw = path.read_bytes()
        snapshot = AppliedDraftSnapshot(
            version=transaction.decision.decision_id,
            path=str(path.resolve()),
            size_bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
        )
        self.transaction_store.write_durable_apply_receipt(
            transaction, snapshot=snapshot
        )
        return snapshot

    def execute_step(self, step: int, *, acceptance: float | None = None) -> None:
        pre_scheduler_state = self.scheduler.state_dict()
        decision = self.scheduler.decide(global_step=step, acceptance=acceptance)
        observation = (
            0.5
            if acceptance is None
            and self.scheduler.state.last_applied_refit_step == step - 1
            else acceptance
        )
        self.terminal_evidence = record_terminal_post_refit_observation(
            self.terminal_evidence,
            decision=decision,
            last_applied_refit_step=self.scheduler.state.last_applied_refit_step,
            acceptance_rate=observation,
        )
        self.save_state.draft_terminal_evidence = self.terminal_evidence.state_dict()
        snapshot_path = self.root / f"applied-draft-v{decision.decision_id}.safetensors"
        transaction = self.transaction_store.begin(
            decision,
            pre_scheduler_state=pre_scheduler_state,
            expected_snapshot_path=snapshot_path,
        )
        snapshot = self.persist_durable_apply_receipt(transaction)
        self.terminal_evidence = self.writer.successful_update_closed(
            decision=decision,
            worker_receipt={
                "successful": True,
                "decision_id": decision.decision_id,
                "global_step": decision.global_step,
                "draft_model_sha256": hashlib.sha256(
                    f"model-{decision.decision_id}".encode()
                ).hexdigest(),
                "draft_optimizer_sha256": hashlib.sha256(
                    f"optimizer-{decision.decision_id}".encode()
                ).hexdigest(),
            },
            evidence=self.terminal_evidence,
            save_state=self.save_state,
        )
        outcome = self.success_outcome(decision)
        outcome_error = close_draft_step_transaction(
            transaction,
            decision=decision,
            outcome=outcome,
            applied_snapshot=snapshot,
            scheduler=self.scheduler,
            decision_ledger=self.ledger,
            save_state=self.save_state,
            transaction_store=self.transaction_store,
        )
        if outcome_error is not None:
            raise outcome_error

    def checkpoint(
        self,
        step: int,
        *,
        model: bool = True,
        optimizer: bool = True,
        dataloader: bool = True,
    ) -> Path:
        checkpoint = self.root / "checkpoints" / f"step_{step}"
        checkpoint.mkdir(parents=True, exist_ok=False)
        components: dict[str, Path] = {}
        for name, enabled in (
            ("model", model),
            ("optimizer", optimizer),
            ("dataloader_rng", dataloader),
        ):
            if enabled:
                path = checkpoint / f"{name}.bin"
                path.write_bytes(f"{name}-step-{step}".encode())
                components[name] = path
        if not optimizer:
            raise RuntimeError("optimizer checkpoint failed")
        self.ledger = self.writer.checkpoint_closed(
            current_step=step,
            checkpoint_path=checkpoint,
            save_state=self.save_state,
            component_paths=components,
            decision_ledger=self.ledger,
            terminal_evidence=self.terminal_evidence,
            resumed_from=self.resumed_from,
        )
        self.last_full_checkpoint_id = f"step_{step}"
        self.transaction_store.base_checkpoint_id = self.last_full_checkpoint_id
        return checkpoint

    def restart_from_checkpoint(self, step: int) -> "CadenceTestRun":
        checkpoint = self.root / "checkpoints" / f"step_{step}"
        opened = open_resume_decision_ledger(checkpoint, self.root)
        save_state = SimpleNamespace(
            draft_update_schedule=None,
            applied_draft_snapshot=None,
            draft_terminal_evidence=None,
            draft_decision_ledger_prefixes=[],
        )
        store = FileDraftStepTransactionStore(
            self.root, base_checkpoint_id=f"step_{step}"
        )
        scheduler = recover_draft_step_transactions(
            config=self.config,
            checkpoint_path=checkpoint,
            transaction_store=store,
            decision_ledger=opened.ledger,
            save_state=save_state,
        )
        assert scheduler is not None
        resumed = object.__new__(CadenceTestRun)
        resumed.root = self.root
        resumed.writer = CadenceRuntimeWriter(
            CadenceRuntimeConfig(enabled=True, result_dir=str(self.root))
        )
        resumed.config = self.config
        resumed.scheduler = scheduler
        resumed.ledger = opened.ledger
        resumed.transaction_store = store
        resumed.terminal_evidence = CadenceTerminalEvidence.from_state(
            save_state.draft_terminal_evidence
        )
        resumed.save_state = save_state
        resumed.last_full_checkpoint_id = f"step_{step}"
        resumed.resumed_from = str(checkpoint.resolve())
        resumed.ledger_quarantine_receipt = opened.quarantine_receipt_path
        return resumed


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
    pre_scheduler_state = scheduler.state_dict()
    decision = scheduler.decide(global_step=1, acceptance=None)
    snapshot_path = tmp_path / "applied-draft-v1.safetensors"
    snapshot_path.write_bytes(b"draft")
    store = FileDraftStepTransactionStore(tmp_path)
    transaction = store.begin(
        decision,
        pre_scheduler_state=pre_scheduler_state,
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
    config, saved = valid_saved_state()
    bad_version = copy.deepcopy(saved)
    bad_version["state_version"] = 2
    with pytest.raises(ValueError, match="outer|state version"):
        DraftUpdateScheduler.create(config, origin_step=0, restored=bad_version)
    bad_config = copy.deepcopy(saved)
    bad_config["config"] = {"mode": "always", "unexpected": True}
    with pytest.raises(ValueError, match="resolved draft update schedule"):
        DraftUpdateScheduler.create(config, origin_step=0, restored=bad_config)


def test_resume_restores_exact_applied_draft_snapshot_before_publication(
    tmp_path: Path,
) -> None:
    scheduler = DraftUpdateScheduler.create(
        FixedDraftUpdateScheduleConfig(
            mode="fixed", action="refit_only", fixed_interval=10
        ),
        origin_step=0,
    )
    for step in range(1, 18):
        decision = scheduler.decide(global_step=step, acceptance=None)
        scheduler.record_outcome(
            decision,
            update_attempted=True,
            update_successful=True,
            draft_refit_attempted=decision.draft_refit_requested,
            draft_refit_successful=decision.draft_refit_requested,
        )
    applied_bytes = b"draft-bytes-applied-at-step-10"
    snapshot_path = tmp_path / "applied-draft-v10.safetensors"
    snapshot_path.write_bytes(applied_bytes)
    snapshot = AppliedDraftSnapshot(
        version=10,
        path=str(snapshot_path),
        size_bytes=len(applied_bytes),
        sha256=hashlib.sha256(applied_bytes).hexdigest(),
    )
    rollout_manager = MagicMock()
    synchronizer = MagicMock()
    synchronizer.sync_applied_draft_snapshot.return_value = {
        "successful": True,
        "version": 10,
        "sha256": snapshot.sha256,
    }
    lifecycle: list[str] = []
    rollout_manager.set_applied_draft_version.side_effect = lambda _version: (
        lifecycle.append("published")
    )
    rollout_manager.enable_reservations.side_effect = lambda: lifecycle.append(
        "reservations"
    )
    save_state = SimpleNamespace(applied_draft_snapshot=None)

    def flush_save_state(state: SimpleNamespace) -> dict[str, object]:
        lifecycle.append("durable")
        installed = AppliedDraftSnapshot(**state.applied_draft_snapshot)
        return {
            "successful": True,
            "version": installed.version,
            "sha256": installed.sha256,
        }

    def install_snapshot(installed: AppliedDraftSnapshot) -> dict[str, object]:
        assert installed == snapshot
        return dict(
            durably_install_startup_snapshot(
                save_state, installed, flush_save_state=flush_save_state
            )
        )

    restore_serving_draft_after_startup_sync(
        scheduler.config,
        scheduler,
        rollout_manager,
        synchronizer,
        snapshot=snapshot,
        snapshot_path=None,
        resuming_from_checkpoint=True,
        install_snapshot=install_snapshot,
    )
    synchronizer.sync_target_from_current_checkpoint.assert_called_once_with()
    synchronizer.sync_applied_draft_snapshot.assert_called_once_with(snapshot)
    synchronizer.sync_current_trainable_draft.assert_not_called()
    assert lifecycle == ["durable", "published", "reservations"]
    assert save_state.applied_draft_snapshot == asdict(snapshot)


@pytest.mark.parametrize("mismatch", ["version", "bytes"])
def test_resume_rejects_snapshot_version_or_bytes_mismatch(
    tmp_path: Path, mismatch: str
) -> None:
    config, saved = valid_saved_state()
    scheduler = DraftUpdateScheduler.create(config, origin_step=0, restored=saved)
    path = tmp_path / "draft.safetensors"
    path.write_bytes(b"right")
    snapshot = AppliedDraftSnapshot(
        version=0 if mismatch == "version" else 1,
        path=str(path),
        size_bytes=5,
        sha256=hashlib.sha256(
            b"wrong" if mismatch == "bytes" else b"right"
        ).hexdigest(),
    )
    with pytest.raises(ValueError, match="snapshot.*version|digest"):
        validate_applied_draft_snapshot(scheduler, snapshot)


def test_resumed_refit_only_without_applied_snapshot_fails_before_sync() -> None:
    config = FixedDraftUpdateScheduleConfig(
        mode="fixed", action="refit_only", fixed_interval=10
    )
    scheduler = DraftUpdateScheduler.create(config, origin_step=0)
    synchronizer = MagicMock()
    with pytest.raises(ValueError, match="resumed.*snapshot"):
        restore_serving_draft_after_startup_sync(
            config,
            scheduler,
            MagicMock(),
            synchronizer,
            snapshot=None,
            snapshot_path=None,
            resuming_from_checkpoint=True,
            install_snapshot=MagicMock(),
        )
    synchronizer.sync_target_from_current_checkpoint.assert_not_called()
    synchronizer.sync_current_trainable_draft.assert_not_called()


def test_startup_apply_must_succeed_before_persistence_or_reservations(
    tmp_path: Path,
) -> None:
    config = AlwaysDraftUpdateScheduleConfig()
    scheduler = DraftUpdateScheduler.create(config, origin_step=0)
    path = tmp_path / "applied-draft-v0.safetensors"
    raw = b"initial"
    path.write_bytes(raw)
    synchronizer = MagicMock()
    synchronizer.sync_current_trainable_draft.return_value = {
        "successful": False,
        "version": 0,
        "snapshot_path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    rollout_manager = MagicMock()
    installer = MagicMock()
    with pytest.raises(RuntimeError, match="initial serving-draft snapshot"):
        restore_serving_draft_after_startup_sync(
            config,
            scheduler,
            rollout_manager,
            synchronizer,
            snapshot=None,
            snapshot_path=path,
            resuming_from_checkpoint=False,
            install_snapshot=installer,
        )
    installer.assert_not_called()
    rollout_manager.set_applied_draft_version.assert_not_called()
    rollout_manager.enable_reservations.assert_not_called()


def test_pre_first_refit_resume_restores_immutable_version_zero_snapshot(
    tmp_path: Path,
) -> None:
    config = FixedDraftUpdateScheduleConfig(
        mode="fixed", action="refit_only", fixed_interval=10
    )
    scheduler = DraftUpdateScheduler.create(config, origin_step=0)
    path = tmp_path / "applied-draft-v0.safetensors"
    raw = b"immutable-initial-serving-draft"
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    first_sync = MagicMock()
    first_sync.sync_current_trainable_draft.return_value = {
        "successful": True,
        "version": 0,
        "snapshot_path": str(path.resolve()),
        "sha256": digest,
    }
    snapshot = restore_serving_draft_after_startup_sync(
        config,
        scheduler,
        MagicMock(),
        first_sync,
        snapshot=None,
        snapshot_path=path,
        resuming_from_checkpoint=False,
        install_snapshot=lambda installed: {
            "successful": True,
            "version": installed.version,
            "sha256": installed.sha256,
        },
    )
    for step in range(1, 6):
        decision = scheduler.decide(global_step=step, acceptance=None)
        scheduler.record_outcome(
            decision,
            update_attempted=True,
            update_successful=True,
            draft_refit_attempted=False,
            draft_refit_successful=False,
        )
    restored = DraftUpdateScheduler.create(
        config, origin_step=0, restored=scheduler.state_dict()
    )
    resume_sync = MagicMock()
    resume_sync.sync_applied_draft_snapshot.return_value = {
        "successful": True,
        "version": 0,
        "sha256": digest,
    }
    restore_serving_draft_after_startup_sync(
        config,
        restored,
        MagicMock(),
        resume_sync,
        snapshot=snapshot,
        snapshot_path=None,
        resuming_from_checkpoint=True,
        install_snapshot=lambda installed: {
            "successful": True,
            "version": installed.version,
            "sha256": installed.sha256,
        },
    )
    resume_sync.sync_applied_draft_snapshot.assert_called_once_with(snapshot)
    resume_sync.sync_current_trainable_draft.assert_not_called()


@pytest.mark.parametrize(
    "crash_after", ["intent", "outcome", "resolution", "bundle", "commit_marker"]
)
def test_draft_step_transaction_recovers_matching_scheduler_snapshot_and_ledger(
    tmp_path: Path, crash_after: str
) -> None:
    run = CadenceTestRun(tmp_path)
    run.execute_step(1)
    run.checkpoint(1)
    run.transaction_store.crash_after = crash_after
    with pytest.raises(RuntimeError, match="injected crash"):
        run.execute_step(2)
    restored = run.restart_from_checkpoint(1)
    bundle = load_checkpoint_bundle(tmp_path / "checkpoints" / "step_1")
    assert bundle["checkpoint_id"] == "step_1"
    assert set(bundle["components"]) == {"model", "optimizer", "dataloader_rng"}
    assert bundle["draft_update_schedule"]["state"]["applied_draft_version"] == 1
    assert bundle["applied_draft_snapshot"]["version"] == 1
    assert bundle["ledger_high_water"] == 1
    assert restored.entries_after(1) == []
    assert restored.scheduler.state.next_decision_id == 2
    assert restored.transaction_store.pending_intents() == ()


@pytest.mark.parametrize("durable_apply_receipt", [False, True])
def test_crash_after_intent_resolves_from_durable_transfer_receipt_then_truncates(
    tmp_path: Path, durable_apply_receipt: bool
) -> None:
    run = CadenceTestRun(tmp_path)
    run.execute_step(1)
    run.checkpoint(1)
    transaction = run.open_step_intent(2)
    if durable_apply_receipt:
        run.persist_durable_apply_receipt(transaction)
    restored = run.restart_from_checkpoint(1)
    resolution_paths = list(
        run.transaction_store.quarantine_root.glob(
            f"resume-*/{transaction.transaction_id}.resolution.json"
        )
    )
    assert len(resolution_paths) == 1
    resolution = json.loads(resolution_paths[0].read_text())
    assert resolution["outcome"]["draft_refit_successful"] is durable_apply_receipt
    assert restored.scheduler.state.next_decision_id == 2
    assert restored.entries_after(1) == []


def test_every_transfer_exception_closes_and_persists_exactly_one_outcome(
    tmp_path: Path,
) -> None:
    run = CadenceTestRun(tmp_path)
    pre_scheduler_state = run.scheduler.state_dict()
    decision = run.scheduler.decide(global_step=1, acceptance=None)
    transaction = run.transaction_store.begin(
        decision, pre_scheduler_state=pre_scheduler_state
    )
    outcome = decision_outcome_payload(
        decision,
        update_attempted=True,
        update_successful=False,
        draft_refit_attempted=True,
        draft_refit_successful=False,
    )
    error = close_draft_step_transaction(
        transaction,
        decision=decision,
        outcome=outcome,
        applied_snapshot=None,
        scheduler=run.scheduler,
        decision_ledger=run.ledger,
        save_state=run.save_state,
        transaction_store=run.transaction_store,
    )
    assert isinstance(error, RuntimeError)
    assert run.entries_for_decision(1) == 1
    row = run.entry(1)
    assert row["outcome"] == outcome


def test_cadence_advances_on_resume_only_after_full_training_checkpoint(
    tmp_path: Path,
) -> None:
    run = CadenceTestRun(tmp_path)
    run.execute_step(1)
    run.checkpoint(1)
    run.execute_step(2)
    with pytest.raises(RuntimeError, match="optimizer checkpoint failed"):
        run.checkpoint(2, optimizer=False)
    restored = run.restart_from_checkpoint(1)
    assert restored.scheduler.state.next_decision_id == 2
    assert restored.scheduler.state.applied_draft_version == 1


@pytest.mark.parametrize("component", ["model", "optimizer", "dataloader_rng"])
def test_checkpoint_bundle_rehashes_every_training_component(
    tmp_path: Path, component: str
) -> None:
    run = CadenceTestRun(tmp_path)
    run.execute_step(1)
    checkpoint = run.checkpoint(1)
    receipt = json.loads((checkpoint / "cadence-checkpoint-receipt.json").read_text())
    artifact = checkpoint / receipt["components"][component]["relative_path"]
    artifact.write_bytes(artifact.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match=f"{component} checkpoint digest"):
        load_checkpoint_bundle(checkpoint)


def test_checkpoint_bundle_rehashes_ledger_scheduler_and_tree(tmp_path: Path) -> None:
    run = CadenceTestRun(tmp_path)
    run.execute_step(1)
    checkpoint = run.checkpoint(1)
    receipt_path = checkpoint / "cadence-checkpoint-receipt.json"
    original = receipt_path.read_bytes()
    receipt = json.loads(original)
    ledger = checkpoint / receipt["decision_ledger"]["relative_path"]
    ledger.write_bytes(ledger.read_bytes() + b"{}\n")
    with pytest.raises(ValueError, match="decision-ledger receipt"):
        load_checkpoint_bundle(checkpoint)
    ledger.write_bytes(ledger.read_bytes()[:-3])
    (checkpoint / "unexpected.bin").write_bytes(b"changes-tree")
    with pytest.raises(ValueError, match="tree digest"):
        load_checkpoint_bundle(checkpoint)
    (checkpoint / "unexpected.bin").unlink()
    receipt = json.loads(original)
    receipt["draft_update_schedule"]["state"]["next_decision_id"] = 3
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="scheduler/ledger high-water"):
        load_checkpoint_bundle(checkpoint)


def test_checkpoint_validates_applied_snapshot_before_sealing_ledger(
    tmp_path: Path,
) -> None:
    run = CadenceTestRun(tmp_path)
    run.execute_step(1)
    snapshot = run.save_state.applied_draft_snapshot
    assert isinstance(snapshot, dict)
    snapshot["sha256"] = "0" * 64
    checkpoint = tmp_path / "checkpoints" / "step_1"
    checkpoint.mkdir(parents=True)
    components: dict[str, Path] = {}
    for name in ("model", "optimizer", "dataloader_rng"):
        path = checkpoint / f"{name}.bin"
        path.write_bytes(name.encode())
        components[name] = path

    with pytest.raises(ValueError, match="snapshot version or digest"):
        run.writer.checkpoint_closed(
            current_step=1,
            checkpoint_path=checkpoint,
            save_state=run.save_state,
            component_paths=components,
            decision_ledger=run.ledger,
            terminal_evidence=run.terminal_evidence,
        )
    decision = DraftUpdateDecision(
        global_step=2,
        decision_id=2,
        update_requested=True,
        draft_refit_requested=True,
        reason="always",
        observed_acceptance=None,
        applied_draft_version=1,
    )
    run.ledger.append_closed(decision, run.success_outcome(decision))
    assert run.ledger.next_decision_id == 3


def test_checkpoint_high_water_is_derived_from_real_scheduler_cursor() -> None:
    _config, saved = valid_saved_state()
    state = saved["state"]
    assert isinstance(state, dict)
    assert "decisions" not in state
    assert scheduler_decision_high_water(saved) == state["next_decision_id"] - 1


def test_disabled_fixed_control_checkpoint_has_explicit_empty_ledger(
    tmp_path: Path,
) -> None:
    writer = CadenceRuntimeWriter(
        CadenceRuntimeConfig(enabled=True, result_dir=str(tmp_path))
    )
    checkpoint = tmp_path / "checkpoints" / "step_100"
    checkpoint.mkdir(parents=True)
    components: dict[str, Path] = {}
    for name in ("model", "optimizer", "dataloader_rng"):
        path = checkpoint / f"{name}.bin"
        path.write_bytes(name.encode())
        components[name] = path
    save_state = SimpleNamespace(
        draft_update_schedule=disabled_draft_schedule_payload(),
        applied_draft_snapshot=None,
        draft_terminal_evidence=None,
        draft_decision_ledger_prefixes=[],
    )
    writer.checkpoint_closed(
        current_step=100,
        checkpoint_path=checkpoint,
        save_state=save_state,
        component_paths=components,
        decision_ledger=DraftDecisionLedger(tmp_path / "disabled-live.jsonl"),
        terminal_evidence=CadenceTerminalEvidence({}, {}),
    )
    bundle = load_checkpoint_bundle(checkpoint)
    assert bundle["draft_update_schedule"]["mode"] == "disabled"
    assert scheduler_decision_high_water(bundle["draft_update_schedule"]) == 0
    assert bundle["decision_ledger"] == {
        "relative_path": "draft-decision-ledger.jsonl",
        "size_bytes": 0,
        "sha256": hashlib.sha256(b"").hexdigest(),
        "first_decision_id": None,
        "last_decision_id": 0,
        "entry_count": 0,
    }
    assert (checkpoint / "draft-decision-ledger.jsonl").read_bytes() == b""


def test_step_100_checkpoint_installs_suffix_and_step_101_continues(
    tmp_path: Path,
) -> None:
    run = CadenceTestRun(tmp_path)
    for step in range(1, 101):
        run.execute_step(step)
    sealed = run.ledger
    run.checkpoint(100)
    assert run.ledger is not sealed
    assert run.ledger.next_decision_id == 101
    decision = DraftUpdateDecision(
        global_step=101,
        decision_id=101,
        update_requested=True,
        draft_refit_requested=True,
        reason="always",
        observed_acceptance=None,
        applied_draft_version=100,
    )
    with pytest.raises(RuntimeError, match="sealed"):
        sealed.append_closed(decision, run.success_outcome(decision))
    run.execute_step(101)
    assert run.entry(101)["decision_id"] == 101
    assert run.scheduler.state.next_decision_id == 102


def test_resume_from_step_100_opens_suffix_at_101(tmp_path: Path) -> None:
    run = CadenceTestRun(tmp_path)
    for step in range(1, 101):
        run.execute_step(step)
    run.checkpoint(100)
    resumed = run.restart_from_checkpoint(100)
    assert resumed.scheduler.state.next_decision_id == 101
    assert resumed.ledger.next_decision_id == 101
    resumed.execute_step(101)
    assert resumed.entry(101)["decision_id"] == 101
    assert resumed.scheduler.state.next_decision_id == 102


def test_resume_rejects_checkpoint_outside_result_root(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    run = CadenceTestRun(source_root)
    run.execute_step(1)
    checkpoint = run.checkpoint(1)

    with pytest.raises(ValueError, match="outside cadence result root"):
        open_resume_decision_ledger(checkpoint, tmp_path / "different-result")


def test_resume_quarantines_written_post_checkpoint_suffix_before_replaying_101(
    tmp_path: Path,
) -> None:
    run = CadenceTestRun(tmp_path)
    for step in range(1, 101):
        run.execute_step(step)
    run.checkpoint(100)
    stale_suffix = run.ledger.path
    run.execute_step(101)
    stale_bytes = stale_suffix.read_bytes()
    resumed = run.restart_from_checkpoint(100)
    assert not stale_suffix.exists()
    quarantine = json.loads(resumed.ledger_quarantine_receipt.read_text())
    assert quarantine["state"] == "resolved"
    assert quarantine["checkpoint_id"] == "step_100"
    artifact = next(
        item
        for item in quarantine["artifacts"]
        if item["original_path"] == str(stale_suffix.resolve())
    )
    assert artifact["size_bytes"] == len(stale_bytes)
    assert artifact["sha256"] == hashlib.sha256(stale_bytes).hexdigest()
    assert Path(artifact["quarantine_path"]).read_bytes() == stale_bytes
    assert resumed.ledger.next_decision_id == 101
    resumed.execute_step(101)
    assert resumed.entry(101)["decision_id"] == 101


def test_ledger_quarantine_fsyncs_recovery_parent_before_first_move(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run = CadenceTestRun(tmp_path)
    run.execute_step(1)
    run.checkpoint(1)
    run.execute_step(2)
    fsynced: list[Path] = []
    real_fsync = runtime_module._fsync_directory
    real_move = runtime_module._move_ledger_to_quarantine

    def record_fsync(path: Path) -> None:
        fsynced.append(path.resolve())
        real_fsync(path)

    def assert_parent_then_move(source: Path, destination: Path) -> None:
        assert (tmp_path / "recovery").resolve() in fsynced
        real_move(source, destination)

    monkeypatch.setattr(runtime_module, "_fsync_directory", record_fsync)
    monkeypatch.setattr(
        runtime_module, "_move_ledger_to_quarantine", assert_parent_then_move
    )

    resumed = run.restart_from_checkpoint(1)

    assert resumed.ledger.next_decision_id == 2


def test_transaction_quarantine_fsyncs_parent_before_first_move(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = FileDraftStepTransactionStore(tmp_path)
    scheduler = DraftUpdateScheduler.create(
        AlwaysDraftUpdateScheduleConfig(), origin_step=0
    )
    pre_scheduler_state = scheduler.state_dict()
    decision = scheduler.decide(global_step=1, acceptance=None)
    transaction = store.begin(decision, pre_scheduler_state=pre_scheduler_state)
    store.resolve(
        transaction,
        decision=decision,
        outcome=decision_outcome_payload(
            decision,
            update_attempted=True,
            update_successful=False,
            draft_refit_attempted=True,
            draft_refit_successful=False,
        ),
        applied_snapshot=None,
    )
    fsynced: list[Path] = []
    real_fsync = schedule_module._fsync_directory
    real_replace = schedule_module.os.replace

    def record_fsync(path: Path) -> None:
        fsynced.append(path.resolve())
        real_fsync(path)

    def assert_parent_then_move(source: Path, destination: Path) -> None:
        assert store.quarantine_root.resolve() in fsynced
        real_replace(source, destination)

    monkeypatch.setattr(schedule_module, "_fsync_directory", record_fsync)
    monkeypatch.setattr(schedule_module.os, "replace", assert_parent_then_move)

    store.discard_after_checkpoint(checkpoint_id="step_0", ledger_high_water=0)

    assert list(
        store.quarantine_root.glob("resume-*/transaction-quarantine-receipt.json")
    )


@pytest.mark.parametrize("crash_phase", ["after_intent", "before_receipt"])
def test_incomplete_quarantine_transaction_reconciles_after_crash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, crash_phase: str
) -> None:
    run = CadenceTestRun(tmp_path)
    run.execute_step(1)
    run.checkpoint(1)
    run.execute_step(2)
    real_move = runtime_module._move_ledger_to_quarantine
    real_write = runtime_module.write_json_exclusive_atomic
    if crash_phase == "after_intent":
        monkeypatch.setattr(
            runtime_module,
            "_move_ledger_to_quarantine",
            lambda *_args: (_ for _ in ()).throw(RuntimeError("after intent")),
        )
    else:

        def crash_before_receipt(path: Path, payload: object) -> None:
            if path.name == "ledger-quarantine-receipt.json":
                raise RuntimeError("before receipt")
            real_write(path, payload)

        monkeypatch.setattr(
            runtime_module, "write_json_exclusive_atomic", crash_before_receipt
        )
    with pytest.raises(RuntimeError, match="after intent|before receipt"):
        run.restart_from_checkpoint(1)
    monkeypatch.setattr(runtime_module, "_move_ledger_to_quarantine", real_move)
    monkeypatch.setattr(runtime_module, "write_json_exclusive_atomic", real_write)
    resumed = run.restart_from_checkpoint(1)
    receipt = json.loads(resumed.ledger_quarantine_receipt.read_text())
    assert receipt["state"] == "resolved"
    assert receipt["checkpoint_id"] == "step_1"
    resumed.execute_step(2)
    assert resumed.entry(2)["decision_id"] == 2


def test_successful_update_receipt_is_exclusive_and_installed_before_return(
    tmp_path: Path,
) -> None:
    writer = CadenceRuntimeWriter(
        CadenceRuntimeConfig(enabled=True, result_dir=str(tmp_path))
    )
    scheduler = DraftUpdateScheduler.create(
        AlwaysDraftUpdateScheduleConfig(), origin_step=0
    )
    decision = scheduler.decide(global_step=1, acceptance=None)
    evidence = CadenceTerminalEvidence({}, {})
    save_state = SimpleNamespace(draft_terminal_evidence=None)
    worker_receipt = {
        "successful": True,
        "decision_id": 1,
        "global_step": 1,
        "draft_model_sha256": "a" * 64,
        "draft_optimizer_sha256": "b" * 64,
    }
    updated = writer.successful_update_closed(
        decision=decision,
        worker_receipt=worker_receipt,
        evidence=evidence,
        save_state=save_state,
    )
    binding = updated.update_receipts_by_decision[1]
    raw = Path(str(binding["path"])).read_bytes()
    assert binding["size_bytes"] == len(raw)
    assert binding["sha256"] == hashlib.sha256(raw).hexdigest()
    assert save_state.draft_terminal_evidence == updated.state_dict()
    with pytest.raises(RuntimeError, match="duplicate"):
        writer.successful_update_closed(
            decision=decision,
            worker_receipt=worker_receipt,
            evidence=updated,
            save_state=save_state,
        )


def test_resume_can_replay_uncheckpointed_decision_without_receipt_collision(
    tmp_path: Path,
) -> None:
    config = CadenceRuntimeConfig(enabled=True, result_dir=str(tmp_path))
    scheduler = DraftUpdateScheduler.create(
        AlwaysDraftUpdateScheduleConfig(), origin_step=0
    )
    decision = scheduler.decide(global_step=1, acceptance=None)
    worker_receipt = {
        "successful": True,
        "decision_id": 1,
        "global_step": 1,
        "draft_model_sha256": "a" * 64,
        "draft_optimizer_sha256": "b" * 64,
    }
    first = CadenceRuntimeWriter(config).successful_update_closed(
        decision=decision,
        worker_receipt=worker_receipt,
        evidence=CadenceTerminalEvidence({}, {}),
        save_state=SimpleNamespace(draft_terminal_evidence=None),
    )
    stale_path = Path(str(first.update_receipts_by_decision[1]["path"]))
    replay = CadenceRuntimeWriter(config).successful_update_closed(
        decision=decision,
        worker_receipt=worker_receipt,
        evidence=CadenceTerminalEvidence({}, {}),
        save_state=SimpleNamespace(draft_terminal_evidence=None),
    )
    replay_path = Path(str(replay.update_receipts_by_decision[1]["path"]))
    assert stale_path.is_file() and replay_path.is_file()
    assert stale_path != replay_path


def test_terminal_payload_maps_decision_id_to_nonzero_origin_step(
    tmp_path: Path,
) -> None:
    run = CadenceTestRun(tmp_path, origin_step=7)
    run.execute_step(8, acceptance=0.70)
    checkpoint = run.checkpoint(8)
    schedule = build_terminal_schedule_payload(
        load_checkpoint_bundle(checkpoint), run.terminal_evidence
    )
    assert schedule["decision_ids"] == [1]
    assert schedule["global_steps"] == [8]
    assert schedule["updated_steps"] == [8]
    assert schedule["update_receipts"][0]["decision_id"] == 1
    assert schedule["refit_versions"] == [{"refit_step": 8, "applied_draft_version": 1}]


def test_resumed_terminal_payload_reports_only_post_boundary_observations(
    tmp_path: Path,
) -> None:
    run = CadenceTestRun(tmp_path)
    run.execute_step(1, acceptance=0.70)
    run.execute_step(2, acceptance=0.69)
    run.checkpoint(2)
    resumed = run.restart_from_checkpoint(2)
    resumed.execute_step(3, acceptance=0.68)
    checkpoint = resumed.checkpoint(3)
    assert set(resumed.terminal_evidence.observations_by_refit_step) == {1, 2}
    schedule = build_terminal_schedule_payload(
        load_checkpoint_bundle(checkpoint), resumed.terminal_evidence
    )
    assert schedule["post_event_observations"] == [
        {
            "refit_step": 2,
            "observation_step": 3,
            "applied_draft_version": 2,
            "acceptance_rate": 0.68,
        }
    ]


def valid_saved_state() -> tuple[AlwaysDraftUpdateScheduleConfig, dict[str, object]]:
    config = AlwaysDraftUpdateScheduleConfig()
    scheduler = DraftUpdateScheduler.create(config, origin_step=0)
    decision = scheduler.decide(global_step=1, acceptance=None)
    scheduler.record_outcome(
        decision,
        update_attempted=True,
        update_successful=True,
        draft_refit_attempted=True,
        draft_refit_successful=True,
    )
    return config, scheduler.state_dict()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("version", 2, "inner state version"),
        ("attempted_updates", -1, "nonnegative"),
        ("successful_updates", 2, "successful updates"),
        ("failed_updates", 1, "attempted updates.*partition"),
        ("skipped_updates", 1, "update partition"),
        ("forced_updates", 2, "forced updates"),
        ("last_update_step", 2, "last update step"),
        ("last_applied_refit_step", 2, "last applied refit step"),
        ("applied_draft_version", 99, "applied draft version"),
        ("next_decision_id", 3, "decision cursor"),
        ("decision_history", [], "history.*nonempty"),
        ("phase", "broken", "phase"),
        ("acceptance_ewma", float("inf"), "EWMA"),
        ("valid_observations", -1, "nonnegative"),
        ("burst_updates", -1, "nonnegative"),
    ],
)
def test_restore_rejects_each_corrupt_scheduler_invariant(
    field: str, value: object, message: str
) -> None:
    config, saved = valid_saved_state()
    corrupt = copy.deepcopy(saved)
    state = corrupt["state"]
    assert isinstance(state, dict)
    state[field] = value
    with pytest.raises(ValueError, match=message):
        validate_scheduler_state_invariants(config, state)


def test_restore_rejects_invalid_history_reason_and_phase_fields() -> None:
    config, saved = valid_saved_state()
    bad_reason = copy.deepcopy(saved)
    reason_state = bad_reason["state"]
    assert isinstance(reason_state, dict)
    history = reason_state["decision_history"]
    assert isinstance(history, list)
    assert isinstance(history[0], dict)
    history[0]["reason"] = "unknown"
    with pytest.raises(ValueError, match="history reason"):
        validate_scheduler_state_invariants(config, reason_state)
    bad_phase = copy.deepcopy(saved)
    phase_state = bad_phase["state"]
    assert isinstance(phase_state, dict)
    phase_state["phase"] = "training_burst"
    with pytest.raises(ValueError, match="non-adaptive.*monitoring"):
        validate_scheduler_state_invariants(config, phase_state)


def test_adaptive_restore_rejects_phase_inconsistent_observation_fields() -> None:
    config = AdaptiveDraftUpdateScheduleConfig()
    saved = DraftUpdateScheduler.create(config, origin_step=0).state_dict()
    state = saved["state"]
    assert isinstance(state, dict)
    state["valid_observations"] = 1
    with pytest.raises(ValueError, match="valid observations require acceptance EWMA"):
        validate_scheduler_state_invariants(config, state)
    saved = DraftUpdateScheduler.create(config, origin_step=0).state_dict()
    state = saved["state"]
    assert isinstance(state, dict)
    state["phase"] = "awaiting_post_refit_observation"
    with pytest.raises(ValueError, match="awaiting phase requires an applied refit"):
        validate_scheduler_state_invariants(config, state)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("last_decided_step", 1.0),
        ("next_decision_id", True),
        ("applied_draft_version", 1.0),
    ],
)
def test_restore_rejects_nonintegral_scheduler_steps_and_versions(
    field: str, value: object
) -> None:
    config, saved = valid_saved_state()
    state = saved["state"]
    assert isinstance(state, dict)
    state[field] = value
    with pytest.raises(ValueError, match="integer"):
        validate_scheduler_state_invariants(config, state)


def test_restore_derives_applied_version_from_last_refit_step() -> None:
    config, saved = valid_saved_state()
    state = saved["state"]
    assert isinstance(state, dict)
    state["applied_draft_version"] = 0
    with pytest.raises(ValueError, match="last applied refit"):
        validate_scheduler_state_invariants(config, state)
    (build_terminal_schedule_payload,)
