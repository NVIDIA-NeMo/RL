# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nemo_rl.algorithms.draft_cadence_runtime import (
    CadenceRuntimeConfig,
    CadenceRuntimeWriter,
    CadenceTerminalEvidence,
)
from nemo_rl.algorithms.draft_update_schedule import (
    DraftDecisionLedger,
    DraftUpdateScheduler,
    FileDraftStepTransactionStore,
)
from nemo_rl.algorithms.grpo_sync import apply_scheduled_refit
from nemo_rl.models.policy.draft_config import (
    AlwaysDraftUpdateScheduleConfig,
    FixedDraftUpdateScheduleConfig,
)
from nemo_rl.weight_sync.interfaces import WeightSyncSelection


class SyncHarness:
    def __init__(self) -> None:
        config = FixedDraftUpdateScheduleConfig(
            mode="fixed", action="sparse_update", fixed_interval=2
        )
        self.scheduler = DraftUpdateScheduler.create(config, origin_step=0)
        self.sync_selections = []
        self.training_decisions = []
        self.worker_result = {"draft_update_successful": True}
        self.target_weight_version = 0
        self.applied_draft_version = 0
        self.root = Path(tempfile.mkdtemp())
        self.transaction_store = FileDraftStepTransactionStore(
            self.root / "transactions"
        )
        self.decision_ledger = DraftDecisionLedger(self.root / "ledger.jsonl")
        self.grpo_save_state = SimpleNamespace(
            draft_update_schedule=None,
            applied_draft_snapshot={
                "version": 0,
                "path": "initial",
                "size_bytes": 0,
                "sha256": hashlib.sha256(b"").hexdigest(),
            },
        )

    def run_one_step(self, fixed_sparse_interval: int = 2) -> None:
        assert fixed_sparse_interval == self.scheduler.config.fixed_interval
        step = self.scheduler.state.last_decided_step + 1
        decision = self.scheduler.decide(global_step=step, acceptance=None)
        transaction = self.transaction_store.begin(decision)
        self.training_decisions.append(decision)

        def sync_weights(*, selection):
            self.sync_selections.append(selection)
            receipt = {"successful": True}
            if selection.draft:
                path = self.root / f"draft-v{decision.decision_id}.bin"
                raw = f"draft-{decision.decision_id}".encode()
                path.write_bytes(raw)
                receipt["draft_apply_receipt"] = {
                    "successful": True,
                    "version": decision.decision_id,
                    "snapshot_path": str(path.resolve()),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            return receipt

        apply_scheduled_refit(
            decision,
            self.worker_result,
            self.scheduler,
            transaction=transaction,
            decision_ledger=self.decision_ledger,
            grpo_save_state=self.grpo_save_state,
            transaction_store=self.transaction_store,
            runtime_writer=None,
            terminal_evidence=None,
            sync_weights=sync_weights,
            publish_target_version=lambda: setattr(
                self, "target_weight_version", self.target_weight_version + 1
            ),
            publish_draft_version=lambda version: setattr(
                self, "applied_draft_version", version
            ),
        )

    def run_two_steps(self, fixed_sparse_interval: int) -> None:
        self.run_one_step(fixed_sparse_interval)
        self.run_one_step(fixed_sparse_interval)


def test_sync_controller_refits_target_every_step() -> None:
    harness = SyncHarness()
    harness.run_two_steps(fixed_sparse_interval=2)
    assert harness.sync_selections == [
        WeightSyncSelection(target=True, draft=False),
        WeightSyncSelection(target=True, draft=True),
    ]
    assert harness.training_decisions[0].update_requested is False
    assert harness.training_decisions[1].update_requested is True


def test_failed_draft_update_stops_before_transfer_or_version_publish() -> None:
    harness = SyncHarness()
    harness.scheduler = DraftUpdateScheduler.create(
        FixedDraftUpdateScheduleConfig(
            mode="fixed", action="sparse_update", fixed_interval=1
        ),
        origin_step=0,
    )
    harness.worker_result["draft_update_successful"] = False
    with pytest.raises(RuntimeError, match="draft update failed across workers"):
        harness.run_one_step(fixed_sparse_interval=1)
    assert harness.sync_selections == []
    assert harness.target_weight_version == 0
    assert harness.applied_draft_version == 0
    assert harness.scheduler.state.attempted_updates == 1
    assert harness.scheduler.state.successful_updates == 0
    assert harness.scheduler.state.failed_updates == 1
    assert harness.scheduler.state.attempted_refits == 0
    assert harness.scheduler.state.failed_refits == 0
    assert harness.scheduler.state.skipped_refits == 1


def test_sync_update_receipt_is_durable_before_transfer_and_publication(
    tmp_path: Path,
) -> None:
    scheduler = DraftUpdateScheduler.create(
        AlwaysDraftUpdateScheduleConfig(), origin_step=0
    )
    decision = scheduler.decide(global_step=1, acceptance=None)
    store = FileDraftStepTransactionStore(tmp_path / "transactions")
    transaction = store.begin(decision)
    ledger = DraftDecisionLedger(tmp_path / "ledger.jsonl")
    evidence = CadenceTerminalEvidence({}, {})
    save_state = SimpleNamespace(
        draft_terminal_evidence=None,
        draft_update_schedule=None,
        applied_draft_snapshot={"version": 0},
    )
    writer = CadenceRuntimeWriter(
        CadenceRuntimeConfig(enabled=True, result_dir=str(tmp_path / "runtime"))
    )
    snapshot_path = tmp_path / "draft-v1.bin"
    snapshot_path.write_bytes(b"draft-v1")
    events = []

    def sync_weights(*, selection):
        assert 1 in evidence.update_receipts_by_decision
        assert save_state.draft_terminal_evidence == evidence.state_dict()
        events.append("transfer")
        return {
            "successful": True,
            "draft_apply_receipt": {
                "successful": True,
                "version": 1,
                "snapshot_path": str(snapshot_path),
                "sha256": hashlib.sha256(b"draft-v1").hexdigest(),
            },
        }

    apply_scheduled_refit(
        decision,
        {
            "draft_update_successful": True,
            "draft_update_receipt": {
                "successful": True,
                "decision_id": 1,
                "global_step": 1,
                "draft_model_sha256": "a" * 64,
                "draft_optimizer_sha256": "b" * 64,
            },
        },
        scheduler,
        transaction=transaction,
        decision_ledger=ledger,
        grpo_save_state=save_state,
        transaction_store=store,
        runtime_writer=writer,
        terminal_evidence=evidence,
        sync_weights=sync_weights,
        publish_target_version=lambda: events.append("publish-target"),
        publish_draft_version=lambda _version: events.append("publish-draft"),
    )
    assert events == ["transfer", "publish-target", "publish-draft"]


def test_missing_update_receipt_closes_without_claiming_refit_attempt(
    tmp_path: Path,
) -> None:
    scheduler = DraftUpdateScheduler.create(
        AlwaysDraftUpdateScheduleConfig(), origin_step=0
    )
    decision = scheduler.decide(global_step=1, acceptance=None)
    store = FileDraftStepTransactionStore(tmp_path / "transactions")
    sync_weights = MagicMock()

    with pytest.raises(RuntimeError, match="lacks worker receipt"):
        apply_scheduled_refit(
            decision,
            {"draft_update_successful": True},
            scheduler,
            transaction=store.begin(decision),
            decision_ledger=DraftDecisionLedger(tmp_path / "ledger.jsonl"),
            grpo_save_state=SimpleNamespace(
                draft_terminal_evidence=None,
                draft_update_schedule=None,
                applied_draft_snapshot={"version": 0},
            ),
            transaction_store=store,
            runtime_writer=CadenceRuntimeWriter(
                CadenceRuntimeConfig(enabled=True, result_dir=str(tmp_path / "runtime"))
            ),
            terminal_evidence=CadenceTerminalEvidence({}, {}),
            sync_weights=sync_weights,
            publish_target_version=MagicMock(),
            publish_draft_version=MagicMock(),
        )

    sync_weights.assert_not_called()
    assert scheduler.state.attempted_updates == 1
    assert scheduler.state.successful_updates == 1
    assert scheduler.state.attempted_refits == 0
    assert scheduler.state.failed_refits == 0
    assert scheduler.state.skipped_refits == 1
