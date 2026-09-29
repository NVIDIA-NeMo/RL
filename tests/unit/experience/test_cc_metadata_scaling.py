# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure durable ledger growth through the paired HTTP capture path on CPU."""

import asyncio
import json
from collections.abc import Callable
from pathlib import Path

import pytest

pytest.importorskip("nemo_gym", reason="requires the paired Gym checkout")

from nemo_gym.token_id_capture.lineage import FileLineageStore
from nemo_gym.token_id_capture.sink import (
    CaptureContext,
    reset_token_sink,
    resolve_parent,
    set_token_sink,
)
from responses_api_models.vllm_model.tests import test_framework_context as gym_harness

from nemo_rl.models.generation.capture_context import decide_capture_input

pytestmark = pytest.mark.nemo_gym


@pytest.mark.parametrize("compact", [False, True])
def test_ledger_growth_and_restart_preserve_segment_chains(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    compact: bool,
    record_property: Callable[[str, object], None],
) -> None:
    harness = gym_harness.make_capture_harness(
        monkeypatch, tmp_path, decide_input=decide_capture_input
    )
    items = [{"role": "user", "content": "task"}]
    sizes = []
    try:
        for turn in range(80):
            if compact and turn == 40:
                items = [{"role": "user", "content": "compacted summary"}]
            response = gym_harness.assert_clean(harness.post("attempt", items))
            items += response["output"] + [
                {"role": "user", "content": f"observation {turn}"}
            ]
            if turn + 1 in (20, 40, 80):
                path = tmp_path / "lineage" / "attempt.lineage.jsonl"
                sizes.append(path.stat().st_size)
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                committed = [row for row in rows if row.get("replay") is not None]
                assert len(committed) == turn + 1
                assert all(row["staging_chain"] == [] for row in committed)
                assert all("items" not in row["replay"] for row in committed)
                # Three fixed digests (including empty-tool-content equivalence)
                # and a growing decimal item count remain bounded per call.
                assert max(len(json.dumps(row["replay"])) for row in committed) < 400

        manifest = harness.manifest("attempt")
        assert len(manifest.records) == len(harness.sink.records) == 80
        assert sum(record.parent_call_id is None for record in manifest.records) == (
            2 if compact else 1
        )
        # Doubling calls approximately doubles the actual durable JSONL bytes.
        # Full historical arrays/chains would approach 4x and fail this bound.
        assert all(
            1.9 < after / before < 2.1 for before, after in zip(sizes, sizes[1:])
        )
        record_property("ledger_bytes_at_20_40_80_calls", json.dumps(sizes))
        expected_chain = [
            record.staging_key for record in manifest.records[40 if compact else 0 :]
        ]

        async def reopen() -> None:
            ledger = FileLineageStore(tmp_path / "lineage")
            context = CaptureContext(
                rollout_id="attempt",
                model_call_id="after-restart",
                token_sink=None,
                lineage_store=ledger,
                external_staging=True,
                framework_owned_context=True,
            )
            token = set_token_sink(context)
            try:
                await resolve_parent(items)
                admission = context.capture_admission
                assert admission.staging_chain == expected_chain
                assert admission.parent_call_id == manifest.records[-1].model_call_id
                # The same prefix decision survives a fresh ledger instance.
                decision = decide_capture_input(
                    admission.model_copy(
                        update={
                            "request_replay": admission.request_replay.model_copy(
                                update={
                                    "render_digest": admission.candidate_replay.render_digest
                                }
                            )
                        }
                    ),
                    messages=items,
                )
                assert decision.storage.mode == "token_in"
            finally:
                reset_token_sink(token)
                await ledger.close()

        asyncio.run(reopen())
    finally:
        harness.client.close()
        asyncio.run(harness.ledger.close())
