# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path

import pytest

from nemo_rl.data_plane.token_staging_wire import (
    TokenCaptureAdmission,
    build_staged_token_record,
    commit_coords,
    compute_extras_digest,
    compute_staging_digest,
)
from nemo_rl.data_plane.tq_token_sink import TQTokenSink

REPO_ROOT = Path(__file__).parents[3]
GOLDEN_VECTORS = (
    REPO_ROOT
    / "3rdparty/Gym-workspace/Gym/nemo_gym/token_id_capture/staging/conformance/golden_vectors.json"
)


def test_dependency_neutral_digest_matches_gym_golden_vectors() -> None:
    vectors = json.loads(GOLDEN_VECTORS.read_text())

    assert compute_extras_digest(vectors["extras"]) == vectors["extras_digest"]
    assert (
        compute_staging_digest(**vectors["staged_call"])
        == vectors["staged_call_digest"]
    )


def test_build_staged_record_and_coords_without_gym_runtime() -> None:
    admission = TokenCaptureAdmission.from_wire(
        {
            "rollout_id": "rollout-1",
            "model_call_id": "call-1",
            "mode": "text",
        }
    )

    record = build_staged_token_record(
        admission=admission,
        prefix_token_ids=[],
        prompt_token_ids=[10, 11],
        generated_token_ids=[20, 21],
        generated_logprobs=[-0.1, -0.2],
        weight_version=3,
        extras=None,
    )

    assert record.token_ids_delta == [10, 11, 20, 21]
    assert record.token_mask_delta == [0.0, 0.0, 1.0, 1.0]
    assert record.generation_log_probs_delta == [0.0, 0.0, -0.1, -0.2]
    assert commit_coords(record) == {
        "schema_version": 2,
        "digest_version": 2,
        "extras_digest_version": 1,
        "rollout_id": "rollout-1",
        "model_call_id": "call-1",
        "parent_call_id": None,
        "prev_len": 0,
        "delta_len": 4,
        "cum_len": 4,
        "weight_version": 3,
        "disposition": "staged",
        "digest": record.digest,
        "extras_digest": record.extras_digest,
        "staging_key": "rollout-1/call-1",
        "chain_hash": record.chain_hash,
        "cumulative_hash": record.cumulative_hash,
    }


def test_token_in_record_requires_exact_admitted_prefix() -> None:
    admission = TokenCaptureAdmission.from_wire(
        {
            "rollout_id": "rollout-1",
            "model_call_id": "call-2",
            "parent_call_id": "call-1",
            "prev_len": 2,
            "mode": "token_in",
            "required_prefix_token_ids": [10, 11],
            "parent_chain_hash": "1" * 64,
        }
    )

    with pytest.raises(ValueError, match="does not begin"):
        build_staged_token_record(
            admission=admission,
            prefix_token_ids=[10, 11],
            prompt_token_ids=[10, 99, 20],
            generated_token_ids=[30],
            generated_logprobs=[-0.1],
            weight_version=0,
            extras=None,
        )


def test_tq_sink_stage_wire_does_not_construct_gym_types() -> None:
    class RecordingClient:
        def __init__(self) -> None:
            self.calls = []

        def put_samples(self, **kwargs) -> None:
            self.calls.append(kwargs)

    admission = TokenCaptureAdmission.from_wire(
        {
            "rollout_id": "rollout-1",
            "model_call_id": "call-1",
            "mode": "text",
        }
    )
    record = build_staged_token_record(
        admission=admission,
        prefix_token_ids=[],
        prompt_token_ids=[10],
        generated_token_ids=[20],
        generated_logprobs=[-0.1],
        weight_version=0,
        extras=None,
    )
    client = RecordingClient()

    result = TQTokenSink(client, staging_partition="staging").stage_wire(record)

    assert result.ok
    assert result.staging_key == "rollout-1/call-1"
    assert len(client.calls) == 1


def test_external_bridge_and_launcher_keep_dependency_boundaries() -> None:
    bridge = (
        REPO_ROOT / "nemo_rl/models/generation/remote_vllm/token_capture_bridge.py"
    ).read_text()
    launcher = (
        REPO_ROOT
        / "tools/external_rollout_vllm/launch_automationbench_super_external_vllm.sh"
    ).read_text()

    assert "nemo_gym" not in bridge
    assert "--extra nemo_gym" not in launcher
