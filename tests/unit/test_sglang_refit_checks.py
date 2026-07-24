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

import pytest

from tests.sglang_refit_checks import (
    CheckError,
    check_cluster_capacity,
    check_log_markers,
    check_training_metrics,
    validate_run_evidence,
)


def _node(*, alive: bool = True, gpus: float = 0) -> dict:
    return {"Alive": alive, "Resources": {"GPU": gpus}}


def test_cluster_capacity_requires_enough_gpus_on_each_node():
    summary = check_cluster_capacity(
        [_node(gpus=4), _node(gpus=4), _node(gpus=2), _node(alive=False, gpus=8)],
        required_nodes=2,
        gpus_per_node=4,
    )

    assert summary["status"] == "passed"
    assert summary["observed"]["eligible_gpu_nodes"] == 2
    assert summary["observed"]["gpu_capacities"] == [2, 4, 4]


def test_cluster_capacity_rejects_aggregate_only_capacity():
    with pytest.raises(CheckError, match="1 eligible nodes"):
        check_cluster_capacity(
            [_node(gpus=8), _node(gpus=2)],
            required_nodes=2,
            gpus_per_node=4,
        )


def test_cluster_capacity_rejects_busy_gpus():
    with pytest.raises(CheckError, match="4 currently available"):
        check_cluster_capacity(
            [_node(gpus=4), _node(gpus=4)],
            required_nodes=2,
            gpus_per_node=4,
            available_gpus=4,
        )


def test_log_markers_require_exact_topology_and_success_count():
    summary = check_log_markers(
        "\n".join(
            [
                "NRL_SGLANG_REFIT_GROUP_READY world_size=5 engines=4",
                "NRL_SGLANG_REFIT_SUCCESS version=1",
                "NRL_SGLANG_REFIT_SUCCESS version=2",
            ]
        ),
        expected_world_size=5,
        expected_engines=4,
        min_refit_successes=2,
    )

    assert summary == {
        "group_ready_count": 1,
        "refit_success_count": 2,
        "world_size": 5,
        "engines": 4,
    }


@pytest.mark.parametrize(
    "log_text, error",
    [
        (
            "NRL_SGLANG_REFIT_GROUP_READY world_size=6 engines=4\n"
            "NRL_SGLANG_REFIT_SUCCESS\nNRL_SGLANG_REFIT_SUCCESS",
            "Unexpected SGLang refit topology",
        ),
        (
            "NRL_SGLANG_REFIT_GROUP_READY world_size=5 engines=4\n"
            "NRL_SGLANG_REFIT_SUCCESS\nNRL_SGLANG_REFIT_FAILURE",
            "Fatal refit evidence",
        ),
        (
            "NRL_SGLANG_REFIT_GROUP_READY world_size=5 engines=4\n"
            "NRL_SGLANG_REFIT_SUCCESS\nNCCL remote error",
            "Fatal refit evidence",
        ),
        (
            "NRL_SGLANG_REFIT_GROUP_READY world_size=5 engines=4\n"
            "NRL_SGLANG_REFIT_SUCCESS\n"
            "Watchdog caught collective operation timeout",
            "Fatal refit evidence",
        ),
        (
            "NRL_SGLANG_REFIT_GROUP_READY world_size=5 engines=4\n"
            "NRL_SGLANG_REFIT_SUCCESS",
            "Expected at least 2",
        ),
    ],
)
def test_log_markers_reject_invalid_evidence(log_text, error):
    with pytest.raises(CheckError, match=error):
        check_log_markers(
            log_text,
            expected_world_size=5,
            expected_engines=4,
            min_refit_successes=2,
        )


def test_training_metrics_require_terminal_step():
    assert check_training_metrics(
        {"train/loss": {"1": 1.0, "2": 0.5, "3": 0.25}},
        expected_max_step=3,
    ) == {"expected_max_step": 3, "max_recorded_step": 3}

    with pytest.raises(CheckError, match="found step 2"):
        check_training_metrics(
            {"train/loss": {"1": 1.0, "2": 0.5}},
            expected_max_step=3,
        )


def test_validate_run_evidence_requires_nonempty_gym_artifact(tmp_path):
    log_text = "\n".join(
        [
            "NRL_SGLANG_REFIT_GROUP_READY world_size=65 engines=32",
            "NRL_SGLANG_REFIT_SUCCESS",
            "NRL_SGLANG_REFIT_SUCCESS",
        ]
    )
    metrics = {"train/loss": {"3": 0.25}}

    with pytest.raises(CheckError, match="No non-empty NeMo-Gym"):
        validate_run_evidence(
            log_text=log_text,
            metrics=metrics,
            expected_max_step=3,
            expected_world_size=65,
            expected_engines=32,
            min_refit_successes=2,
            gym_log_dir=tmp_path,
        )

    artifact = tmp_path / "nested" / "train_data_step3.jsonl"
    artifact.parent.mkdir()
    artifact.write_text('{"reward": 1}\n')
    summary = validate_run_evidence(
        log_text=log_text,
        metrics=metrics,
        expected_max_step=3,
        expected_world_size=65,
        expected_engines=32,
        min_refit_successes=2,
        gym_log_dir=tmp_path,
    )

    assert summary["gym"] == {
        "artifact_count": 1,
        "artifacts": ["nested/train_data_step3.jsonl"],
    }
