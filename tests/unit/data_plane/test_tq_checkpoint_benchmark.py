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

"""Unit tests for the standalone TQ checkpoint benchmark helpers."""

from __future__ import annotations

import pytest

from tools.tq_checkpoint_benchmark import (
    BenchmarkConfig,
    _make_payload,
    _producer_key,
    _producer_row_id,
    _producer_validation_upper_bounds,
    _row_id_from_key,
    logical_tensor_bytes,
    logical_tensor_bytes_for_length,
    payload_field_names,
    percentile,
    sequence_length,
    summarize_records,
)


def _config(**overrides) -> BenchmarkConfig:
    values = {
        "checkpoint_root": "/tmp/checkpoints",
        "run_dir": "/tmp/checkpoints/run",
        "num_rows": 8,
        "min_seq_len": 4,
        "max_seq_len": 8,
        "payload_profile": "generation",
        "batch_rows": 4,
        "num_storage_units": 2,
        "producer_mode": "quiescent",
        "num_producers": 4,
        "producer_batch_rows": 2,
        "producer_max_rows": 8,
        "producer_warmup_s": 0.1,
        "producer_cooldown_s": 0.1,
        "producer_sleep_ms": 0.0,
        "verify_mode": "sample",
        "verify_samples": 4,
        "verify_batch_rows": 2,
        "group_size": 2,
        "weight_version": 3,
        "seed": 42,
        "ray_address": "",
        "phase_timeout_s": 30.0,
        "torch_num_threads": 1,
    }
    values.update(overrides)
    return BenchmarkConfig(**values)


def test_sequence_lengths_are_deterministic_and_bounded() -> None:
    first = [sequence_length(row, 128, 512, 7) for row in range(100)]
    second = [sequence_length(row, 128, 512, 7) for row in range(100)]

    assert first == second
    assert min(first) >= 128
    assert max(first) <= 512
    assert len(set(first)) > 1


def test_logical_bytes_match_profiles() -> None:
    assert logical_tensor_bytes_for_length(10, "generation") == 152
    assert logical_tensor_bytes_for_length(10, "train-ready") == 212
    row_ids = [0, 1, 2]
    assert (
        logical_tensor_bytes(
            row_ids,
            min_seq_len=10,
            max_seq_len=10,
            seed=0,
            profile="generation",
        )
        == 3 * 152
    )


def test_payload_profiles_match_nemo_rl_train_fields() -> None:
    assert payload_field_names("generation") == [
        "input_ids",
        "input_lengths",
        "generation_logprobs",
        "token_mask",
        "sample_mask",
    ]
    assert payload_field_names("train-ready") == [
        "input_ids",
        "input_lengths",
        "generation_logprobs",
        "prev_logprobs",
        "reference_policy_logprobs",
        "advantages",
        "token_mask",
        "sample_mask",
    ]


def test_key_round_trip() -> None:
    key = _producer_key(12, 345)
    assert _row_id_from_key(key) == _producer_row_id(12, 345)
    assert _row_id_from_key("base-000000000123") == 123
    with pytest.raises(ValueError, match="unrecognized"):
        _row_id_from_key("not-a-benchmark-key")


def test_producer_validation_uses_final_acknowledgements() -> None:
    metrics = {
        # Producer 3 has completed its TQ put, but its local counter was not
        # visible in the immediate post-checkpoint snapshot.
        "acknowledged_by_checkpoint_return": {"0": 32, "3": 0},
        "final_acknowledged": {"0": 64, "3": 32},
    }

    assert _producer_validation_upper_bounds(metrics) == {"0": 64, "3": 32}


def test_percentile_and_window_metrics() -> None:
    assert percentile([], 0.5) is None
    assert percentile([10.0], 0.5) == 10.0
    assert percentile([0.0, 10.0], 0.5) == 5.0

    records = [
        {
            "completed_at": 1.5,
            "rows": 2,
            "logical_tensor_bytes": 100,
            "put_duration_s": 0.010,
        },
        {
            "completed_at": 2.5,
            "rows": 4,
            "logical_tensor_bytes": 200,
            "put_duration_s": 0.030,
        },
    ]
    summary = summarize_records(records, window_start=1.0, window_end=2.0)
    assert summary["batches"] == 1
    assert summary["rows"] == 2
    assert summary["logical_tensor_bytes"] == 100
    assert summary["rows_per_s"] == 2.0
    assert summary["put_latency_p50_ms"] == 10.0


def test_payload_is_jagged_deterministic_and_sized() -> None:
    torch = pytest.importorskip("torch")
    config = _config()
    row_ids = [0, 1, 2]

    first, first_tags, first_bytes = _make_payload(row_ids, config)
    second, second_tags, second_bytes = _make_payload(row_ids, config)

    assert set(first.keys()) == set(payload_field_names("generation"))
    assert first["input_ids"].is_nested
    assert first_tags == second_tags
    assert first_bytes == second_bytes
    assert first_bytes == logical_tensor_bytes(
        row_ids,
        min_seq_len=config.min_seq_len,
        max_seq_len=config.max_seq_len,
        seed=config.seed,
        profile=config.payload_profile,
    )
    for field in payload_field_names("generation"):
        for left, right in zip(
            first[field].unbind(),
            second[field].unbind(),
            strict=True,
        ):
            assert torch.equal(left, right)


def test_train_ready_payload_adds_policy_fields() -> None:
    pytest.importorskip("torch")
    config = _config(payload_profile="train-ready")
    payload, _, _ = _make_payload([0, 1], config)

    assert list(payload.keys()) == payload_field_names("train-ready")
