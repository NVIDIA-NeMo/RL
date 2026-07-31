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
import torch

from nemo_rl.algorithms.rollout_collection import (
    _aggregate_sample_metrics,
    assemble_group_payload,
    assigned_indices,
    build_group_payload,
    existing_group_indices,
    existing_group_indices_all,
    group_filename,
    load_group,
    parse_group_index,
    resolve_collection_config,
    write_group_atomic,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def test_assigned_indices_partition():
    """Shards partition the index range: disjoint and complete."""
    n, num_shards = 103, 7
    seen = []
    for k in range(num_shards):
        idx = assigned_indices(n, k, num_shards)
        assert all(i % num_shards == k for i in idx)
        seen.extend(idx)
    assert sorted(seen) == list(range(n))


def test_assigned_indices_range_and_cap():
    """index_start/index_end clamp the range; max_groups caps the count."""
    idx = assigned_indices(100, 0, 2, index_start=10, index_end=20)
    assert idx == [10, 12, 14, 16, 18]
    assert assigned_indices(100, 0, 1, max_groups=3) == [0, 1, 2]
    # index_end beyond dataset length is clamped
    assert assigned_indices(5, 0, 1, index_end=100) == [0, 1, 2, 3, 4]


def test_group_filename_roundtrip():
    """parse_group_index inverts group_filename and rejects non-group files."""
    assert parse_group_index(group_filename(42)) == 42
    assert parse_group_index(group_filename(12345678)) == 12345678
    assert parse_group_index("group_00000042.pt.tmp.123") is None
    assert parse_group_index("meta.json") is None
    assert parse_group_index("group_abc.pt") is None


def test_existing_group_indices_ignores_tmp(tmp_path):
    """Only finalized group files count toward resume state."""
    write_group_atomic(tmp_path, 3, {"x": 1})
    write_group_atomic(tmp_path, 7, {"x": 2})
    torch.save({"x": 3}, tmp_path / "group_00000009.pt.tmp.999")  # crashed write
    (tmp_path / "meta.json").write_text("{}")
    assert existing_group_indices(tmp_path) == {3, 7}
    assert existing_group_indices(tmp_path / "does_not_exist") == set()


def test_existing_group_indices_all_spans_shard_layouts(tmp_path):
    """Resume unions across all shard dirs so num_shards can be rescaled."""
    (tmp_path / "shard_000").mkdir()
    (tmp_path / "shard_001").mkdir()
    (tmp_path / "shard_042").mkdir()
    write_group_atomic(tmp_path / "shard_000", 0, {"x": 1})
    write_group_atomic(tmp_path / "shard_001", 5, {"x": 1})
    write_group_atomic(tmp_path / "shard_042", 9, {"x": 1})
    assert existing_group_indices_all(tmp_path) == {0, 5, 9}
    assert existing_group_indices_all(tmp_path / "missing") == set()


def test_write_load_group_roundtrip(tmp_path):
    """Payloads round-trip through the atomic writer, tensors intact."""
    batch = BatchedDataDict(
        {
            "total_reward": torch.tensor([0.0, 1.0]),
            "message_log": [[{"role": "assistant"}], [{"role": "assistant"}]],
        }
    )
    payload = {"format_version": 1, "dataset_idx": 5, "batch": batch}
    path = write_group_atomic(tmp_path, 5, payload)
    loaded = load_group(path)
    assert loaded["dataset_idx"] == 5
    torch.testing.assert_close(
        loaded["batch"]["total_reward"], batch["total_reward"]
    )


def test_build_group_payload_grafts_input_keys():
    """extra_env_info/idx/task_name are grafted from the input batch (the
    NeMo-Gym rollout path drops them from final_batch)."""
    input_batch = BatchedDataDict(
        {
            "extra_env_info": [{"gt": "a"}, {"gt": "b"}],
            "idx": [4, 4],
            "task_name": ["nemo_gym", "nemo_gym"],
            "loss_multiplier": torch.ones(2),
        }
    )
    final_batch = BatchedDataDict(
        {
            "total_reward": torch.tensor([1.0, 0.0]),
            "loss_multiplier": torch.ones(2),
        }
    )
    payload = build_group_payload(4, input_batch, final_batch, {"m": 1})
    assert payload["batch"]["extra_env_info"] == [{"gt": "a"}, {"gt": "b"}]
    assert payload["batch"]["idx"] == [4, 4]
    assert payload["batch"]["task_name"] == ["nemo_gym", "nemo_gym"]
    assert payload["dataset_idx"] == 4
    assert payload["rollout_metrics"] == {"m": 1}


def test_build_group_payload_strips_rowidx():
    """The rollout's per-call _rowidx scratch field is not persisted."""
    input_batch = BatchedDataDict(
        {
            "extra_env_info": [{"gt": "a", "_rowidx": 0}, {"gt": "b", "_rowidx": 0}],
            "loss_multiplier": torch.ones(2),
        }
    )
    final_batch = BatchedDataDict({"total_reward": torch.zeros(2)})
    payload = build_group_payload(9, input_batch, final_batch, {})
    assert payload["batch"]["extra_env_info"] == [{"gt": "a"}, {"gt": "b"}]


def test_resolve_collection_config_defaults_and_required():
    """Defaults are filled from the ppo block; out_dir is mandatory."""
    ppo_cfg = {"num_generations_per_prompt": 16}
    cfg = resolve_collection_config({"out_dir": "/tmp/x"}, ppo_cfg)
    assert cfg["gens_per_prompt"] == 16
    assert cfg["num_shards"] == 1 and cfg["shard_id"] == 0
    assert cfg["max_inflight_samples"] == 48  # 3 groups-equivalent x 16
    cfg = resolve_collection_config(
        {"out_dir": "/tmp/x", "gens_per_prompt": "4", "shard_id": "2",
         "num_shards": "8"},
        ppo_cfg,
    )
    assert cfg["gens_per_prompt"] == 4 and cfg["shard_id"] == 2
    assert cfg["max_inflight_samples"] == 12  # 3 x gens_per_prompt=4
    with pytest.raises(AssertionError):
        resolve_collection_config({}, ppo_cfg)


def test_resolve_collection_config_inflight_knobs():
    """max_inflight_samples wins; legacy max_inflight_groups converts."""
    ppo_cfg = {"num_generations_per_prompt": 16}
    cfg = resolve_collection_config(
        {"out_dir": "/tmp/x", "max_inflight_samples": "64"}, ppo_cfg
    )
    assert cfg["max_inflight_samples"] == 64
    cfg = resolve_collection_config(
        {"out_dir": "/tmp/x", "max_inflight_groups": 2, "gens_per_prompt": 8},
        ppo_cfg,
    )
    assert cfg["max_inflight_samples"] == 16
    # explicit samples takes precedence over the legacy knob
    cfg = resolve_collection_config(
        {"out_dir": "/tmp/x", "max_inflight_groups": 2, "max_inflight_samples": 5},
        ppo_cfg,
    )
    assert cfg["max_inflight_samples"] == 5


def test_aggregate_sample_metrics():
    """Numeric fields are averaged; non-numeric/non-finite dropped."""
    agg = _aggregate_sample_metrics(
        [
            {"reward": 1.0, "turns": 4, "note": "text", "hit_max": True,
             "len_stddev": float("nan")},
            {"reward": 0.0, "turns": 6, "hit_max": False,
             "len_stddev": float("nan")},
        ]
    )
    assert agg["reward"] == 0.5
    assert agg["turns"] == 5.0
    assert agg["hit_max"] == 0.5
    assert "note" not in agg
    assert "len_stddev" not in agg  # single-sample NaN stats must not survive
    assert agg["aggregated_from_samples"] == 2.0


def test_assemble_group_payload_orders_and_concats():
    """Per-sample results reassemble into a whole-group-shaped payload."""
    def sample_batch(reward, truncated):
        return BatchedDataDict(
            {
                "message_log": [[{"role": "assistant"}]],
                "total_reward": torch.tensor([reward]),
                "truncated": torch.tensor([truncated]),
                "loss_multiplier": torch.ones(1),
            }
        )

    input_batch = BatchedDataDict(
        {
            "extra_env_info": [{"k": 1}, {"k": 2}],
            "idx": [7, 7],
            "task_name": ["nemo_gym", "nemo_gym"],
            "loss_multiplier": torch.ones(2),
        }
    )
    payload = assemble_group_payload(
        7,
        input_batch,
        [sample_batch(1.0, False), sample_batch(0.0, True)],
        [{"reward": 1.0}, {"reward": 0.0}],
    )
    assert payload["dataset_idx"] == 7
    assert payload["batch"].size == 2
    torch.testing.assert_close(
        payload["batch"]["total_reward"], torch.tensor([1.0, 0.0])
    )
    assert payload["batch"]["extra_env_info"] == [{"k": 1}, {"k": 2}]
    assert payload["rollout_metrics"]["reward"] == 0.5
