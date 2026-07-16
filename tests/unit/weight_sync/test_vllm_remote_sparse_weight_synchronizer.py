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

from unittest.mock import ANY, MagicMock, patch

import pytest

from nemo_rl.models.generation.vllm.config import VllmDeltaCompressionConfig
from nemo_rl.weight_sync.vllm_remote_sparse_weight_synchronizer import (
    VllmRemoteSparseWeightSynchronizer,
    validate_vllm_remote_sparse_refit,
)

_MODULE = "nemo_rl.weight_sync.vllm_remote_sparse_weight_synchronizer"


@pytest.fixture
def mock_ray():
    with patch(f"{_MODULE}.ray") as value:
        yield value


@pytest.fixture
def post():
    with patch(f"{_MODULE}.post_vllm_refit_endpoints") as value:
        yield value


def _remote_sparse_sync(
    mock_ray: MagicMock,
    transport: str,
    stream_result: list[dict[str, int]] | RuntimeError,
) -> tuple[VllmRemoteSparseWeightSynchronizer, MagicMock, MagicMock]:
    init_refs, stream_refs, commit_refs = [MagicMock()], [MagicMock()], [MagicMock()]
    policy = MagicMock()
    policy.worker_group.workers = [object(), object()]
    policy.worker_group.run_all_workers_multiple_data.side_effect = [
        init_refs,
        stream_refs,
    ]
    policy.worker_group.run_all_workers_single_data.return_value = commit_refs

    generation = MagicMock()
    generation.worker_group.run_all_workers_single_data.return_value = [MagicMock()]
    generation.invalidate_kv_cache.return_value = True

    get_results: list[object] = [["http://receiver"]]
    if transport == "zmq":
        get_results.extend([["tcp://relay:19090"], [None]])
    get_results.extend([[{"weight": ((8,), "float32")}], stream_result])
    mock_ray.get.side_effect = get_results

    sync = VllmRemoteSparseWeightSynchronizer(policy, generation, transport=transport)
    with patch(f"{_MODULE}.post_vllm_refit_endpoints"):
        sync.init_communicator()
    return sync, policy, generation


def _valid_config() -> dict:
    return {
        "refit_transport": "vllm_s3_sparse",
        "delta_compression": {"encoding": "overwrite"},
        "vllm_cfg": {"precision": "bfloat16", "kv_cache_dtype": "auto"},
    }


def test_validate_remote_sparse_refit_accepts_supported_scope():
    config = _valid_config()
    assert (
        validate_vllm_remote_sparse_refit(
            config, colocated=False, megatron_enabled=True
        )
        == "s3"
    )
    assert config["delta_compression"] == VllmDeltaCompressionConfig(
        encoding="overwrite"
    )


@pytest.mark.parametrize(
    ("change", "kwargs"),
    [
        ({"refit_transport": "unknown"}, {}),
        ({}, {"colocated": True}),
        ({}, {"megatron_enabled": False}),
        ({"delta_compression": None}, {}),
        ({"quant_cfg": "fp8"}, {}),
        ({"vllm_cfg": {"precision": "fp8", "kv_cache_dtype": "auto"}}, {}),
        (
            {"vllm_cfg": {"precision": "bfloat16", "kv_cache_dtype": "fp8_e4m3"}},
            {},
        ),
    ],
)
def test_validate_remote_sparse_refit_rejects_unsupported_scope(change, kwargs):
    config = _valid_config()
    config.update(change)
    arguments = {"colocated": False, "megatron_enabled": True}
    arguments.update(kwargs)

    with pytest.raises(ValueError):
        validate_vllm_remote_sparse_refit(config, **arguments)


class TestVllmRemoteSparseWeightSynchronizer:
    def test_init_communicator_joins_prelaunched_baseline(self, mock_ray, post):
        baseline_ref = MagicMock()
        policy = MagicMock()
        generation = MagicMock()
        generation.worker_group.run_all_workers_single_data.return_value = [MagicMock()]
        mock_ray.get.side_effect = [
            ["http://receiver"],
            [{"weight": ((8,), "float32")}],
        ]
        sync = VllmRemoteSparseWeightSynchronizer(
            policy,
            generation,
            transport="s3",
            baseline_init_refs=[baseline_ref],
        )
        post.return_value = [{"overwrite_names": ["weight"]}]

        sync.init_communicator()

        policy.worker_group.run_all_workers_multiple_data.assert_not_called()
        mock_ray.get.assert_any_call([baseline_ref])
        post.assert_called_once_with(
            ["http://receiver/nemo-rl/refit/prepare"],
            {"tensors": {"weight": [[8], "float32"]}},
            api_key=None,
            timeout_s=600.0,
        )
        assert sync._baseline_init_refs == []
        assert sync._overwrite_names == ["weight"]

    def test_merge_refit_info_rejects_conflicting_metadata(self):
        with pytest.raises(ValueError, match="Conflicting sparse refit metadata"):
            VllmRemoteSparseWeightSynchronizer._merge_refit_info(
                [
                    {"weight": ((8,), "float32")},
                    {"weight": ((16,), "float32")},
                ]
            )

    def test_init_communicator_requires_receiver_endpoints(self, mock_ray):
        policy = MagicMock()
        policy.worker_group.workers = [object()]
        policy.worker_group.run_all_workers_multiple_data.return_value = [MagicMock()]
        generation = MagicMock()
        generation.worker_group.workers = [object()]
        generation.worker_group.run_all_workers_single_data.return_value = [MagicMock()]
        mock_ray.get.return_value = []
        sync = VllmRemoteSparseWeightSynchronizer(policy, generation, transport="s3")

        with pytest.raises(ValueError, match="endpoints are missing"):
            sync.init_communicator()

    def test_shutdown_cancels_pending_work_and_stops_zmq(self, mock_ray):
        policy = MagicMock()
        generation = MagicMock()
        generation.worker_group.workers = [object()]
        generation.worker_group.run_all_workers_single_data.return_value = [MagicMock()]
        sync = VllmRemoteSparseWeightSynchronizer(policy, generation, transport="zmq")
        init_ref, commit_ref = MagicMock(), MagicMock()
        sync._baseline_init_refs = [init_ref]
        sync._baseline_commit_refs = [commit_ref]
        sync._refit_urls = ["http://receiver"]
        sync._targets = ["tcp://relay"]
        sync._stale = False

        sync.mark_stale()
        sync.shutdown()

        assert mock_ray.cancel.call_count == 2
        mock_ray.cancel.assert_any_call(init_ref, force=False)
        mock_ray.cancel.assert_any_call(commit_ref, force=False)
        generation.worker_group.run_all_workers_single_data.assert_called_once_with(
            "stop_zmq_sparse_refit_relay",
            run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
        )
        assert sync.is_stale
        assert sync._baseline_init_refs == []
        assert sync._baseline_commit_refs == []
        assert sync._refit_urls == []
        assert sync._targets == []
        assert sync._overwrite_names == []

    def test_fails_before_transfer_when_kv_cache_invalidation_fails(self, mock_ray):
        policy = MagicMock()
        generation = MagicMock()
        generation.invalidate_kv_cache.return_value = False
        sync = VllmRemoteSparseWeightSynchronizer(policy, generation, transport="s3")

        with pytest.raises(RuntimeError, match="KV cache invalidation failed"):
            sync.sync_weights()
        policy.worker_group.run_all_workers_multiple_data.assert_not_called()

    def test_initializes_streams_commits_and_updates_baseline(
        self, mock_ray, post, capsys
    ):
        sync, policy, generation = _remote_sparse_sync(
            mock_ray,
            "zmq",
            [{"payloads": 3, "changed_elements": 3, "total_elements": 100}],
        )
        post.return_value = [
            {
                "verification_candidates": 4,
                "verification_samples": 4,
                "verification_exact_mismatches": 1,
                "verification_mismatches": 0,
                "verification_abs_sum": 1e-9,
                "verification_max_abs": 1e-9,
            }
        ]
        verification = post.return_value
        post.side_effect = [
            [{"ok": True, "payloads": 3, "receiver_relay_fanout_s": 1.0}],
            verification,
        ]
        metrics = sync.sync_weights()

        assert [
            entry.args[0]
            for entry in policy.worker_group.run_all_workers_multiple_data.call_args_list
        ] == ["init_remote_sparse_delta_baseline", "stream_remote_sparse_weights"]
        assert (
            policy.worker_group.run_all_workers_multiple_data.call_args_list[1].kwargs[
                "common_kwargs"
            ]["overwrite_names"]
            == []
        )
        assert [
            entry.args[0]
            for entry in generation.worker_group.run_all_workers_single_data.call_args_list
        ] == [
            "report_refit_server_base_url",
            "start_zmq_sparse_refit_relay",
            "configure_zmq_sparse_refit_relay",
        ]
        generation.worker_group.run_all_workers_single_data.assert_any_call(
            "start_zmq_sparse_refit_relay",
            run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
        )
        generation.worker_group.run_all_workers_single_data.assert_any_call(
            "configure_zmq_sparse_refit_relay",
            run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
            relay_addresses=["tcp://relay:19090"],
        )
        post.assert_any_call(
            ["http://receiver/nemo-rl/refit/zmq-flush"],
            {"transfer_id": ANY, "expected_payloads": 3},
            api_key=None,
            timeout_s=600.0,
        )
        post.assert_any_call(
            ["http://receiver/nemo-rl/refit/flush"],
            {},
            api_key=None,
            timeout_s=600.0,
        )
        policy.worker_group.run_all_workers_single_data.assert_called_once_with(
            "finish_remote_sparse_delta_sync", succeeded=True
        )
        assert (
            "REFIT_ZMQ_DELTA_CHANGE changed_elements=3 total_elements=100 "
            "changed_pct=3" in capsys.readouterr().out
        )
        assert metrics["delta/changed_pct"] == 3.0
        assert metrics["delta_verify/candidates"] == 4.0
        assert metrics["delta_verify/samples"] == 4.0
        assert metrics["delta_verify/exact_mismatches"] == 1.0
        assert metrics["delta_verify/mismatches"] == 0.0
        assert metrics["delta_verify/mean_abs"] == 2.5e-10
        assert metrics["delta_verify/max_abs"] == 1e-9
        assert metrics["transfer/payloads"] == 3.0
        assert metrics["transfer/relay_flush_s"] >= 0.0
        assert not sync.is_stale

    def test_sample_mismatch_does_not_commit_baseline(self, mock_ray, post):
        sync, policy, _ = _remote_sparse_sync(
            mock_ray,
            "zmq",
            [{"payloads": 3, "changed_elements": 3, "total_elements": 100}],
        )
        post.return_value = [
            {
                "verification_samples": 4,
                "verification_mismatches": 1,
                "verification_abs_sum": 0.5,
                "verification_max_abs": 0.5,
            }
        ]

        verification = post.return_value
        post.side_effect = [
            [{"ok": True, "payloads": 3}],
            verification,
            verification,
        ]
        with pytest.raises(RuntimeError, match="1 mismatched deltas out of 4"):
            sync.sync_weights()

        policy.worker_group.run_all_workers_single_data.assert_called_once_with(
            "finish_remote_sparse_delta_sync", succeeded=False
        )

    def test_failure_drains_receivers_without_committing_baseline(self, mock_ray, post):
        sync, policy, _ = _remote_sparse_sync(
            mock_ray, "s3", RuntimeError("stream failed")
        )

        with pytest.raises(RuntimeError, match="stream failed"):
            sync.sync_weights()

        post.assert_called_once_with(
            ["http://receiver/nemo-rl/refit/flush"],
            {},
            api_key=None,
            timeout_s=60.0,
        )
        policy.worker_group.run_all_workers_single_data.assert_called_once_with(
            "finish_remote_sparse_delta_sync", succeeded=False
        )

    def test_zmq_relay_failure_drains_before_rejecting_baseline(self, mock_ray, post):
        sync, policy, _ = _remote_sparse_sync(
            mock_ray,
            "zmq",
            [{"payloads": 3, "changed_elements": 3, "total_elements": 100}],
        )
        mock_ray.get.side_effect = [
            [{"payloads": 3, "changed_elements": 3, "total_elements": 100}],
        ]
        post.side_effect = [
            RuntimeError("fanout failed"),
            [{"ok": True, "payloads": 3}],
            [{"ok": True}],
        ]

        with pytest.raises(RuntimeError, match="fanout failed"):
            sync.sync_weights()

        assert [call.args[0][0] for call in post.call_args_list] == [
            "http://receiver/nemo-rl/refit/zmq-flush",
            "http://receiver/nemo-rl/refit/zmq-flush",
            "http://receiver/nemo-rl/refit/flush",
        ]
        policy.worker_group.run_all_workers_single_data.assert_called_once_with(
            "finish_remote_sparse_delta_sync", succeeded=False
        )
