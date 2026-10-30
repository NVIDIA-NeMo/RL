# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from pydantic import ValidationError

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import DisaggConfig
from nemo_rl.models.generation.trtllm import trtllm_generation
from nemo_rl.models.generation.trtllm.config import (
    TrtllmDisaggServerConfig,
    replica_width,
    role_engine_config,
    resolve_disagg_layout,
    resolve_disagg_server_config,
    resolve_role_overrides,
)
from nemo_rl.models.generation.trtllm.trtllm_generation import TrtllmGeneration

pytestmark = pytest.mark.trtllm


def _config(disaggregation=None, **trtllm_overrides):
    trtllm_cfg = {
        "tensor_parallel_size": 1,
        "max_model_len": 128,
        "precision": "bfloat16",
        "max_batch_size": 8,
        "max_num_tokens": 256,
        "async_engine": True,
    }
    trtllm_cfg.update(trtllm_overrides)
    return {
        "backend": "trtllm",
        "model_name": "test/model",
        "max_new_tokens": 8,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "val_temperature": 1.0,
        "val_top_p": 1.0,
        "val_top_k": None,
        "stop_token_ids": None,
        "stop_strings": None,
        "_pad_token_id": 0,
        "colocated": {
            "enabled": True,
            "resources": {"gpus_per_node": None, "num_nodes": None},
        },
        "trtllm_cfg": trtllm_cfg,
        **({} if disaggregation is None else {"disaggregation": disaggregation}),
    }


class _Cluster:
    def __init__(self, *, world_size=2, gpus_per_node=2):
        self._world_size = world_size
        self.num_gpus_per_node = gpus_per_node
        self._init_placement_groups = MagicMock()

    def world_size(self):
        return self._world_size


def _bare_generation(*, colocated=True, dp_size=2, **trtllm_overrides):
    generation = TrtllmGeneration.__new__(TrtllmGeneration)
    generation.cfg = _config(**trtllm_overrides)
    generation.cfg["colocated"]["enabled"] = colocated
    generation.colocated_enabled = colocated
    generation.current_generate_dp_shard_idx = 0
    generation.worker_group = MagicMock()
    generation.worker_group.dp_size = dp_size
    generation.worker_group.workers = [object()] * dp_size
    # Aggregated shape: a replica is one engine, one worker wide, so engine i
    # is owned by worker i. __init__ derives these; these tests bypass it.
    generation._disagg_cfg = resolve_disagg_layout(generation.cfg)
    generation._disagg_server_cfg = resolve_disagg_server_config(generation.cfg)
    generation.num_replicas = dp_size
    generation.num_engines = dp_size
    generation._engine_tps = [1] * dp_size
    generation._engine_owner_indices = list(range(dp_size))
    generation._replica_owner_indices = list(range(dp_size))
    return generation


@pytest.mark.parametrize(
    ("cluster", "config_update", "error"),
    [
        (_Cluster(world_size=3), {"tensor_parallel_size": 2}, "must be divisible"),
        (
            _Cluster(world_size=4),
            {
                "tensor_parallel_size": 4,
                "moe_tensor_parallel_size": 2,
                "moe_expert_parallel_size": 1,
            },
            "must equal tensor_parallel_size",
        ),
        (_Cluster(), {"async_engine": False}, "requires trtllm_cfg.async_engine=true"),
        (
            _Cluster(world_size=4, gpus_per_node=2),
            {"tensor_parallel_size": 4},
            "only supported for non-colocated",
        ),
    ],
)
def test_invalid_async_engine_configuration_fails_before_worker_start(
    cluster, config_update, error
):
    with pytest.raises(AssertionError, match=error):
        TrtllmGeneration(cluster, _config(**config_update))


def _placement_config(tp_size: int, *, colocated: bool = False) -> dict:
    config = _config(tensor_parallel_size=tp_size, pipeline_parallel_size=1)
    config["colocated"]["enabled"] = colocated
    return config


def test_init_cluster_placement_groups_uses_unified_pg_for_cross_node_tp():
    cluster = MagicMock(num_gpus_per_node=4)

    TrtllmGeneration.init_cluster_placement_groups(cluster, _placement_config(8))

    cluster._init_placement_groups.assert_called_once_with(
        strategy="PACK",
        use_unified_pg=True,
    )


def test_init_cluster_placement_groups_uses_per_node_pgs_for_node_local_tp():
    cluster = MagicMock(num_gpus_per_node=4)

    TrtllmGeneration.init_cluster_placement_groups(cluster, _placement_config(4))

    cluster._init_placement_groups.assert_called_once_with(
        strategy="PACK",
        use_unified_pg=False,
    )


def test_init_cluster_placement_groups_rejects_colocated_cross_node_tp():
    cluster = MagicMock(num_gpus_per_node=4)

    with pytest.raises(AssertionError, match="only supported for non-colocated"):
        TrtllmGeneration.init_cluster_placement_groups(
            cluster, _placement_config(8, colocated=True)
        )

    cluster._init_placement_groups.assert_not_called()


def test_cross_node_tp_replicas_use_unified_placement_group():
    generation = TrtllmGeneration.__new__(TrtllmGeneration)
    generation.model_parallel_size = 8
    # Bundles are handed out engine by engine, so the widths are what the
    # slicing follows -- two engines of 8 across the 16-bundle unified PG.
    generation._engine_tps = [8, 8]
    generation.worker_group = MagicMock()

    unified_pg = MagicMock()
    bundle_to_node = {i: f"node-{i // 4}" for i in range(16)}
    cluster = MagicMock()
    cluster.get_placement_groups.return_value = [unified_pg]
    cluster._sorted_bundle_indices = list(range(16))

    with patch.object(
        trtllm_generation.ray.util,
        "placement_group_table",
        return_value={"bundles_to_node_id": bundle_to_node},
    ):
        result = generation._get_tied_worker_bundle_indices(cluster)

    assert result == [
        (0, list(range(8))),
        (0, list(range(8, 16))),
    ]


@pytest.mark.asyncio
async def test_generate_async_dispatches_round_robin_and_returns_leader_index():
    generation = _bare_generation(dp_size=2)
    # Dispatch follows _engine_owner_indices, not the worker group's DP
    # leaders: the grid cannot model engines of differing width. Non-identity
    # owners so a regression to worker_idx == engine_idx would show.
    generation._engine_owner_indices = [3, 7]

    async def worker_result():
        return BatchedDataDict(
            {
                "output_ids": torch.tensor([[1, 2, 3]]),
                "logprobs": torch.zeros((1, 3)),
                "generation_lengths": torch.tensor([1]),
                "unpadded_sequence_lengths": torch.tensor([3]),
            }
        )

    generation.worker_group.run_single_worker_single_data.side_effect = lambda **_: (
        worker_result()
    )
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2]]),
            "input_lengths": torch.tensor([2]),
        }
    )

    first = [item async for item in generation.generate_async(data, greedy=True)]
    second = [item async for item in generation.generate_async(data)]

    assert first[0][0] == 0
    assert first[0][1]["gen_leader_worker_idx"] == [3]
    assert second[0][1]["gen_leader_worker_idx"] == [7]
    assert generation.current_generate_dp_shard_idx == 0
    calls = generation.worker_group.run_single_worker_single_data.call_args_list
    assert calls[0].kwargs["worker_idx"] == 3
    assert calls[0].kwargs["greedy"] is True
    assert calls[1].kwargs["worker_idx"] == 7


@pytest.mark.asyncio
async def test_generate_async_validates_input_shape_and_empty_batch():
    generation = _bare_generation()

    missing_lengths = BatchedDataDict({"input_ids": torch.tensor([[1]])})
    with pytest.raises(AssertionError, match="input_ids and input_lengths"):
        async for _ in generation.generate_async(missing_lengths):
            pass

    multi_sample = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1], [2]]),
            "input_lengths": torch.tensor([1, 1]),
        }
    )
    with pytest.raises(AssertionError, match="single-sample"):
        async for _ in generation.generate_async(multi_sample):
            pass

    empty = BatchedDataDict(
        {
            "input_ids": torch.empty((0, 0), dtype=torch.long),
            "input_lengths": torch.empty(0, dtype=torch.long),
        }
    )
    assert [item async for item in generation.generate_async(empty)] == []
    generation.worker_group.run_single_worker_single_data.assert_not_called()


@pytest.mark.asyncio
async def test_generate_async_surfaces_timeout(monkeypatch):
    generation = _bare_generation()
    generation.worker_group.get_dp_leader_worker_idx.return_value = 0

    async def pending_result():
        await asyncio.Future()

    generation.worker_group.run_single_worker_single_data.return_value = (
        pending_result()
    )

    async def raise_timeout(awaitable, timeout):
        awaitable.close()
        assert timeout == 12.5
        raise asyncio.TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", raise_timeout)
    monkeypatch.setenv("NRL_TRTLLM_ASYNC_TIMEOUT_SECONDS", "12.5")
    data = BatchedDataDict(
        {"input_ids": torch.tensor([[1]]), "input_lengths": torch.tensor([1])}
    )

    with pytest.raises(RuntimeError, match="timed out after 12.5s"):
        async for _ in generation.generate_async(data):
            pass


@pytest.mark.parametrize(
    ("colocated", "prepare_method", "finish_method"),
    [
        (True, "wake_up_async", "sleep_async"),
        (False, None, "reset_prefix_cache_async"),
    ],
)
def test_generation_lifecycle_routes_by_colocation(
    monkeypatch, colocated, prepare_method, finish_method
):
    generation = _bare_generation(colocated=colocated, dp_size=2)
    run_one = generation.worker_group.run_single_worker_single_data
    run_one.side_effect = lambda **_: True
    monkeypatch.setattr(trtllm_generation.ray, "get", lambda values: values)

    assert generation.prepare_for_generation(tags=["weights"]) is True
    if prepare_method is None:
        run_one.assert_not_called()
    else:
        # One call per engine, each addressed to that engine's owner.
        assert [c.kwargs["method_name"] for c in run_one.call_args_list] == [
            prepare_method
        ] * 2
        assert [c.kwargs["worker_idx"] for c in run_one.call_args_list] == [0, 1]
        assert all(c.kwargs["tags"] == ["weights"] for c in run_one.call_args_list)
        assert not generation.worker_group.run_all_workers_single_data.called

    run_one.reset_mock()
    assert generation.finish_generation() is True
    assert [c.kwargs["method_name"] for c in run_one.call_args_list] == [
        finish_method
    ] * 2
    assert [c.kwargs["worker_idx"] for c in run_one.call_args_list] == [0, 1]


def test_prepare_refit_info_skips_missing_metadata_and_dispatches_present_metadata(
    monkeypatch,
):
    generation = _bare_generation()
    generation._engine_owner_indices = [3, 7]
    run_one = generation.worker_group.run_single_worker_single_data
    ray_get = MagicMock()
    monkeypatch.setattr(trtllm_generation.ray, "get", ray_get)

    assert generation.prepare_refit_info(None) is None
    run_one.assert_not_called()
    ray_get.assert_not_called()

    state_dict_info = {"weight": {"shape": [2, 2]}}
    futures = [SimpleNamespace(), SimpleNamespace()]
    run_one.side_effect = futures

    assert generation.prepare_refit_info(state_dict_info) is None
    assert [call.kwargs for call in run_one.call_args_list] == [
        {
            "method_name": "prepare_refit_info_async",
            "worker_idx": worker_idx,
            "state_dict_info": state_dict_info,
        }
        for worker_idx in [3, 7]
    ]
    ray_get.assert_called_once_with(futures)


@pytest.mark.parametrize(
    ("in_flight", "recompute_kv", "expected_drain"),
    [(False, False, True), (True, False, False), (True, True, False)],
)
def test_collective_refit_forwards_async_update_policy(
    in_flight, recompute_kv, expected_drain
):
    generation = _bare_generation(
        colocated=False,
        dp_size=2,
        in_flight_weight_updates=in_flight,
        recompute_kv_cache_after_weight_updates=recompute_kv,
    )
    refs = [SimpleNamespace(), SimpleNamespace()]
    run_one = generation.worker_group.run_single_worker_single_data
    run_one.side_effect = refs

    assert generation.update_weights_from_collective() == refs
    # One call per engine: drain and recompute_kv are the same for all of them,
    # but the owner is not, so the fan-out has to be per engine.
    assert [c.kwargs["method_name"] for c in run_one.call_args_list] == [
        "update_weights_from_collective_async"
    ] * 2
    assert [c.kwargs["worker_idx"] for c in run_one.call_args_list] == [0, 1]
    for call in run_one.call_args_list:
        assert call.kwargs["drain"] is expected_drain
        assert call.kwargs["recompute_kv"] is recompute_kv


def test_ipc_refit_and_missing_worker_group():
    generation = _bare_generation(colocated=True, dp_size=2)
    refs = [SimpleNamespace(), SimpleNamespace()]
    run_one = generation.worker_group.run_single_worker_single_data
    run_one.side_effect = refs

    assert generation.update_weights_via_ipc_zmq() == refs
    assert [c.kwargs["method_name"] for c in run_one.call_args_list] == [
        "update_weights_via_ipc_zmq_async"
    ] * 2
    assert [c.kwargs["worker_idx"] for c in run_one.call_args_list] == [0, 1]

    broken = _bare_generation(colocated=True)
    broken.worker_group.workers = []
    with pytest.raises(RuntimeError, match="Worker group not initialised"):
        broken.update_weights_via_ipc_zmq()


# -------------------------------------------------------------------------- #
#  Disaggregation config and engine planning
# -------------------------------------------------------------------------- #


def test_disagg_defaults_come_from_the_schema_not_the_call_sites():
    # Absent and present-but-empty must agree: every default lives on the model,
    # so no consumer has to re-derive one per key.
    for raw in (None, {}):
        cfg = _config(disaggregation=raw) if raw is not None else _config()
        layout = resolve_disagg_layout(cfg)
        assert layout.enabled is False
        assert layout.num_prefill_engines == 1
        assert layout.num_decode_engines == 1
        assert layout.num_frontend_workers == 1
        assert layout.frontend_tokenize is False
        assert layout.frontend_base_port == 8300

        server = resolve_disagg_server_config(cfg)
        assert server.ctx_router == "conversation"
        assert server.gen_router == "load_balancing"
        assert server.gen_tokids_ctxbytes is False
        assert server.gen_strip_message_history is False

        assert "prefill_engine" not in cfg["trtllm_cfg"]


def test_the_layout_and_the_server_block_carry_disjoint_keys():
    # The two schemas describe different things, so neither should be able to
    # declare the other's keys -- a layout key under disagg_server would read as
    # configuration while nothing applied it.
    assert not set(DisaggConfig.model_fields) & set(
        TrtllmDisaggServerConfig.model_fields
    )


def test_a_layout_key_under_disagg_server_is_rejected():
    with pytest.raises(ValueError, match="generation.disaggregation"):
        resolve_disagg_server_config(_config(disagg_server={"num_prefill_engines": 2}))


def test_an_unforwarded_disagg_server_key_is_rejected():
    # build_config forwards only the declared fields, so anything else would
    # validate and then silently have no effect.
    with pytest.raises(ValidationError, match="max_retries"):
        resolve_disagg_server_config(_config(disagg_server={"max_retries": 5}))


def test_a_plausible_but_wrong_router_is_rejected():
    # round_robin parses as a str but silently discards the prefix affinity the
    # prefill engines depend on, so the schema has to reject it.
    with pytest.raises(ValidationError):
        resolve_disagg_server_config(
            _config(disagg_server={"ctx_router": "round_robin"})
        )


def test_unknown_disagg_keys_are_preserved_for_older_configs():
    layout = resolve_disagg_layout(
        _config(disaggregation={"enabled": True, "some_future_key": 3})
    )
    assert layout.enabled is True
    assert layout.model_extra["some_future_key"] == 3


@pytest.mark.parametrize(
    "trtllm_overrides, world_size, expected_roles, expected_tps, expected_replicas",
    [
        # Disabled: a replica is an engine, so the count follows TP as before.
        ({}, 8, ["aggregated"] * 8, [1] * 8, 8),
        # Enabled, 1:1 at TP1 -- the exemplar's defaults on one 8-GPU node.
        (
            {"disaggregation": {"enabled": True}},
            8,
            ["prefill", "decode"] * 4,
            [1, 1] * 4,
            4,
        ),
        # Per-role TP and engine counts are independent, and same-role engines
        # stay contiguous within a replica. The role overrides sit beside the
        # shared engine config, not under disaggregation.
        (
            {
                "disaggregation": {
                    "enabled": True,
                    "num_prefill_engines": 2,
                    "num_decode_engines": 1,
                },
                "prefill_engine": {"tensor_parallel_size": 1},
                "decode_engine": {"tensor_parallel_size": 2},
            },
            8,
            ["prefill", "prefill", "decode"] * 2,
            [1, 1, 2] * 2,
            2,
        ),
    ],
)
def test_plan_engines_lays_replicas_out_contiguously(
    trtllm_overrides, world_size, expected_roles, expected_tps, expected_replicas
):
    cfg = resolve_role_overrides(_config(**trtllm_overrides))
    generation = TrtllmGeneration.__new__(TrtllmGeneration)
    generation.cfg = cfg
    generation.tp_size = cfg["trtllm_cfg"]["tensor_parallel_size"]
    generation._disagg_cfg = resolve_disagg_layout(cfg)

    roles, tps, num_replicas = generation._plan_engines(world_size)

    assert roles == expected_roles
    assert tps == expected_tps
    assert num_replicas == expected_replicas


@pytest.mark.parametrize("as_model", [False, True], ids=["dict", "DisaggConfig"])
def test_a_disaggregated_replica_spans_all_of_its_engines(as_model):
    # MasterConfig(**cfg) validates the layout into a DisaggConfig before
    # grpo.setup() sizes NVLink-domain placement with this, so both shapes
    # must work.
    layout = {"enabled": True, "num_prefill_engines": 1, "num_decode_engines": 2}
    cfg = resolve_role_overrides(
        _config(
            disaggregation=DisaggConfig.model_validate(layout) if as_model else layout,
            prefill_engine={"tensor_parallel_size": 2},
        )
    )

    # One TP2 prefill engine plus two TP1 decode engines.
    assert replica_width(cfg) == 4


def test_an_aggregated_replica_is_one_engine():
    assert replica_width(_config(tensor_parallel_size=2)) == 2


def test_plan_engines_rejects_a_replica_width_that_does_not_tile_the_cluster():
    cfg = resolve_role_overrides(
        _config(
            disaggregation={"enabled": True},
            prefill_engine={"tensor_parallel_size": 2},
            decode_engine={"tensor_parallel_size": 1},
        )
    )
    generation = TrtllmGeneration.__new__(TrtllmGeneration)
    generation.cfg = cfg
    generation.tp_size = cfg["trtllm_cfg"]["tensor_parallel_size"]
    generation._disagg_cfg = resolve_disagg_layout(cfg)

    with pytest.raises(AssertionError, match="replica width 3 GPUs"):
        generation._plan_engines(8)


# -------------------------------------------------------------------------- #
#  Role overrides
# -------------------------------------------------------------------------- #


def _disagg_config(**trtllm_overrides):
    return _config(disaggregation={"enabled": True}, **trtllm_overrides)


def test_a_role_block_resolves_to_its_whole_engine_config():
    cfg = resolve_role_overrides(
        _disagg_config(
            tensor_parallel_size=2,
            prefill_engine={"tensor_parallel_size": 4, "max_batch_size": 2},
            disagg_server={"ctx_router": "conversation"},
        )
    )
    prefill = cfg["trtllm_cfg"]["prefill_engine"]
    decode = cfg["trtllm_cfg"]["decode_engine"]

    # The role's keys win; everything else comes from the shared block.
    assert prefill["tensor_parallel_size"] == 4
    assert prefill["max_batch_size"] == 2
    assert prefill["precision"] == "bfloat16"
    assert decode["tensor_parallel_size"] == 2
    assert decode["max_batch_size"] == 8
    # Neither the sibling role nor the disagg server is part of an engine.
    for block in (prefill, decode):
        assert not {"prefill_engine", "decode_engine", "disagg_server"} & set(block)


def test_role_trtllm_kwargs_merge_key_by_key_over_the_shared_ones():
    cfg = _disagg_config(
        decode_engine={
            "trtllm_kwargs": {
                "kv_cache_config": {"max_tokens": 4096},
                "cache_transceiver_config": {"kv_transfer_timeout_ms": 600000},
            }
        }
    )
    cfg["trtllm_kwargs"] = {
        "kv_cache_config": {"enable_block_reuse": True, "enable_partial_reuse": False},
        "cache_transceiver_config": {"backend": "NIXL"},
        "enable_chunked_prefill": True,
    }

    resolved = resolve_role_overrides(cfg)

    assert resolved["trtllm_cfg"]["decode_engine"]["trtllm_kwargs"] == {
        "kv_cache_config": {
            "enable_block_reuse": True,
            "enable_partial_reuse": False,
            "max_tokens": 4096,
        },
        "cache_transceiver_config": {
            "backend": "NIXL",
            "kv_transfer_timeout_ms": 600000,
        },
        "enable_chunked_prefill": True,
    }
    assert (
        resolved["trtllm_cfg"]["prefill_engine"]["trtllm_kwargs"]
        == cfg["trtllm_kwargs"]
    )
    # The shared block itself is left alone.
    assert cfg["trtllm_kwargs"]["kv_cache_config"] == {
        "enable_block_reuse": True,
        "enable_partial_reuse": False,
    }


def test_a_role_key_outside_the_engine_knobs_is_rejected():
    with pytest.raises(ValueError, match=r"prefill_engine may only override"):
        resolve_role_overrides(_disagg_config(prefill_engine={"precision": "float16"}))


def test_resolving_twice_is_the_same_as_once():
    cfg = _disagg_config(prefill_engine={"tensor_parallel_size": 2})
    cfg["trtllm_kwargs"] = {"kv_cache_config": {"enable_partial_reuse": False}}

    once = resolve_role_overrides(cfg)

    assert resolve_role_overrides(once) == once


@pytest.mark.parametrize("disaggregation", [None, {"enabled": False}])
def test_role_blocks_are_left_as_written_without_disaggregation(disaggregation):
    # Inert without disaggregation, so not even validated.
    cfg = _config(disaggregation=disaggregation, prefill_engine={"precision": "x"})

    assert resolve_role_overrides(cfg) is cfg


class _LoadFormatTokenizer:
    pad_token_id = 0
    eos_token_id = 1


def test_configure_generation_config_selects_trtllm_load_format() -> None:
    """Training starts on dummy weights; the initial refit supplies the real ones."""
    from nemo_rl.models.generation import configure_generation_config

    training = configure_generation_config(_config(), _LoadFormatTokenizer())
    assert training["trtllm_cfg"]["load_format"] == "dummy"

    evaluation = configure_generation_config(
        _config(), _LoadFormatTokenizer(), is_eval=True
    )
    assert evaluation["trtllm_cfg"]["load_format"] == "auto"


def test_configure_generation_config_forces_dummy_for_fp8() -> None:
    """An FP8 engine has no loadable checkpoint: the BF16 refit builds its weights."""
    from nemo_rl.models.generation import configure_generation_config

    for is_eval in (False, True):
        configured = configure_generation_config(
            _config(precision="fp8"), _LoadFormatTokenizer(), is_eval=is_eval
        )
        assert configured["trtllm_cfg"]["load_format"] == "dummy"


@pytest.mark.parametrize("precision", ["bf16", "fp8"])
def test_configure_generation_config_load_format_reaches_both_disagg_engines(
    precision,
) -> None:
    """Both role blocks are copied from the shared block, so they must inherit it."""
    from nemo_rl.models.generation import configure_generation_config

    configured = configure_generation_config(
        _disagg_config(precision=precision), _LoadFormatTokenizer()
    )
    for role in ("prefill", "decode"):
        assert role_engine_config(configured, role)["load_format"] == "dummy"
