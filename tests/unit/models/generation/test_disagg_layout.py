"""The backend-agnostic prefill/decode layout helpers shared by TRT-LLM and vLLM."""

import pytest

from nemo_rl.models.generation.disagg_layout import (
    engine_owner_indices,
    plan_engines,
    replica_width,
    resolve_disagg_layout,
    resolve_role_overrides,
    slice_engines_on_unified_pg,
    tile_engines_on_per_node_pgs,
)
from nemo_rl.models.generation.interfaces import DisaggConfig

_ALLOWED = frozenset({"tensor_parallel_size", "gpu_memory_utilization", "vllm_kwargs"})
_EXCLUDED = frozenset(
    {"prefill_engine", "decode_engine", "disagg_frontend", "kv_transfer"}
)


def _config(disaggregation=None, **vllm_overrides):
    vllm_cfg = {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "gpu_memory_utilization": 0.6,
        "precision": "bfloat16",
    }
    vllm_cfg.update(vllm_overrides)
    cfg = {"backend": "vllm", "vllm_cfg": vllm_cfg}
    if disaggregation is not None:
        cfg["disaggregation"] = disaggregation
    return cfg


def _resolve(cfg):
    return resolve_role_overrides(
        cfg,
        backend_cfg_key="vllm_cfg",
        allowed_keys=_ALLOWED,
        excluded_keys=_EXCLUDED,
        kwargs_key="vllm_kwargs",
    )


# -------------------------------------------------------------------------- #
#  Role overrides
# -------------------------------------------------------------------------- #


@pytest.mark.parametrize("disaggregation", [None, {"enabled": False}])
def test_role_blocks_are_inert_and_the_config_untouched_without_disaggregation(
    disaggregation,
):
    cfg = _config(disaggregation=disaggregation, prefill_engine={"precision": "x"})
    assert _resolve(cfg) is cfg


def test_a_role_block_resolves_to_its_whole_engine_config():
    cfg = _config(
        disaggregation={"enabled": True},
        tensor_parallel_size=2,
        prefill_engine={"tensor_parallel_size": 4, "gpu_memory_utilization": 0.5},
        disagg_frontend={"prefill_router": "conversation"},
    )
    resolved = _resolve(cfg)
    prefill = resolved["vllm_cfg"]["prefill_engine"]
    decode = resolved["vllm_cfg"]["decode_engine"]
    assert prefill["tensor_parallel_size"] == 4
    assert prefill["gpu_memory_utilization"] == 0.5
    assert prefill["precision"] == "bfloat16"
    assert decode["tensor_parallel_size"] == 2
    assert decode["gpu_memory_utilization"] == 0.6
    for block in (prefill, decode):
        assert not _EXCLUDED & set(block)
    # The input is not mutated.
    assert "decode_engine" not in cfg["vllm_cfg"]


def test_role_kwargs_merge_key_by_key_over_the_shared_ones():
    cfg = _config(
        disaggregation={"enabled": True},
        decode_engine={
            "vllm_kwargs": {"max_num_seqs": 512, "compilation_config": {"level": 3}}
        },
    )
    cfg["vllm_kwargs"] = {
        "compilation_config": {"backend": "eager", "level": 0},
        "enable_chunked_prefill": True,
    }
    resolved = _resolve(cfg)
    assert resolved["vllm_cfg"]["decode_engine"]["vllm_kwargs"] == {
        "compilation_config": {"backend": "eager", "level": 3},
        "enable_chunked_prefill": True,
        "max_num_seqs": 512,
    }
    assert resolved["vllm_cfg"]["prefill_engine"]["vllm_kwargs"] == cfg["vllm_kwargs"]
    assert cfg["vllm_kwargs"]["compilation_config"] == {"backend": "eager", "level": 0}


def test_a_role_key_outside_the_allowed_set_is_rejected():
    with pytest.raises(ValueError, match=r"vllm_cfg\.prefill_engine may only override"):
        _resolve(
            _config(
                disaggregation={"enabled": True}, prefill_engine={"precision": "fp8"}
            )
        )


def test_resolving_twice_is_the_same_as_once():
    cfg = _config(
        disaggregation={"enabled": True}, prefill_engine={"tensor_parallel_size": 2}
    )
    once = _resolve(cfg)
    assert _resolve(once) == once


# -------------------------------------------------------------------------- #
#  Replica width and engine planning
# -------------------------------------------------------------------------- #


def test_replica_width_counts_pp_and_both_roles():
    cfg = _resolve(
        _config(
            disaggregation={
                "enabled": True,
                "num_prefill_engines": 1,
                "num_decode_engines": 2,
            },
            pipeline_parallel_size=1,
            prefill_engine={"tensor_parallel_size": 2},
        )
    )
    assert replica_width(cfg, backend_cfg_key="vllm_cfg") == 4
    assert (
        replica_width(
            _config(tensor_parallel_size=2, pipeline_parallel_size=2),
            backend_cfg_key="vllm_cfg",
        )
        == 4
    )


@pytest.mark.parametrize(
    "layout, world, aggregated, prefill, decode, roles, widths, replicas",
    [
        (DisaggConfig(), 8, 1, 1, 1, ["aggregated"] * 8, [1] * 8, 8),
        (
            DisaggConfig(enabled=True),
            8,
            1,
            1,
            1,
            ["prefill", "decode"] * 4,
            [1, 1] * 4,
            4,
        ),
        (
            DisaggConfig(enabled=True, num_prefill_engines=2, num_decode_engines=1),
            8,
            1,
            1,
            2,
            ["prefill", "prefill", "decode"] * 2,
            [1, 1, 2] * 2,
            2,
        ),
    ],
)
def test_plan_engines_lays_replicas_out_contiguously(
    layout, world, aggregated, prefill, decode, roles, widths, replicas
):
    got = plan_engines(
        world,
        layout=layout,
        aggregated_width=aggregated,
        prefill_width=prefill,
        decode_width=decode,
        backend_cfg_key="vllm_cfg",
    )
    assert got == (roles, widths, replicas)


def test_plan_engines_rejects_a_replica_width_that_does_not_tile_the_cluster():
    with pytest.raises(AssertionError, match="replica width 3 GPUs"):
        plan_engines(
            8,
            layout=DisaggConfig(enabled=True),
            aggregated_width=1,
            prefill_width=2,
            decode_width=1,
            backend_cfg_key="vllm_cfg",
        )


def test_engine_owners_are_prefix_sums_of_the_widths():
    assert engine_owner_indices([4, 2, 2, 4, 2, 2]) == [0, 4, 6, 8, 12, 14]


def test_resolve_disagg_layout_reads_absent_as_disabled():
    assert resolve_disagg_layout({}).enabled is False
    assert resolve_disagg_layout({"disaggregation": DisaggConfig(enabled=True)}).enabled


# -------------------------------------------------------------------------- #
#  Bundle tiling
# -------------------------------------------------------------------------- #


def test_per_node_tiling_packs_whole_replicas_in_topology_order():
    tied = tile_engines_on_per_node_pgs(
        pg_bundle_counts=[8, 8],
        pg_order=[1, 0],
        engine_widths=[4, 2, 2, 4, 2, 2],
        replica_width_gpus=8,
    )
    assert tied == [
        (1, [0, 1, 2, 3]),
        (1, [4, 5]),
        (1, [6, 7]),
        (0, [0, 1, 2, 3]),
        (0, [4, 5]),
        (0, [6, 7]),
    ]


def test_per_node_tiling_refuses_to_split_an_engine_across_nodes():
    # 1xTP4 + 1xTP2 = 6 of 8 GPUs; the next engine (TP4) does not fit.
    with pytest.raises(ValueError, match="does not tile placement group 0"):
        tile_engines_on_per_node_pgs(
            pg_bundle_counts=[8, 8],
            pg_order=[0, 1],
            engine_widths=[4, 2, 4, 2],
            replica_width_gpus=6,
        )


def test_per_node_tiling_skips_empty_groups():
    tied = tile_engines_on_per_node_pgs(
        pg_bundle_counts=[0, 4],
        pg_order=[0, 1],
        engine_widths=[2, 2],
        replica_width_gpus=4,
    )
    assert tied == [(1, [0, 1]), (1, [2, 3])]


def test_unified_pg_slicing_consumes_the_sorted_bundles_per_engine_width():
    assert slice_engines_on_unified_pg([7, 3, 5, 1, 0, 2], [4, 2]) == [
        (0, [7, 3, 5, 1]),
        (0, [0, 2]),
    ]
    with pytest.raises(ValueError, match="needs 6 bundles"):
        slice_engines_on_unified_pg([0, 1, 2, 3], [4, 2])
