# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spec derivation for the main-native Megatron reshard publisher.

Everything here runs on CPU with no MX server, no NIXL and no Megatron: the
translation from a classified publish set to MX alias inputs is pure geometry and
name resolution, which is the part that is wrong silently. A bad shard range
publishes an address that reads real but wrong bytes, and a bad HF name publishes
one expert's weights under another's.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

pytest.importorskip(
    "modelexpress",
    reason="ModelExpress is an optional integration dependency",
)

from nemo_rl.distributed.mx_reshard_config import (
    DEFAULT_PUBLISHER_LISTEN_PORT_BASE,
    DEFAULT_RECEIVER_LISTEN_PORT_BASE,
    maybe_preinit_mx_reshard_nixl,
    resolve_mx_reshard_listen_port_bases,
    resolve_mx_reshard_publisher_listen_port_base,
    resolve_mx_reshard_receiver_listen_port_base,
    validate_mx_reshard_listen_port_ranges,
)
from nemo_rl.distributed.mx_megatron_helpers import (
    ROLE_EXPERT_COLUMN,
    ROLE_GATED_MLP_COLUMN,
    ROLE_QKV_COLUMN,
    ROLE_REPLICATED,
    ROLE_ROW,
    MegatronRoleSpec,
)
from nemo_rl.distributed.mx_reshard_publisher import (
    _gated_mlp_extras,
    PLACEMENT_REPLICATE,
    PLACEMENT_SHARD,
    MxMegatronPublisher,
    UnmappedMegatronTensor,
    build_bridge_name_map,
    build_megatron_alias_inputs,
    make_bridge_resolver,
    publish_megatron_hf_aliases,
    published_byte_count,
)


def _entry(name, tensor, role, extras=None, **spec_kwargs):
    """One (name, tensor, spec, full_extras) tuple as the collector yields it."""
    descriptor_extras = dict(extras or {})
    spec = MegatronRoleSpec(
        role=role, descriptor_extras=descriptor_extras, **spec_kwargs
    )
    # The collector merges the mesh position into the per-tensor extras; the
    # alias builder only reads the geometry keys, but carry both so the fixture
    # matches the real shape of the input.
    full = {"megatron_role": role, "tp_size": "1", "ep_size": "8"}
    full.update(descriptor_extras)
    return name, tensor, spec, full


def test_publisher_step_is_forwarded_to_mx_rendezvous_encoder():
    entry = _entry("norm.weight", torch.zeros(16), ROLE_REPLICATED)
    resolver = make_bridge_resolver({"norm.weight": ["model.norm.weight"]})
    items = list(
        build_megatron_alias_inputs(
            [entry], resolve_hf_names=resolver, tp_size=1, tp_rank=0
        )
    )
    manager = MagicMock(agent_name="trainer-r0", nixl_metadata=b"metadata")
    rendezvous = MagicMock()
    rendezvous.publish.return_value = "source-id"

    with patch(
        "nemo_rl.distributed.mx_reshard_publisher.wrap_rendezvous_blob",
        return_value=b"stamped",
    ) as wrap:
        source_id, _ = publish_megatron_hf_aliases(
            manager=manager,
            rendezvous=rendezvous,
            items=items,
            metadata_endpoint="trainer:5555",
            publisher_step=17,
        )

    assert source_id == "source-id"
    assert wrap.call_args.kwargs["publisher_step"] == 17
    rendezvous.publish.assert_called_once_with(b"stamped")


def test_bridge_map_uses_local_pp_name_and_local_tasks_only():
    local = SimpleNamespace(
        param_name="decoder.layers.0.self_attention.linear_qkv.weight",
        global_param_name="decoder.layers.24.self_attention.linear_qkv.weight",
        param_weight=torch.zeros(1),
        mapping=SimpleNamespace(
            hf_param={
                "k": "model.layers.24.self_attn.k_proj.weight",
                "q": "model.layers.24.self_attn.q_proj.weight",
                "v": "model.layers.24.self_attn.v_proj.weight",
            }
        ),
    )
    remote_pp = SimpleNamespace(
        param_name="decoder.layers.0.input_layernorm.weight",
        global_param_name="decoder.layers.0.input_layernorm.weight",
        param_weight=None,
        mapping=SimpleNamespace(hf_param="model.layers.0.input_layernorm.weight"),
    )

    name_map = build_bridge_name_map([local, remote_pp])

    assert name_map == {
        "decoder.layers.0.self_attention.linear_qkv.weight": (
            "model.layers.24.self_attn.q_proj.weight",
            "model.layers.24.self_attn.k_proj.weight",
            "model.layers.24.self_attn.v_proj.weight",
        )
    }


def test_bridge_map_rejects_conflicting_local_names():
    def task(hf_name):
        return SimpleNamespace(
            param_name="decoder.layers.0.weight",
            global_param_name="decoder.layers.0.weight",
            param_weight=torch.zeros(1),
            mapping=SimpleNamespace(hf_param=hf_name),
        )

    with pytest.raises(ValueError, match="conflicting Bridge mappings"):
        build_bridge_name_map([task("model.layers.0.a"), task("model.layers.0.b")])


def test_default_publisher_and_receiver_port_ranges_do_not_overlap():
    publisher, receiver = validate_mx_reshard_listen_port_ranges(
        {},
        train_world_size=8,
        inference_world_size=8,
    )

    assert publisher == DEFAULT_PUBLISHER_LISTEN_PORT_BASE
    assert receiver == DEFAULT_RECEIVER_LISTEN_PORT_BASE
    assert publisher + 7 < receiver


def test_legacy_port_base_falls_back_to_shifted_receiver_range():
    assert resolve_mx_reshard_listen_port_bases({"listen_port_base": 7000}) == (
        7000,
        17000,
    )


def test_role_specific_port_bases_forward_to_the_correct_side():
    config = {
        "publisher_listen_port_base": 19000,
        "receiver_listen_port_base": 29000,
    }
    assert resolve_mx_reshard_publisher_listen_port_base(config) == 19000
    assert resolve_mx_reshard_receiver_listen_port_base(config) == 29000


def test_mx_reshard_preinitializes_nixl_only_for_selected_transport(monkeypatch):
    from nemo_rl.utils.checkpoint_engines import nixl

    calls = []
    agent = object()
    monkeypatch.setattr(
        nixl,
        "preinit_nixl_agent",
        lambda **kwargs: calls.append(kwargs) or agent,
    )

    assert maybe_preinit_mx_reshard_nixl({}) is None
    assert (
        maybe_preinit_mx_reshard_nixl({"generation": {"refit_transport": "mx_reshard"}})
        is agent
    )
    assert calls == [{}]


def test_overlapping_explicit_port_ranges_are_rejected():
    with pytest.raises(ValueError, match="listen port ranges overlap"):
        validate_mx_reshard_listen_port_ranges(
            {
                "publisher_listen_port_base": 19000,
                "receiver_listen_port_base": 19004,
            },
            train_world_size=8,
            inference_world_size=8,
        )


def test_publisher_is_constructed_with_the_expected_mx_arguments():
    item = SimpleNamespace(name="weight", tensor=torch.zeros(1))
    manager = MagicMock()
    client = MagicMock()
    rendezvous = MagicMock()
    with (
        patch(
            "modelexpress.nixl_transfer.NixlTransferManager",
            return_value=manager,
        ) as manager_cls,
        patch("modelexpress.client.MxClient", return_value=client) as client_cls,
        patch(
            "modelexpress.refit.reshard.rendezvous.MxReshardRendezvous",
            return_value=rendezvous,
        ) as rendezvous_cls,
    ):
        publisher = MxMegatronPublisher(
            items=[item],
            model_name="Qwen/Qwen3-30B",
            server_url="mx.example:8001",
            rank=11,
            device_id=3,
            listen_port=19011,
            metadata_endpoint="trainer-1:19011",
        )

    manager_cls.assert_called_once_with(
        agent_name="nemo-rl-trainer-11",
        device_id=3,
        listen_port=19011,
    )
    manager.initialize.assert_called_once()
    manager.register_tensors.assert_called_once_with({"weight": item.tensor})
    client_cls.assert_called_once_with(server_url="mx.example:8001")
    rendezvous_cls.assert_called_once_with(
        client,
        role="trainer",
        rank=11,
        model_name="Qwen/Qwen3-30B",
        worker_id="nemo-rl-trainer-11",
    )

    publisher.shutdown()
    rendezvous.close.assert_called_once()
    manager.shutdown.assert_called_once()
    client.close.assert_called_once()


def test_dp_replica_ranks_receive_unique_worker_ids():
    item = SimpleNamespace(name="weight", tensor=torch.zeros(1))
    worker_ids = []

    def rendezvous(*_args, **kwargs):
        worker_ids.append(kwargs["worker_id"])
        return MagicMock()

    with (
        patch("modelexpress.nixl_transfer.NixlTransferManager"),
        patch("modelexpress.client.MxClient"),
        patch(
            "modelexpress.refit.reshard.rendezvous.MxReshardRendezvous",
            side_effect=rendezvous,
        ),
    ):
        first = MxMegatronPublisher(
            items=[item],
            model_name="model",
            server_url="mx:8001",
            rank=0,
            device_id=0,
            listen_port=19000,
            metadata_endpoint="host:19000",
        )
        second = MxMegatronPublisher(
            items=[item],
            model_name="model",
            server_url="mx:8001",
            rank=8,
            device_id=0,
            listen_port=19008,
            metadata_endpoint="host:19008",
        )

    assert worker_ids == ["nemo-rl-trainer-0", "nemo-rl-trainer-8"]
    first.shutdown()
    second.shutdown()


# --- placement -------------------------------------------------------------


def test_replicated_tensor_publishes_its_whole_shape():
    entry = _entry(
        "decoder.layers.0.input_layernorm.weight", torch.zeros(16), ROLE_REPLICATED
    )
    resolver = make_bridge_resolver(
        {
            "decoder.layers.0.input_layernorm.weight": [
                "model.layers.0.input_layernorm.weight"
            ]
        }
    )

    (item,) = build_megatron_alias_inputs(
        [entry], resolve_hf_names=resolver, tp_size=1, tp_rank=0
    )

    assert item.placement_kind == PLACEMENT_REPLICATE
    assert item.global_shape == (16,)
    assert item.shard_axis is None
    assert item.local_shard_range is None
    assert item.hf_names == ("model.layers.0.input_layernorm.weight",)


def test_tp_sharded_column_carries_its_global_shape_and_range():
    """TP2 rank 1 of a column-parallel weight owns the second row band."""
    entry = _entry(
        "decoder.layers.0.mlp.linear_fc2.weight", torch.zeros(8, 32), ROLE_ROW
    )
    resolver = make_bridge_resolver(
        {
            "decoder.layers.0.mlp.linear_fc2.weight": [
                "model.layers.0.mlp.down_proj.weight"
            ]
        }
    )

    (item,) = build_megatron_alias_inputs(
        [entry], resolve_hf_names=resolver, tp_size=2, tp_rank=1
    )

    # ROLE_ROW shards axis 1.
    assert item.placement_kind == PLACEMENT_SHARD
    assert item.shard_axis == 1
    assert item.global_shape == (8, 64)
    assert item.local_shard_range == (32, 64)


def test_tp1_leaves_every_tensor_whole():
    """The Topology B publisher runs TP1/ETP1, so nothing is TP-sharded."""
    entries = [
        _entry("a.weight", torch.zeros(4, 8), ROLE_ROW),
        _entry("b.weight", torch.zeros(4, 8), ROLE_GATED_MLP_COLUMN),
    ]
    resolver = make_bridge_resolver({"a.weight": ["hf.a"], "b.weight": ["hf.b"]})

    items = list(
        build_megatron_alias_inputs(
            entries, resolve_hf_names=resolver, tp_size=1, tp_rank=0
        )
    )

    assert [item.placement_kind for item in items] == [PLACEMENT_REPLICATE] * 2
    assert all(item.local_shard_range is None for item in items)


def test_expert_tp_geometry_is_taken_from_the_expert_mesh():
    """An expert tensor shards on the expert-TP mesh, not the dense TP mesh."""
    entry = _entry(
        "decoder.layers.0.mlp.experts.linear_fc2.weight0",
        torch.zeros(4, 16),
        ROLE_EXPERT_COLUMN,
        extras={"expert_layout": "grouped", "expert_id": "0", "local_expert_id": "0"},
        is_expert=True,
    )
    resolver = make_bridge_resolver(
        {
            "decoder.layers.0.mlp.experts.linear_fc2.weight0": [
                "model.layers.0.mlp.experts.0.down_proj.weight"
            ]
        }
    )

    (item,) = build_megatron_alias_inputs(
        [entry],
        resolve_hf_names=resolver,
        tp_size=4,
        tp_rank=3,
        expert_tp_size=2,
        expert_tp_rank=1,
    )

    # Grouped expert_column shards axis 0, and the extent comes from ETP=2 not TP=4.
    assert item.shard_axis == 0
    assert item.global_shape == (8, 16)
    assert item.local_shard_range == (4, 8)


# --- expert name resolution ------------------------------------------------


def test_global_expert_id_is_substituted_into_the_hf_name():
    """EP rank 1's first local expert is global expert 4 and must publish as such.

    The Bridge inspects one rank's module tree, so its map is keyed on the
    EP-local leaf (`weight0`) and its HF names describe expert 0. Publishing
    those names unchanged would have every EP rank overwrite rank 0's experts.
    """
    resolver = make_bridge_resolver(
        {
            "decoder.layers.0.mlp.experts.linear_fc1.weight0": [
                "model.layers.0.mlp.experts.0.gate_proj.weight"
            ]
        }
    )

    names = resolver(
        "decoder.layers.0.mlp.experts.linear_fc1.weight4",
        {"expert_id": "4", "local_expert_id": "0"},
    )

    assert names == ("model.layers.0.mlp.experts.4.gate_proj.weight",)


def test_exact_key_wins_over_the_local_expert_retry():
    """A map already keyed on the global name is used as-is."""
    resolver = make_bridge_resolver(
        {
            "e.linear_fc1.weight4": ["model.layers.0.mlp.experts.4.gate_proj.weight"],
            "e.linear_fc1.weight0": ["model.layers.0.mlp.experts.0.gate_proj.weight"],
        }
    )

    names = resolver("e.linear_fc1.weight4", {"expert_id": "4", "local_expert_id": "0"})

    assert names == ("model.layers.0.mlp.experts.4.gate_proj.weight",)


def test_expert_substitution_rewrites_every_hf_name_of_a_fused_pair():
    """A fused gate/up expert tensor maps to two HF names, both needing the id."""
    resolver = make_bridge_resolver(
        {
            "e.linear_fc1.weight0": [
                "model.layers.0.mlp.experts.0.gate_proj.weight",
                "model.layers.0.mlp.experts.0.up_proj.weight",
            ]
        }
    )

    names = resolver("e.linear_fc1.weight7", {"expert_id": "7", "local_expert_id": "0"})

    assert names == (
        "model.layers.0.mlp.experts.7.gate_proj.weight",
        "model.layers.0.mlp.experts.7.up_proj.weight",
    )


def test_substitution_only_touches_the_expert_index():
    """A layer number that equals the old expert id must not be rewritten."""
    resolver = make_bridge_resolver(
        {"e.linear_fc1.weight3": ["model.layers.3.mlp.experts.3.gate_proj.weight"]}
    )

    names = resolver("e.linear_fc1.weight5", {"expert_id": "5", "local_expert_id": "3"})

    assert names == ("model.layers.3.mlp.experts.5.gate_proj.weight",)


def test_unmapped_tensor_raises_instead_of_being_skipped():
    """A silently dropped source becomes a coverage shortfall far from its cause."""
    resolver = make_bridge_resolver({"known.weight": ["hf.known"]})

    with pytest.raises(UnmappedMegatronTensor, match="unknown.weight"):
        resolver("unknown.weight", {})


def test_non_strict_resolver_reports_no_names_and_the_tensor_is_dropped():
    entry = _entry("unknown.weight", torch.zeros(4), ROLE_REPLICATED)
    resolver = make_bridge_resolver({"known.weight": ["hf.known"]}, strict=False)

    items = list(
        build_megatron_alias_inputs(
            [entry], resolve_hf_names=resolver, tp_size=1, tp_rank=0
        )
    )

    assert items == []


# --- the extras contract that build_hf_aliases depends on -----------------


def test_qkv_head_extras_survive_translation():
    """Global QKV geometry must reach MX unchanged when KV heads are below TP."""
    entry = _entry(
        "decoder.layers.0.self_attention.linear_qkv.weight",
        torch.zeros(1088, 2048),
        ROLE_QKV_COLUMN,
        extras={
            "qkv_interleave": "by_head",
            "head_dim": "128",
            "num_heads": "64",
            "num_kv_heads": "2",
        },
    )
    resolver = make_bridge_resolver(
        {
            "decoder.layers.0.self_attention.linear_qkv.weight": [
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.k_proj.weight",
                "model.layers.0.self_attn.v_proj.weight",
            ]
        }
    )

    (item,) = build_megatron_alias_inputs(
        [entry], resolve_hf_names=resolver, tp_size=8, tp_rank=3
    )

    assert item.extras["head_dim"] == "128"
    assert item.extras["num_heads"] == "64"
    assert item.extras["num_kv_heads"] == "2"
    assert "num_kv_heads_local" not in item.extras
    assert item.global_shape == (8704, 2048)
    assert item.local_shard_range == (3264, 4352)
    assert len(item.hf_names) == 3


def test_qkv_names_keep_q_k_v_order():
    """build_hf_aliases assigns hf_names[0..2] to the Q, K and V row bands, so a
    reordering here transposes the projections with no error anywhere."""
    hf = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
    ]
    resolver = make_bridge_resolver({"qkv": hf})

    assert resolver("qkv", {}) == tuple(hf)


def test_gated_order_extra_survives_translation():
    entry = _entry(
        "decoder.layers.0.mlp.linear_fc1.weight",
        torch.zeros(64, 32),
        ROLE_GATED_MLP_COLUMN,
        extras={"gated_mlp_order": "gate_then_up"},
    )
    resolver = make_bridge_resolver(
        {
            "decoder.layers.0.mlp.linear_fc1.weight": [
                "model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.mlp.up_proj.weight",
            ]
        }
    )

    (item,) = build_megatron_alias_inputs(
        [entry], resolve_hf_names=resolver, tp_size=1, tp_rank=0
    )

    assert item.extras["gated_mlp_order"] == "gate_then_up"


# --- end-to-end against the real MX alias builder -------------------------


def test_translated_items_are_accepted_by_build_hf_aliases():
    """The contract test: MX main's builder consumes what we produce.

    Runs the real `build_hf_aliases`, so a drift in either the role vocabulary or
    the extras keys fails here rather than on a cluster.
    """
    from modelexpress.refit.reshard.megatron_aliases import build_hf_aliases

    entries = [
        _entry("norm.weight", torch.zeros(16, dtype=torch.bfloat16), ROLE_REPLICATED),
        _entry(
            "mlp.linear_fc1.weight",
            torch.zeros(64, 32, dtype=torch.bfloat16),
            ROLE_GATED_MLP_COLUMN,
            extras={"gated_mlp_order": "gate_then_up"},
        ),
    ]
    resolver = make_bridge_resolver(
        {
            "norm.weight": ["model.norm.weight"],
            "mlp.linear_fc1.weight": [
                "model.mlp.gate_proj.weight",
                "model.mlp.up_proj.weight",
            ],
        }
    )

    items = list(
        build_megatron_alias_inputs(
            entries, resolve_hf_names=resolver, tp_size=1, tp_rank=0
        )
    )
    published = build_hf_aliases(items, agent_name="trainer-r0")

    # The fused gate/up parent becomes two HF tensors, so three in total.
    assert sorted(tensor.name for tensor in published) == [
        "model.mlp.gate_proj.weight",
        "model.mlp.up_proj.weight",
        "model.norm.weight",
    ]
    for tensor in published:
        assert tensor.shards
        for shard in tensor.shards:
            assert shard.agent_name == "trainer-r0"


def test_publish_reports_its_shard_and_byte_counts(capsys):
    """The synchronizer times the publish half but not what it described, so a
    publish that slowed down cannot be told from one that grew without this."""
    item = SimpleNamespace(name="weight", tensor=torch.zeros(1))
    published = [SimpleNamespace(shards=[SimpleNamespace(shape=(8,))], elsize=2)]
    with (
        patch("modelexpress.nixl_transfer.NixlTransferManager"),
        patch("modelexpress.client.MxClient"),
        patch("modelexpress.refit.reshard.rendezvous.MxReshardRendezvous"),
    ):
        publisher = MxMegatronPublisher(
            items=[item],
            model_name="model",
            server_url="mx:8001",
            rank=11,
            device_id=0,
            listen_port=19011,
            metadata_endpoint="host:19011",
        )
    with patch(
        "nemo_rl.distributed.mx_reshard_publisher.publish_megatron_hf_aliases",
        return_value=("source-1", published),
    ):
        publisher.publish(7)

    record = json.loads(capsys.readouterr().out.split("MX_PUBLISH_PHASE ", 1)[1])
    assert record["step"] == 7
    assert record["rank"] == 11
    assert record["tensors"] == 1
    assert record["shards"] == 1
    assert record["bytes"] == 8 * 2
    publisher.shutdown()


def test_publish_telemetry_cannot_fail_a_successful_publish():
    """Reporting is never worth a failed publish: the bytes are already out."""
    item = SimpleNamespace(name="weight", tensor=torch.zeros(1))
    with (
        patch("modelexpress.nixl_transfer.NixlTransferManager"),
        patch("modelexpress.client.MxClient"),
        patch("modelexpress.refit.reshard.rendezvous.MxReshardRendezvous"),
    ):
        publisher = MxMegatronPublisher(
            items=[item],
            model_name="model",
            server_url="mx:8001",
            rank=0,
            device_id=0,
            listen_port=19000,
            metadata_endpoint="host:19000",
        )
    # A payload whose shards cannot be counted at all.
    with patch(
        "nemo_rl.distributed.mx_reshard_publisher.publish_megatron_hf_aliases",
        return_value=("source-1", [SimpleNamespace()]),
    ):
        publisher.publish(3)

    publisher.shutdown()


def test_published_byte_count_sums_shard_boxes():
    from modelexpress.refit.reshard.megatron_aliases import build_hf_aliases

    entry = _entry(
        "norm.weight", torch.zeros(16, dtype=torch.bfloat16), ROLE_REPLICATED
    )
    resolver = make_bridge_resolver({"norm.weight": ["model.norm.weight"]})
    items = list(
        build_megatron_alias_inputs(
            [entry], resolve_hf_names=resolver, tp_size=1, tp_rank=0
        )
    )

    published = build_hf_aliases(items, agent_name="trainer-r0")

    assert published_byte_count(published) == 16 * 2


# --- whole-model name coverage --------------------------------------------
#
# Qwen3-30B-A3B-Instruct-2507's real geometry. The expected HF name set below is
# generated from these constants rather than shipped as a fixture, and the
# generator was checked against the checkpoint's own
# model.safetensors.index.json on 2026-08-12: 18,867 names, exact match, nothing
# missing and nothing extra.
QWEN3_30B_A3B = {
    "layers": 48,
    "experts": 128,
    "tie_word_embeddings": False,
}


def _expected_hf_names(
    layers: int, experts: int, tie_word_embeddings: bool
) -> set[str]:
    """The tensor names the HF checkpoint actually contains."""
    names = {"model.embed_tokens.weight", "model.norm.weight"}
    if not tie_word_embeddings:
        names.add("lm_head.weight")
    for layer in range(layers):
        head = f"model.layers.{layer}"
        names.update(
            {
                f"{head}.input_layernorm.weight",
                f"{head}.post_attention_layernorm.weight",
                f"{head}.mlp.gate.weight",
                f"{head}.self_attn.q_proj.weight",
                f"{head}.self_attn.k_proj.weight",
                f"{head}.self_attn.v_proj.weight",
                f"{head}.self_attn.o_proj.weight",
                f"{head}.self_attn.q_norm.weight",
                f"{head}.self_attn.k_norm.weight",
            }
        )
        for expert in range(experts):
            stem = f"{head}.mlp.experts.{expert}"
            names.update(
                {
                    f"{stem}.gate_proj.weight",
                    f"{stem}.up_proj.weight",
                    f"{stem}.down_proj.weight",
                }
            )
    return names


def _bridge_map_for_rank(layers: int, local_experts: int) -> dict[str, list[str]]:
    """A Bridge map as seen from ONE EP rank.

    The Bridge walks one rank's module tree, so grouped-expert keys carry the
    **EP-local** leaf index and the HF names it returns describe that local
    expert. Every rank's map therefore looks identical and describes experts
    0..local_experts-1 -- which is exactly why the resolver has to substitute the
    global id, and why a map keyed on the global name would be the easier thing
    to test and the wrong thing to test.
    """
    name_map: dict[str, list[str]] = {
        "embedding.word_embeddings.weight": ["model.embed_tokens.weight"],
        "decoder.final_layernorm.weight": ["model.norm.weight"],
        "output_layer.weight": ["lm_head.weight"],
    }
    for layer in range(layers):
        megatron = f"decoder.layers.{layer}"
        hf = f"model.layers.{layer}"
        name_map[f"{megatron}.self_attention.linear_qkv.weight"] = [
            f"{hf}.self_attn.q_proj.weight",
            f"{hf}.self_attn.k_proj.weight",
            f"{hf}.self_attn.v_proj.weight",
        ]
        name_map[f"{megatron}.self_attention.linear_proj.weight"] = [
            f"{hf}.self_attn.o_proj.weight"
        ]
        name_map[f"{megatron}.self_attention.q_layernorm.weight"] = [
            f"{hf}.self_attn.q_norm.weight"
        ]
        name_map[f"{megatron}.self_attention.k_layernorm.weight"] = [
            f"{hf}.self_attn.k_norm.weight"
        ]
        name_map[f"{megatron}.input_layernorm.weight"] = [
            f"{hf}.input_layernorm.weight"
        ]
        name_map[f"{megatron}.pre_mlp_layernorm.weight"] = [
            f"{hf}.post_attention_layernorm.weight"
        ]
        name_map[f"{megatron}.mlp.router.weight"] = [f"{hf}.mlp.gate.weight"]
        for local in range(local_experts):
            name_map[f"{megatron}.mlp.experts.linear_fc1.weight{local}"] = [
                f"{hf}.mlp.experts.{local}.gate_proj.weight",
                f"{hf}.mlp.experts.{local}.up_proj.weight",
            ]
            name_map[f"{megatron}.mlp.experts.linear_fc2.weight{local}"] = [
                f"{hf}.mlp.experts.{local}.down_proj.weight"
            ]
    return name_map


def _publish_whole_model(ep_size: int) -> dict[str, set[int]]:
    """Resolve every rank's publish set; return hf name -> owning EP ranks."""
    layers = QWEN3_30B_A3B["layers"]
    local_experts = QWEN3_30B_A3B["experts"] // ep_size
    owners: dict[str, set[int]] = {}

    for ep_rank in range(ep_size):
        resolve = make_bridge_resolver(_bridge_map_for_rank(layers, local_experts))
        entries: list[tuple[str, dict[str, str]]] = [
            (key, {})
            for key in _bridge_map_for_rank(layers, local_experts)
            if ".experts.linear_fc" not in key
        ]
        for layer in range(layers):
            megatron = f"decoder.layers.{layer}"
            for local in range(local_experts):
                global_id = ep_rank * local_experts + local
                for fused in ("linear_fc1", "linear_fc2"):
                    entries.append(
                        (
                            f"{megatron}.mlp.experts.{fused}.weight{global_id}",
                            {
                                "expert_id": str(global_id),
                                "local_expert_id": str(local),
                                "expert_layout": "grouped",
                            },
                        )
                    )
        for name, extras in entries:
            for hf_name in resolve(name, extras):
                owners.setdefault(hf_name, set()).add(ep_rank)
    return owners


def test_ep8_publishes_every_checkpoint_tensor_exactly_once_per_owner():
    """Whole-model coverage: the union of 8 EP ranks is the checkpoint, exactly.

    A name the fleet never publishes is a coverage shortfall the receiver reports
    far from its cause; a name it publishes that the model does not have is an
    install into a buffer nothing owns.
    """
    owners = _publish_whole_model(ep_size=8)
    expected = _expected_hf_names(**QWEN3_30B_A3B)

    assert set(owners) - expected == set(), "published names absent from the checkpoint"
    assert expected - set(owners) == set(), "checkpoint tensors nobody publishes"
    assert len(owners) == 18867


def test_every_expert_has_exactly_one_owner_under_ep8():
    """EP partitions experts, so two ranks claiming one expert means the global id
    substitution failed and one rank is publishing another's weights."""
    owners = _publish_whole_model(ep_size=8)

    experts = {name: rank for name, rank in owners.items() if ".experts." in name}
    assert len(experts) == 48 * 128 * 3
    assert {len(rank) for rank in experts.values()} == {1}


def test_non_expert_tensors_are_published_by_every_rank():
    """Not a defect: with TP1 the collector keeps replicated tensors on all ranks,
    so the fleet offers 8 byte-identical copies and the receiver's merge picks one.
    Recorded because it is the DP amplification c2 removes, and because a change
    here silently changes the wire bytes of every measurement."""
    owners = _publish_whole_model(ep_size=8)

    non_expert = {
        name: rank for name, rank in owners.items() if ".experts." not in name
    }
    assert len(non_expert) == 435
    assert {len(rank) for rank in non_expert.values()} == {8}

    # 18,432 expert offers + 435 x 8 replicated offers.
    assert sum(len(rank) for rank in owners.values()) == 21912


# --------------------------------------------------------- gated MLP fusion order
# MX assigns the two halves of a fused gate/up parameter to hf_names positionally
# and refuses to infer which half is which, because getting it wrong publishes the
# gate's bytes under the up projection's name with every digest agreeing.


def test_fused_expert_gate_up_is_stamped_gate_then_up():
    extras = _gated_mlp_extras(
        "decoder.layers.0.mlp.experts.linear_fc1.weight0",
        "expert_column",
        (
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.0.mlp.experts.0.up_proj.weight",
        ),
    )
    assert extras == {"gated_mlp_order": "gate_then_up"}


def test_dense_gated_mlp_is_stamped_too():
    extras = _gated_mlp_extras(
        "decoder.layers.0.mlp.linear_fc1.weight",
        "gated_mlp_column",
        ("model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.up_proj.weight"),
    )
    assert extras == {"gated_mlp_order": "gate_then_up"}


def test_reversed_hf_name_order_is_refused_not_guessed():
    with pytest.raises(ValueError, match="do not read as"):
        _gated_mlp_extras(
            "decoder.layers.0.mlp.experts.linear_fc1.weight0",
            "expert_column",
            (
                "model.layers.0.mlp.experts.0.up_proj.weight",
                "model.layers.0.mlp.experts.0.gate_proj.weight",
            ),
        )


def test_unfused_roles_are_not_stamped():
    # linear_fc2 maps to one HF name, so there is nothing to order; stamping it
    # anyway would assert a layout claim about a tensor that has no halves.
    assert (
        _gated_mlp_extras(
            "decoder.layers.0.mlp.experts.linear_fc2.weight0",
            "expert_row",
            ("model.layers.0.mlp.experts.0.down_proj.weight",),
        )
        == {}
    )
    assert (
        _gated_mlp_extras(
            "decoder.layers.0.self_attention.linear_qkv.weight",
            "qkv_column",
            ("q.weight", "k.weight", "v.weight"),
        )
        == {}
    )
