from types import SimpleNamespace

import pytest
import torch

from nemo_rl.distributed.mx_megatron_helpers import (
    ROLE_EXPERT_COLUMN,
    ROLE_EXPERT_ROW,
    ROLE_QKV_COLUMN,
    canonicalize_grouped_expert_name,
    collect_megatron_publish_set,
    detect_megatron_role,
    infer_megatron_tp_shard_geometry,
    resolve_qkv_geometry_from_param,
)


class ReplicatedOnlyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))


class ModuleWrapper(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module


class ColumnParallelLinear(torch.nn.Module):
    def __init__(self, rows: int, q_heads: int, kv_heads: int, head_dim: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(rows, 16))
        self.config = SimpleNamespace(
            num_attention_heads=q_heads,
            num_query_groups=kv_heads,
            kv_channels=head_dim,
        )


class SelfAttention(torch.nn.Module):
    def __init__(self, rows: int, q_heads: int, kv_heads: int, head_dim: int):
        super().__init__()
        self.linear_qkv = ColumnParallelLinear(
            rows, q_heads, kv_heads, head_dim
        )


class TransformerLayer(torch.nn.Module):
    def __init__(self, rows: int, q_heads: int, kv_heads: int, head_dim: int):
        super().__init__()
        self.self_attention = SelfAttention(rows, q_heads, kv_heads, head_dim)


class HeterogeneousAttentionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                TransformerLayer(1088, 64, 2, 128),
                TransformerLayer(384, 32, 8, 64),
            ]
        )


def _published_names(*, tp_rank: int) -> list[str]:
    model = ReplicatedOnlyModule()
    published = collect_megatron_publish_set(
        model,
        tp_size=2,
        pp_size=1,
        pp_rank=0,
        ep_size=1,
        ep_rank=0,
        tp_rank=tp_rank,
    )
    return [name for name, _, _, _ in published]


def test_collect_megatron_publish_set_skips_replicated_on_nonzero_tp_rank():
    assert _published_names(tp_rank=1) == []


def test_collect_megatron_publish_set_publishes_replicated_on_zero_tp_rank():
    assert _published_names(tp_rank=0) == ["weight"]


def test_collect_megatron_publish_set_strips_all_leading_module_wrappers():
    model = ModuleWrapper(ModuleWrapper(ReplicatedOnlyModule()))

    published = collect_megatron_publish_set(
        model,
        tp_size=1,
        pp_size=1,
        pp_rank=0,
        ep_size=1,
        ep_rank=0,
        tp_rank=0,
    )

    assert [name for name, _, _, _ in published] == ["weight"]


def test_grouped_expert_column_records_tp_geometry():
    geometry = infer_megatron_tp_shard_geometry(
        local_shape=(4864, 2048),
        role=ROLE_EXPERT_COLUMN,
        tp_size=4,
        tp_rank=2,
        expert_tp_size=4,
        expert_tp_rank=2,
        descriptor_extras={"expert_layout": "grouped"},
    )

    assert geometry is not None
    assert geometry.global_shape == (19456, 2048)
    assert geometry.shard_axis == 0
    assert geometry.local_shard_range == (9728, 14592)


def test_grouped_expert_row_records_tp_geometry():
    geometry = infer_megatron_tp_shard_geometry(
        local_shape=(2048, 2432),
        role=ROLE_EXPERT_ROW,
        tp_size=4,
        tp_rank=1,
        expert_tp_size=4,
        expert_tp_rank=1,
        descriptor_extras={"expert_layout": "grouped"},
    )

    assert geometry is not None
    assert geometry.global_shape == (2048, 9728)
    assert geometry.shard_axis == 1
    assert geometry.local_shard_range == (2432, 4864)


def test_leading_axis_expert_geometry_keeps_expert_axis():
    column = infer_megatron_tp_shard_geometry(
        local_shape=(4, 4864, 2048),
        role=ROLE_EXPERT_COLUMN,
        tp_size=4,
        tp_rank=3,
        expert_tp_size=4,
        expert_tp_rank=3,
        descriptor_extras={"expert_layout": "leading_axis"},
    )
    row = infer_megatron_tp_shard_geometry(
        local_shape=(4, 2048, 2432),
        role=ROLE_EXPERT_ROW,
        tp_size=4,
        tp_rank=3,
        expert_tp_size=4,
        expert_tp_rank=3,
        descriptor_extras={"expert_layout": "leading_axis"},
    )

    assert column is not None and row is not None
    assert column.global_shape == (4, 19456, 2048)
    assert column.shard_axis == 1
    assert column.local_shard_range == (14592, 19456)
    assert row.global_shape == (4, 2048, 9728)
    assert row.shard_axis == 2
    assert row.local_shard_range == (7296, 9728)


def test_etp1_expert_is_not_mislabeled_as_trainer_tp_sharded():
    geometry = infer_megatron_tp_shard_geometry(
        local_shape=(19456, 2048),
        role=ROLE_EXPERT_COLUMN,
        tp_size=2,
        tp_rank=1,
        expert_tp_size=1,
        expert_tp_rank=0,
        descriptor_extras={"expert_layout": "grouped"},
    )

    assert geometry is None


def test_grouped_expert_publish_name_uses_global_id():
    assert (
        canonicalize_grouped_expert_name(
            "decoder.layers.3.mlp.experts.linear_fc1.weight0",
            {
                "expert_layout": "grouped",
                "local_expert_id": "0",
                "expert_id": "64",
            },
        )
        == "decoder.layers.3.mlp.experts.linear_fc1.weight64"
    )


def test_qkv_descriptor_carries_global_heads_when_kv_heads_are_below_tp():
    model = HeterogeneousAttentionModel()
    name = "layers.0.self_attention.linear_qkv.weight"

    spec = detect_megatron_role(
        name,
        model.layers[0].self_attention.linear_qkv.weight,
        model=model,
        tp_size=8,
        ep_size=1,
        ep_rank=0,
        qkv_geometry=(64, 2, 128),
    )

    assert spec.role == ROLE_QKV_COLUMN
    assert spec.descriptor_extras == {
        "qkv_interleave": "by_head",
        "num_heads": "64",
        "num_kv_heads": "2",
        "head_dim": "128",
    }


def test_divisible_qkv_descriptor_retains_legacy_local_head_fields():
    model = HeterogeneousAttentionModel()
    name = "layers.1.self_attention.linear_qkv.weight"

    spec = detect_megatron_role(
        name,
        model.layers[1].self_attention.linear_qkv.weight,
        model=model,
        tp_size=8,
        ep_size=1,
        ep_rank=0,
        qkv_geometry=(32, 8, 64),
    )

    assert spec.descriptor_extras == {
        "qkv_interleave": "by_head",
        "num_heads": "32",
        "num_kv_heads": "8",
        "head_dim": "64",
        "num_heads_local": "4",
        "num_kv_heads_local": "1",
    }


def test_collect_reads_per_layer_geometry_from_the_live_qkv_module():
    model = HeterogeneousAttentionModel()

    published = list(
        collect_megatron_publish_set(
            model,
            tp_size=8,
            pp_size=1,
            pp_rank=0,
            ep_size=1,
            ep_rank=0,
            tp_rank=3,
            qkv_geometry_resolver=resolve_qkv_geometry_from_param,
        )
    )

    assert len(published) == 2
    extras_by_name = {name: extras for name, _, _, extras in published}
    assert extras_by_name[
        "layers.0.self_attention.linear_qkv.weight"
    ]["num_kv_heads"] == "2"
    assert extras_by_name[
        "layers.1.self_attention.linear_qkv.weight"
    ]["num_kv_heads"] == "8"
    assert "num_kv_heads_local" not in extras_by_name[
        "layers.0.self_attention.linear_qkv.weight"
    ]
    assert extras_by_name[
        "layers.1.self_attention.linear_qkv.weight"
    ]["num_kv_heads_local"] == "1"


def test_malformed_qkv_geometry_fails_closed():
    model = HeterogeneousAttentionModel()
    with pytest.raises(ValueError, match="invalid global Q/KV geometry"):
        detect_megatron_role(
            "layers.0.self_attention.linear_qkv.weight",
            model.layers[0].self_attention.linear_qkv.weight,
            model=model,
            tp_size=8,
            ep_size=1,
            ep_rank=0,
            qkv_geometry=(63, 2, 128),
        )


def test_qkv_geometry_must_match_the_fused_weight_rows():
    model = HeterogeneousAttentionModel()
    with pytest.raises(ValueError, match="fused QKV rows"):
        detect_megatron_role(
            "layers.0.self_attention.linear_qkv.weight",
            model.layers[0].self_attention.linear_qkv.weight,
            model=model,
            tp_size=8,
            ep_size=1,
            ep_rank=0,
            qkv_geometry=(64, 4, 128),
        )
