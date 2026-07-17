import torch

from nemo_rl.distributed.mx_megatron_helpers import (
    ROLE_EXPERT_COLUMN,
    ROLE_QKV_COLUMN,
    collect_megatron_publish_set,
)


class ReplicatedOnlyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))


class ColumnParallelLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(6, 2))


class TEColumnParallelGroupedLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight0 = torch.nn.Parameter(torch.ones(4, 2))


class AttentionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_qkv = ColumnParallelLinear()


class ExpertModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = TEColumnParallelGroupedLinear()


def _published_names(*, tp_rank: int) -> list[str]:
    model = ReplicatedOnlyModule()
    published = collect_megatron_publish_set(
        model,
        tp_size=2,
        ep_size=1,
        ep_rank=0,
        tp_rank=tp_rank,
    )
    return [name for name, _, _ in published]


def test_collect_megatron_publish_set_skips_replicated_on_nonzero_tp_rank():
    assert _published_names(tp_rank=1) == []


def test_collect_megatron_publish_set_publishes_replicated_on_zero_tp_rank():
    assert _published_names(tp_rank=0) == ["weight"]


def test_collect_megatron_publish_set_classifies_fused_qkv():
    published = list(
        collect_megatron_publish_set(
            AttentionModule(),
            tp_size=2,
            ep_size=1,
            ep_rank=0,
            tp_rank=0,
            num_attention_heads=8,
            num_kv_heads=4,
            head_dim=16,
        )
    )

    name, _, spec = published[0]
    assert name == "linear_qkv.weight"
    assert spec.role == ROLE_QKV_COLUMN
    assert spec.descriptor_extras == {
        "qkv_interleave": "by_head",
        "num_heads_local": "4",
        "num_kv_heads_local": "2",
        "head_dim": "16",
    }


def test_collect_megatron_publish_set_uses_global_grouped_expert_id():
    published = list(
        collect_megatron_publish_set(
            ExpertModule(),
            tp_size=1,
            ep_size=2,
            ep_rank=1,
            tp_rank=0,
            num_local_experts=4,
        )
    )

    name, _, spec = published[0]
    assert name == "experts.weight0"
    assert spec.role == ROLE_EXPERT_COLUMN
    assert spec.owned_expert_ids == {4}
