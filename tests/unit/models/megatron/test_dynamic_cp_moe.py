# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
"""Runtime binding contracts shared by Qwen MoE and Nemotron hybrid models."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.mcore


class _Group:
    def __init__(self, size, rank=0):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


class _CPHelper:
    def __init__(
        self,
        *,
        cp_group,
        d_inner_local_tp=32,
        nheads_local_tp=16,
        ngroups_local_tp=4,
        d_state=8,
        conv1d_weight_cp1=None,
        conv1d_bias_cp1=None,
        conv1d_padding=3,
        dt_bias_cp1=None,
        A_log_cp1=None,
        D_cp1=None,
        D_has_hdim=False,
    ):
        self.cp_group = cp_group
        self.d_inner_local_tp = d_inner_local_tp
        self.nheads_local_tp = nheads_local_tp
        self.ngroups_local_tp = ngroups_local_tp
        self.d_state = d_state
        self.conv1d_weight_cp1 = conv1d_weight_cp1
        self.conv1d_bias_cp1 = conv1d_bias_cp1
        self.conv1d_padding = conv1d_padding
        self.dt_bias_cp1 = dt_bias_cp1
        self.A_log_cp1 = A_log_cp1
        self.D_cp1 = D_cp1
        self.D_has_hdim = D_has_hdim


class _GDPHelper:
    def __init__(
        self,
        *,
        cp_group,
        d_inner_local_tp=32,
        nheads_local_tp=16,
        ngroups_local_tp=4,
        d_state=8,
        num_householder=2,
        headdim=2,
        conv1d_cp1=None,
        dt_bias_cp1=None,
        A_log_cp1=None,
        D_cp1=None,
        D_has_hdim=False,
        sequence_is_contiguous=False,
    ):
        self.cp_group = cp_group
        self.d_inner_local_tp = d_inner_local_tp
        self.nheads_local_tp = nheads_local_tp
        self.ngroups_local_tp = ngroups_local_tp
        self.d_state = d_state
        self.num_householder = num_householder
        self.headdim = headdim
        self.conv1d_cp1 = conv1d_cp1
        self.dt_bias_cp1 = dt_bias_cp1
        self.A_log_cp1 = A_log_cp1
        self.D_cp1 = D_cp1
        self.D_has_hdim = D_has_hdim
        self.sequence_is_contiguous = sequence_is_contiguous
        self.d_inner_local_tpcp = d_inner_local_tp // cp_group.size()
        self.nheads_local_tpcp = nheads_local_tp // cp_group.size()
        self.ngroups_local_tpcp = max(1, ngroups_local_tp // cp_group.size())


class _Mamba(torch.nn.Module):
    def __init__(self, group):
        super().__init__()
        self.pg_collection = SimpleNamespace(cp=group)
        self.cp = _CPHelper(cp_group=group)


class _GatedDelta(torch.nn.Module):
    def __init__(self, group):
        super().__init__()
        self.pg_collection = SimpleNamespace(cp=group)
        self.cp_size = group.size()
        self.feat_dim_split = (32, 16, 8, 8)


_GatedDelta.__module__ = "megatron.core.ssm.gated_delta_net"


class _GatedDeltaProduct(torch.nn.Module):
    def __init__(self, group):
        super().__init__()
        self.pg_collection = SimpleNamespace(cp=group)
        self.cp = _GDPHelper(cp_group=group)
        self.d_inner_local_cp = self.cp.d_inner_local_tpcp
        self.nheads_local_cp = self.cp.nheads_local_tpcp
        self.ngroups_local_cp = self.cp.ngroups_local_tpcp


class _Router(torch.nn.Module):
    def __init__(self, group, config):
        super().__init__()
        self.cp_group = group
        self.tp_cp_group = group
        self.config = config


def test_dynamic_binding_updates_router_and_ssm_then_restores(monkeypatch):
    from nemo_rl.models.megatron import dynamic_cp

    original_cp = _Group(1)
    active_cp = _Group(2, rank=1)
    original_tp_cp = _Group(2)
    active_tp_cp = _Group(4)
    config = SimpleNamespace(moe_aux_loss_coeff=[0.1, 0.2], moe_z_loss_coeff=0.01)

    model = torch.nn.Module()
    model.add_module("mamba", _Mamba(original_cp))
    model.add_module("gdn", _GatedDelta(original_cp))
    model.add_module("gdp", _GatedDeltaProduct(original_cp))
    model.add_module("router", _Router(original_tp_cp, config))
    # Real transformer layers share one config across many routers.
    model.add_module("router2", _Router(original_tp_cp, config))
    packed = SimpleNamespace(
        local_cp_size=2,
        cp_group=active_cp,
        dynamic_cp_padding_only=True,
    )

    monkeypatch.setattr(dynamic_cp, "Router", _Router)
    monkeypatch.setattr(
        dynamic_cp.parallel_state,
        "get_tensor_model_parallel_world_size",
        lambda: 2,
    )
    monkeypatch.setitem(dynamic_cp._DYNAMIC_TP_CP_GROUPS, 2, active_tp_cp)

    original_mamba_helper = model.mamba.cp
    original_gdp_helper = model.gdp.cp
    with dynamic_cp.preserve_attention_cp_groups(model):
        model_packed = dynamic_cp.bind_attention_cp_group(model, packed)
        assert model_packed.cp_group is active_cp
        assert model.router.cp_group is active_cp
        assert model.router.tp_cp_group is active_tp_cp
        assert model.router2.cp_group is active_cp
        assert model.router2.tp_cp_group is active_tp_cp
        assert model.mamba.pg_collection.cp is active_cp
        assert model.mamba.cp is not original_mamba_helper
        assert model.mamba.cp.cp_group is active_cp
        assert model.gdn.pg_collection.cp is active_cp
        assert model.gdn.cp_size == 2
        assert model.gdn.feat_dim_split == (16, 8, 4, 4)
        assert model.gdp.cp is not original_gdp_helper
        assert model.gdp.cp.cp_group is active_cp
        assert model.gdp.d_inner_local_cp == 16
        assert config.moe_aux_loss_coeff == [0.0, 0.0]
        assert config.moe_z_loss_coeff is None

    assert model.router.cp_group is original_tp_cp
    assert model.router.tp_cp_group is original_tp_cp
    assert model.router2.cp_group is original_tp_cp
    assert model.router2.tp_cp_group is original_tp_cp
    assert model.mamba.pg_collection.cp is original_cp
    assert model.mamba.cp is original_mamba_helper
    assert model.gdn.pg_collection.cp is original_cp
    assert model.gdn.cp_size == 1
    assert model.gdn.feat_dim_split == (32, 16, 8, 8)
    assert model.gdp.cp is original_gdp_helper
    assert model.gdp.d_inner_local_cp == 32
    assert config.moe_aux_loss_coeff == [0.1, 0.2]
    assert config.moe_z_loss_coeff == 0.01


def test_dynamic_padding_task_keeps_only_global_aux_loss(monkeypatch):
    from nemo_rl.models.megatron import dynamic_cp

    group = _Group(1)
    config = SimpleNamespace(
        moe_aux_loss_coeff=[0.1, 0.2],
        moe_z_loss_coeff=0.01,
        moe_router_load_balancing_type=["aux_loss", "global_aux_loss"],
    )
    router = _Router(group, config)
    model = torch.nn.Module()
    model.add_module("router", router)
    packed = SimpleNamespace(
        local_cp_size=1,
        cp_group=None,
        dynamic_cp_padding_only=True,
    )

    monkeypatch.setattr(dynamic_cp, "Router", _Router)
    monkeypatch.setattr(
        dynamic_cp.parallel_state, "get_pipeline_model_parallel_group", lambda: group
    )
    monkeypatch.setattr(
        dynamic_cp.parallel_state, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setitem(dynamic_cp._DYNAMIC_TP_CP_GROUPS, 1, group)

    with dynamic_cp.preserve_attention_cp_groups(model):
        dynamic_cp.bind_attention_cp_group(model, packed)
        assert config.moe_aux_loss_coeff == [0.0, 0.2]
        assert config.moe_z_loss_coeff is None

    assert config.moe_aux_loss_coeff == [0.1, 0.2]
    assert config.moe_z_loss_coeff == 0.01


def test_dynamic_binding_setup_failure_cleans_global_state(monkeypatch):
    from nemo_rl.models.megatron import dynamic_cp

    group = _Group(1)
    config = SimpleNamespace(moe_aux_loss_coeff=0.1, moe_z_loss_coeff=0.01)
    # The second config fails after the first baseline has been installed.
    invalid_config = SimpleNamespace(moe_aux_loss_coeff=0.2)
    model = torch.nn.Module()
    model.add_module("router", _Router(group, config))
    model.add_module("invalid_router", _Router(group, invalid_config))

    monkeypatch.setattr(dynamic_cp, "Router", _Router)

    with pytest.raises(AttributeError):
        with dynamic_cp.preserve_attention_cp_groups(model):
            pytest.fail("setup unexpectedly completed")

    assert id(model) not in dynamic_cp._ACTIVE_BIND_TARGETS
    assert id(config) not in dynamic_cp._ROUTER_CONFIG_BASELINES
    assert id(invalid_config) not in dynamic_cp._ROUTER_CONFIG_BASELINES
    assert model.router.cp_group is group
    assert model.router.tp_cp_group is group


@pytest.mark.parametrize("active_size", [1, 4])
def test_dynamic_moe_scaling_is_applied_by_router_attachment(monkeypatch, active_size):
    from nemo_rl.models.megatron import dynamic_cp

    active_tp_cp = _Group(active_size)
    config = SimpleNamespace(moe_aux_loss_coeff=0.1, moe_z_loss_coeff=0.02)
    model = torch.nn.Module()
    model.add_module("router", _Router(active_tp_cp, config))
    padding_mask = torch.tensor([[False, False, False, True]])

    monkeypatch.setattr(dynamic_cp, "Router", _Router)

    monkeypatch.setattr(
        dynamic_cp.torch.distributed,
        "all_reduce",
        lambda *_args, **_kwargs: pytest.fail(
            "pre-forward validation performed a token reduction"
        ),
    )

    with dynamic_cp.preserve_attention_cp_groups(model):
        dynamic_cp.configure_dynamic_moe_loss_scaling(model, padding_mask)
        assert config.moe_z_loss_coeff == 0.02

    assert config.moe_aux_loss_coeff == 0.1
    assert config.moe_z_loss_coeff == 0.02


def test_dynamic_router_attachment_uses_exact_task_token_count(monkeypatch):
    from megatron.core.transformer.moe import moe_logging, moe_utils
    from nemo_rl.models.megatron import dynamic_cp

    tracker = SimpleNamespace(record=lambda *args, **kwargs: None)
    attached = {}

    def _apply(_activation, aux_loss):
        attached["aux_loss"] = aux_loss
        return _activation

    monkeypatch.setattr(moe_logging, "get_moe_metrics_tracker", lambda: tracker)
    monkeypatch.setattr(moe_utils.MoEAuxLossAutoScaler, "apply", _apply)

    router = SimpleNamespace(
        is_mtp_layer=False,
        layer_number=1,
        calculate_per_token_loss=True,
        config=SimpleNamespace(
            mtp_use_repeated_layer=False,
            mtp_num_layers=None,
            num_layers=2,
        ),
        _nemo_dynamic_aux_scale_tokens=torch.tensor(10.0),
    )
    activation = torch.ones(1)
    result = dynamic_cp._dynamic_attach_and_log_load_balancing_loss(
        router,
        activation,
        aux_loss_coeff=0.1,
        aux_loss=torch.tensor(2.0),
        aux_loss_name="load_balancing_loss",
        reduce_group=_Group(1),
        valid_token_count=torch.tensor(3.0),
    )

    assert result is activation
    assert attached["aux_loss"].item() == pytest.approx(20.0)


def test_dynamic_moe_scaling_is_noop_for_dense_model(monkeypatch):
    from nemo_rl.models.megatron import dynamic_cp

    monkeypatch.setattr(
        dynamic_cp.torch.distributed,
        "all_reduce",
        lambda *_args, **_kwargs: pytest.fail("dense model performed MoE reduction"),
    )

    dynamic_cp.configure_dynamic_moe_loss_scaling(torch.nn.Linear(2, 2), None)


def test_runtime_mtp_counts_reduce_before_division(monkeypatch):
    from nemo_rl.models.megatron import dynamic_cp

    group = _Group(4)

    def _all_reduce(counts, *, op, group):
        del op
        assert group.size() == 4
        counts.copy_(torch.tensor([11.0, 7.0]))

    monkeypatch.setattr(dynamic_cp.torch.distributed, "all_reduce", _all_reduce)
    main_tokens, mtp_tokens = dynamic_cp._runtime_mtp_token_counts(
        torch.tensor(3.0), torch.tensor(1.0), group
    )

    assert main_tokens.item() == 11
    assert mtp_tokens.item() == 7


def test_dynamic_mtp_backward_uses_task_wide_token_ratio(monkeypatch):
    from megatron.core.transformer import multi_token_prediction as mcore_mtp
    from nemo_rl.models.megatron import dynamic_cp

    group = _Group(2)

    def _roll_tensor(tensor, **kwargs):
        return tensor, tensor.sum() if kwargs.get("return_sum", True) else None

    def _all_reduce(counts, *, op, group):
        del op
        assert group.size() == 2
        counts.copy_(torch.tensor([8.0, 4.0]))

    monkeypatch.setattr(mcore_mtp, "roll_tensor", _roll_tensor)
    monkeypatch.setattr(dynamic_cp.torch.distributed, "all_reduce", _all_reduce)
    mcore_mtp.MTPLossAutoScaler.set_loss_scale(torch.tensor(1.0))

    hidden_states = torch.ones(4, 1, requires_grad=True)
    labels = torch.zeros(2, 1, dtype=torch.long)
    loss_mask = torch.ones(2, 1)
    config = SimpleNamespace(
        mtp_num_layers=1,
        mtp_detach_heads=False,
        mtp_loss_scaling_factor=1.0,
        calculate_per_token_loss=True,
    )

    output = dynamic_cp._dynamic_process_mtp_loss(
        hidden_states=hidden_states,
        labels=labels,
        loss_mask=loss_mask,
        output_layer=lambda value, **_kwargs: (value, None),
        output_weight=None,
        runtime_gather_output=False,
        is_training=False,
        compute_language_model_loss=lambda _labels, logits: logits,
        config=config,
        cp_group=group,
    )
    output.sum().backward()

    torch.testing.assert_close(hidden_states.grad[:2], torch.ones(2, 1))
    torch.testing.assert_close(hidden_states.grad[2:], torch.full((2, 1), 2.0))


def test_dynamic_mtp_metrics_are_token_weighted(monkeypatch):
    from nemo_rl.models.megatron import dynamic_cp

    dynamic_cp._DYNAMIC_MTP_METRICS.clear()
    dynamic_cp._save_dynamic_mtp_metrics(
        loss_sum=torch.tensor(6.0),
        num_tokens=torch.tensor(2.0),
        correct=torch.tensor(1.0),
        total=torch.tensor(2.0),
        layer_number=0,
        num_layers=1,
    )
    dynamic_cp._save_dynamic_mtp_metrics(
        loss_sum=torch.tensor(9.0),
        num_tokens=torch.tensor(3.0),
        correct=torch.tensor(2.0),
        total=torch.tensor(3.0),
        layer_number=0,
        num_layers=1,
    )
    monkeypatch.setattr(
        dynamic_cp.torch.distributed,
        "all_reduce",
        lambda _value, *, op, group: None,
    )

    metrics = dynamic_cp.get_dynamic_mtp_metrics(parallel_group=_Group(4))

    assert metrics["mtp_1_loss"] == pytest.approx(3.0)
    assert metrics["mtp_1_acceptance_rate"] == pytest.approx(60.0)
    assert dynamic_cp._DYNAMIC_MTP_METRICS == {}


def test_dynamic_hybrid_mtp_receives_router_padding_mask(monkeypatch):
    from nemo_rl.models.megatron import dynamic_cp

    class _MTPRouter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                calculate_per_token_loss=False,
                moe_aux_loss_coeff=0.0,
            )
            self.tp_cp_group = _Group(1)
            self.seen_padding_mask = None

        def forward(self, hidden_states, padding_mask=None):
            self.seen_padding_mask = padding_mask
            return hidden_states

    class _MTP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.router = _MTPRouter()
            self.seen_padding_mask = None

        def forward(self, *, padding_mask=None):
            self.seen_padding_mask = padding_mask
            self.router(torch.ones(1), padding_mask=padding_mask)
            return self.router.seen_padding_mask

    class _HybridModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mtp = _MTP()

        def forward(self, *, padding_mask=None):
            # This intentionally mirrors the pinned MCore bug: HybridModel has
            # the mask, but its MTP call omits it.
            return self.mtp()

    _HybridModel.__module__ = "megatron.core.models.hybrid.hybrid_model"
    model = _HybridModel()
    padding_mask = torch.tensor([[False, True]])
    monkeypatch.setattr(dynamic_cp, "Router", _MTPRouter)

    with dynamic_cp._patch_hybrid_mtp_padding_masks(model):
        result = model(padding_mask=padding_mask)

    assert torch.equal(result, padding_mask)
    assert torch.equal(model.mtp.seen_padding_mask, ~padding_mask)
    model.mtp.seen_padding_mask = None
    assert model(padding_mask=padding_mask) is None


def test_dynamic_model_validation_rejects_unmerged_mla_support():
    from nemo_rl.models.megatron import dynamic_cp

    class MLASelfAttention(torch.nn.Module):
        pass

    model = torch.nn.Module()
    model.add_module("mla", MLASelfAttention())

    with pytest.raises(ValueError, match="unmerged MCore"):
        dynamic_cp.validate_dynamic_cp_model(model)


def test_dynamic_model_validation_rejects_chunkwise_ssm():
    from nemo_rl.models.megatron import dynamic_cp

    class ChunkwiseMamba(_Mamba):
        pass

    ChunkwiseMamba.__module__ = "megatron.core.ssm.mamba_mixer"
    mamba = ChunkwiseMamba(_Group(1))
    mamba.config = SimpleNamespace(linear_cp_mode="chunkwise")
    model = torch.nn.Module()
    model.add_module("mamba", mamba)

    with pytest.raises(ValueError, match="headwise"):
        dynamic_cp.validate_dynamic_cp_model(model)
