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


_GatedDelta.__module__ = "megatron.core.ssm.gated_delta_net"


class _Router(torch.nn.Module):
    def __init__(self, group, config):
        super().__init__()
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
    model.add_module("router", _Router(original_tp_cp, config))
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
    with dynamic_cp.preserve_attention_cp_groups(model):
        model_packed = dynamic_cp.bind_attention_cp_group(model, packed)
        assert model_packed.cp_group is active_cp
        assert model.router.tp_cp_group is active_tp_cp
        assert model.mamba.pg_collection.cp is active_cp
        assert model.mamba.cp is not original_mamba_helper
        assert model.mamba.cp.cp_group is active_cp
        assert model.gdn.pg_collection.cp is active_cp
        assert model.gdn.cp_size == 2
        assert config.moe_aux_loss_coeff == [0.0, 0.0]
        assert config.moe_z_loss_coeff is None

    assert model.router.tp_cp_group is original_tp_cp
    assert model.mamba.pg_collection.cp is original_cp
    assert model.mamba.cp is original_mamba_helper
    assert model.gdn.pg_collection.cp is original_cp
    assert model.gdn.cp_size == 1
    assert config.moe_aux_loss_coeff == [0.1, 0.2]
    assert config.moe_z_loss_coeff == 0.01
