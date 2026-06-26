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

"""Unit test for the train-side expert stacking (``_group_experts``).

``_group_experts`` (``MegatronPolicyWorkerImpl``) stacks this rank's local
per-expert tensors for one projection into ``[E_local, ...]``.  It doesn't use
``self`` and operates on plain tensors, so a dummy ``self`` + CPU tensors suffice.

Importing ``megatron_policy_worker`` pulls in megatron.core, so this is
mcore-marked and skipped where mcore is unavailable.
"""

from types import SimpleNamespace

import pytest
import torch

# megatron_policy_worker imports both megatron.core and megatron.bridge at
# module top, so guard on both: an env can have megatron.core but not
# megatron.bridge, and importing this test module would otherwise raise a
# collection error (not skip) in non-mcore lanes.
pytest.importorskip("megatron.core")
pytest.importorskip("megatron.bridge")

from nemo_rl.models.policy.workers.megatron_policy_worker import (  # noqa: E402
    MegatronPolicyWorkerImpl,
)

pytestmark = pytest.mark.mcore


def _group(proj, grouped_name, expert_groups):
    # _group_experts ignores self; pass a dummy.
    return MegatronPolicyWorkerImpl._group_experts(
        SimpleNamespace(), proj, grouped_name, expert_groups
    )


def test_group_experts_stacks_in_order():
    prefix = "model.layers.0.mlp.experts"
    e0 = torch.randn(1536, 4096)
    e1 = torch.randn(1536, 4096)
    e2 = torch.randn(1536, 4096)
    groups = {(prefix, "gate_proj"): [e0, e1, e2]}
    out = _group("gate_proj", f"{prefix}.gate_proj.weight", groups)
    assert out.shape == (3, 1536, 4096)
    # Order preserved (expert 0 first).
    assert torch.equal(out[0], e0)
    assert torch.equal(out[1], e1)
    assert torch.equal(out[2], e2)


def test_group_experts_down_proj_uses_correct_key():
    prefix = "model.layers.5.mlp.experts"
    tensors = [torch.randn(4096, 1536), torch.randn(4096, 1536)]
    groups = {(prefix, "down_proj"): tensors}
    out = _group("down_proj", f"{prefix}.down_proj.weight", groups)
    assert out.shape == (2, 4096, 1536)


def test_group_experts_missing_group_raises():
    groups = {("other.experts", "gate_proj"): [torch.randn(8, 8)]}
    with pytest.raises(AssertionError):
        _group("gate_proj", "model.layers.0.mlp.experts.gate_proj.weight", groups)


def test_group_experts_empty_group_raises():
    prefix = "model.layers.0.mlp.experts"
    with pytest.raises(AssertionError):
        _group("gate_proj", f"{prefix}.gate_proj.weight", {(prefix, "gate_proj"): []})


# --------------------------------------------------------------------------
# build_hf_to_local_param_map (train/src side) — folds this rank's local
# shards (_iter_local_hf_param_shards) into LocalParamSpecs.  Fake the shard
# iterator; _build_expert_groups / _group_experts run for real.
# --------------------------------------------------------------------------
def test_build_hf_to_local_param_map_train_side():
    from nemo_rl.distributed.nccl_xfer_utils import HFToLocalParamMap

    w = object.__new__(MegatronPolicyWorkerImpl)  # no __init__ / no megatron state
    prefix = "model.layers.0.mlp.experts"
    direct = torch.randn(8, 16)  # a non-expert local shard view
    e0 = torch.randn(128, 16)  # this rank's local expert 0 gate_proj
    e1 = torch.randn(128, 16)  # local expert 1 gate_proj
    w._iter_local_hf_param_shards = lambda: [
        ("model.layers.0.self_attn.o_proj.weight", direct),
        (f"{prefix}.0.gate_proj.weight", e0),
        (f"{prefix}.1.gate_proj.weight", e1),
    ]
    refit_info = {
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": "model.layers.0.self_attn.o_proj.weight",
                    "global_shape": [8, 16],
                },
                {
                    "name": f"{prefix}.gate_proj.weight",
                    "global_shape": [2, 128, 16],
                    "grouped_expert_proj": "gate_proj",
                },
            ]
        },
    }

    pmap = w.build_hf_to_local_param_map(refit_info)
    assert isinstance(pmap, HFToLocalParamMap)

    # Direct: base is the live local view, sent as-is (no hooks).
    d = pmap.get("model.layers.0.self_attn.o_proj.weight")
    assert d.base is direct and d.pre is None and d.post is None

    # Grouped expert: pre stacks this rank's per-expert views into [E_local, ...]
    # fresh each refit (base unused — the views are captured in the hook).
    g = pmap.get(f"{prefix}.gate_proj.weight")
    assert g.pre is not None
    ctx = g.pre(g.base)
    assert ctx.buf.shape == (2, 128, 16)
    assert torch.equal(ctx.buf[0], e0) and torch.equal(ctx.buf[1], e1)
