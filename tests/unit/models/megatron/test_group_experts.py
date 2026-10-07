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

"""Unit tests for train-side expert source grouping and materialization.

``_group_experts`` (``MegatronPolicyWorkerImpl``) creates a ``pre`` hook that
materializes this rank's local per-expert sources for one projection and stacks
them into ``[E_local, ...]``. Plain CPU tensors suffice for the BF16 path.

Importing ``megatron_policy_worker`` pulls in megatron.core, so this is
mcore-marked and skipped where mcore is unavailable.
"""

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
    _materialize_refit_spec,
)
from nemo_rl.weight_sync.nccl_reshard_utils import LocalParamSpec  # noqa: E402

pytestmark = pytest.mark.mcore


def _group(proj, grouped_name, expert_groups):
    worker = object.__new__(MegatronPolicyWorkerImpl)
    grouped = worker._group_experts(proj, grouped_name, expert_groups)
    return _materialize_refit_spec(grouped).buf


def test_group_experts_stacks_in_order():
    prefix = "model.layers.0.mlp.experts"
    e0 = torch.randn(1536, 4096, dtype=torch.bfloat16)
    e1 = torch.randn(1536, 4096, dtype=torch.bfloat16)
    e2 = torch.randn(1536, 4096, dtype=torch.bfloat16)
    groups = {
        (prefix, "gate_proj"): [
            LocalParamSpec(base=e0),
            LocalParamSpec(base=e1),
            LocalParamSpec(base=e2),
        ]
    }
    out = _group("gate_proj", f"{prefix}.gate_proj.weight", groups)
    assert out.shape == (3, 1536, 4096)
    assert out.dtype == torch.bfloat16
    # Order preserved (expert 0 first).
    assert torch.equal(out[0], e0)
    assert torch.equal(out[1], e1)
    assert torch.equal(out[2], e2)


def test_group_experts_missing_group_raises():
    groups = {("other.experts", "gate_proj"): [LocalParamSpec(base=torch.randn(8, 8))]}
    with pytest.raises(AssertionError, match="no local experts.*PP-filter"):
        _group("gate_proj", "model.layers.0.mlp.experts.gate_proj.weight", groups)


def test_group_experts_empty_group_raises():
    prefix = "model.layers.0.mlp.experts"
    with pytest.raises(AssertionError, match="no local experts.*PP-filter"):
        _group("gate_proj", f"{prefix}.gate_proj.weight", {(prefix, "gate_proj"): []})


# --------------------------------------------------------------------------
# build_hf_to_local_param_map (train/src side) — folds this rank's local
# shards (_iter_local_hf_param_shards) into LocalParamSpecs.  Fake the shard
# iterator; _build_expert_groups / _group_experts run for real.
# --------------------------------------------------------------------------
def test_build_hf_to_local_param_map_train_side():
    from nemo_rl.weight_sync.nccl_reshard_utils import HFToLocalParamMap

    w = object.__new__(MegatronPolicyWorkerImpl)  # no __init__ / no megatron state
    w.my_pp_stage = 0
    # opd_full off, mirroring __init__ when the config block is absent.
    w._opd_full_enabled = False
    w._opd_full_lm_head_lifecycle = None
    w._opd_full_teacher_lm_heads = {}
    w._opd_full_teacher_checkpoint_paths = {}
    w._opd_full_lm_head_evicted = False
    prefix = "model.layers.0.mlp.experts"
    direct = torch.randn(
        8, 16, dtype=torch.bfloat16
    )  # a dense FFN down_proj local shard view
    e0 = torch.randn(
        128, 16, dtype=torch.bfloat16
    )  # this rank's local expert 0 gate_proj
    e1 = torch.randn(128, 16, dtype=torch.bfloat16)  # local expert 1 gate_proj
    w._iter_local_hf_param_shards = lambda: [
        ("model.layers.0.mlp.down_proj.weight", LocalParamSpec(base=direct)),
        (f"{prefix}.1.gate_proj.weight", LocalParamSpec(base=e1)),
        (f"{prefix}.0.gate_proj.weight", LocalParamSpec(base=e0)),
    ]
    refit_info = {
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": "model.layers.0.mlp.down_proj.weight",
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
    d = pmap.get("model.layers.0.mlp.down_proj.weight")
    assert d.base is direct and d.pre is None and d.post is None

    # Grouped expert: pre stacks live views in expert-index order on each refit,
    # even when the shard iterator supplies those experts in a different order.
    g = pmap.get(f"{prefix}.gate_proj.weight")
    assert g.base is None and g.pre is not None and g.post is None
    ctx = _materialize_refit_spec(g)
    assert ctx.buf.shape == (2, 128, 16)
    assert torch.equal(ctx.buf[0], e0) and torch.equal(ctx.buf[1], e1)

    direct.add_(1)
    e0.add_(2)
    e1.add_(3)
    assert _materialize_refit_spec(d).buf is direct
    refreshed = _materialize_refit_spec(g).buf
    assert torch.equal(refreshed[0], e0) and torch.equal(refreshed[1], e1)
    assert not torch.equal(refreshed, ctx.buf)
