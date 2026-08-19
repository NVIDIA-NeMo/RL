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

"""Hidden-state capture must tap the aux layers the draft was built for.

The drafter checkpoint's ``target_layer_ids`` (or
``policy.draft.aux_layer_indices``) decide which policy layers feed the draft
at serving time; the trainer capture must hook the same layers, not the
hard-coded defaults (silent feature mismatch when the counts match, fc width
mismatch when they don't).
"""

import types

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.mcore

from megatron.core import parallel_state  # noqa: E402

from nemo_rl.models.megatron.draft.hidden_capture import (  # noqa: E402
    get_capture_context,
    get_eagle3_aux_hidden_state_layers,
)

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA (parallel_state init)"
)


def _make_policy_stub(num_layers: int) -> torch.nn.Module:
    model = torch.nn.Module()
    model.config = types.SimpleNamespace(num_layers=num_layers)
    model.decoder = torch.nn.Module()
    layers = []
    for layer_idx in range(num_layers):
        layer = torch.nn.Identity()
        layer.layer_number = layer_idx + 1  # mcore layer_number is 1-based
        layers.append(layer)
    model.decoder.layers = torch.nn.ModuleList(layers)
    return model


@requires_gpu
def test_capture_hooks_follow_configured_aux_layers(tmp_path):
    created_process_group = False
    try:
        torch.cuda.set_device(0)
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                rank=0,
                world_size=1,
                init_method=f"file://{tmp_path / 'capture_pg_init'}",
            )
            created_process_group = True
        parallel_state.destroy_model_parallel()
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

        model = _make_policy_stub(num_layers=8)

        # Configured ids (e.g. an external checkpoint's target_layer_ids)
        # must reach the hooks verbatim.
        context, capture = get_capture_context(
            model, enabled=True, aux_layer_indices=(2, 5)
        )
        assert capture.aux_layer_indices == (2, 5)
        assert capture._local_aux_indices == [2, 5]
        with context:
            # One hook per aux layer (the stub has no embedding module).
            assert len(capture._hooks) == 2

        # Without configured ids the default formula applies.
        _, default_capture = get_capture_context(model, enabled=True)
        assert default_capture.aux_layer_indices == get_eagle3_aux_hidden_state_layers(
            8
        )
    finally:
        parallel_state.destroy_model_parallel()
        if created_process_group and dist.is_initialized():
            dist.destroy_process_group()
