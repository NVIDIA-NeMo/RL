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

import pytest
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)

from nemo_rl.models.automodel.refit import refit_state_dict


class Selected(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = checkpoint_wrapper(torch.nn.Linear(3, 2))
        self.frozen = torch.nn.Linear(2, 1)
        self.frozen.requires_grad_(False)
        self.register_buffer("router", torch.zeros(2))

    def refit_parameter_names(self):
        return ("layer.weight", "layer.bias")


def test_refit_surface_keeps_canonical_parameter_names_and_excludes_frozen_state():
    model = Selected()
    selected = refit_state_dict(model)
    assert list(selected) == ["layer.weight", "layer.bias"]
    torch.testing.assert_close(selected["layer.weight"], model.layer.weight)


def test_ordinary_model_keeps_full_state_including_frozen_tensors_and_buffers():
    model = torch.nn.Linear(3, 2)
    model.bias.requires_grad_(False)
    model.register_buffer("router", torch.zeros(2))
    assert set(refit_state_dict(model)) == set(model.state_dict())


def test_missing_selected_refit_parameter_fails_before_transport():
    model = Selected()
    model.layer = torch.nn.Linear(3, 2, bias=False)
    with pytest.raises(ValueError, match="missing"):
        refit_state_dict(model)
