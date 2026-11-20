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
"""Verify gradient and storage ownership before Megatron allocates the inference context."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytestmark = pytest.mark.mcore


@pytest.mark.parametrize(
    "colocated,dedicated", [(False, False), (True, True), (True, False)]
)
@pytest.mark.parametrize("media", [False, True])
def test_inference_model_ownership(
    monkeypatch: pytest.MonkeyPatch, colocated: bool, dedicated: bool, media: bool
) -> None:
    # The optional Megatron environment provides the actual worker and model unwrapping.
    from nemo_rl.models.generation.megatron.megatron_worker import (
        MegatronGenerationMixin,
    )

    def model() -> torch.nn.Module:
        if not media:
            return torch.nn.Linear(4, 3)
        root = torch.nn.Module()
        root.language_model = torch.nn.Linear(4, 3)
        root.vision_model = torch.nn.Linear(4, 3)
        return root

    learner = model()
    shared_model = colocated and not dedicated
    engine = learner if shared_model else model()
    for learner_param, engine_param in zip(
        learner.parameters(), engine.parameters(), strict=True
    ):
        engine_param.data = learner_param.data
    learner_parameters = list(learner.parameters())
    learner_parameters[-1].requires_grad_(False)
    expected_grad = [p.requires_grad for p in learner_parameters]
    engine_parameters = list(engine.parameters())
    pointers = [p.data_ptr() for p in engine_parameters]

    worker = MegatronGenerationMixin()
    worker.is_generation_colocated = colocated
    worker._inference_engine_initialized = False
    worker.model = learner if colocated else engine
    worker.inference_model = engine if dedicated else None
    worker._get_megatron_inference_wrapper_cls = lambda: SimpleNamespace(
        supports_image=media
    )

    class BeforeProcessGroups(Exception):
        pass

    monkeypatch.setattr(
        "megatron.core.utils.get_attr_wrapped_model",
        Mock(side_effect=BeforeProcessGroups),
    )
    with pytest.raises(BeforeProcessGroups):
        worker._initialize_inference_engine({})

    assert all(
        current is original
        for current, original in zip(
            engine.parameters(), engine_parameters, strict=True
        )
    )
    assert [p.data_ptr() for p in engine_parameters] == pointers
    assert [p.requires_grad for p in engine_parameters] == (
        expected_grad if shared_model else [False] * len(engine_parameters)
    )
    inputs = torch.ones(1, 4)
    output = (
        learner.language_model(inputs) + learner.vision_model(inputs)
        if media
        else learner(inputs)
    )
    output.sum().backward()
    assert [p.grad is not None for p in learner_parameters] == expected_grad


def test_initialized_engine_does_not_revisit_ownership() -> None:
    from nemo_rl.models.generation.megatron.megatron_worker import (
        MegatronGenerationMixin,
    )

    worker = MegatronGenerationMixin()
    worker._inference_engine_initialized = True
    worker._initialize_inference_engine({})
