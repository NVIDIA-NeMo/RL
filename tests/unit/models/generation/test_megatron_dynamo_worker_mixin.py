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
import sys
import types
from unittest.mock import MagicMock

import pytest

from nemo_rl.models.generation.megatron import dynamo_worker_mixin
from nemo_rl.models.generation.megatron.dynamo_worker_mixin import (
    MegatronDynamoGenerationMixin,
)
from nemo_rl.models.generation.megatron.megatron_worker import MegatronGenerationMixin


class _Worker(MegatronDynamoGenerationMixin, MegatronGenerationMixin):
    """MegatronDynamoPolicyWorker's generation bases, in the order it composes them."""


def _bare_worker(role, cls=_Worker):
    worker = object.__new__(cls)
    worker.cfg = {"generation": {}}
    if role is not None:
        worker.cfg["generation"]["_dynamo_role"] = role
    return worker


@pytest.fixture
def engine_modules(monkeypatch):
    """Fake the plain and the Dynamo handoff engine classes the hooks pick between."""
    plain = MagicMock(name="DynamicInferenceEngine")
    handoff = MagicMock(name="DynamoDynamicInferenceEngine")
    monkeypatch.setattr(
        "megatron.core.inference.engines.dynamic_engine.DynamicInferenceEngine", plain
    )
    for name in (
        "megatron.inference",
        "megatron.inference.integrations",
        "megatron.inference.integrations.dynamo",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    module = types.ModuleType("megatron.inference.integrations.dynamo.dynamic_engine")
    module.DynamoDynamicInferenceEngine = handoff
    monkeypatch.setitem(
        sys.modules, "megatron.inference.integrations.dynamo.dynamic_engine", module
    )
    monkeypatch.setattr(dynamo_worker_mixin, "_get_node_ip_local", lambda: "10.0.0.5")
    return plain, handoff


def test_plain_worker_builds_the_plain_engine(engine_modules):
    plain, handoff = engine_modules
    worker = _bare_worker(role=None, cls=MegatronGenerationMixin)
    controller, context = MagicMock(), MagicMock()

    assert worker._inference_config_overrides() == {}
    assert worker._coordinator_kwargs() == {}
    assert worker._build_dynamic_inference_engine(controller, context) is (
        plain.return_value
    )
    plain.assert_called_once_with(controller, context)
    handoff.assert_not_called()


def test_dynamo_worker_needs_its_role():
    worker = _bare_worker(role=None)

    with pytest.raises(RuntimeError, match="_dynamo_role"):
        worker._dynamo_role


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_dynamo_worker_builds_the_handoff_engine(engine_modules, role):
    plain, handoff = engine_modules
    worker = _bare_worker(role=role)
    controller, context = MagicMock(), MagicMock()

    assert worker._dynamo_role == role
    # The peer role lives in other processes: NIXL carries the handoff and the
    # coordinator must listen on a routable address.
    assert worker._inference_config_overrides() == {"kv_transport_backend": "nixl"}
    assert worker._coordinator_kwargs() == {"hostname": "10.0.0.5"}

    engine = worker._build_dynamic_inference_engine(controller, context)

    assert engine is handoff.return_value
    handoff.assert_called_once_with(controller, context)
    engine.setup_kv_transfer.assert_called_once_with(
        role=role, require_handoff_claim=True
    )
    plain.assert_not_called()
