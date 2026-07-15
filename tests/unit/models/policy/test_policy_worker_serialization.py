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

"""Tests for policy worker construction and serialization."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.models.policy.workers.tokenizer_serialization import (
    resolve_worker_tokenizer,
    worker_tokenizer_kwargs,
)
from tests.unit.models.policy.test_dtensor_worker import create_test_config


def test_dtensor_policy_reconstructs_tokenizer_in_worker(monkeypatch):
    from nemo_rl.models.policy import lm_policy as lm_policy_mod

    captured_builders = []

    class FakeCluster:
        _sorted_bundle_indices = None
        num_gpus_per_node = 1

        def world_size(self):
            return 1

    class FakeWorkerGroup:
        def __init__(self, _cluster, worker_builder, **_kwargs):
            captured_builders.append(worker_builder)

        def shutdown(self, *_args, **_kwargs):
            pass

    config = create_test_config("dummy-model")
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0

    monkeypatch.setattr(lm_policy_mod, "RayQueue", lambda: object())
    monkeypatch.setattr(lm_policy_mod, "RayWorkerGroup", FakeWorkerGroup)
    monkeypatch.setattr(lm_policy_mod, "get_default_hf_config", lambda _name: {})
    monkeypatch.setattr(
        lm_policy_mod.FLOPTracker,
        "from_config",
        lambda *_args, **_kwargs: object(),
    )

    Policy(FakeCluster(), config, tokenizer, init_reference_model=False)

    assert captured_builders
    worker_kwargs = captured_builders[0].kwargs
    assert "tokenizer" not in worker_kwargs
    assert "processor" not in worker_kwargs
    assert config["tokenizer"]["use_processor"] is False


def test_worker_tokenizer_kwargs_preserves_non_dtensor_arguments():
    config = {"tokenizer": {}}
    tokenizer = object()
    processor = object()

    assert worker_tokenizer_kwargs(
        config,
        tokenizer,
        processor,
        reconstruct_in_worker=False,
    ) == {"tokenizer": tokenizer, "processor": processor}
    assert "use_processor" not in config["tokenizer"]


def test_resolve_worker_tokenizer_reconstructs_processor(monkeypatch):
    from nemo_rl.algorithms import utils as algorithm_utils

    processor = SimpleNamespace(tokenizer=object())
    get_tokenizer = MagicMock(return_value=processor)
    monkeypatch.setattr(algorithm_utils, "get_tokenizer", get_tokenizer)
    config = {"tokenizer": {"name": "model", "use_processor": True}}

    tokenizer, resolved_processor = resolve_worker_tokenizer(config)

    assert tokenizer is processor.tokenizer
    assert resolved_processor is processor
    get_tokenizer.assert_called_once_with(config["tokenizer"], get_processor=True)
