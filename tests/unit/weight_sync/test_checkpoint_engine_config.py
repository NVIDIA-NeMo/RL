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

"""Tests for checkpoint-engine configuration selection."""

import pytest

from nemo_rl.weight_sync.checkpoint_engine_config import (
    checkpoint_engine_refit_config,
)


@pytest.mark.parametrize(
    "generation_config",
    [{}, {"refit_transport": None}, {"refit_transport": "vllm_zmq_sparse"}],
)
def test_checkpoint_engine_refit_config_returns_none_for_other_transports(
    generation_config,
):
    assert checkpoint_engine_refit_config(generation_config) is None


def test_checkpoint_engine_refit_config_selects_nixl_without_a_refit_cfg():
    generation_config = {"refit_transport": "nixl", "refit_cfg": None}

    config = checkpoint_engine_refit_config(generation_config)
    assert config is not None
    assert config["backend"] == "nixl"
    assert set(config["engine_kwargs"]) == {"nixl"}


def test_checkpoint_engine_refit_config_resolves_custom_backend_scope():
    backend = "package.engine:CustomCheckpointEngine"
    generation_config = {
        "refit_transport": backend,
        "refit_cfg": {
            backend: {
                "update_weights_bucket_memory_ratio": 0.1,
                "release_after_refit": True,
                "custom_option": 7,
            }
        },
    }

    assert checkpoint_engine_refit_config(generation_config) == {
        "backend": backend,
        "update_weights_bucket_memory_ratio": 0.1,
        "engine_kwargs": {backend: {"release_after_refit": True, "custom_option": 7}},
    }


def test_checkpoint_engine_refit_config_rejects_legacy_block():
    with pytest.raises(ValueError, match="checkpoint_engine was replaced"):
        checkpoint_engine_refit_config(
            {"checkpoint_engine": {"enabled": True, "backend": "nixl"}}
        )


def test_checkpoint_engine_refit_config_rejects_unknown_selector():
    with pytest.raises(ValueError, match="Unknown vLLM refit transport"):
        checkpoint_engine_refit_config({"refit_transport": "nixll"})
