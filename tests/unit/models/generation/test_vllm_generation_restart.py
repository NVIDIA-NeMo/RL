# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""restart_shard's interaction with the two model-load paths."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.vllm


class TestRestartShardLoadPath:
    """restart_shard must not re-load a model the recreated worker already loaded.

    The two startup paths differ. defer_model_load=True leaves the model unloaded and
    stashes bundle_indices and seed for load_and_start(). The default False loads inside
    __init__ and leaves _deferred_seed at None, because the branch that assigns it returns
    early. Calling load_model on that worker re-enters _create_engine with seed=None, and
    vLLM's ModelConfig requires an int -- job 6722014 failed five restarts with
    "ValidationError: seed - Input should be a valid integer" while the recreated worker
    already had a working engine.
    """

    @staticmethod
    def _generation(*, defer: bool):
        from nemo_rl.models.generation.vllm import vllm_generation

        gen = vllm_generation.VllmGeneration.__new__(vllm_generation.VllmGeneration)
        leader = SimpleNamespace(
            load_model=SimpleNamespace(remote=MagicMock(return_value="load")),
            post_init_async=SimpleNamespace(remote=MagicMock(return_value="post")),
            post_init=SimpleNamespace(remote=MagicMock(return_value="post")),
            report_dp_openai_server_base_url=SimpleNamespace(
                remote=MagicMock(return_value="url")
            ),
        )
        gen.worker_group = SimpleNamespace(
            workers=[leader],
            get_dp_leader_worker_idx=lambda shard: 0,
            recreate_worker=MagicMock(),
        )
        gen.model_parallel_size = 1
        gen.cfg = {"vllm_cfg": {"async_engine": True}}
        gen.dp_openai_server_base_urls = ["http://old:8000/v1"]
        gen._defer_model_load = defer
        return gen, leader

    def test_the_eager_path_does_not_reload(self, monkeypatch):
        from nemo_rl.models.generation.vllm import vllm_generation

        gen, leader = self._generation(defer=False)
        monkeypatch.setattr(vllm_generation.ray, "get", lambda x: "url")
        gen.restart_shard(0)

        leader.load_model.remote.assert_not_called()
        gen.worker_group.recreate_worker.assert_called_once_with(0)
        leader.post_init_async.remote.assert_called_once()

    def test_the_deferred_path_still_loads(self, monkeypatch):
        from nemo_rl.models.generation.vllm import vllm_generation

        gen, leader = self._generation(defer=True)
        monkeypatch.setattr(vllm_generation.ray, "get", lambda x: "url")
        gen.restart_shard(0)

        leader.load_model.remote.assert_called_once()
        leader.post_init_async.remote.assert_called_once()
