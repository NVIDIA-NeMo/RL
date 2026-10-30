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
    vLLM's ModelConfig requires an int: "ValidationError: seed - Input should be a valid
    integer", raised against a recreated worker that already had a working engine.
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
        gen._token_capture_setup = None
        gen._token_capture_media = None
        gen._rollout_weight_version = None
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


class TestRestartShardTokenCapture:
    """restart_shard must give a replacement leader the token capture the fleet has.

    setup_token_capture runs once, at setup. A recreated leader starts with capture
    dormant, so without a re-install it serves calls without staging them, and the
    rollouts it serves are never captured.
    """

    SETUP = {
        "dp_cfg": {"backend": "simple"},
        "staging_partition": "rollout_staging",
        "capture_media": False,
    }

    @staticmethod
    def _generation(*, media_reported=None):
        from nemo_rl.models.generation.vllm import vllm_generation

        gen = vllm_generation.VllmGeneration.__new__(vllm_generation.VllmGeneration)
        calls = MagicMock()
        leader = SimpleNamespace(
            post_init_async=SimpleNamespace(remote=calls.post_init_async),
            setup_token_capture=SimpleNamespace(remote=calls.setup_token_capture),
            set_rollout_weight_version=SimpleNamespace(
                remote=calls.set_rollout_weight_version
            ),
            report_dp_openai_server_base_url=SimpleNamespace(
                remote=calls.report_dp_openai_server_base_url
            ),
        )
        calls.setup_token_capture.return_value = media_reported
        calls.report_dp_openai_server_base_url.return_value = "http://new:9000/v1"
        gen.worker_group = SimpleNamespace(
            workers=[leader],
            get_dp_leader_worker_idx=lambda shard: 0,
            recreate_worker=MagicMock(),
        )
        gen.model_parallel_size = 1
        gen.cfg = {"vllm_cfg": {"async_engine": True}}
        gen.dp_openai_server_base_urls = ["http://old:8000/v1"]
        gen._defer_model_load = False
        gen._token_capture_setup = None
        gen._token_capture_media = None
        gen._rollout_weight_version = None
        return gen, calls

    def test_capture_and_weight_version_are_reinstalled_before_the_url(
        self, monkeypatch
    ):
        from nemo_rl.models.generation.vllm import vllm_generation

        gen, calls = self._generation()
        gen._token_capture_setup = vllm_generation._TokenCaptureSetup(**self.SETUP)
        gen._rollout_weight_version = 3
        monkeypatch.setattr(vllm_generation.ray, "get", lambda ref: ref)

        assert gen.restart_shard(0) == "http://new:9000/v1"

        calls.setup_token_capture.assert_called_once_with(**self.SETUP)
        calls.set_rollout_weight_version.assert_called_once_with(version=3)
        assert [name for name, _, _ in calls.mock_calls] == [
            "post_init_async",
            "setup_token_capture",
            "set_rollout_weight_version",
            "report_dp_openai_server_base_url",
        ]

    def test_nothing_is_installed_when_capture_was_never_set_up(self, monkeypatch):
        from nemo_rl.models.generation.vllm import vllm_generation

        gen, calls = self._generation()
        monkeypatch.setattr(vllm_generation.ray, "get", lambda ref: ref)

        gen.restart_shard(0)

        calls.setup_token_capture.assert_not_called()
        calls.set_rollout_weight_version.assert_not_called()

    def test_a_replacement_that_pins_a_different_media_spec_is_refused(
        self, monkeypatch
    ):
        from nemo_rl.models.generation.vllm import vllm_generation

        gen, calls = self._generation(media_reported="other-spec")
        gen._token_capture_setup = vllm_generation._TokenCaptureSetup(**self.SETUP)
        monkeypatch.setattr(vllm_generation.ray, "get", lambda ref: ref)

        with pytest.raises(RuntimeError, match="pinned media column spec"):
            gen.restart_shard(0)
        calls.report_dp_openai_server_base_url.assert_not_called()
