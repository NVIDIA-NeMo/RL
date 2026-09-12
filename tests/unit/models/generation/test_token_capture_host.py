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

"""Capture host isolation and disabled-path behavior without Gym installed."""

import builtins
from types import SimpleNamespace
from unittest.mock import MagicMock

from nemo_rl.models.generation.vllm.token_capture_host import TokenCaptureHost


def test_disabled_host_never_imports_gym(monkeypatch):
    original_import = builtins.__import__

    def reject_gym(name, *args, **kwargs):
        if name.startswith("nemo_gym"):
            raise AssertionError("disabled capture imported Gym")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_gym)
    host = TokenCaptureHost()
    request = SimpleNamespace(ng_capture={"rollout_id": "r"}, stream=False)
    host.set_weight_version(7)
    assert host.admission(request) is None
    host.begin_request(request, [1, 2])
    response = {"choices": []}
    assert host.finish_request(request, response) is response
    host.abort_request(request, reason="engine_error")


def test_active_calls_are_isolated_and_abort_only_once():
    first, second = TokenCaptureHost(), TokenCaptureHost()
    capture = MagicMock()
    first.install_token_capture(capture)
    request = SimpleNamespace(stream=False)
    first.begin_request(request, [1, 2], admission=SimpleNamespace())

    # An unrelated host cannot finish or abort another worker's active call.
    response = {"choices": []}
    assert second.finish_request(request, response) is response
    second.abort_request(request, reason="engine_error")
    capture.fail_call.assert_not_called()
    first.abort_request(request, reason="engine_error")
    first.abort_request(request, reason="engine_error")
    capture.fail_call.assert_called_once_with(
        capture.begin_call.return_value, reason="engine_error"
    )


def test_prefix_cache_is_local_and_returned_values_do_not_mutate_cache():
    first, second = TokenCaptureHost(), TokenCaptureHost()
    first._staging_source = MagicMock()
    second._staging_source = MagicMock()
    first._staging_source.fetch_prefix_token_ids.return_value = [1, 2]
    second._staging_source.fetch_prefix_token_ids.return_value = [3, 4]
    assert first.fetch_chain_prefix(["same/key"]) == [1, 2]
    cached = first.fetch_chain_prefix(["same/key"])
    cached.append(99)
    assert first.fetch_chain_prefix(["same/key"]) == [1, 2]
    assert second.fetch_chain_prefix(["same/key"]) == [3, 4]
    first._staging_source.fetch_prefix_token_ids.assert_called_once()
    second._staging_source.fetch_prefix_token_ids.assert_called_once()
