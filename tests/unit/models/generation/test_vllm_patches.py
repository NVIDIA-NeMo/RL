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

from unittest.mock import MagicMock

from nemo_rl.models.generation.vllm import patches

_ORIGINAL_SESSION_UPDATE = (
    "        session.arrival_time = update.arrival_time\n"
    "        session.sampling_params = update.sampling_params\n"
    "        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:\n"
)


def test_streaming_session_max_tokens_patch_is_guarded_and_idempotent(
    tmp_path, monkeypatch
) -> None:
    scheduler = tmp_path / "scheduler.py"
    scheduler.write_text(f"before\n{_ORIGINAL_SESSION_UPDATE}after\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _: str(scheduler))
    logger = MagicMock()

    assert patches._patch_vllm_streaming_session_max_tokens(logger)
    assert patches._patch_vllm_streaming_session_max_tokens(logger)

    content = scheduler.read_text()
    assert (
        content.count(
            "        session.max_tokens = update.sampling_params.max_tokens\n"
        )
        == 1
    )
    assert "        assert update.sampling_params.max_tokens is not None\n" in content


def test_streaming_session_max_tokens_patch_rejects_unknown_scheduler(
    tmp_path, monkeypatch
) -> None:
    scheduler = tmp_path / "scheduler.py"
    scheduler.write_text("unexpected future implementation\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _: str(scheduler))
    logger = MagicMock()

    assert not patches._patch_vllm_streaming_session_max_tokens(logger)
    logger.warning.assert_called_once()
    assert scheduler.read_text() == "unexpected future implementation\n"
