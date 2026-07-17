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


def test_streaming_session_priority_patch_is_guarded_and_idempotent(
    tmp_path, monkeypatch
) -> None:
    sources = {
        "engine/protocol.py": (
            "    prompt: EngineInput\n"
            "    sampling_params: SamplingParams | None = None\n"
        ),
        "v1/engine/async_llm.py": (
            "                    # TODO(nick): Avoid re-validating reused sampling parameters\n"
            "                    req = self.input_processor.process_inputs(\n"
            "                        request_id=internal_req_id,\n"
            "                        prompt=input_chunk.prompt,\n"
            "                        params=sp,\n"
            "                        resumable=True,\n"
            "                        **inputs,  # type: ignore[arg-type]\n"
            "                    )\n"
        ),
        "v1/request.py": (
            "    sampling_params: SamplingParams | None\n\n"
            "    @classmethod\n"
            "            arrival_time=request.arrival_time,\n"
            "            sampling_params=request.sampling_params,\n"
            "        )\n"
        ),
        "v1/core/sched/scheduler.py": (
            "        assert update.sampling_params.max_tokens is not None\n"
            "        session.max_tokens = update.sampling_params.max_tokens\n"
            "        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:\n"
        ),
    }
    paths = {}
    for relative_path, content in sources.items():
        path = tmp_path / relative_path.replace("/", "-")
        path.write_text(content)
        paths[relative_path] = path
    monkeypatch.setattr(
        patches,
        "_get_vllm_file",
        lambda relative_path: str(paths[relative_path]),
    )
    logger = MagicMock()

    assert patches._patch_vllm_streaming_session_priority(logger)
    assert patches._patch_vllm_streaming_session_priority(logger)

    assert (
        "    priority: int | None = None\n" in paths["engine/protocol.py"].read_text()
    )
    assert "input_chunk.priority" in paths["v1/engine/async_llm.py"].read_text()
    request_content = paths["v1/request.py"].read_text()
    assert "    priority: int\n" in request_content
    assert "            priority=request.priority,\n" in request_content
    assert (
        "        session.priority = update.priority\n"
        in paths["v1/core/sched/scheduler.py"].read_text()
    )


def test_streaming_session_priority_patch_rejects_unknown_source(
    tmp_path, monkeypatch
) -> None:
    unknown = tmp_path / "unknown.py"
    unknown.write_text("unexpected future implementation\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _: str(unknown))
    logger = MagicMock()

    assert not patches._patch_vllm_streaming_session_priority(logger)
    logger.warning.assert_called_once()
    assert unknown.read_text() == "unexpected future implementation\n"


def test_streaming_session_priority_patch_is_atomic_on_unknown_source(
    tmp_path, monkeypatch
) -> None:
    sources = {
        "engine/protocol.py": (
            "    prompt: EngineInput\n"
            "    sampling_params: SamplingParams | None = None\n"
        ),
        "v1/engine/async_llm.py": "unexpected future implementation\n",
        "v1/request.py": "request source is never reached\n",
        "v1/core/sched/scheduler.py": "scheduler source is never reached\n",
    }
    paths = {}
    for relative_path, content in sources.items():
        path = tmp_path / relative_path.replace("/", "-")
        path.write_text(content)
        paths[relative_path] = path
    monkeypatch.setattr(
        patches,
        "_get_vllm_file",
        lambda relative_path: str(paths[relative_path]),
    )
    logger = MagicMock()

    assert not patches._patch_vllm_streaming_session_priority(logger)
    assert paths["engine/protocol.py"].read_text() == sources["engine/protocol.py"]
