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


def test_streaming_session_max_tokens_patch_accepts_priority_extension(
    tmp_path, monkeypatch
) -> None:
    scheduler = tmp_path / "scheduler.py"
    priority_extended_update = (
        "        session.arrival_time = update.arrival_time\n"
        "        session.sampling_params = update.sampling_params\n"
        "        assert update.sampling_params.max_tokens is not None\n"
        "        session.max_tokens = update.sampling_params.max_tokens\n"
        "        session.priority = update.priority\n"
        "        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:\n"
    )
    scheduler.write_text(priority_extended_update)
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _: str(scheduler))
    logger = MagicMock()

    assert patches._patch_vllm_streaming_session_max_tokens(logger)
    assert scheduler.read_text() == priority_extended_update


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
            "            if existing.status != RequestStatus.WAITING_FOR_STREAMING_REQ:\n"
            '                assert existing.streaming_queue is not None, "duplicate request id"\n'
            "                # Queue next input chunk (or finished sentinel).\n"
            "                existing.streaming_queue.append(update)\n"
            "            elif update is not None:\n"
            "                # Commence next input chunk.\n"
            "                self._update_request_as_session(existing, update)\n"
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
    scheduler_content = paths["v1/core/sched/scheduler.py"].read_text()
    assert "        session.priority = update.priority\n" in scheduler_content
    assert "update.priority < existing.priority" in scheduler_content
    assert "queued_request_queue.remove_request(existing)" in scheduler_content
    assert "queued_request_queue.add_request(existing)" in scheduler_content
    assert "self.skipped_waiting.remove_request(existing)" in scheduler_content
    assert "self._enqueue_waiting_request(existing)" in scheduler_content

    waiting_requeue_v2 = (
        "            elif update is not None:\n"
        "                # WAITING_FOR_STREAMING_REQ lives in skipped_waiting. A\n"
        "                # priority-changing continuation must be removed before\n"
        "                # mutating the heap key, then inserted into the queue\n"
        "                # selected by its new WAITING status and priority.\n"
        "                self.skipped_waiting.remove_request(existing)\n"
        "                self._update_request_as_session(existing, update)\n"
        "                self._enqueue_waiting_request(existing)\n"
    )
    waiting_requeue_v1 = (
        "            elif update is not None:\n"
        "                # Commence next input chunk.\n"
        "                self._update_request_as_session(existing, update)\n"
    )
    paths["v1/core/sched/scheduler.py"].write_text(
        scheduler_content.replace(waiting_requeue_v2, waiting_requeue_v1, 1)
    )

    assert patches._patch_vllm_streaming_session_priority(logger)
    upgraded_scheduler = paths["v1/core/sched/scheduler.py"].read_text()
    assert waiting_requeue_v2 in upgraded_scheduler
    assert waiting_requeue_v1 not in upgraded_scheduler


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


_ORIGINAL_PRIORITY_SCHEDULER = (
    "import itertools\n"
    "import time\n"
    "        self.scheduler_config = vllm_config.scheduler_config\n"
    "        self.cache_config = vllm_config.cache_config\n"
    "        token_budget = self.max_num_scheduled_tokens\n"
    "        if self._pause_state == PauseState.PAUSED_ALL:\n"
    "            request = self.running[req_index]\n"
    "\n"
    "            if (\n"
    "                existing_running_condition\n"
    "            )\n"
    "            num_new_tokens = min(num_new_tokens, token_budget)\n"
    "\n"
    "            # Make sure the input position does not exceed the max model len.\n"
    "            token_budget -= num_new_tokens\n"
    "            req_index += 1\n"
    "                request = request_queue.peek_request()\n"
    "                request_id = request.request_id\n"
    "\n"
    "                # try to promote blocked statuses while traversing skipped queue.\n"
    "                    num_new_tokens = min(num_new_tokens, token_budget)\n"
    "                    assert num_new_tokens > 0\n"
    "                token_budget -= num_new_tokens\n"
    "                request.status = RequestStatus.RUNNING\n"
)


def test_strict_priority_scheduling_patch_is_guarded_and_idempotent(
    tmp_path, monkeypatch
) -> None:
    scheduler = tmp_path / "scheduler.py"
    scheduler.write_text(_ORIGINAL_PRIORITY_SCHEDULER)
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _: str(scheduler))
    logger = MagicMock()

    assert patches._patch_vllm_strict_priority_scheduling(logger)
    assert patches._patch_vllm_strict_priority_scheduling(logger)

    content = scheduler.read_text()
    assert content.count("candidate.priority < request.priority") == 2
    assert "self.waiting.peek_request().priority" in content
    assert "self.skipped_waiting.peek_request().priority" in content
    assert content.count("self.policy == SchedulingPolicy.PRIORITY") == 3
    assert "self.running.sort(key=lambda request: request.priority)" in content
    assert content.count("nemo_rl_background_prefills_scheduled += 1") == 2
    assert (
        content.count("self._nemo_rl_background_prefill_max_foreground_requests = None")
        == 1
    )
    assert (
        content.count("self._nemo_rl_background_prefill_max_tokens_per_step = None")
        == 1
    )
    assert content.count("self._nemo_rl_background_prefill_max_tokens_per_step,") == 2


def test_strict_priority_scheduling_patch_repairs_malformed_legacy_upgrade(
    tmp_path, monkeypatch
) -> None:
    scheduler = tmp_path / "scheduler.py"
    scheduler.write_text(_ORIGINAL_PRIORITY_SCHEDULER)
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _: str(scheduler))
    logger = MagicMock()

    assert patches._patch_vllm_strict_priority_scheduling(logger)
    content = scheduler.read_text()
    bounded_tail = (
        "                    req_index += 1\n"
        "                    continue\n"
        "\n"
        "            if (\n"
        "                existing_running_condition\n"
    )
    malformed_tail = (
        "                    req_index += 1\n"
        "                    continue\n"
        "\n"
        "            if (\n"
        "                self.policy == SchedulingPolicy.PRIORITY\n"
        "                and (\n"
        "                    any(\n"
        "                        candidate.priority < request.priority\n"
        "                        for candidate in self.running\n"
        "                    )\n"
        "                    or (\n"
        "                        self.waiting\n"
        "                        and self.waiting.peek_request().priority\n"
        "                        < request.priority\n"
        "                    )\n"
        "                    or (\n"
        "                        self.skipped_waiting\n"
        "                        and self.skipped_waiting.peek_request().priority\n"
        "                        < request.priority\n"
        "                    )\n"
        "                )\n"
        "            ):\n"
        "                req_index += 1\n"
        "                continue\n"
        "\n"
        "            if (\n"
        "                existing_running_condition\n"
    )
    assert bounded_tail in content
    scheduler.write_text(content.replace(bounded_tail, malformed_tail, 1))

    assert patches._patch_vllm_strict_priority_scheduling(logger)
    repaired = scheduler.read_text()
    assert malformed_tail not in repaired
    assert bounded_tail in repaired
    assert repaired.count("candidate.priority < request.priority") == 2


def test_strict_priority_scheduling_patch_rejects_unknown_source(
    tmp_path, monkeypatch
) -> None:
    scheduler = tmp_path / "scheduler.py"
    scheduler.write_text("unexpected future implementation\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _: str(scheduler))
    logger = MagicMock()

    assert not patches._patch_vllm_strict_priority_scheduling(logger)
    logger.warning.assert_called_once()
    assert scheduler.read_text() == "unexpected future implementation\n"


_ORIGINAL_OUTPUT_PROCESSOR = (
    "import asyncio\n"
    "    prompt_token_ids: list[int] | None\n"
    "    arrival_time: float\n"
    "    final: bool = False\n"
    "    def __init__(\n"
    "        self,\n"
    "        request_id: str,\n"
    "    ):\n"
    "        self.request_id = request_id\n"
    "        self.external_req_id = external_req_id\n"
    "    def apply_streaming_update(self, update: StreamingUpdate) -> None:\n"
    "        # Apply the update to the request state.\n"
    "        self.streaming_input = not update.final\n"
    "        # TODO also include relevant output tokens in new prompt here\n"
    "        #     (match scheduler behavior).\n"
    "        if update.prompt:\n"
    "            self.prompt = (\n"
    "                (self.prompt + update.prompt) if self.prompt else update.prompt\n"
    "            )\n"
    "        if self.prompt_token_ids:\n"
    "            self.prompt_token_ids.extend(update.prompt_token_ids or ())\n"
    "        else:\n"
    "            self.prompt_token_ids = update.prompt_token_ids or []\n"
    "        assert self.prompt_token_ids is not None\n"
    "        self.prompt_len = len(self.prompt_token_ids)\n"
    "        if self.stats is not None:\n"
    "            self.stats.arrival_time = update.arrival_time\n"
    "        self.is_prefilling = True\n"
    "                req_state.logprobs_processor.update_from_output(engine_core_output)\n"
    "            if request_output := req_state.make_request_output(\n"
    "                new_token_ids,\n"
    "                pooling_output,\n"
    "                finish_reason,\n"
    "                stop_reason,\n"
    "                kv_transfer_params,\n"
    "                routed_experts,\n"
    "            ):\n"
    "        return cls(\n"
    "            request_id=request.request_id,\n"
    "        update = StreamingUpdate(\n"
    "            prompt=prompt,\n"
    "            prompt_token_ids=request.prompt_token_ids,\n"
    "            arrival_time=request.arrival_time,\n"
    "        )\n"
)


def test_streaming_session_output_state_patch_is_guarded_and_idempotent(
    tmp_path, monkeypatch
) -> None:
    output_processor = tmp_path / "output_processor.py"
    output_processor.write_text(_ORIGINAL_OUTPUT_PROCESSOR)
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _: str(output_processor))
    logger = MagicMock()

    assert patches._patch_vllm_streaming_session_output_state(logger)
    assert patches._patch_vllm_streaming_session_output_state(logger)

    content = output_processor.read_text()
    assert content.count("from copy import copy\n") == 1
    assert content.count("    request: EngineCoreRequest\n") == 1
    assert content.count("        self.tokenizer = tokenizer\n") == 1
    assert content.count("            tokenizer=tokenizer,\n") == 1
    assert content.count("            request=request,\n") == 1
    assert "output_request = copy(request)" in content
    assert "output_request.prompt_token_ids = list(self.prompt_token_ids)" in content
    assert (
        "\n        request.prompt_token_ids = list(self.prompt_token_ids)"
        not in content
    )
    assert (
        "tokenizer = self.tokenizer if sampling_params.detokenize else None" in content
    )
    assert "LogprobsProcessor.from_new_request" in content
    assert "IncrementalDetokenizer.from_new_request" in content
    assert "sampling_params.output_kind == RequestOutputKind.DELTA" in content
    assert "transition_dummy_logprobs" in content
    assert "transition_dummy_output" in content
    assert "and queued_streaming_update.final" in content
    assert "if pooling_output is None and transition_dummy_output" in content
    assert "TODO also include relevant output tokens" not in content


def test_streaming_session_output_state_patch_rejects_unknown_source(
    tmp_path, monkeypatch
) -> None:
    output_processor = tmp_path / "output_processor.py"
    output_processor.write_text("unexpected future implementation\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _: str(output_processor))
    logger = MagicMock()

    assert not patches._patch_vllm_streaming_session_output_state(logger)
    logger.warning.assert_called_once()
    assert output_processor.read_text() == "unexpected future implementation\n"


def test_streaming_session_output_state_patch_upgrades_mutating_version(
    tmp_path, monkeypatch
) -> None:
    output_processor = tmp_path / "output_processor.py"
    output_processor.write_text(_ORIGINAL_OUTPUT_PROCESSOR)
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _: str(output_processor))
    logger = MagicMock()

    assert patches._patch_vllm_streaming_session_output_state(logger)
    unsafe_content = (
        output_processor.read_text()
        .replace("from copy import copy\n", "")
        .replace(
            "        output_request = copy(request)\n"
            "        output_request.prompt_token_ids = list(self.prompt_token_ids)\n"
            "        output_request.prompt_embeds = None\n",
            "        request.prompt_token_ids = list(self.prompt_token_ids)\n"
            "        request.prompt_embeds = None\n",
        )
        .replace("request=output_request", "request=request")
    )
    output_processor.write_text(unsafe_content)

    assert patches._patch_vllm_streaming_session_output_state(logger)
    upgraded_content = output_processor.read_text()
    assert upgraded_content.count("from copy import copy\n") == 1
    assert "output_request = copy(request)" in upgraded_content
    assert "request=output_request" in upgraded_content
    assert "\n        request.prompt_token_ids = list(self.prompt_token_ids)" not in (
        upgraded_content
    )


def test_streaming_session_output_state_patch_upgrades_logprob_only_guard(
    tmp_path, monkeypatch
) -> None:
    previous_guard = (
        "                # The scheduler may adopt a queued final streaming chunk\n"
        "                # before this processor consumes the prior dummy completion.\n"
        "                # That can attach the final chunk's logprob contract to the\n"
        "                # dummy output even though the old processor requested none.\n"
        "                # The dummy token is discarded when the chunk is applied.\n"
        "                transition_dummy_logprobs = (\n"
        "                    req_state.streaming_input\n"
        "                    and bool(req_state.input_chunk_queue)\n"
        "                    and finish_reason is not None\n"
        "                    and req_state.logprobs_processor.num_logprobs is None\n"
        "                    and engine_core_output.new_logprobs is not None\n"
        "                )\n"
        "                if not transition_dummy_logprobs:\n"
        "                    req_state.logprobs_processor.update_from_output(\n"
        "                        engine_core_output\n"
        "                    )\n"
    )
    output_processor = tmp_path / "output_processor.py"
    output_processor.write_text(
        _ORIGINAL_OUTPUT_PROCESSOR.replace(
            "                req_state.logprobs_processor.update_from_output("
            "engine_core_output)\n",
            previous_guard,
        )
    )
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _: str(output_processor))
    logger = MagicMock()

    assert patches._patch_vllm_streaming_session_output_state(logger)

    upgraded_content = output_processor.read_text()
    assert "and bool(req_state.input_chunk_queue)" not in upgraded_content
    assert "and queued_streaming_update.final" in upgraded_content
    assert "if pooling_output is None and transition_dummy_output" in upgraded_content


def test_streaming_session_output_state_patch_is_atomic(tmp_path, monkeypatch) -> None:
    output_processor = tmp_path / "output_processor.py"
    source_with_one_unknown_snippet = _ORIGINAL_OUTPUT_PROCESSOR.replace(
        "        update = StreamingUpdate(\n",
        "        update = FutureStreamingUpdate(\n",
    )
    output_processor.write_text(source_with_one_unknown_snippet)
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _: str(output_processor))
    logger = MagicMock()

    assert not patches._patch_vllm_streaming_session_output_state(logger)
    assert output_processor.read_text() == source_with_one_unknown_snippet


def test_streaming_session_patch_suite_holds_one_feature_lock(
    tmp_path, monkeypatch
) -> None:
    scheduler = tmp_path / "scheduler.py"
    scheduler.write_text("scheduler source\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _: str(scheduler))

    feature_lock = MagicMock()
    monkeypatch.setattr(
        patches,
        "_exclusive_patch_lock",
        lambda lock_path: (
            feature_lock
            if lock_path == f"{scheduler}.streaming_session_patch_suite_lock"
            else MagicMock()
        ),
    )

    patch_names = (
        "_patch_vllm_streaming_session_max_tokens",
        "_patch_vllm_streaming_session_priority",
        "_patch_vllm_strict_priority_scheduling",
        "_patch_vllm_streaming_session_output_state",
    )
    calls: list[str] = []

    def apply_patch(name: str):
        def apply(_logger) -> bool:
            assert feature_lock.__enter__.called
            assert not feature_lock.__exit__.called
            calls.append(name)
            return True

        return apply

    for patch_name in patch_names:
        monkeypatch.setattr(patches, patch_name, apply_patch(patch_name))

    capabilities = patches._patch_vllm_streaming_session_support(MagicMock())

    assert calls == list(patch_names)
    assert capabilities == {
        "streaming_session_max_tokens": True,
        "streaming_session_priority": True,
        "strict_priority_scheduling": True,
        "streaming_session_output_state": True,
    }
    feature_lock.__enter__.assert_called_once_with()
    feature_lock.__exit__.assert_called_once()
