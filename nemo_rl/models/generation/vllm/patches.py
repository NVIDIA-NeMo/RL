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

import os
from contextlib import contextmanager
from importlib.util import find_spec


def _get_vllm_file(relative_path: str) -> str:
    """Return absolute path to a vLLM file or raise if it cannot be found.

    The relative_path should be a POSIX-style path under the vllm
    package root, e.g. "v1/executor/ray_executor.py" or
    "attention/layer.py".
    """
    spec = find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            "vLLM package not found while attempting to patch "
            f"'{relative_path}'. Ensure vLLM is installed and "
            "available in this environment."
        )

    base_dir = next(iter(spec.submodule_search_locations))
    file_path = os.path.join(base_dir, *relative_path.split("/"))

    if not os.path.exists(file_path):
        raise RuntimeError(
            "Failed to locate expected vLLM file to patch. "
            f"Looked for '{relative_path}' at '{file_path}'. "
            "This likely indicates an unexpected vLLM installation "
            "layout or version mismatch."
        )

    return file_path


@contextmanager
def _locked_file_patch(file_path: str):
    """Yield (content, writer) under an exclusive file lock."""
    import fcntl

    lock_path = file_path + ".patch_lock"
    lock_fd = open(lock_path, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

        with open(file_path, "r") as f:
            content = f.read()

        def write_back(new_content: str):
            with open(file_path, "w") as f:
                f.write(new_content)

        yield content, write_back
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


@contextmanager
def _exclusive_patch_lock(lock_path: str):
    """Serialize one logical patch that spans multiple source files."""
    import fcntl

    lock_fd = open(lock_path, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


def _patch_vllm_init_workers_ray(
    py_executable: str, extra_env_vars: list[str] | None
) -> None:
    """Patch the vLLM ray_distributed_executor.py file.

    1. Pass custom runtime_env in _init_workers_ray call.
        - This allows passing custom py_executable to worker initialization.
    2. Add NCCL_CUMEM_ENABLE and NCCL_NVLS_ENABLE to vLLM ADDITIONAL_ENV_VARS.
        - This is a workaround to fix async vllm in some scenarios.
        - See https://github.com/NVIDIA-NeMo/RL/pull/898 for more details.
    """
    file_to_patch = _get_vllm_file("v1/executor/ray_executor.py")

    old_lines = [
        "self._init_workers_ray(placement_group)",
        'ADDITIONAL_ENV_VARS = {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}',
    ]
    additional_env_vars = [
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "NCCL_CUMEM_ENABLE",
        "NCCL_NVLS_ENABLE",
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV",
        *(extra_env_vars or []),
    ]
    additional_env_str = ", ".join(f'"{env_var}"' for env_var in additional_env_vars)

    new_lines = [
        (
            "self._init_workers_ray(placement_group, "
            f'runtime_env={{"py_executable": "{py_executable}"}})'
        ),
        f"ADDITIONAL_ENV_VARS = {{{additional_env_str}}}",
    ]

    with _locked_file_patch(file_to_patch) as (content, write_back):
        need_replace = False
        for old_line, new_line in zip(old_lines, new_lines):
            if new_line in content or old_line not in content:
                continue
            content = content.replace(old_line, new_line)
            need_replace = True

        if need_replace:
            write_back(content)


def _patch_vllm_llama_eagle3_own_lm_head(logger) -> None:
    """Patch LlamaEagle3 to keep truncated draft lm_head ownership."""
    try:
        file_to_patch = _get_vllm_file("model_executor/models/llama_eagle3.py")
    except RuntimeError:
        logger.warning("Could not locate llama_eagle3.py for lm_head ownership patch.")
        return

    old_snippet = (
        "        self.lm_head = ParallelLMHead(\n"
        "            self.config.draft_vocab_size,\n"
        "            self.config.hidden_size,\n"
        "            quant_config=get_draft_quant_config(vllm_config),\n"
        '            prefix=maybe_prefix(prefix, "lm_head"),\n'
        "        )\n"
        "        self.logits_processor = LogitsProcessor(\n"
    )

    new_snippet = (
        "        self.lm_head = ParallelLMHead(\n"
        "            self.config.draft_vocab_size,\n"
        "            self.config.hidden_size,\n"
        "            quant_config=get_draft_quant_config(vllm_config),\n"
        '            prefix=maybe_prefix(prefix, "lm_head"),\n'
        "        )\n"
        "        self.has_own_lm_head = (\n"
        "            self.config.draft_vocab_size != self.config.vocab_size\n"
        "        )\n"
        "        self.logits_processor = LogitsProcessor(\n"
    )

    with _locked_file_patch(file_to_patch) as (content, write_back):
        if "self.has_own_lm_head = (" in content:
            logger.info("llama_eagle3 lm_head ownership patch already applied.")
            return

        if old_snippet not in content:
            logger.warning(
                "Could not apply llama_eagle3 lm_head ownership patch: "
                "expected code snippet not found in %s. "
                "The vLLM version may have changed.",
                file_to_patch,
            )
            return

        content = content.replace(old_snippet, new_snippet, 1)
        write_back(content)

    logger.info("Successfully patched llama_eagle3 lm_head ownership.")


def _patch_vllm_hermes_tool_parser_thread_safety(logger) -> None:
    """Patch Hermes2ProToolParser.__init__ to cache tokenizer calls.

    The HuggingFace tokenizer's Rust backend does not support concurrent
    access. When multiple async requests call _preprocess_chat concurrently,
    each one constructs a new Hermes2ProToolParser which calls
    tokenizer.encode() and tokenizer.decode() in __init__, causing
    "RuntimeError: Already borrowed".

    A lock alone is insufficient because the tool parser's encode() can
    race with render_chat_async() in another concurrent request - two
    different codepaths sharing the same tokenizer instance.

    This patch caches the encode/decode results so only the first
    instantiation (protected by a lock) touches the tokenizer. All
    subsequent instantiations read from cache without any tokenizer
    access.

    Related:
    - https://github.com/vllm-project/vllm/pull/30264
    - https://github.com/huggingface/tokenizers/issues/537
    - https://github.com/PrimeIntellect-ai/prime-rl/pull/1837
    """
    file_to_patch = _get_vllm_file("tool_parsers/hermes_tool_parser.py")

    old_import = "import json\nfrom collections.abc import Sequence"
    new_import = "import json\nimport threading\nfrom collections.abc import Sequence"

    old_class_line = "class Hermes2ProToolParser(ToolParser):"
    new_class_line = (
        "class Hermes2ProToolParser(ToolParser):\n"
        "    _tokenizer_lock = threading.Lock()\n"
        "    _tokenizer_cache = {}"
    )

    old_init_snippet = (
        "        self.tool_call_start_token_ids = self.model_tokenizer.encode(\n"
        "            self.tool_call_start_token, add_special_tokens=False\n"
        "        )\n"
        "        self.tool_call_end_token_ids = self.model_tokenizer.encode(\n"
        "            self.tool_call_end_token, add_special_tokens=False\n"
        "        )\n"
        "\n"
        "        self.tool_call_start_token_array = [\n"
        "            self.model_tokenizer.decode([token_id])\n"
        "            for token_id in self.tool_call_start_token_ids\n"
        "        ]\n"
        "\n"
        "        self.tool_call_end_token_array = [\n"
        "            self.model_tokenizer.decode([token_id])\n"
        "            for token_id in self.tool_call_end_token_ids\n"
        "        ]"
    )

    new_init_snippet = (
        "        _tid = id(self.model_tokenizer)\n"
        "        if _tid in Hermes2ProToolParser._tokenizer_cache:\n"
        "            _cached = Hermes2ProToolParser._tokenizer_cache[_tid]\n"
        "            self.tool_call_start_token_ids = _cached['start_ids']\n"
        "            self.tool_call_end_token_ids = _cached['end_ids']\n"
        "            self.tool_call_start_token_array = _cached['start_array']\n"
        "            self.tool_call_end_token_array = _cached['end_array']\n"
        "        else:\n"
        "            with Hermes2ProToolParser._tokenizer_lock:\n"
        "                if _tid in Hermes2ProToolParser._tokenizer_cache:\n"
        "                    _cached = Hermes2ProToolParser._tokenizer_cache[_tid]\n"
        "                    self.tool_call_start_token_ids = _cached['start_ids']\n"
        "                    self.tool_call_end_token_ids = _cached['end_ids']\n"
        "                    self.tool_call_start_token_array = _cached['start_array']\n"
        "                    self.tool_call_end_token_array = _cached['end_array']\n"
        "                else:\n"
        "                    self.tool_call_start_token_ids = self.model_tokenizer.encode(\n"
        "                        self.tool_call_start_token, add_special_tokens=False\n"
        "                    )\n"
        "                    self.tool_call_end_token_ids = self.model_tokenizer.encode(\n"
        "                        self.tool_call_end_token, add_special_tokens=False\n"
        "                    )\n"
        "                    self.tool_call_start_token_array = [\n"
        "                        self.model_tokenizer.decode([token_id])\n"
        "                        for token_id in self.tool_call_start_token_ids\n"
        "                    ]\n"
        "                    self.tool_call_end_token_array = [\n"
        "                        self.model_tokenizer.decode([token_id])\n"
        "                        for token_id in self.tool_call_end_token_ids\n"
        "                    ]\n"
        "                    Hermes2ProToolParser._tokenizer_cache[_tid] = {\n"
        "                        'start_ids': self.tool_call_start_token_ids,\n"
        "                        'end_ids': self.tool_call_end_token_ids,\n"
        "                        'start_array': self.tool_call_start_token_array,\n"
        "                        'end_array': self.tool_call_end_token_array,\n"
        "                    }"
    )

    with _locked_file_patch(file_to_patch) as (content, write_back):
        if "_tokenizer_cache" in content:
            logger.info("Hermes tool parser thread-safety patch already applied.")
            return

        if old_init_snippet not in content:
            logger.warning(
                "Could not apply hermes tool parser thread-safety patch: "
                "expected code snippet not found in %s. "
                "The vLLM version may have changed.",
                file_to_patch,
            )
            return

        content = content.replace(old_import, new_import, 1)
        content = content.replace(old_class_line, new_class_line, 1)
        content = content.replace(old_init_snippet, new_init_snippet, 1)
        write_back(content)

    logger.info("Successfully patched hermes tool parser for thread-safety.")


def _patch_vllm_streaming_session_max_tokens(logger) -> bool:
    """Make per-chunk max_tokens effective for vLLM streaming input sessions.

    vLLM 0.20 updates ``session.sampling_params`` when a resumable request gets
    another input chunk, but ``Request.max_tokens`` remains fixed to the first
    chunk. A one-token prefill phase therefore also limits the authoritative
    final decode to one token. Keep the request's scheduler budget consistent
    with the sampling parameters that vLLM already replaces.

    Returns whether the installed source already contains or accepted the
    guarded patch. A false result lets feature-specific callers fail closed
    while unrelated vLLM generation remains usable on future versions.
    """
    try:
        file_to_patch = _get_vllm_file("v1/core/sched/scheduler.py")
    except RuntimeError:
        logger.warning("Could not locate vLLM scheduler for streaming-input patch.")
        return False

    old_snippet = (
        "        session.arrival_time = update.arrival_time\n"
        "        session.sampling_params = update.sampling_params\n"
        "        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:\n"
    )
    new_snippet = (
        "        session.arrival_time = update.arrival_time\n"
        "        session.sampling_params = update.sampling_params\n"
        "        assert update.sampling_params.max_tokens is not None\n"
        "        session.max_tokens = update.sampling_params.max_tokens\n"
        "        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:\n"
    )
    priority_extended_snippet = (
        "        session.arrival_time = update.arrival_time\n"
        "        session.sampling_params = update.sampling_params\n"
        "        assert update.sampling_params.max_tokens is not None\n"
        "        session.max_tokens = update.sampling_params.max_tokens\n"
        "        session.priority = update.priority\n"
        "        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:\n"
    )

    with _locked_file_patch(file_to_patch) as (content, write_back):
        if new_snippet in content or priority_extended_snippet in content:
            logger.info("vLLM streaming-session max_tokens patch already applied.")
            return True
        if old_snippet not in content:
            logger.warning(
                "Could not apply vLLM streaming-session max_tokens patch: "
                "expected scheduler snippet not found in %s.",
                file_to_patch,
            )
            return False
        write_back(content.replace(old_snippet, new_snippet, 1))

    logger.info("Successfully patched vLLM streaming-session max_tokens updates.")
    return True


def _patch_vllm_streaming_session_priority(logger) -> bool:
    """Allow a final streaming-input chunk to promote its session priority.

    Background prefill uses a lower scheduler priority than foreground model
    calls. vLLM 0.20 fixes streaming-input priority at request creation, so a
    same-request final decode would otherwise remain background work. Extend
    the public ``StreamingInput`` value and the internal continuation update so
    the authoritative final chunk can explicitly restore priority zero.
    """
    try:
        protocol_file = _get_vllm_file("engine/protocol.py")
        async_llm_file = _get_vllm_file("v1/engine/async_llm.py")
        request_file = _get_vllm_file("v1/request.py")
        scheduler_file = _get_vllm_file("v1/core/sched/scheduler.py")
    except RuntimeError:
        logger.warning(
            "Could not locate vLLM streaming-input sources for priority patch."
        )
        return False

    scheduler_add_original = (
        "            if existing.status != RequestStatus.WAITING_FOR_STREAMING_REQ:\n"
        '                assert existing.streaming_queue is not None, "duplicate request id"\n'
        "                # Queue next input chunk (or finished sentinel).\n"
        "                existing.streaming_queue.append(update)\n"
        "            elif update is not None:\n"
        "                # Commence next input chunk.\n"
        "                self._update_request_as_session(existing, update)\n"
    )
    scheduler_add_v1 = (
        "            if existing.status != RequestStatus.WAITING_FOR_STREAMING_REQ:\n"
        '                assert existing.streaming_queue is not None, "duplicate request id"\n'
        "                # Queue next input chunk (or finished sentinel).\n"
        "                existing.streaming_queue.append(update)\n"
        "                # A final streaming chunk turns speculative prefill into\n"
        "                # foreground model-call work. Promote the live request\n"
        "                # immediately instead of waiting for its current dummy\n"
        "                # decode to finish; otherwise strict priority can block\n"
        "                # the request that the foreground caller is awaiting.\n"
        "                if update is not None and update.priority < existing.priority:\n"
        "                    queued_request_queue = None\n"
        "                    if existing.status == RequestStatus.WAITING:\n"
        "                        queued_request_queue = self.waiting\n"
        "                    elif self._is_blocked_waiting_status(existing.status):\n"
        "                        queued_request_queue = self.skipped_waiting\n"
        "                    if queued_request_queue is not None:\n"
        "                        queued_request_queue.remove_request(existing)\n"
        "                    existing.priority = update.priority\n"
        "                    existing.arrival_time = update.arrival_time\n"
        "                    if queued_request_queue is not None:\n"
        "                        queued_request_queue.add_request(existing)\n"
        "            elif update is not None:\n"
        "                # Commence next input chunk.\n"
        "                self._update_request_as_session(existing, update)\n"
    )
    scheduler_add_v2 = (
        "            if existing.status != RequestStatus.WAITING_FOR_STREAMING_REQ:\n"
        '                assert existing.streaming_queue is not None, "duplicate request id"\n'
        "                # Queue next input chunk (or finished sentinel).\n"
        "                existing.streaming_queue.append(update)\n"
        "                # A final streaming chunk turns speculative prefill into\n"
        "                # foreground model-call work. Promote the live request\n"
        "                # immediately instead of waiting for its current dummy\n"
        "                # decode to finish; otherwise strict priority can block\n"
        "                # the request that the foreground caller is awaiting.\n"
        "                if update is not None and update.priority < existing.priority:\n"
        "                    queued_request_queue = None\n"
        "                    if existing.status == RequestStatus.WAITING:\n"
        "                        queued_request_queue = self.waiting\n"
        "                    elif self._is_blocked_waiting_status(existing.status):\n"
        "                        queued_request_queue = self.skipped_waiting\n"
        "                    if queued_request_queue is not None:\n"
        "                        queued_request_queue.remove_request(existing)\n"
        "                    existing.priority = update.priority\n"
        "                    existing.arrival_time = update.arrival_time\n"
        "                    if queued_request_queue is not None:\n"
        "                        queued_request_queue.add_request(existing)\n"
        "            elif update is not None:\n"
        "                # WAITING_FOR_STREAMING_REQ lives in skipped_waiting. A\n"
        "                # priority-changing continuation must be removed before\n"
        "                # mutating the heap key, then inserted into the queue\n"
        "                # selected by its new WAITING status and priority.\n"
        "                self.skipped_waiting.remove_request(existing)\n"
        "                self._update_request_as_session(existing, update)\n"
        "                self._enqueue_waiting_request(existing)\n"
    )

    replacements = (
        (
            protocol_file,
            (
                "    prompt: EngineInput\n"
                "    sampling_params: SamplingParams | None = None\n",
            ),
            "    prompt: EngineInput\n"
            "    sampling_params: SamplingParams | None = None\n"
            "    priority: int | None = None\n",
        ),
        (
            async_llm_file,
            (
                "                    # TODO(nick): Avoid re-validating reused sampling parameters\n"
                "                    req = self.input_processor.process_inputs(\n"
                "                        request_id=internal_req_id,\n"
                "                        prompt=input_chunk.prompt,\n"
                "                        params=sp,\n"
                "                        resumable=True,\n"
                "                        **inputs,  # type: ignore[arg-type]\n"
                "                    )\n",
            ),
            "                    chunk_inputs = inputs\n"
            "                    if input_chunk.priority is not None:\n"
            "                        chunk_inputs = dict(\n"
            "                            inputs, priority=input_chunk.priority\n"
            "                        )\n"
            "                    # TODO(nick): Avoid re-validating reused sampling parameters\n"
            "                    req = self.input_processor.process_inputs(\n"
            "                        request_id=internal_req_id,\n"
            "                        prompt=input_chunk.prompt,\n"
            "                        params=sp,\n"
            "                        resumable=True,\n"
            "                        **chunk_inputs,  # type: ignore[arg-type]\n"
            "                    )\n",
        ),
        (
            request_file,
            ("    sampling_params: SamplingParams | None\n\n    @classmethod\n",),
            "    sampling_params: SamplingParams | None\n"
            "    priority: int\n\n"
            "    @classmethod\n",
        ),
        (
            request_file,
            (
                "            arrival_time=request.arrival_time,\n"
                "            sampling_params=request.sampling_params,\n"
                "        )\n",
            ),
            "            arrival_time=request.arrival_time,\n"
            "            sampling_params=request.sampling_params,\n"
            "            priority=request.priority,\n"
            "        )\n",
        ),
        (
            scheduler_file,
            (
                "        assert update.sampling_params.max_tokens is not None\n"
                "        session.max_tokens = update.sampling_params.max_tokens\n"
                "        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:\n",
            ),
            "        assert update.sampling_params.max_tokens is not None\n"
            "        session.max_tokens = update.sampling_params.max_tokens\n"
            "        session.priority = update.priority\n"
            "        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:\n",
        ),
        (
            scheduler_file,
            (scheduler_add_original, scheduler_add_v1),
            scheduler_add_v2,
        ),
    )

    # All vLLM worker processes patch the same shared environment. A single
    # feature-level lock both serializes them and lets us preflight every file
    # before changing any of them.
    lock_path = protocol_file + ".streaming_session_priority_patch_lock"
    with _exclusive_patch_lock(lock_path):
        source_contents: dict[str, str] = {}
        patched_contents: dict[str, str] = {}
        for file_path, old_snippets, new_snippet in replacements:
            if file_path not in source_contents:
                with open(file_path) as source_file:
                    source_contents[file_path] = source_file.read()
                patched_contents[file_path] = source_contents[file_path]
            content = patched_contents[file_path]
            if new_snippet in content:
                continue
            if not any(old_snippet in content for old_snippet in old_snippets):
                logger.warning(
                    "Could not apply vLLM streaming-session priority patch: "
                    "expected snippet not found in %s.",
                    file_path,
                )
                return False
            old_snippet = next(
                old_snippet for old_snippet in old_snippets if old_snippet in content
            )
            patched_contents[file_path] = content.replace(old_snippet, new_snippet, 1)

        for file_path, content in patched_contents.items():
            if content == source_contents[file_path]:
                continue
            with open(file_path, "w") as source_file:
                source_file.write(content)

    logger.info("Successfully patched vLLM streaming-session priority updates.")
    return True


def _patch_vllm_strict_priority_scheduling(logger) -> bool:
    """Bound lower-priority prefill sharing with foreground work.

    vLLM 0.20 orders newly admitted requests by priority, but once a request is
    RUNNING it is scheduled before the WAITING queue without another priority
    comparison. A background streaming prefill can therefore consume the
    token budget before a newly arrived foreground request, or share a large
    prefill batch with foreground decode. Enforce priority across both queues.

    The NeMo RL worker exposes a non-negative foreground-slack limit through
    ``NEMO_RL_VLLM_BACKGROUND_PREFILL_MAX_FOREGROUND_REQUESTS``. Zero keeps
    strict idle-only behavior. Positive values allow one background request
    per scheduler step only after all foreground requests have been ordered
    first, and only while the active foreground count is within the limit.
    ``NEMO_RL_VLLM_BACKGROUND_PREFILL_MAX_TOKENS_PER_STEP`` can additionally
    cap that request's tokens per scheduler step so a cache-page fill is spread
    across multiple foreground decode steps. An unset foreground limit leaves
    non-streaming vLLM scheduling unchanged.

    Returns whether the installed source already contains or accepted the
    guarded patch.
    """
    try:
        scheduler_file = _get_vllm_file("v1/core/sched/scheduler.py")
    except RuntimeError:
        logger.warning("Could not locate vLLM scheduler for strict priority patch.")
        return False

    import_original = "import itertools\nimport time\n"
    import_patched = "import itertools\nimport os\nimport time\n"
    init_original = (
        "        self.scheduler_config = vllm_config.scheduler_config\n"
        "        self.cache_config = vllm_config.cache_config\n"
    )
    init_bounded_legacy = (
        "        self.scheduler_config = vllm_config.scheduler_config\n"
        "        _nemo_rl_background_slack = os.environ.get(\n"
        '            "NEMO_RL_VLLM_BACKGROUND_PREFILL_MAX_FOREGROUND_REQUESTS"\n'
        "        )\n"
        "        self._nemo_rl_background_prefill_max_foreground_requests = None\n"
        "        if _nemo_rl_background_slack is not None:\n"
        "            try:\n"
        "                _nemo_rl_background_slack = int(_nemo_rl_background_slack)\n"
        "            except ValueError as exc:\n"
        "                raise ValueError(\n"
        '                    "NEMO_RL_VLLM_BACKGROUND_PREFILL_MAX_FOREGROUND_REQUESTS "\n'
        '                    "must be a non-negative integer"\n'
        "                ) from exc\n"
        "            if _nemo_rl_background_slack < 0:\n"
        "                raise ValueError(\n"
        '                    "NEMO_RL_VLLM_BACKGROUND_PREFILL_MAX_FOREGROUND_REQUESTS "\n'
        '                    "must be a non-negative integer"\n'
        "                )\n"
        "            self._nemo_rl_background_prefill_max_foreground_requests = (\n"
        "                _nemo_rl_background_slack\n"
        "            )\n"
        "        self.cache_config = vllm_config.cache_config\n"
    )
    init_patched = (
        "        self.scheduler_config = vllm_config.scheduler_config\n"
        "        _nemo_rl_background_slack = os.environ.get(\n"
        '            "NEMO_RL_VLLM_BACKGROUND_PREFILL_MAX_FOREGROUND_REQUESTS"\n'
        "        )\n"
        "        _nemo_rl_background_max_tokens = os.environ.get(\n"
        '            "NEMO_RL_VLLM_BACKGROUND_PREFILL_MAX_TOKENS_PER_STEP"\n'
        "        )\n"
        "        self._nemo_rl_background_prefill_max_foreground_requests = None\n"
        "        self._nemo_rl_background_prefill_max_tokens_per_step = None\n"
        "        if _nemo_rl_background_slack is not None:\n"
        "            try:\n"
        "                _nemo_rl_background_slack = int(_nemo_rl_background_slack)\n"
        "            except ValueError as exc:\n"
        "                raise ValueError(\n"
        '                    "NEMO_RL_VLLM_BACKGROUND_PREFILL_MAX_FOREGROUND_REQUESTS "\n'
        '                    "must be a non-negative integer"\n'
        "                ) from exc\n"
        "            if _nemo_rl_background_slack < 0:\n"
        "                raise ValueError(\n"
        '                    "NEMO_RL_VLLM_BACKGROUND_PREFILL_MAX_FOREGROUND_REQUESTS "\n'
        '                    "must be a non-negative integer"\n'
        "                )\n"
        "            self._nemo_rl_background_prefill_max_foreground_requests = (\n"
        "                _nemo_rl_background_slack\n"
        "            )\n"
        "        if _nemo_rl_background_max_tokens is not None:\n"
        "            try:\n"
        "                _nemo_rl_background_max_tokens = int(\n"
        "                    _nemo_rl_background_max_tokens\n"
        "                )\n"
        "            except ValueError as exc:\n"
        "                raise ValueError(\n"
        '                    "NEMO_RL_VLLM_BACKGROUND_PREFILL_MAX_TOKENS_PER_STEP "\n'
        '                    "must be a non-negative integer"\n'
        "                ) from exc\n"
        "            if _nemo_rl_background_max_tokens < 0:\n"
        "                raise ValueError(\n"
        '                    "NEMO_RL_VLLM_BACKGROUND_PREFILL_MAX_TOKENS_PER_STEP "\n'
        '                    "must be a non-negative integer"\n'
        "                )\n"
        "            if _nemo_rl_background_max_tokens > 0:\n"
        "                self._nemo_rl_background_prefill_max_tokens_per_step = (\n"
        "                    _nemo_rl_background_max_tokens\n"
        "                )\n"
        "        self.cache_config = vllm_config.cache_config\n"
    )
    schedule_original = (
        "        token_budget = self.max_num_scheduled_tokens\n"
        "        if self._pause_state == PauseState.PAUSED_ALL:\n"
    )
    schedule_patched = (
        "        token_budget = self.max_num_scheduled_tokens\n"
        "        nemo_rl_background_prefills_scheduled = 0\n"
        "        if (\n"
        "            self.policy == SchedulingPolicy.PRIORITY\n"
        "            and self._nemo_rl_background_prefill_max_foreground_requests\n"
        "            is not None\n"
        "            and any(request.priority > 0 for request in self.running)\n"
        "        ):\n"
        "            # Stable ordering guarantees foreground work consumes the\n"
        "            # token budget before an admitted background prefill.\n"
        "            self.running.sort(key=lambda request: request.priority)\n"
        "        if self._pause_state == PauseState.PAUSED_ALL:\n"
    )
    running_original = (
        "            request = self.running[req_index]\n\n            if (\n"
    )
    running_legacy = (
        "            request = self.running[req_index]\n"
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
    )
    running_patched = (
        "            request = self.running[req_index]\n"
        "\n"
        "            if (\n"
        "                self.policy == SchedulingPolicy.PRIORITY\n"
        "                and request.priority > 0\n"
        "                and self._nemo_rl_background_prefill_max_foreground_requests\n"
        "                is not None\n"
        "            ):\n"
        "                nemo_rl_higher_priority_running = sum(\n"
        "                    candidate.priority < request.priority\n"
        "                    for candidate in self.running\n"
        "                )\n"
        "                nemo_rl_higher_priority_waiting = (\n"
        "                    self.waiting\n"
        "                    and self.waiting.peek_request().priority < request.priority\n"
        "                ) or (\n"
        "                    self.skipped_waiting\n"
        "                    and self.skipped_waiting.peek_request().priority\n"
        "                    < request.priority\n"
        "                )\n"
        "                if (\n"
        "                    nemo_rl_higher_priority_running\n"
        "                    > self._nemo_rl_background_prefill_max_foreground_requests\n"
        "                    or nemo_rl_higher_priority_waiting\n"
        "                    or nemo_rl_background_prefills_scheduled >= 1\n"
        "                ):\n"
        "                    req_index += 1\n"
        "                    continue\n"
        "\n"
        "            if (\n"
    )
    # A prior bounded-patch revision checked the short pristine prefix before
    # the full legacy guard. Because the pristine prefix is also the start of
    # the legacy guard, it inserted the bounded guard ahead of strict priority
    # instead of replacing strict priority. Detect and repair that exact
    # composition before the normal idempotence checks below.
    running_malformed_upgrade = (
        running_patched + running_legacy[len(running_original) :]
    )
    running_account_original = (
        "            token_budget -= num_new_tokens\n            req_index += 1\n"
    )
    running_account_patched = (
        "            token_budget -= num_new_tokens\n"
        "            if (\n"
        "                self._nemo_rl_background_prefill_max_foreground_requests\n"
        "                is not None\n"
        "                and request.priority > 0\n"
        "            ):\n"
        "                nemo_rl_background_prefills_scheduled += 1\n"
        "            req_index += 1\n"
    )
    running_token_cap_original = (
        "            num_new_tokens = min(num_new_tokens, token_budget)\n"
        "\n"
        "            # Make sure the input position does not exceed the max model len.\n"
    )
    running_token_cap_patched = (
        "            num_new_tokens = min(num_new_tokens, token_budget)\n"
        "            if (\n"
        "                request.priority > 0\n"
        "                and self._nemo_rl_background_prefill_max_foreground_requests\n"
        "                is not None\n"
        "                and self._nemo_rl_background_prefill_max_tokens_per_step\n"
        "                is not None\n"
        "            ):\n"
        "                num_new_tokens = min(\n"
        "                    num_new_tokens,\n"
        "                    self._nemo_rl_background_prefill_max_tokens_per_step,\n"
        "                )\n"
        "\n"
        "            # Make sure the input position does not exceed the max model len.\n"
    )
    waiting_original = (
        "                request = request_queue.peek_request()\n"
        "                request_id = request.request_id\n"
        "\n"
        "                # try to promote blocked statuses while traversing skipped queue.\n"
    )
    waiting_legacy = (
        "                request = request_queue.peek_request()\n"
        "                request_id = request.request_id\n"
        "\n"
        "                if (\n"
        "                    self.policy == SchedulingPolicy.PRIORITY\n"
        "                    and any(\n"
        "                        candidate.priority < request.priority\n"
        "                        for candidate in self.running\n"
        "                    )\n"
        "                ):\n"
        "                    break\n"
        "\n"
        "                # try to promote blocked statuses while traversing skipped queue.\n"
    )
    waiting_patched = (
        "                request = request_queue.peek_request()\n"
        "                request_id = request.request_id\n"
        "\n"
        "                if (\n"
        "                    self.policy == SchedulingPolicy.PRIORITY\n"
        "                    and request.priority > 0\n"
        "                    and self._nemo_rl_background_prefill_max_foreground_requests\n"
        "                    is not None\n"
        "                ):\n"
        "                    nemo_rl_higher_priority_running = sum(\n"
        "                        candidate.priority < request.priority\n"
        "                        for candidate in self.running\n"
        "                    )\n"
        "                    if (\n"
        "                        nemo_rl_higher_priority_running\n"
        "                        > self._nemo_rl_background_prefill_max_foreground_requests\n"
        "                        or nemo_rl_background_prefills_scheduled >= 1\n"
        "                    ):\n"
        "                        break\n"
        "\n"
        "                # try to promote blocked statuses while traversing skipped queue.\n"
    )
    waiting_account_original = (
        "                token_budget -= num_new_tokens\n"
        "                request.status = RequestStatus.RUNNING\n"
    )
    waiting_account_patched = (
        "                token_budget -= num_new_tokens\n"
        "                if (\n"
        "                    self._nemo_rl_background_prefill_max_foreground_requests\n"
        "                    is not None\n"
        "                    and request.priority > 0\n"
        "                ):\n"
        "                    nemo_rl_background_prefills_scheduled += 1\n"
        "                request.status = RequestStatus.RUNNING\n"
    )
    waiting_token_cap_original = (
        "                    num_new_tokens = min(num_new_tokens, token_budget)\n"
        "                    assert num_new_tokens > 0\n"
    )
    waiting_token_cap_patched = (
        "                    num_new_tokens = min(num_new_tokens, token_budget)\n"
        "                    if (\n"
        "                        request.priority > 0\n"
        "                        and self._nemo_rl_background_prefill_max_foreground_requests\n"
        "                        is not None\n"
        "                        and self._nemo_rl_background_prefill_max_tokens_per_step\n"
        "                        is not None\n"
        "                    ):\n"
        "                        num_new_tokens = min(\n"
        "                            num_new_tokens,\n"
        "                            self._nemo_rl_background_prefill_max_tokens_per_step,\n"
        "                        )\n"
        "                    assert num_new_tokens > 0\n"
    )

    with _locked_file_patch(scheduler_file) as (content, write_back):
        if running_malformed_upgrade in content:
            content = content.replace(running_malformed_upgrade, running_patched, 1)
        replacements = (
            ((import_original,), import_patched),
            ((init_original, init_bounded_legacy), init_patched),
            ((schedule_original,), schedule_patched),
            ((running_legacy, running_original), running_patched),
            ((running_token_cap_original,), running_token_cap_patched),
            ((running_account_original,), running_account_patched),
            ((waiting_original, waiting_legacy), waiting_patched),
            ((waiting_token_cap_original,), waiting_token_cap_patched),
            ((waiting_account_original,), waiting_account_patched),
        )
        for originals, patched in replacements:
            if patched in content:
                continue
            if not any(original in content for original in originals):
                logger.warning(
                    "Could not apply vLLM bounded priority patch: expected snippet "
                    "not found in %s.",
                    scheduler_file,
                )
                return False

        for originals, patched in replacements:
            if patched in content:
                continue
            original = next(original for original in originals if original in content)
            content = content.replace(original, patched, 1)
        write_back(content)

    logger.info("Successfully patched vLLM bounded priority scheduling.")
    return True


def _patch_vllm_streaming_session_output_state(logger) -> bool:
    """Keep vLLM output state consistent across streaming-input chunks.

    vLLM 0.20 updates the scheduler-side sampling parameters when a resumable
    request receives another chunk, but its output processor only appends the
    prompt tokens. This leaves the original detokenizer and logprob processor
    alive. For same-request prefill, that state contains the discarded dummy
    token and was created without logprobs; the authoritative final decode then
    requests logprobs and crashes in ``LogprobsProcessor``.

    Rebuild the output-side processors from the accumulated authoritative
    prompt and the new chunk's sampling parameters. Use a shallow request copy
    for that output-only view: AsyncLLM sends the original request to the
    scheduler after updating the output processor, and the scheduler must still
    receive only the incremental chunk. The scheduler can adopt a queued final
    chunk before the output processor consumes the prior dummy completion. In
    that one transition, ignore logprobs attached under the new sampling
    contract and suppress the discarded dummy before AsyncLLM's aggregate
    queue. Otherwise the queue either requires logprobs for the dummy or merges
    its token into the authoritative final output. This mirrors the scheduler
    without paying to calculate dummy-token logprobs. Guard every replacement
    so future vLLM versions fail closed instead of receiving a partial source
    patch.
    """
    try:
        output_processor_file = _get_vllm_file("v1/engine/output_processor.py")
    except RuntimeError:
        logger.warning(
            "Could not locate vLLM output processor for streaming-input patch."
        )
        return False

    original_apply_snippet = (
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
    )
    unsafe_apply_snippet = (
        "    def apply_streaming_update(self, update: StreamingUpdate) -> None:\n"
        "        # Mirror Scheduler._update_request_as_session: the prior chunk's\n"
        "        # sampled dummy token is not part of the resumed prompt. Recreate\n"
        "        # output-side state from the accumulated real prompt so neither\n"
        "        # detokenization nor logprobs retain that discarded token.\n"
        "        self.streaming_input = not update.final\n"
        "        if update.prompt:\n"
        "            self.prompt = (\n"
        "                (self.prompt + update.prompt) if self.prompt else update.prompt\n"
        "            )\n"
        "        if self.prompt_token_ids:\n"
        "            self.prompt_token_ids.extend(update.prompt_token_ids or ())\n"
        "        else:\n"
        "            self.prompt_token_ids = update.prompt_token_ids or []\n"
        "        assert self.prompt_token_ids is not None\n"
        "\n"
        "        request = update.request\n"
        "        request.prompt_token_ids = list(self.prompt_token_ids)\n"
        "        request.prompt_embeds = None\n"
        "        sampling_params = request.sampling_params\n"
        "        assert sampling_params is not None\n"
        "        tokenizer = self.tokenizer if sampling_params.detokenize else None\n"
        "        self.output_kind = sampling_params.output_kind\n"
        "        self.logprobs_processor = LogprobsProcessor.from_new_request(\n"
        "            tokenizer=tokenizer, request=request\n"
        "        )\n"
        "        self.detokenizer = IncrementalDetokenizer.from_new_request(\n"
        "            tokenizer=tokenizer, request=request\n"
        "        )\n"
        "        self.prompt_embeds = None\n"
        "        self.prompt_len = len(self.prompt_token_ids)\n"
        "        self.max_tokens_param = sampling_params.max_tokens\n"
        "        self.top_p = sampling_params.top_p\n"
        "        self.n = sampling_params.n\n"
        "        self.temperature = sampling_params.temperature\n"
        "        self.sent_tokens_offset = 0\n"
        "        if self.queue is not None:\n"
        "            self.queue.aggregate = (\n"
        "                sampling_params.output_kind == RequestOutputKind.DELTA\n"
        "            )\n"
        "        if self.stats is not None:\n"
        "            self.stats.arrival_time = update.arrival_time\n"
        "        self.is_prefilling = True\n"
    )
    safe_apply_snippet = unsafe_apply_snippet.replace(
        "        request = update.request\n"
        "        request.prompt_token_ids = list(self.prompt_token_ids)\n"
        "        request.prompt_embeds = None\n"
        "        sampling_params = request.sampling_params\n",
        "        request = update.request\n"
        "        output_request = copy(request)\n"
        "        output_request.prompt_token_ids = list(self.prompt_token_ids)\n"
        "        output_request.prompt_embeds = None\n"
        "        sampling_params = request.sampling_params\n",
    ).replace(
        "            tokenizer=tokenizer, request=request\n",
        "            tokenizer=tokenizer, request=output_request\n",
    )
    transition_logprobs_original = (
        "                req_state.logprobs_processor.update_from_output("
        "engine_core_output)\n"
    )
    transition_logprobs_previous = (
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
    transition_output_patched = (
        "                # The scheduler may adopt a queued final streaming chunk\n"
        "                # before this processor consumes the prior dummy completion.\n"
        "                # That can attach the final chunk's logprob contract to the\n"
        "                # dummy output even though the old processor requested none.\n"
        "                # Suppress that dummy before AsyncLLM's aggregate queue; the\n"
        "                # authoritative final chunk immediately replaces it.\n"
        "                queued_streaming_update = (\n"
        "                    req_state.input_chunk_queue[0]\n"
        "                    if req_state.input_chunk_queue\n"
        "                    else None\n"
        "                )\n"
        "                transition_dummy_output = (\n"
        "                    req_state.streaming_input\n"
        "                    and queued_streaming_update is not None\n"
        "                    and queued_streaming_update.final\n"
        "                    and finish_reason is not None\n"
        "                )\n"
        "                transition_dummy_logprobs = (\n"
        "                    transition_dummy_output\n"
        "                    and req_state.logprobs_processor.num_logprobs is None\n"
        "                    and engine_core_output.new_logprobs is not None\n"
        "                )\n"
        "                if not transition_dummy_logprobs:\n"
        "                    req_state.logprobs_processor.update_from_output(\n"
        "                        engine_core_output\n"
        "                    )\n"
    )
    request_output_original = (
        "            if request_output := req_state.make_request_output(\n"
        "                new_token_ids,\n"
        "                pooling_output,\n"
        "                finish_reason,\n"
        "                stop_reason,\n"
        "                kv_transfer_params,\n"
        "                routed_experts,\n"
        "            ):\n"
    )
    request_output_patched = (
        "            request_output = (\n"
        "                None\n"
        "                if pooling_output is None and transition_dummy_output\n"
        "                else req_state.make_request_output(\n"
        "                    new_token_ids,\n"
        "                    pooling_output,\n"
        "                    finish_reason,\n"
        "                    stop_reason,\n"
        "                    kv_transfer_params,\n"
        "                    routed_experts,\n"
        "                )\n"
        "            )\n"
        "            if request_output:\n"
    )

    replacements = (
        ("import asyncio\n", "import asyncio\nfrom copy import copy\n"),
        (
            "    prompt_token_ids: list[int] | None\n"
            "    arrival_time: float\n"
            "    final: bool = False\n",
            "    prompt_token_ids: list[int] | None\n"
            "    arrival_time: float\n"
            "    request: EngineCoreRequest\n"
            "    final: bool = False\n",
        ),
        (
            "    def __init__(\n        self,\n        request_id: str,\n",
            "    def __init__(\n"
            "        self,\n"
            "        tokenizer: TokenizerLike | None,\n"
            "        request_id: str,\n",
        ),
        (
            "    ):\n"
            "        self.request_id = request_id\n"
            "        self.external_req_id = external_req_id\n",
            "    ):\n"
            "        self.tokenizer = tokenizer\n"
            "        self.request_id = request_id\n"
            "        self.external_req_id = external_req_id\n",
        ),
        (original_apply_snippet, safe_apply_snippet),
        (
            "        return cls(\n            request_id=request.request_id,\n",
            "        return cls(\n"
            "            tokenizer=tokenizer,\n"
            "            request_id=request.request_id,\n",
        ),
        (
            "        update = StreamingUpdate(\n"
            "            prompt=prompt,\n"
            "            prompt_token_ids=request.prompt_token_ids,\n"
            "            arrival_time=request.arrival_time,\n"
            "        )\n",
            "        update = StreamingUpdate(\n"
            "            prompt=prompt,\n"
            "            prompt_token_ids=request.prompt_token_ids,\n"
            "            arrival_time=request.arrival_time,\n"
            "            request=request,\n"
            "        )\n",
        ),
        (transition_logprobs_original, transition_output_patched),
        (request_output_original, request_output_patched),
    )

    lock_path = output_processor_file + ".streaming_session_output_state_patch_lock"
    with _exclusive_patch_lock(lock_path):
        with open(output_processor_file) as source_file:
            content = source_file.read()

        # Upgrade the first implementation without ever publishing a partial
        # patch. It rebuilt output state correctly for one continuation, but
        # mutated the EngineCoreRequest before AsyncLLM sent that same object
        # to the scheduler, turning later incremental chunks into accumulated
        # chunks and duplicating the prompt.
        if unsafe_apply_snippet in content:
            content = content.replace(unsafe_apply_snippet, safe_apply_snippet, 1)

        # Upgrade the low-overhead logprob-only guard. It prevented the old
        # LogprobsProcessor assertion but still allowed the discarded dummy to
        # enter AsyncLLM's aggregate queue.
        if transition_logprobs_previous in content:
            content = content.replace(
                transition_logprobs_previous, transition_output_patched, 1
            )

        for old_snippet, new_snippet in replacements:
            if new_snippet in content:
                continue
            if old_snippet not in content:
                logger.warning(
                    "Could not apply vLLM streaming-session output-state patch: "
                    "expected snippet not found in %s.",
                    output_processor_file,
                )
                return False

        for old_snippet, new_snippet in replacements:
            if new_snippet in content:
                continue
            content = content.replace(old_snippet, new_snippet, 1)

        with open(output_processor_file, "w") as source_file:
            source_file.write(content)

    logger.info("Successfully patched vLLM streaming-session output state.")
    return True


def _patch_vllm_streaming_session_support(logger) -> dict[str, bool]:
    """Apply the cross-file streaming patch suite as one transaction.

    Every vLLM worker imports and patches the same shared environment. The
    individual helpers protect their own files, but that is not sufficient
    when two helpers both rewrite ``scheduler.py``: another worker can observe
    a partially composed suite and report a false incompatibility. Serialize
    the complete suite with one feature-level lock while retaining the
    per-file locks for callers that exercise an individual helper.
    """
    try:
        scheduler_file = _get_vllm_file("v1/core/sched/scheduler.py")
    except RuntimeError:
        logger.warning(
            "Could not locate the vLLM scheduler for the streaming patch suite."
        )
        return {
            "streaming_session_max_tokens": False,
            "streaming_session_priority": False,
            "strict_priority_scheduling": False,
            "streaming_session_output_state": False,
        }

    lock_path = scheduler_file + ".streaming_session_patch_suite_lock"
    with _exclusive_patch_lock(lock_path):
        return {
            "streaming_session_max_tokens": (
                _patch_vllm_streaming_session_max_tokens(logger)
            ),
            "streaming_session_priority": (
                _patch_vllm_streaming_session_priority(logger)
            ),
            "strict_priority_scheduling": (
                _patch_vllm_strict_priority_scheduling(logger)
            ),
            "streaming_session_output_state": (
                _patch_vllm_streaming_session_output_state(logger)
            ),
        }


def _apply_vllm_patches(
    py_executable: str, *, extra_env_vars: list[str] | None = None
) -> dict[str, bool]:
    # Import lazily so importing the worker module does not import vLLM.
    from vllm.logger import init_logger

    patch_logger = init_logger("vllm_patch")

    _patch_vllm_init_workers_ray(py_executable, extra_env_vars)
    patch_logger.info("Successfully patched vllm _init_workers_ray.")

    _patch_vllm_llama_eagle3_own_lm_head(patch_logger)
    _patch_vllm_hermes_tool_parser_thread_safety(patch_logger)
    return _patch_vllm_streaming_session_support(patch_logger)
