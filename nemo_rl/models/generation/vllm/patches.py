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

    with _locked_file_patch(file_to_patch) as (content, write_back):
        if new_snippet in content:
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

    replacements = (
        (
            protocol_file,
            "    prompt: EngineInput\n"
            "    sampling_params: SamplingParams | None = None\n",
            "    prompt: EngineInput\n"
            "    sampling_params: SamplingParams | None = None\n"
            "    priority: int | None = None\n",
        ),
        (
            async_llm_file,
            "                    # TODO(nick): Avoid re-validating reused sampling parameters\n"
            "                    req = self.input_processor.process_inputs(\n"
            "                        request_id=internal_req_id,\n"
            "                        prompt=input_chunk.prompt,\n"
            "                        params=sp,\n"
            "                        resumable=True,\n"
            "                        **inputs,  # type: ignore[arg-type]\n"
            "                    )\n",
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
            "    sampling_params: SamplingParams | None\n\n    @classmethod\n",
            "    sampling_params: SamplingParams | None\n"
            "    priority: int\n\n"
            "    @classmethod\n",
        ),
        (
            request_file,
            "            arrival_time=request.arrival_time,\n"
            "            sampling_params=request.sampling_params,\n"
            "        )\n",
            "            arrival_time=request.arrival_time,\n"
            "            sampling_params=request.sampling_params,\n"
            "            priority=request.priority,\n"
            "        )\n",
        ),
        (
            scheduler_file,
            "        assert update.sampling_params.max_tokens is not None\n"
            "        session.max_tokens = update.sampling_params.max_tokens\n"
            "        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:\n",
            "        assert update.sampling_params.max_tokens is not None\n"
            "        session.max_tokens = update.sampling_params.max_tokens\n"
            "        session.priority = update.priority\n"
            "        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:\n",
        ),
    )

    # All vLLM worker processes patch the same shared environment. A single
    # feature-level lock both serializes them and lets us preflight every file
    # before changing any of them.
    lock_path = protocol_file + ".streaming_session_priority_patch_lock"
    with _exclusive_patch_lock(lock_path):
        source_contents = {}
        for file_path, old_snippet, new_snippet in replacements:
            with open(file_path) as source_file:
                content = source_file.read()
            source_contents[file_path] = content
            if new_snippet in content:
                continue
            if old_snippet not in content:
                logger.warning(
                    "Could not apply vLLM streaming-session priority patch: "
                    "expected snippet not found in %s.",
                    file_path,
                )
                return False

        for file_path, old_snippet, new_snippet in replacements:
            content = source_contents[file_path]
            if new_snippet in content:
                continue
            with open(file_path, "w") as source_file:
                source_file.write(content.replace(old_snippet, new_snippet, 1))

    logger.info("Successfully patched vLLM streaming-session priority updates.")
    return True


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
    max_tokens_supported = _patch_vllm_streaming_session_max_tokens(patch_logger)
    priority_supported = _patch_vllm_streaming_session_priority(patch_logger)
    return {
        "streaming_session_max_tokens": max_tokens_supported,
        "streaming_session_priority": priority_supported,
    }
