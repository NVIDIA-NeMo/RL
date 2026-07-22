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


def _patch_vllm_init_workers_ray(
    py_executable: str, extra_env_vars: list[str] | None
) -> None:
    """Patch vLLM's Ray executor env propagation and worker runtime_env.

    1. Pass custom runtime_env in _init_workers_ray call (file patch).
        - This allows passing custom py_executable to worker initialization.
    2. Forward extra env vars to the Ray workers via vLLM's additive
       VLLM_RAY_EXTRA_ENV_VARS_TO_COPY hook (vLLM >= 0.25). NCCL_*, HF_*, and
       HUGGING_FACE_* vars are already copied by vLLM's default prefix list
       (this includes the NCCL_CUMEM_ENABLE/NCCL_NVLS_ENABLE workaround from
       https://github.com/NVIDIA-NeMo/RL/pull/898).
    """
    file_to_patch = _get_vllm_file("v1/executor/ray_executor.py")

    old_line = "self._init_workers_ray(placement_group)"
    new_line = (
        "self._init_workers_ray(placement_group, "
        f'runtime_env={{"py_executable": "{py_executable}"}})'
    )

    with _locked_file_patch(file_to_patch) as (content, write_back):
        if new_line not in content and old_line in content:
            write_back(content.replace(old_line, new_line))

    env_vars_to_copy = ["RAY_ENABLE_UV_RUN_RUNTIME_ENV", *(extra_env_vars or [])]
    existing = os.environ.get("VLLM_RAY_EXTRA_ENV_VARS_TO_COPY", "")
    merged = {
        var.strip() for var in (*existing.split(","), *env_vars_to_copy) if var.strip()
    }
    os.environ["VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"] = ",".join(sorted(merged))


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


def _patch_vllm_tool_parser_namespace_tool(logger) -> None:
    """Guard vLLM's NamespaceTool import for openai < 2.25.

    vLLM 0.25 imports ``openai.types.responses.NamespaceTool`` (added in
    openai 2.25.0) at the top of ``tool_parsers/utils.py``, but nemo-gym pins
    ``openai<=2.7.2`` and its child server venvs must match the parent's
    openai version exactly. NamespaceTool is only used in isinstance checks
    for Responses-API namespace tools, which cannot be constructed by an
    openai client that predates the feature, so a never-matching stub is a
    faithful fallback.
    """
    try:
        file_to_patch = _get_vllm_file("tool_parsers/utils.py")
    except RuntimeError:
        logger.warning(
            "Could not locate tool_parsers/utils.py for openai compat patch."
        )
        return

    old_snippet = (
        "from openai.types.responses import (\n"
        "    FunctionTool,\n"
        "    NamespaceTool,\n"
        "    ToolChoiceFunction,\n"
        ")\n"
    )

    new_snippet = (
        "from openai.types.responses import (\n"
        "    FunctionTool,\n"
        "    ToolChoiceFunction,\n"
        ")\n"
        "\n"
        "try:\n"
        "    from openai.types.responses import NamespaceTool\n"
        "except ImportError:  # openai < 2.25.0 predates namespace tools\n"
        "\n"
        "    class NamespaceTool:  # type: ignore[no-redef]\n"
        '        """Stub: openai<2.25 clients cannot construct namespace tools."""\n'
        "\n"
    )

    with _locked_file_patch(file_to_patch) as (content, write_back):
        if "except ImportError:  # openai < 2.25.0 predates namespace tools" in content:
            logger.info("vLLM NamespaceTool openai compat patch already applied.")
            return

        if old_snippet not in content:
            logger.warning(
                "Could not apply NamespaceTool openai compat patch: "
                "expected import block not found in %s. "
                "The vLLM version may have changed.",
                file_to_patch,
            )
            return

        content = content.replace(old_snippet, new_snippet, 1)
        write_back(content)

    logger.info("Successfully patched vLLM NamespaceTool import for openai compat.")


def _patch_vllm_ray_executor_v2_tcpstore_port(logger) -> None:
    """Keep RayExecutorV2's TCPStore port out of the MessageQueue's scan range.

    vLLM 0.25's ``RayExecutorV2._init_executor`` picks the torch.distributed
    TCPStore port with a bind-probe (Step 3) but only binds it much later, in
    the rank-0 worker's ``init_process_group``. In between, Step 4 builds the
    broadcast ``MessageQueue``; when the engine spans nodes that queue needs a
    real TCP socket, so it calls ``get_open_port()`` and *binds and holds* the
    result (``shm_broadcast.py``: ``remote_subscribe_port = get_open_port()``
    then ``remote_socket.bind(...)``). Both searches start at ``VLLM_PORT``, so
    the queue deterministically takes the very port the probe just released and
    engine startup dies with ``EADDRINUSE`` (DeepSeek-V3 generation TP=32,
    observed on port 7000). Engines that fit on one node use a shm/ipc socket
    instead and never allocate a TCP port here, which is why only node-spanning
    engines are affected.

    Offsetting the TCPStore search past the queue's scan range removes the
    collision while keeping both ports inside the engine's 100-port window, and
    therefore below the OS ephemeral floor. That band is deliberate: leaving
    ``VLLM_PORT`` unset would send vLLM to kernel-assigned ephemeral ports and
    reintroduce the TOCTOU contention this layout exists to prevent (#2380,
    #3103).

    The offset must be applied *before* the ``local_dp_rank is None`` test, not
    inside it. vLLM's own disjoint-window branch below reads as if it only
    applies to DP engines, but ``ParallelConfig.__post_init__`` takes the
    "offline SPMD" path for every engine NeMo-RL builds and assigns
    ``data_parallel_rank_local = envs.VLLM_DP_RANK_LOCAL`` (0 by default) and
    ``data_parallel_master_port = envs.VLLM_DP_MASTER_PORT`` (0 by default). So
    a plain non-DP engine arrives here with ``local_dp_rank=0``, not ``None``:
    the ``None`` branch is dead, and the DP branch searches from
    ``0 + 100 + 0 * 32 = 100``, fails all 32 attempts on the privileged range,
    and falls through to ``get_open_port()`` — straight back to ``VLLM_PORT``.
    That is exactly the port the MessageQueue takes. See RL-1104.

    Returns without raising when the snippet is missing, but logs at warning
    level so a silent no-op is visible in worker logs.
    """
    try:
        file_to_patch = _get_vllm_file("v1/executor/ray_executor_v2.py")
    except RuntimeError:
        logger.warning(
            "Could not locate ray_executor_v2.py; TCPStore port patch NOT applied. "
            "Engines spanning nodes may fail with EADDRINUSE at startup."
        )
        return

    marker = "start_port=envs.VLLM_PORT + 32"
    old_snippet = (
        "        if local_dp_rank is None:\n            return get_open_port()\n"
    )
    new_snippet = (
        "        if envs.VLLM_PORT is not None:\n"
        "            # NeMo-RL: this port and the broadcast MessageQueue's remote\n"
        "            # socket are both allocated from VLLM_PORT, but the queue\n"
        "            # binds and holds its port before this one is bound in the\n"
        "            # rank-0 worker, so a shared search collides. Search a window\n"
        "            # past the queue's, still inside the engine's reserved\n"
        "            # 100-port band.\n"
        "            #\n"
        "            # This has to run *before* the local_dp_rank test below:\n"
        "            # ParallelConfig leaves a non-DP engine with\n"
        "            # data_parallel_rank_local=0 (not None) and\n"
        "            # data_parallel_master_port=0, so that branch searches from\n"
        "            # port 100, fails on the privileged range, and falls back to\n"
        "            # get_open_port() -- straight back to VLLM_PORT.\n"
        "            try:\n"
        "                return _get_open_port(\n"
        "                    start_port=envs.VLLM_PORT + 32, max_attempts=32\n"
        "                )\n"
        "            except RuntimeError:\n"
        "                pass\n"
        "        if local_dp_rank is None:\n"
        "            return get_open_port()\n"
    )

    with _locked_file_patch(file_to_patch) as (content, write_back):
        if marker in content:
            logger.info("vLLM RayExecutorV2 TCPStore port patch already applied.")
            return

        if old_snippet not in content:
            logger.warning(
                "Could not apply RayExecutorV2 TCPStore port patch: expected "
                "snippet not found in %s. The vLLM version may have changed. "
                "Engines spanning nodes may fail with EADDRINUSE at startup.",
                file_to_patch,
            )
            return

        content = content.replace(old_snippet, new_snippet, 1)
        write_back(content)

    # Read back so a patch that silently failed to land is not reported as
    # applied; this is the failure mode that previously went unnoticed.
    try:
        with open(file_to_patch) as handle:
            applied = marker in handle.read()
    except OSError as error:
        logger.warning("Could not verify TCPStore port patch: %s", error)
        return

    if applied:
        logger.info("Successfully patched vLLM RayExecutorV2 TCPStore port selection.")
    else:
        logger.warning(
            "RayExecutorV2 TCPStore port patch did not persist to %s. Engines "
            "spanning nodes may fail with EADDRINUSE at startup.",
            file_to_patch,
        )


def ensure_vllm_source_compat() -> None:
    """Apply interpreter-independent vLLM source-compat patches.

    Safe to call from any process that imports vLLM directly (e.g. the
    tools/model_diagnostics scripts, which construct ``vllm.LLM`` without
    going through a NeMo-RL generation worker). Must be called BEFORE the
    first ``import vllm`` submodule that pulls in ``vllm.tool_parsers``.
    Worker processes get this via ``_apply_vllm_patches`` at init.
    """
    from vllm.logger import init_logger

    _patch_vllm_tool_parser_namespace_tool(init_logger("vllm_patch"))


def _apply_vllm_patches(
    py_executable: str,
    *,
    extra_env_vars: list[str] | None = None,
) -> None:
    # Import lazily so importing the worker module does not import vLLM.
    from vllm.logger import init_logger

    patch_logger = init_logger("vllm_patch")

    _patch_vllm_init_workers_ray(py_executable, extra_env_vars)
    patch_logger.info("Successfully patched vllm _init_workers_ray.")

    _patch_vllm_llama_eagle3_own_lm_head(patch_logger)
    _patch_vllm_tool_parser_namespace_tool(patch_logger)
    _patch_vllm_ray_executor_v2_tcpstore_port(patch_logger)
