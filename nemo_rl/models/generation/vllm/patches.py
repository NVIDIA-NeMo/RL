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

import hashlib
import os
from contextlib import contextmanager
from functools import wraps
from importlib.util import find_spec
from typing import Any

G_FLASHINFER_TRTLLM_W13_KERNEL_ATTR = "_nrl_flashinfer_trtllm_w13_weight"
G_FLASHINFER_TRTLLM_W2_KERNEL_ATTR = "_nrl_flashinfer_trtllm_w2_weight"
G_FLASHINFER_TRTLLM_PATCH_ATTR = "_nrl_flashinfer_trtllm_refit_patch_applied"
G_FLASHINFER_TRTLLM_INTERMEDIATE_ALIGNMENT = 128
G_FLASHINFER_TRTLLM_ENGINE_CORE_PATCH_ATTR = (
    "_nrl_flashinfer_trtllm_refit_engine_core_patch_applied"
)
G_FLASHINFER_TRTLLM_ENGINE_CORE_ORIGINAL_ATTR = "_nrl_original_run_engine_core"
G_FLASHINFER_TRTLLM_RAY_WORKER_PATCH_ATTR = (
    "_nrl_flashinfer_trtllm_refit_ray_worker_patch_applied"
)
G_FLASHINFER_TRTLLM_RAY_WORKER_ORIGINAL_ATTR = "_nrl_original_initialize_worker"
G_FLASHINFER_TRTLLM_RAY_EXECUTOR_PATCH_ATTR = (
    "_nrl_flashinfer_trtllm_refit_ray_executor_patch_applied"
)
G_FLASHINFER_TRTLLM_RAY_EXECUTOR_ORIGINAL_ATTR = "_nrl_original_collective_rpc"
G_FLASHINFER_TRTLLM_RAY_EXECUTOR_WORKERS_PATCHED_ATTR = (
    "_nrl_flashinfer_trtllm_refit_workers_patched"
)


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


_ROUTED_EXPERTS_CAPTURER_SOURCE = (
    "model_executor/layers/fused_moe/routed_experts_capturer.py"
)
# This blob is shared by the live image's b31cc131 build and uv.lock's
# 6f11a9c5 source revision, despite other differences between those commits.
_ROUTED_EXPERTS_CAPTURER_V0251_SHA256 = (
    "e2e02d6322fd45edc143bdf2553a0fe6e2a010e1da3fbcea0df5887485a211c1"
)
_ROUTED_EXPERTS_CAPTURER_COMPACT_SHA256 = (
    "7aada566f750ae6c46d214b03f3b012757ea07f81ac3f0c2d6ce15956a4ad25b"
)


def _compact_routed_experts_capturer_source(content: str) -> str:
    """Apply the pinned vLLM 0.25.1 compact-layer capturer source edit."""
    replacements = (
        (
            "routed-layer resolver",
            """        "or 'num_local_experts'."
    )


class RoutedExpertsCapturer:""",
            '''        "or 'num_local_experts'."
    )


def get_routed_experts_layer_indices(hf_config) -> tuple[int, ...]:
    """Return backbone layer IDs represented by the routing payload.

    Hybrid Nemotron configs explicitly identify MoE blocks in
    ``layers_block_type``. Compress those sparse global layer IDs into a dense
    payload axis. Other model families retain the historical full-layer layout
    unless they expose an equivalent explicit block list.

    MTP block types intentionally live in ``mtp_layers_block_type`` and are not
    included: rollout routes replay only backbone MoE layers.
    """
    layer_types = getattr(hf_config, "layers_block_type", None)
    if isinstance(layer_types, (list, tuple)) and layer_types:
        moe_layer_ids = tuple(
            layer_id
            for layer_id, layer_type in enumerate(layer_types)
            if str(layer_type).lower() == "moe"
        )
        if moe_layer_ids:
            return moe_layer_ids

    num_hidden_layers = int(hf_config.num_hidden_layers)
    return tuple(range(num_hidden_layers))


class RoutedExpertsCapturer:''',
        ),
        (
            "capturer allocation",
            """        num_experts_per_tok = _get_num_experts_per_tok(hf_config)
        num_layers = hf_config.num_hidden_layers
        logger.info(
            "RoutedExpertsCapturer: allocating buffer with "
            "max_tokens=%d, num_layers=%d, num_experts_per_tok=%d "
            "(hf_config.model_type=%s)",
            max_num_batched_tokens,
            num_layers,
            num_experts_per_tok,
            getattr(hf_config, "model_type", "unknown"),
        )""",
            """        num_experts_per_tok = _get_num_experts_per_tok(hf_config)
        routed_layer_ids = get_routed_experts_layer_indices(hf_config)
        num_layers = len(routed_layer_ids)
        logger.info(
            "RoutedExpertsCapturer: allocating buffer with "
            "max_tokens=%d, routed_layers=%d, num_experts_per_tok=%d "
            "(hf_config.model_type=%s, global_layer_ids=%s)",
            max_num_batched_tokens,
            num_layers,
            num_experts_per_tok,
            getattr(hf_config, "model_type", "unknown"),
            routed_layer_ids,
        )""",
        ),
        (
            "global-to-compact layer map",
            """        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size

    def capture""",
            """        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.layer_id_to_capture_index = {
            layer_id: capture_index
            for capture_index, layer_id in enumerate(routed_layer_ids)
        }

    def capture""",
        ),
        (
            "capturer layer lookup",
            """        # Defensive: model may expose more layers than the capture buffer
        # was sized for (unusual, but guards against miss-config).
        if layer_id >= self.device_buffer.shape[1]:
            return

        self.device_buffer[:token_num_per_dp, layer_id, :] = topk_ids[""",
            """        layer_id_to_capture_index = getattr(self, "layer_id_to_capture_index", None)
        capture_index = (
            layer_id
            if layer_id_to_capture_index is None
            else layer_id_to_capture_index.get(layer_id)
        )
        # Missing IDs include non-MoE backbone blocks and MTP routers.
        if (
            capture_index is None
            or capture_index < 0
            or capture_index >= self.device_buffer.shape[1]
        ):
            return

        self.device_buffer[:token_num_per_dp, capture_index, :] = topk_ids[""",
        ),
        (
            "manager routed-layer resolver",
            """        num_experts = get_num_experts(hf_config)
        num_experts_per_tok = _get_num_experts_per_tok(hf_config)
        max_num_slots = kv_cache_config.num_blocks * self.block_size""",
            """        num_experts = get_num_experts(hf_config)
        num_experts_per_tok = _get_num_experts_per_tok(hf_config)
        routed_layer_ids = get_routed_experts_layer_indices(hf_config)
        max_num_slots = kv_cache_config.num_blocks * self.block_size""",
        ),
        (
            "manager allocation",
            """                max_num_slots,
                hf_config.num_hidden_layers,
                num_experts_per_tok,""",
            """                max_num_slots,
                len(routed_layer_ids),
                num_experts_per_tok,""",
        ),
        (
            "manager allocation log",
            """            max_num_slots,
            hf_config.num_hidden_layers,
            hf_config.num_experts_per_tok,
            self.routed_experts_by_slot.dtype.name,""",
            """            max_num_slots,
            len(routed_layer_ids),
            num_experts_per_tok,
            self.routed_experts_by_slot.dtype.name,""",
        ),
    )

    for description, old_snippet, new_snippet in replacements:
        occurrences = content.count(old_snippet)
        if occurrences != 1:
            raise ValueError(
                f"expected exactly one {description} source anchor, found {occurrences}"
            )
        content = content.replace(old_snippet, new_snippet, 1)
    return content


def _patch_vllm_routed_experts_compact_layers(logger) -> bool:
    """Compact vLLM's routed-experts buffers to backbone MoE layers only.

    vLLM's engine processes import their own module state, so a normal Python
    monkeypatch in the NeMo-RL actor does not propagate. Patch the installed
    source before those processes start instead. The edit is pinned by source
    hashes to the image's known vLLM 0.25.1 implementation and to the desired
    result, keeping this independent of a mounted custom vLLM checkout while
    refusing to rewrite an unknown implementation.

    Returns whether the compact implementation is present after this call.
    """
    try:
        file_to_patch = _get_vllm_file(_ROUTED_EXPERTS_CAPTURER_SOURCE)
    except RuntimeError as error:
        logger.error("Could not locate routed-experts capturer: %s", error)
        return False

    with _locked_file_patch(file_to_patch) as (content, write_back):
        source_sha256 = hashlib.sha256(content.encode()).hexdigest()
        if source_sha256 == _ROUTED_EXPERTS_CAPTURER_COMPACT_SHA256:
            logger.info("vLLM compact routed-experts capturer source already present.")
            return True

        if source_sha256 != _ROUTED_EXPERTS_CAPTURER_V0251_SHA256:
            logger.error(
                "Refusing to patch unknown vLLM routed-experts capturer at %s "
                "(sha256=%s; expected stock=%s or compact=%s).",
                file_to_patch,
                source_sha256,
                _ROUTED_EXPERTS_CAPTURER_V0251_SHA256,
                _ROUTED_EXPERTS_CAPTURER_COMPACT_SHA256,
            )
            return False

        try:
            patched_content = _compact_routed_experts_capturer_source(content)
            compile(patched_content, file_to_patch, "exec")
        except (SyntaxError, ValueError) as error:
            logger.error("Could not build compact routed-experts source: %s", error)
            return False

        patched_sha256 = hashlib.sha256(patched_content.encode()).hexdigest()
        if patched_sha256 != _ROUTED_EXPERTS_CAPTURER_COMPACT_SHA256:
            logger.error(
                "Compact routed-experts source had unexpected sha256=%s "
                "(expected %s); source was not modified.",
                patched_sha256,
                _ROUTED_EXPERTS_CAPTURER_COMPACT_SHA256,
            )
            return False

        write_back(patched_content)

    logger.info(
        "Applied compact routed-experts source patch to %s (sha256=%s).",
        file_to_patch,
        _ROUTED_EXPERTS_CAPTURER_COMPACT_SHA256,
    )
    return True


def _patch_vllm_init_workers_ray(
    py_executable: str, extra_env_vars: list[str] | None
) -> bool:
    """Patch vLLM's Ray executor env propagation and worker runtime_env.

    1. Pass custom runtime_env in _init_workers_ray call (file patch).
        - This allows passing custom py_executable to worker initialization.
    2. Forward extra env vars to the Ray workers via vLLM's additive
       VLLM_RAY_EXTRA_ENV_VARS_TO_COPY hook (vLLM >= 0.25). NCCL_*, HF_*, and
       HUGGING_FACE_* vars are already copied by vLLM's default prefix list
       (this includes the NCCL_CUMEM_ENABLE/NCCL_NVLS_ENABLE workaround from
       https://github.com/NVIDIA-NeMo/RL/pull/898).

    .. note::
        Step 1 patches the **v1 Ray executor**, which vLLM 0.25 no longer
        selects by default: ``VLLM_USE_RAY_V2_EXECUTOR_BACKEND`` flipped from
        ``"0"`` (0.20) to ``"1"`` (0.25), so ``Executor.get_class`` returns
        ``RayExecutorV2`` for ray-backed engines. ``RayExecutorV2`` has no
        ``_init_workers_ray`` at all -- it creates workers inline, and its
        ``_build_runtime_env`` never sets ``py_executable``.

        The patch is kept because it is still load-bearing when
        ``VLLM_USE_RAY_V2_EXECUTOR_BACKEND=0`` selects the v1 executor. Under
        the 0.25 default it is inert, and workers get the right interpreter
        from Ray's per-field ``runtime_env`` inheritance instead: the parent
        NeMo-RL actor sets ``py_executable``, and a child created with a
        ``runtime_env`` that omits it inherits the parent's value.

        So a ``True`` return means "the anchor is in place", not "this is what
        put the workers on the right interpreter". The caller logs
        accordingly.

    Returns:
        Whether the v1 runtime_env source patch is in place. The env-var merge
        in step 2 cannot fail, but step 1 is anchored on a call-site string; if
        that moves upstream the py_executable injection silently stops
        happening, so the caller must not report success unconditionally.
    """
    file_to_patch = _get_vllm_file("v1/executor/ray_executor.py")

    old_line = "self._init_workers_ray(placement_group)"
    new_line = (
        "self._init_workers_ray(placement_group, "
        f'runtime_env={{"py_executable": "{py_executable}"}})'
    )

    applied = False
    with _locked_file_patch(file_to_patch) as (content, write_back):
        if new_line in content:
            applied = True  # already patched by another worker on this node
        elif old_line in content:
            write_back(content.replace(old_line, new_line))
            applied = True

    env_vars_to_copy = ["RAY_ENABLE_UV_RUN_RUNTIME_ENV", *(extra_env_vars or [])]
    existing = os.environ.get("VLLM_RAY_EXTRA_ENV_VARS_TO_COPY", "")
    merged = {
        var.strip() for var in (*existing.split(","), *env_vars_to_copy) if var.strip()
    }
    os.environ["VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"] = ",".join(sorted(merged))

    return applied


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


def _patch_vllm_shm_broadcast_bind_retry(logger) -> None:
    """Make MessageQueue's remote socket survive losing a port race.

    ``MessageQueue.__init__`` picks the port for its remote (TCP) socket with
    ``remote_subscribe_port = get_open_port()``, which *probes a port and
    releases it*, and only binds it with ZMQ several statements later
    (``shm_broadcast.py``: ``self.remote_socket.bind(socket_addr)``). The
    window between the probe and the bind is a TOCTOU race.

    On vLLM 0.25 that race is lost reliably, not occasionally. Every
    ``RayWorkerProc`` on a **non-driver** node takes ``n_local_reader=0``
    (``ray_executor_v2.py::_init_message_queues``), so every one of them needs
    a real TCP port, and they all scan from the same ``VLLM_PORT`` -- 7000 for
    a node-spanning engine. ``_init_message_queues`` runs immediately after
    ``init_device()``, whose process-group setup is a collective barrier, so
    all workers on the node arrive at the probe within microseconds of each
    other, all see the same port free, and all but one die with::

        zmq.error.ZMQError: Address already in use (addr='tcp://10.65.1.9:7000')

    Workers on the driver node take ``n_local_reader=1`` and use an ``ipc://``
    socket instead, which is why only node-spanning engines are affected --
    and why no nightly test catches it (none runs an engine whose
    ``tensor_parallel_size * pipeline_parallel_size`` exceeds
    ``cluster.gpus_per_node``). See RL-1111.

    Fix the race at the bind rather than the probe: retry, advancing past the
    port that was lost. This is safe and terminating because a port a peer
    already holds with ZMQ *is* visible to the next ``_get_open_port`` probe
    (a plain ``bind(("", port))`` on it fails with ``EADDRINUSE``), so each
    retry makes forward progress.

    Deliberately keeps the search anchored at ``VLLM_PORT`` instead of letting
    vLLM fall back to ``bind(("", 0))``: kernel-assigned ephemeral ports are
    exactly the TOCTOU contention the reserved sub-ephemeral band exists to
    prevent (#2380, #3103).

    Patching the bind (rather than handing each worker a private start port)
    also covers every other ``MessageQueue`` with a remote reader -- notably
    the executor's own ``rpc_broadcast_mq`` -- instead of the one call site
    that happens to be failing today.

    Returns without raising when the snippet is missing, but logs at warning
    level so a silent no-op is visible in worker logs.
    """
    try:
        file_to_patch = _get_vllm_file(
            "distributed/device_communicators/shm_broadcast.py"
        )
    except RuntimeError:
        logger.warning(
            "Could not locate shm_broadcast.py; MessageQueue bind-retry patch "
            "NOT applied. Engines spanning nodes may fail with EADDRINUSE at "
            "startup."
        )
        return

    marker = "_nrl_bind_attempts"
    old_snippet = (
        '            socket_addr = f"tcp://{connect_ip}:{remote_subscribe_port}"\n'
        "            self.remote_socket.bind(socket_addr)\n"
    )
    new_snippet = (
        "            # NeMo-RL: get_open_port() above probed this port and then\n"
        "            # released it; ZMQ only binds it for real here. Every worker\n"
        "            # on a non-driver node builds its response queue at the same\n"
        "            # instant (init_device()'s collective releases them together)\n"
        "            # scanning from the same VLLM_PORT, so they all probe the same\n"
        "            # free port and all but one die with EADDRINUSE. Retry around\n"
        "            # the bind instead of trusting the probe: a port a peer already\n"
        "            # holds IS visible to the next probe, so advancing past the\n"
        "            # loser terminates. Ports stay in the reserved VLLM_PORT band\n"
        "            # rather than falling back to kernel-ephemeral ones, which is\n"
        "            # the contention that band exists to avoid (#2380, #3103).\n"
        "            _nrl_bind_attempts = 64\n"
        "            for _nrl_bind_attempt in range(_nrl_bind_attempts):\n"
        '                socket_addr = f"tcp://{connect_ip}:{remote_subscribe_port}"\n'
        "                try:\n"
        "                    self.remote_socket.bind(socket_addr)\n"
        "                    break\n"
        "                except zmq.ZMQError:\n"
        "                    if _nrl_bind_attempt == _nrl_bind_attempts - 1:\n"
        "                        raise\n"
        "                    from vllm.utils.network_utils import _get_open_port\n"
        "\n"
        "                    logger.info(\n"
        '                        "Port %s was taken between probe and bind; '
        'retrying.",\n'
        "                        remote_subscribe_port,\n"
        "                    )\n"
        "                    remote_subscribe_port = (\n"
        "                        _get_open_port(start_port=remote_subscribe_port + 1)\n"
        "                        if envs.VLLM_PORT is not None\n"
        "                        else get_open_port()\n"
        "                    )\n"
    )

    with _locked_file_patch(file_to_patch) as (content, write_back):
        if marker in content:
            logger.info("vLLM MessageQueue bind-retry patch already applied.")
            return

        if old_snippet not in content:
            logger.warning(
                "Could not apply MessageQueue bind-retry patch: expected "
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
        logger.warning("Could not verify MessageQueue bind-retry patch: %s", error)
        return

    if applied:
        logger.info("Successfully patched vLLM MessageQueue remote socket bind.")
    else:
        logger.warning(
            "MessageQueue bind-retry patch did not persist to %s. Engines "
            "spanning nodes may fail with EADDRINUSE at startup.",
            file_to_patch,
        )


def _patch_vllm_radio_layerscale_loader(logger) -> None:
    """Load explicit RADIO LayerScale weights and initialize folded weights.

    vLLM 0.25.1 uses ``ls1`` and ``ls2`` in ``RadioVisionEncoderLayer`` but
    skips them in ``RadioModel.load_weights``. Explicit checkpoint values are
    therefore ignored, while folded checkpoints leave the parameters at dummy
    initialization. Patch the loader so explicit values are loaded and absent
    values are initialized to RADIO's configured identity factor.
    """
    try:
        file_to_patch = _get_vllm_file("model_executor/models/radio.py")
    except RuntimeError:
        logger.warning("Could not locate radio.py for the LayerScale loader patch.")
        return

    old_snippet = """            elif sub.startswith("model.blocks."):
                # Encoder blocks: HF 'model.blocks.{i}.' ->
                # vLLM 'model.encoder.layers.{i}.'
                parts = sub.split(".")
                if len(parts) >= 4:
                    layer_idx = parts[2]
                    suffix = ".".join(parts[3:])
                    # Skip layer-scale entries that vLLM doesn't use
                    if suffix in {"ls1", "ls2"} or suffix.startswith(("ls1.", "ls2.")):
                        continue
                    vllm_key = f"model.encoder.layers.{layer_idx}.{suffix}"

            if vllm_key and vllm_key in params_dict:
                param = params_dict[vllm_key]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, weight)
                loaded_params.add(vllm_key)

        return loaded_params
"""
    new_snippet = """            elif sub.startswith("model.blocks."):
                # Encoder blocks: HF 'model.blocks.{i}.' ->
                # vLLM 'model.encoder.layers.{i}.'
                parts = sub.split(".")
                if len(parts) >= 4:
                    layer_idx = parts[2]
                    suffix = ".".join(parts[3:])
                    vllm_key = f"model.encoder.layers.{layer_idx}.{suffix}"

            if vllm_key and vllm_key in params_dict:
                param = params_dict[vllm_key]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, weight)
                loaded_params.add(vllm_key)

        initializer_factor = self.config.initializer_factor
        for name, param in params_dict.items():
            if name.endswith((".ls1", ".ls2")) and name not in loaded_params:
                param.data.fill_(initializer_factor)
                loaded_params.add(name)

        return loaded_params
"""

    with _locked_file_patch(file_to_patch) as (content, write_back):
        if new_snippet in content:
            logger.info("vLLM RADIO LayerScale loader patch already applied.")
            return
        if old_snippet not in content:
            logger.warning(
                "Could not apply vLLM RADIO LayerScale loader patch: expected "
                "vLLM 0.25.1 source shape was not found in %s.",
                file_to_patch,
            )
            return
        write_back(content.replace(old_snippet, new_snippet, 1))

    logger.info("Successfully patched vLLM RADIO LayerScale loading.")


def _patch_vllm_nemotron_h_fp32_lm_head(logger) -> None:
    """Compute NemotronH logits with an fp32 LM head (MiniMax-M1-style).

    bf16 rounding of the logits GEMM output is the dominant contributor to
    generation/training logprob mismatch (train/token_mult_prob_error). With
    this patch the sampled-token logprobs come from fp32 logits, matching a
    trainer that enables megatron_cfg.fp32_lm_head.

    This must be a source patch (not a monkeypatch): the model executes in
    vLLM's EngineCore worker subprocesses, which import vllm independently of
    this process. The patched code is opt-in at runtime via
    NRL_VLLM_FP32_LM_HEAD=1 (set it through policy.generation.vllm_cfg.env_vars
    so worker processes inherit it). Costs one fp32 copy of the vocab-sharded
    head weight per rank.
    """
    try:
        file_to_patch = _get_vllm_file("model_executor/models/nemotron_h.py")
    except RuntimeError:
        logger.warning("Could not locate nemotron_h.py for the fp32 LM head patch.")
        return

    old_snippet = """        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits"""
    new_snippet = """        import os as _os

        if _os.environ.get("NRL_VLLM_FP32_LM_HEAD", "0") == "1":
            # NeMo-RL patch: fp32 LM head (MiniMax-M1-style). bf16 rounding of
            # the logits is the dominant gen/train logprob mismatch source.
            _fp32_head = getattr(self, "_nrl_lm_head_fp32", None)
            if _fp32_head is None and not torch.cuda.is_current_stream_capturing():
                # Skipped under graph capture: an allocation there lives in the
                # graph's memory pool and is not valid for later eager replays.
                # Capture output is discarded anyway, so bf16 is fine for it.
                import copy as _copy

                _fp32_head = _copy.deepcopy(self.lm_head).float()
                # object.__setattr__ bypasses nn.Module.__setattr__: registering
                # this as a submodule would add a vocab-sized parameter to
                # named_parameters(), which the refit weight mapping is built from.
                object.__setattr__(self, "_nrl_lm_head_fp32", _fp32_head)
                self._nrl_lm_head_fp32_dirty = False
                print(
                    "[fp32_lm_head] built fp32 head in forward shape=%s"
                    % (tuple(_fp32_head.weight.shape),),
                    flush=True,
                )
            elif _fp32_head is not None and getattr(
                self, "_nrl_lm_head_fp32_dirty", False
            ):
                # Refreshed in place: replacing the module would leave any
                # captured CUDA graph pointing at the old storage.
                _fp32_head.weight.data.copy_(self.lm_head.weight)
                if getattr(_fp32_head, "bias", None) is not None:
                    _fp32_head.bias.data.copy_(self.lm_head.bias)
                self._nrl_lm_head_fp32_dirty = False
                print("[fp32_lm_head] refreshed cached head in forward", flush=True)
            if _fp32_head is not None:
                return self.logits_processor(_fp32_head, hidden_states.float())
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits"""

    with _locked_file_patch(file_to_patch) as (content, write_back):
        if "NRL_VLLM_FP32_LM_HEAD" in content:
            logger.info("NemotronH fp32 LM head patch already present.")
            return
        if content.count(old_snippet) != 1:
            logger.warning(
                "NemotronH fp32 LM head patch anchor not found exactly once "
                "in %s; patch not applied.",
                file_to_patch,
            )
            return
        write_back(content.replace(old_snippet, new_snippet, 1))

    logger.info("Applied NemotronH fp32 LM head source patch.")


def _apply_vllm_flashinfer_trtllm_refit_buffer_runtime_patch(
    logger: Any | None = None,
) -> None:
    """Keep FlashInfer TRTLLM MoE kernel weights stable across vLLM refits.

    vLLM 0.25.1 converts unquantized FlashInfer TRTLLM MoE weights from the
    canonical loader shape to the kernel's block layout by replacing
    ``layer.w13_weight`` and ``layer.w2_weight``. NeMo-RL later refits by
    calling vLLM's HF loader again, which expects those parameters to still
    have the canonical 3D expert-weight shapes. Keep the registered
    Parameters as canonical views for loading, and store the converted
    block-layout tensors separately for the kernels.
    """
    # These imports require vLLM, so keep them out of module import time.
    import torch
    from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
        UnquantizedMoeBackend,
        convert_to_unquantized_kernel_format,
        make_unquantized_moe_kernel,
    )
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )

    if logger is None:
        from vllm.logger import init_logger

        logger = init_logger("vllm_patch")

    if getattr(UnquantizedFusedMoEMethod, G_FLASHINFER_TRTLLM_PATCH_ATTR, False):
        logger.info("vLLM FlashInfer TRTLLM refit buffer patch already applied.")
        return

    original_setup_kernel = UnquantizedFusedMoEMethod._setup_kernel
    original_forward_native = UnquantizedFusedMoEMethod.forward_native
    original_apply_monolithic = UnquantizedFusedMoEMethod.apply_monolithic

    def _load_shape_numel(load_shape: tuple[int, ...]) -> int:
        numel = 1
        for dim in load_shape:
            numel *= dim
        return numel

    def _raise_if_unsupported_flashinfer_trtllm_padding(
        layer: Any,
        w2: torch.Tensor,
    ) -> None:
        moe_config = getattr(layer, "moe_config", None)
        if getattr(moe_config, "is_act_and_mul", None) is not False:
            return

        intermediate_size_per_partition = getattr(
            moe_config, "intermediate_size_per_partition", None
        )
        if intermediate_size_per_partition is None and w2.dim() >= 3:
            intermediate_size_per_partition = w2.shape[-1]
        if intermediate_size_per_partition is None:
            return

        intermediate_size_per_partition = int(intermediate_size_per_partition)
        remainder = (
            intermediate_size_per_partition % G_FLASHINFER_TRTLLM_INTERMEDIATE_ALIGNMENT
        )
        if remainder == 0:
            return

        padded_intermediate_size = (
            intermediate_size_per_partition
            + G_FLASHINFER_TRTLLM_INTERMEDIATE_ALIGNMENT
            - remainder
        )
        raise ValueError(
            "NeMo-RL's FlashInfer TRTLLM refit patch does not support "
            "padded non-gated MoE layouts. "
            f"intermediate_size_per_partition={intermediate_size_per_partition} "
            "is not divisible by "
            f"{G_FLASHINFER_TRTLLM_INTERMEDIATE_ALIGNMENT}, so vLLM pads it "
            f"to {padded_intermediate_size} before converting weights. "
            "The padded kernel tensor has more elements than the load-layout "
            "tensor, which requires a dedicated padded-layout refit path. "
            "Use expert_parallel_size to avoid the MoE TP split, or choose a "
            "backend that does not expand the MoE weights."
        )

    def _ensure_kernel_weight(
        layer: Any,
        *,
        weight_name: str,
        kernel_attr: str,
        kernel_weight: torch.Tensor,
        load_shape: tuple[int, ...],
    ) -> torch.Tensor:
        load_numel = _load_shape_numel(load_shape)
        if kernel_weight.numel() != load_numel:
            raise ValueError(
                "NeMo-RL's FlashInfer TRTLLM refit patch currently supports "
                "only conversions that preserve the number of elements. "
                f"{weight_name} load shape is {load_shape} "
                f"({load_numel} elements), but the kernel layout has shape "
                f"{tuple(kernel_weight.shape)} ({kernel_weight.numel()} elements). "
                "This is likely a padded or otherwise expanded MoE shape."
            )

        existing_kernel_weight = getattr(layer, kernel_attr, None)
        if existing_kernel_weight is None:
            stable_kernel_weight = kernel_weight.contiguous()
            setattr(layer, kernel_attr, stable_kernel_weight)
        else:
            if existing_kernel_weight.shape != kernel_weight.shape:
                raise ValueError(
                    "NeMo-RL's FlashInfer TRTLLM refit patch cannot preserve "
                    f"CUDA graph addresses for {weight_name}: existing kernel "
                    f"shape is {tuple(existing_kernel_weight.shape)}, but the "
                    f"new kernel shape is {tuple(kernel_weight.shape)}."
                )
            if existing_kernel_weight.dtype != kernel_weight.dtype:
                raise ValueError(
                    "NeMo-RL's FlashInfer TRTLLM refit patch cannot preserve "
                    f"CUDA graph addresses for {weight_name}: existing kernel "
                    f"dtype is {existing_kernel_weight.dtype}, but the new "
                    f"kernel dtype is {kernel_weight.dtype}."
                )
            if existing_kernel_weight.device != kernel_weight.device:
                raise ValueError(
                    "NeMo-RL's FlashInfer TRTLLM refit patch cannot preserve "
                    f"CUDA graph addresses for {weight_name}: existing kernel "
                    f"device is {existing_kernel_weight.device}, but the new "
                    f"kernel device is {kernel_weight.device}."
                )
            stable_kernel_weight = existing_kernel_weight
            with torch.no_grad():
                stable_kernel_weight.copy_(kernel_weight)

        getattr(layer, weight_name).data = stable_kernel_weight.view(load_shape)
        return stable_kernel_weight

    def _flashinfer_trtllm_kernel_weights(
        layer: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            return (
                getattr(layer, G_FLASHINFER_TRTLLM_W13_KERNEL_ATTR),
                getattr(layer, G_FLASHINFER_TRTLLM_W2_KERNEL_ATTR),
            )
        except AttributeError as error:
            raise RuntimeError(
                "NeMo-RL's FlashInfer TRTLLM refit patch did not find the "
                "separate kernel-layout MoE weights. "
                "process_weights_after_loading must run before inference."
            ) from error

    @wraps(original_setup_kernel)
    def patched_setup_kernel(
        self: Any,
        layer: Any,
        w13: torch.Tensor,
        w2: torch.Tensor,
    ) -> None:
        if self.unquantized_backend != UnquantizedMoeBackend.FLASHINFER_TRTLLM:
            return original_setup_kernel(self, layer, w13, w2)

        _raise_if_unsupported_flashinfer_trtllm_padding(layer, w2)

        w13_load_shape = tuple(w13.shape)
        w2_load_shape = tuple(w2.shape)
        w13_new, w2_new = convert_to_unquantized_kernel_format(
            self.unquantized_backend,
            moe_config=layer.moe_config,
            w13_weight=w13,
            w2_weight=w2,
        )

        _ensure_kernel_weight(
            layer,
            weight_name="w13_weight",
            kernel_attr=G_FLASHINFER_TRTLLM_W13_KERNEL_ATTR,
            kernel_weight=w13_new,
            load_shape=w13_load_shape,
        )
        _ensure_kernel_weight(
            layer,
            weight_name="w2_weight",
            kernel_attr=G_FLASHINFER_TRTLLM_W2_KERNEL_ATTR,
            kernel_weight=w2_new,
            load_shape=w2_load_shape,
        )

        if self.moe_kernel is None:
            self.moe_quant_config = self.get_fused_moe_quant_config(layer)
            assert self.moe_quant_config is not None
            assert self.experts_cls is not None
            self.moe_kernel = make_unquantized_moe_kernel(
                quant_config=self.moe_quant_config,
                moe_config=self.moe,
                backend=self.unquantized_backend,
                experts_cls=self.experts_cls,
                routing_tables=layer._expert_routing_tables(),
            )

    @wraps(original_forward_native)
    def patched_forward_native(
        self: Any,
        layer: Any,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: Any,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.unquantized_backend != UnquantizedMoeBackend.FLASHINFER_TRTLLM:
            return original_forward_native(
                self,
                layer,
                x,
                topk_weights,
                topk_ids,
                shared_experts,
                shared_experts_input,
            )

        assert self.moe_kernel is not None
        w13_kernel, w2_kernel = _flashinfer_trtllm_kernel_weights(layer)
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=w13_kernel,
            w2=w2_kernel,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )

    @wraps(original_apply_monolithic)
    def patched_apply_monolithic(
        self: Any,
        layer: Any,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.unquantized_backend != UnquantizedMoeBackend.FLASHINFER_TRTLLM:
            return original_apply_monolithic(self, layer, x, router_logits, input_ids)

        assert self.is_monolithic
        assert self.moe_kernel is not None
        w13_kernel, w2_kernel = _flashinfer_trtllm_kernel_weights(layer)
        return self.moe_kernel.apply_monolithic(
            x,
            w13_kernel,
            w2_kernel,
            router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            e_score_correction_bias=layer.e_score_correction_bias,
            routed_scaling_factor=layer.routed_scaling_factor,
        )

    UnquantizedFusedMoEMethod._nrl_original_setup_kernel = original_setup_kernel
    UnquantizedFusedMoEMethod._nrl_original_forward_native = original_forward_native
    UnquantizedFusedMoEMethod._nrl_original_apply_monolithic = original_apply_monolithic
    UnquantizedFusedMoEMethod._setup_kernel = patched_setup_kernel
    UnquantizedFusedMoEMethod.forward_native = patched_forward_native
    UnquantizedFusedMoEMethod.apply_monolithic = patched_apply_monolithic
    setattr(UnquantizedFusedMoEMethod, G_FLASHINFER_TRTLLM_PATCH_ATTR, True)
    logger.info("Successfully patched vLLM FlashInfer TRTLLM MoE refit buffers.")


def _run_engine_core_with_flashinfer_trtllm_refit_patch(
    *args: Any, **kwargs: Any
) -> Any:
    """Apply the refit patch inside EngineCore before model construction."""
    _apply_vllm_flashinfer_trtllm_refit_buffer_runtime_patch()
    _patch_vllm_flashinfer_trtllm_refit_from_collective_rpc()

    # Import here because this function is also the multiprocessing target for
    # spawned EngineCore processes, where module import state is rebuilt.
    from vllm.v1.engine.core import EngineCoreProc

    original_run_engine_core = getattr(
        EngineCoreProc, G_FLASHINFER_TRTLLM_ENGINE_CORE_ORIGINAL_ATTR, None
    )
    if original_run_engine_core is None:
        original_run_engine_core = EngineCoreProc.run_engine_core
    if original_run_engine_core is _run_engine_core_with_flashinfer_trtllm_refit_patch:
        raise RuntimeError(
            "NeMo-RL's FlashInfer TRTLLM refit EngineCore entrypoint patch "
            "could not find the original vLLM EngineCoreProc.run_engine_core."
        )
    return original_run_engine_core(*args, **kwargs)


def _apply_vllm_flashinfer_trtllm_refit_buffer_runtime_patch_on_worker(
    _worker: Any | None = None,
) -> None:
    """Adapter for vLLM v1 Ray workers that pass their worker as arg 0."""
    del _worker
    _apply_vllm_flashinfer_trtllm_refit_buffer_runtime_patch()


def _ray_worker_initialize_with_flashinfer_trtllm_refit_patch(
    self: Any, *args: Any, **kwargs: Any
) -> Any:
    """Apply the refit patch before RayExecutorV2 initializes a worker model."""
    _apply_vllm_flashinfer_trtllm_refit_buffer_runtime_patch()

    original_initialize_worker = getattr(
        type(self), G_FLASHINFER_TRTLLM_RAY_WORKER_ORIGINAL_ATTR, None
    )
    if original_initialize_worker is None:
        from vllm.v1.executor.ray_executor_v2 import RayWorkerProc

        original_initialize_worker = RayWorkerProc.initialize_worker
    if (
        original_initialize_worker
        is _ray_worker_initialize_with_flashinfer_trtllm_refit_patch
    ):
        raise RuntimeError(
            "NeMo-RL's FlashInfer TRTLLM refit RayExecutorV2 worker patch could "
            "not find the original vLLM RayWorkerProc.initialize_worker."
        )
    return original_initialize_worker(self, *args, **kwargs)


def _collective_rpc_with_flashinfer_trtllm_refit_patch(
    self: Any, *args: Any, **kwargs: Any
) -> Any:
    """Patch vLLM v1 Ray workers before their first collective RPC."""
    original_collective_rpc = getattr(
        type(self), G_FLASHINFER_TRTLLM_RAY_EXECUTOR_ORIGINAL_ATTR, None
    )
    if original_collective_rpc is None:
        from vllm.v1.executor.ray_executor import RayDistributedExecutor

        original_collective_rpc = RayDistributedExecutor.collective_rpc
    if original_collective_rpc is _collective_rpc_with_flashinfer_trtllm_refit_patch:
        raise RuntimeError(
            "NeMo-RL's FlashInfer TRTLLM refit Ray executor patch could not "
            "find the original vLLM RayDistributedExecutor.collective_rpc."
        )

    if not getattr(self, G_FLASHINFER_TRTLLM_RAY_EXECUTOR_WORKERS_PATCHED_ATTR, False):
        from vllm.v1.executor.ray_utils import ray

        futures = [
            worker.execute_method.remote(
                _apply_vllm_flashinfer_trtllm_refit_buffer_runtime_patch_on_worker
            )
            for worker in self.workers
        ]
        ray.get(futures)
        setattr(self, G_FLASHINFER_TRTLLM_RAY_EXECUTOR_WORKERS_PATCHED_ATTR, True)

    return original_collective_rpc(self, *args, **kwargs)


def _patch_vllm_flashinfer_trtllm_refit_from_collective_rpc(
    logger: Any | None = None,
) -> None:
    """Patch vLLM process entrypoints that can construct the model."""
    if logger is None:
        from vllm.logger import init_logger

        logger = init_logger("vllm_patch")

    try:
        from vllm.v1.engine.core import EngineCoreProc
    except (AttributeError, ImportError) as error:
        logger.warning(
            "Could not patch vLLM EngineCoreProc for FlashInfer TRTLLM refits: %s",
            error,
        )
    else:
        if getattr(EngineCoreProc, G_FLASHINFER_TRTLLM_ENGINE_CORE_PATCH_ATTR, False):
            logger.info(
                "vLLM EngineCoreProc FlashInfer TRTLLM refit patch already applied."
            )
        else:
            setattr(
                EngineCoreProc,
                G_FLASHINFER_TRTLLM_ENGINE_CORE_ORIGINAL_ATTR,
                EngineCoreProc.run_engine_core,
            )
            EngineCoreProc.run_engine_core = staticmethod(
                _run_engine_core_with_flashinfer_trtllm_refit_patch
            )
            setattr(EngineCoreProc, G_FLASHINFER_TRTLLM_ENGINE_CORE_PATCH_ATTR, True)
            logger.info(
                "Successfully patched vLLM EngineCoreProc for FlashInfer TRTLLM refits."
            )

    try:
        from vllm.v1.executor.ray_executor_v2 import RayWorkerProc
    except (AttributeError, ImportError) as error:
        logger.info(
            "Skipping RayExecutorV2 FlashInfer TRTLLM refit worker patch: %s",
            error,
        )
    else:
        if getattr(RayWorkerProc, G_FLASHINFER_TRTLLM_RAY_WORKER_PATCH_ATTR, False):
            logger.info(
                "vLLM RayWorkerProc FlashInfer TRTLLM refit patch already applied."
            )
        else:
            setattr(
                RayWorkerProc,
                G_FLASHINFER_TRTLLM_RAY_WORKER_ORIGINAL_ATTR,
                RayWorkerProc.initialize_worker,
            )
            RayWorkerProc.initialize_worker = (
                _ray_worker_initialize_with_flashinfer_trtllm_refit_patch
            )
            setattr(RayWorkerProc, G_FLASHINFER_TRTLLM_RAY_WORKER_PATCH_ATTR, True)
            logger.info(
                "Successfully patched vLLM RayWorkerProc for FlashInfer TRTLLM refits."
            )

    try:
        from vllm.v1.executor.ray_executor import RayDistributedExecutor
    except (AttributeError, ImportError) as error:
        logger.info(
            "Skipping vLLM v1 Ray executor FlashInfer TRTLLM refit patch: %s",
            error,
        )
    else:
        if getattr(
            RayDistributedExecutor, G_FLASHINFER_TRTLLM_RAY_EXECUTOR_PATCH_ATTR, False
        ):
            logger.info(
                "vLLM RayDistributedExecutor FlashInfer TRTLLM refit patch "
                "already applied."
            )
        else:
            patched_collective_rpc = _collective_rpc_with_flashinfer_trtllm_refit_patch
            setattr(
                RayDistributedExecutor,
                G_FLASHINFER_TRTLLM_RAY_EXECUTOR_ORIGINAL_ATTR,
                RayDistributedExecutor.collective_rpc,
            )
            RayDistributedExecutor.collective_rpc = patched_collective_rpc
            setattr(
                RayDistributedExecutor,
                G_FLASHINFER_TRTLLM_RAY_EXECUTOR_PATCH_ATTR,
                True,
            )
            logger.info(
                "Successfully patched vLLM RayDistributedExecutor for "
                "FlashInfer TRTLLM refits."
            )


def _patch_vllm_flashinfer_trtllm_refit_buffers(logger: Any) -> None:
    """Patch current and child vLLM processes for FlashInfer TRTLLM refits."""
    _apply_vllm_flashinfer_trtllm_refit_buffer_runtime_patch(logger)
    _patch_vllm_flashinfer_trtllm_refit_from_collective_rpc(logger)


def ensure_vllm_source_compat() -> None:
    """Apply interpreter-independent vLLM source-compat patches.

    Safe to call from any process that imports vLLM directly (e.g. the
    tools/model_diagnostics scripts, which construct ``vllm.LLM`` without
    going through a NeMo-RL generation worker). Must be called BEFORE the
    first ``import vllm`` submodule that pulls in ``vllm.tool_parsers``.
    Worker processes get this via ``_apply_vllm_patches`` at init.
    """
    from vllm.logger import init_logger

    patch_logger = init_logger("vllm_patch")
    _patch_vllm_tool_parser_namespace_tool(patch_logger)
    _patch_vllm_radio_layerscale_loader(patch_logger)


def _apply_external_vllm_patches() -> None:
    """Apply only compatibility patches required by external vLLM servers.

    External Gym model servers are inference-only. Keep this list explicit so
    policy-specific runtime patches, such as refit support, are not installed
    in their vLLM engine processes.
    """
    from vllm.logger import init_logger

    patch_logger = init_logger("vllm_patch")
    _patch_vllm_tool_parser_namespace_tool(patch_logger)
    _patch_vllm_ray_executor_v2_tcpstore_port(patch_logger)
    _patch_vllm_shm_broadcast_bind_retry(patch_logger)


def _apply_vllm_patches(
    py_executable: str,
    *,
    extra_env_vars: list[str] | None = None,
    require_compact_routed_experts: bool = False,
) -> None:
    # Import lazily so importing the worker module does not import vLLM.
    import vllm.envs as envs
    from vllm.logger import init_logger

    patch_logger = init_logger("vllm_patch")

    # Whether the v1 patch matters at all depends on which executor vLLM will
    # select. 0.25 defaults this to "1" (RayExecutorV2), which has no
    # _init_workers_ray; the patch is only load-bearing when it is set to "0".
    # Reporting the same way in both cases either cries wolf or hides a real
    # break, so branch on it.
    uses_v1_executor = not envs.VLLM_USE_RAY_V2_EXECUTOR_BACKEND
    applied = _patch_vllm_init_workers_ray(py_executable, extra_env_vars)

    if applied and uses_v1_executor:
        patch_logger.info(
            "Successfully patched vllm v1 _init_workers_ray; Ray workers will "
            "launch under %s.",
            py_executable,
        )
    elif applied:
        patch_logger.info(
            "Patched vllm v1 _init_workers_ray, but VLLM_USE_RAY_V2_EXECUTOR_"
            "BACKEND selects RayExecutorV2, which has no such method. The "
            "patch is inert here; workers inherit py_executable from this "
            "actor's runtime_env instead."
        )
    elif uses_v1_executor:
        patch_logger.error(
            "vllm v1 _init_workers_ray patch did NOT apply: the "
            "'self._init_workers_ray(placement_group)' anchor was not found, "
            "and VLLM_USE_RAY_V2_EXECUTOR_BACKEND=0 selects the v1 executor "
            "that depends on it. Ray workers will launch under the wrong "
            "interpreter. Either the anchor moved upstream, or unset "
            "VLLM_USE_RAY_V2_EXECUTOR_BACKEND to use RayExecutorV2."
        )
    else:
        patch_logger.info(
            "vllm v1 _init_workers_ray anchor not found, which is harmless "
            "here: RayExecutorV2 is selected and does not use it."
        )

    _patch_vllm_llama_eagle3_own_lm_head(patch_logger)
    _patch_vllm_tool_parser_namespace_tool(patch_logger)
    _patch_vllm_ray_executor_v2_tcpstore_port(patch_logger)
    _patch_vllm_shm_broadcast_bind_retry(patch_logger)
    _patch_vllm_radio_layerscale_loader(patch_logger)
    _patch_vllm_nemotron_h_fp32_lm_head(patch_logger)
    compact_routed_experts_ready = _patch_vllm_routed_experts_compact_layers(
        patch_logger
    )
    if require_compact_routed_experts and not compact_routed_experts_ready:
        raise RuntimeError(
            "Router replay requires NeMo-RL's compact routed-experts vLLM "
            "source patch, but it could not be applied. See the preceding "
            "vllm_patch error for the installed source path and hash."
        )
    _patch_vllm_flashinfer_trtllm_refit_buffers(patch_logger)
