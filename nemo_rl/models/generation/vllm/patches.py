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


def _patch_vllm_glm_decoder_sequence_parallel_moe(logger) -> None:
    """Restore the vLLM 0.24 decoder boundary for GLM DSA models.

    vLLM 0.25.1 keeps hidden states sequence-parallel across attention and MoE
    decoder layers when TP, DP, and EP are all enabled. GLM-5.1/5.2 decode
    diverges on that new path: the first generated token is correct, while
    subsequent decode-token logprobs collapse. Keep vLLM's existing MoE-local
    sequence parallelism, but disable the new decoder-level optimization for
    ``glm_moe_dsa`` so the MoE gathers its output as it did in vLLM 0.24.

    The upstream bug and proposed fix are tracked at
    https://github.com/vllm-project/vllm/issues/50154 and
    https://github.com/vllm-project/vllm/pull/50155. Remove this patch after
    upgrading to a vLLM release containing the fix and validating iterative
    GLM-5.1/5.2 decode with TP, DP, and EP all enabled.
    """
    try:
        file_to_patch = _get_vllm_file("model_executor/models/deepseek_v2.py")
    except RuntimeError:
        logger.warning(
            "Could not locate deepseek_v2.py for the GLM decoder SP-MoE patch."
        )
        return

    old_snippet = """        self.use_sequence_parallel_moe = (
            parallel_config.use_sequence_parallel_moe
            and parallel_config.pipeline_parallel_size == 1
            and is_moe_layer
        )
"""
    new_snippet = """        self.use_sequence_parallel_moe = (
            parallel_config.use_sequence_parallel_moe
            and parallel_config.pipeline_parallel_size == 1
            and is_moe_layer
            # vLLM 0.25.1's decoder-level SP-MoE path corrupts iterative
            # decoding for GLM-5.1/5.2. Retain the vLLM 0.24 MoE-local path.
            and getattr(config, "model_type", None) != "glm_moe_dsa"
        )
"""

    with _locked_file_patch(file_to_patch) as (content, write_back):
        if new_snippet in content:
            logger.info("vLLM GLM decoder SP-MoE patch already applied.")
            return
        if old_snippet not in content:
            logger.warning(
                "Could not apply vLLM GLM decoder SP-MoE patch: expected "
                "vLLM 0.25.1 source shape was not found in %s.",
                file_to_patch,
            )
            return
        write_back(content.replace(old_snippet, new_snippet, 1))

    logger.info("Successfully disabled decoder-level SP-MoE for GLM DSA models.")


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
    _patch_vllm_glm_decoder_sequence_parallel_moe(patch_logger)


def _apply_vllm_patches(
    py_executable: str,
    *,
    extra_env_vars: list[str] | None = None,
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
    _patch_vllm_glm_decoder_sequence_parallel_moe(patch_logger)
