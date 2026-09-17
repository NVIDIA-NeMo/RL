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


def _patch_vllm_monolithic_routing_replay_base(logger) -> None:
    """Add routing-replay buffer plumbing to ``FusedMoEExpertsMonolithic``.

    Backports the base-class half of vllm-project/vllm#44214 ("Enable router
    replay output from FlashInfer monolithic MoE kernel"). On vLLM 0.25.1,
    routed-experts capture (``enable_return_routed_experts`` /
    ``router_replay``) only works through the modular MoE kernel path via
    ``router.set_capture_fn()`` -- the monolithic path (FlashInfer TRT-LLM,
    used by MXFP8/NVFP4/MXFP4 monolithic kernels) fuses routing into the
    kernel itself, so ``module.router`` isn't a ``BaseRouter`` there and
    nothing ever captures it. The routed-experts buffer then silently stays
    at its zero-initialized default (see ``RoutedExpertsCapturer.__init__``),
    which NeMo-RL's router_replay then replays into Megatron's MoE
    all-to-all dispatch as if every token routed to expert 0 -- observed as
    ``RuntimeError: Split sizes doesn't match total dim 0 size`` in
    ``megatron/core/tensor_parallel/mappings.py``'s ``all_to_all`` on a
    policy using ``vllm_kwargs.moe_backend=flashinfer_trtllm`` with MXFP8 and
    ``policy.router_replay.enabled=true``.

    Paired with ``_patch_vllm_trtllm_fp8_routing_replay`` (which opts
    ``TrtLlmFp8ExpertsMonolithic`` -- the fp8 block-scale/MXFP8 monolithic
    kernel NeMo-RL's MXFP8-rollout recipes use -- into this) and
    ``_patch_vllm_bind_routed_experts_capturer_monolithic`` (the
    ``GPUModelRunner`` side that wires it up and hard-fails instead of
    silently corrupting routing for any other monolithic kernel that hasn't
    opted in, backported from the same PR plus vllm-project/vllm#48622).

    Remove all three patches after upgrading to a vLLM release containing
    #44214 and #48622, and re-validate router_replay + MXFP8 rollouts.
    """
    try:
        file_to_patch = _get_vllm_file(
            "model_executor/layers/fused_moe/modular_kernel.py"
        )
    except RuntimeError:
        logger.warning(
            "Could not locate modular_kernel.py for the monolithic routing "
            "replay base patch."
        )
        return

    old_snippet = """    @staticmethod
    def is_monolithic() -> bool:
        return True

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        # grouped topk + fused topk bias parameters
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        \"\"\"
        Same as apply(), except uses router_logits as opposed
        to the topk_ids and topk_weights. This is useful for kernels
        with fused router and fused_experts (e.g. FLASHINFER_TRTLLM).
        \"\"\"
        raise NotImplementedError
"""

    new_snippet = """    @staticmethod
    def is_monolithic() -> bool:
        return True

    # Backport of vllm-project/vllm#44214: lets a monolithic kernel opt into
    # routing-replay capture and stages the per-token expert-ID buffer that
    # NeMo-RL's router_replay reads via ``enable_return_routed_experts``.
    routing_replay_capture_fn: Callable[[torch.Tensor], None] | None = None
    _routing_replay_buffer: torch.Tensor | None = None

    def supports_routing_replay_capture(self) -> bool:
        \"\"\"Whether this expert supports routing replay capture.

        Subclasses backed by a kernel that exposes routed expert IDs
        (e.g. FlashInfer's ``routing_replay_out``) should override.
        \"\"\"
        return False

    def set_capture_fn(
        self,
        capture_fn: Callable[[torch.Tensor], None] | None,
    ) -> None:
        self.routing_replay_capture_fn = capture_fn
        if capture_fn is None:
            self._routing_replay_buffer = None
            return
        self._routing_replay_buffer = torch.empty(
            (self.moe_config.max_num_tokens, self.moe_config.experts_per_token),
            dtype=torch.int16,
            device=self.moe_config.device,
        )

    def _maybe_make_routing_replay_buffer(
        self,
        num_tokens: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if self.routing_replay_capture_fn is None:
            return None
        buf = self._routing_replay_buffer
        assert buf is not None
        if buf.shape[0] < num_tokens or buf.device != device:
            raise ValueError(
                "Routing replay buffer was initialized for "
                f"{buf.shape[0]} tokens on {buf.device}, but the kernel "
                f"received {num_tokens} tokens on {device}."
            )
        return buf

    def _maybe_dispatch_routing_replay(
        self,
        routing_replay_out: torch.Tensor | None,
        num_tokens: int,
    ) -> None:
        if routing_replay_out is None or self.routing_replay_capture_fn is None:
            return
        self.routing_replay_capture_fn(routing_replay_out[:num_tokens])

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        # grouped topk + fused topk bias parameters
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        \"\"\"
        Same as apply(), except uses router_logits as opposed
        to the topk_ids and topk_weights. This is useful for kernels
        with fused router and fused_experts (e.g. FLASHINFER_TRTLLM).
        \"\"\"
        raise NotImplementedError
"""

    with _locked_file_patch(file_to_patch) as (content, write_back):
        if new_snippet in content:
            logger.info("vLLM monolithic routing-replay base patch already applied.")
            return
        if old_snippet not in content:
            logger.warning(
                "Could not apply vLLM monolithic routing-replay base patch: "
                "expected vLLM 0.25.1 source shape was not found in %s.",
                file_to_patch,
            )
            return
        write_back(content.replace(old_snippet, new_snippet, 1))

    logger.info(
        "Successfully added routing-replay buffer support to "
        "FusedMoEExpertsMonolithic."
    )


def _patch_vllm_trtllm_fp8_routing_replay(logger) -> None:
    """Wire routing-replay capture through the fp8/MXFP8 monolithic TRT-LLM kernel.

    Backports the ``TrtLlmFp8ExpertsMonolithic`` half of
    vllm-project/vllm#44214 onto vLLM 0.25.1. This is the monolithic kernel
    NeMo-RL's ``vllm_kwargs.moe_backend=flashinfer_trtllm`` MXFP8-rollout
    recipes select (both the ``[128, 128]`` fp8 block-scale and ``[1, 32]``
    MXFP8 block shapes route through ``_apply_block_scale``). Requires
    ``_patch_vllm_monolithic_routing_replay_base`` to have run first, since
    it adds the ``_maybe_make_routing_replay_buffer`` /
    ``_maybe_dispatch_routing_replay`` methods this patch calls.

    See ``_patch_vllm_monolithic_routing_replay_base`` for why this matters:
    without it, MXFP8 rollouts under ``policy.router_replay.enabled=true``
    silently replay all-zero routing into Megatron's training-side MoE
    dispatch instead of raising or, after this patch, actually capturing
    real per-token expert IDs.
    """
    try:
        file_to_patch = _get_vllm_file(
            "model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py"
        )
    except RuntimeError:
        logger.warning(
            "Could not locate trtllm_fp8_moe.py for the fp8/MXFP8 routing "
            "replay patch."
        )
        return

    class_old = """class TrtLlmFp8ExpertsMonolithic(TrtLlmFp8ExpertsBase, mk.FusedMoEExpertsMonolithic):
    \"\"\"
    Fp8 TRTLLM-Gen MoE kernels. Supports monolithic interface.
    \"\"\"

    def __init__(
"""
    class_new = """class TrtLlmFp8ExpertsMonolithic(TrtLlmFp8ExpertsBase, mk.FusedMoEExpertsMonolithic):
    \"\"\"
    Fp8 TRTLLM-Gen MoE kernels. Supports monolithic interface.
    \"\"\"

    def supports_routing_replay_capture(self) -> bool:
        return True

    def __init__(
"""

    block_scale_old = """            n_group = num_expert_group or 0
            selected_topk_group = topk_group or 0

        kwargs = dict(
"""
    block_scale_new = """            n_group = num_expert_group or 0
            selected_topk_group = topk_group or 0

        routing_replay_out = self._maybe_make_routing_replay_buffer(
            num_tokens=hidden_states.shape[0],
            device=hidden_states.device,
        )

        kwargs = dict(
"""

    block_scale_tail_old = """            use_shuffled_weight=use_shuffled_weight,
            weight_layout=weight_layout,
            fp8_quantization_type=fp8_quant_type,
        )
        if is_mxfp8 or activation == MoEActivation.RELU2_NO_MUL:
            kwargs["activation_type"] = activation_type
        return flashinfer.fused_moe.trtllm_fp8_block_scale_moe(**kwargs)
"""
    block_scale_tail_new = """            use_shuffled_weight=use_shuffled_weight,
            weight_layout=weight_layout,
            fp8_quantization_type=fp8_quant_type,
            routing_replay_out=routing_replay_out,
        )
        if is_mxfp8 or activation == MoEActivation.RELU2_NO_MUL:
            kwargs["activation_type"] = activation_type
        result = flashinfer.fused_moe.trtllm_fp8_block_scale_moe(**kwargs)
        self._maybe_dispatch_routing_replay(
            routing_replay_out, num_tokens=hidden_states.shape[0]
        )
        return result
"""

    per_tensor_old = """        out = flashinfer.fused_moe.trtllm_fp8_per_tensor_scale_moe(
            routing_logits=router_logits,
            routing_bias=e_score_correction_bias,
            hidden_states=hidden_states,
            gemm1_weights=w1,
            output1_scales_scalar=self._g1_scale_c,
            output1_scales_gate_scalar=self._g1_alphas,
            gemm2_weights=w2,
            output2_scales_scalar=self._g2_alphas,
            num_experts=global_num_experts,
            top_k=self.topk,
            n_group=num_expert_group or 0,
            topk_group=topk_group or 0,
            intermediate_size=self.intermediate_size_per_partition,
            local_expert_offset=self.ep_rank * self.local_num_experts,
            local_num_experts=self.local_num_experts,
            routed_scaling_factor=routed_scaling_factor,
            use_routing_scales_on_input=apply_router_weight_on_input,
            routing_method_type=self.routing_method_type,
            activation_type=activation_type,
        )
        return out
"""
    per_tensor_new = """        routing_replay_out = self._maybe_make_routing_replay_buffer(
            num_tokens=hidden_states.shape[0],
            device=hidden_states.device,
        )
        out = flashinfer.fused_moe.trtllm_fp8_per_tensor_scale_moe(
            routing_logits=router_logits,
            routing_bias=e_score_correction_bias,
            hidden_states=hidden_states,
            gemm1_weights=w1,
            output1_scales_scalar=self._g1_scale_c,
            output1_scales_gate_scalar=self._g1_alphas,
            gemm2_weights=w2,
            output2_scales_scalar=self._g2_alphas,
            num_experts=global_num_experts,
            top_k=self.topk,
            n_group=num_expert_group or 0,
            topk_group=topk_group or 0,
            intermediate_size=self.intermediate_size_per_partition,
            local_expert_offset=self.ep_rank * self.local_num_experts,
            local_num_experts=self.local_num_experts,
            routed_scaling_factor=routed_scaling_factor,
            use_routing_scales_on_input=apply_router_weight_on_input,
            routing_method_type=self.routing_method_type,
            activation_type=activation_type,
            routing_replay_out=routing_replay_out,
        )
        self._maybe_dispatch_routing_replay(
            routing_replay_out, num_tokens=hidden_states.shape[0]
        )
        return out
"""

    hunks = [
        ("class", class_old, class_new),
        ("block_scale_buffer", block_scale_old, block_scale_new),
        ("block_scale_tail", block_scale_tail_old, block_scale_tail_new),
        ("per_tensor", per_tensor_old, per_tensor_new),
    ]

    with _locked_file_patch(file_to_patch) as (content, write_back):
        if all(new in content for _, _, new in hunks):
            logger.info("vLLM fp8/MXFP8 routing-replay patch already applied.")
            return
        missing = [name for name, old, _ in hunks if old not in content]
        if missing:
            logger.warning(
                "Could not apply vLLM fp8/MXFP8 routing-replay patch: "
                "expected vLLM 0.25.1 source shape for %s was not found in "
                "%s.",
                ", ".join(missing),
                file_to_patch,
            )
            return
        for _, old, new in hunks:
            content = content.replace(old, new, 1)
        write_back(content)

    logger.info(
        "Successfully wired routing-replay capture through "
        "TrtLlmFp8ExpertsMonolithic (fp8 block-scale and MXFP8)."
    )


def _patch_vllm_bind_routed_experts_capturer_monolithic(logger) -> None:
    """Bind routed-experts capture to monolithic MoE kernels, and stop
    silently skipping any monolithic kernel that doesn't support it.

    Backports vllm-project/vllm#48622 ("Exclude draft routers from expert
    capture") and the ``GPUModelRunner`` half of #44214 onto vLLM 0.25.1.

    #48622 alone: ``_bind_routed_experts_capturer`` walks
    ``self.compilation_config.static_forward_context.values()``, which
    includes MTP/draft-model MoE layers. NeMo-RL's recipes run
    ``megatron_cfg.mtp_num_layers`` heads that get loaded into vLLM too (see
    ``quantization_ignore_patterns`` excluding ``mtp.*`` in the MXFP8-rollout
    recipes), and a draft MoE layer can share ``layer_id=0`` with the
    target's first MoE layer, corrupting the shared capture buffer. The fix
    scopes capture binding to ``self.model.modules()`` (the target model
    only) instead.

    #44214 on top of that: without an explicit ``is_monolithic`` branch, a
    monolithic kernel's ``module.router`` is never a ``BaseRouter`` (routing
    is fused into the kernel), so the ``isinstance(module.router,
    BaseRouter)`` check just silently skips it -- the routed-experts buffer
    for that layer stays at its zero-initialized default with no warning.
    This raises instead, unless the kernel has opted in via
    ``supports_routing_replay_capture()`` (see
    ``_patch_vllm_monolithic_routing_replay_base`` /
    ``_patch_vllm_trtllm_fp8_routing_replay``).

    Requires the other two patches in this file to have run first.
    """
    try:
        file_to_patch = _get_vllm_file("v1/worker/gpu_model_runner.py")
    except RuntimeError:
        logger.warning(
            "Could not locate gpu_model_runner.py for the routed-experts "
            "capturer binding patch."
        )
        return

    old_snippet = """    def _bind_routed_experts_capturer(self, capturer: RoutedExpertsCapturer) -> None:
        from vllm.model_executor.layers.fused_moe.layer import MoERunner
        from vllm.model_executor.layers.fused_moe.router.base_router import (
            BaseRouter,
        )

        for module in self.compilation_config.static_forward_context.values():
            if isinstance(module, MoERunner) and isinstance(module.router, BaseRouter):
                layer_id = module.layer_id

                def _capture_fn(topk_ids, _layer_id=layer_id, _capturer=capturer):
                    _capturer.capture(_layer_id, topk_ids)

                module.router.set_capture_fn(_capture_fn)
"""
    new_snippet = """    def _bind_routed_experts_capturer(self, capturer: RoutedExpertsCapturer) -> None:
        from vllm.model_executor.layers.fused_moe.layer import MoERunner
        from vllm.model_executor.layers.fused_moe.modular_kernel import (
            FusedMoEExpertsMonolithic,
        )
        from vllm.model_executor.layers.fused_moe.router.base_router import (
            BaseRouter,
        )

        for module in self.model.modules():
            if not isinstance(module, MoERunner):
                continue
            layer_id = module.layer_id

            def _capture_fn(topk_ids, _layer_id=layer_id, _capturer=capturer):
                _capturer.capture(_layer_id, topk_ids)

            quant_method = module._quant_method
            moe_kernel = getattr(quant_method, "moe_kernel", None)
            impl = getattr(moe_kernel, "impl", None)
            fused_experts = getattr(impl, "fused_experts", None)
            if quant_method.is_monolithic:
                if not (
                    isinstance(fused_experts, FusedMoEExpertsMonolithic)
                    and fused_experts.supports_routing_replay_capture()
                ):
                    raise ValueError(
                        "--enable-return-routed-experts is not supported with "
                        f"monolithic MoE kernel {type(fused_experts).__name__}; "
                        "routed expert IDs would be silently all-zero."
                    )
                fused_experts.set_capture_fn(_capture_fn)
            elif isinstance(module.router, BaseRouter):
                module.router.set_capture_fn(_capture_fn)
"""

    with _locked_file_patch(file_to_patch) as (content, write_back):
        if new_snippet in content:
            logger.info(
                "vLLM routed-experts capturer binding patch already applied."
            )
            return
        if old_snippet not in content:
            logger.warning(
                "Could not apply vLLM routed-experts capturer binding patch: "
                "expected vLLM 0.25.1 source shape was not found in %s.",
                file_to_patch,
            )
            return
        write_back(content.replace(old_snippet, new_snippet, 1))

    logger.info(
        "Successfully rebound routed-experts capture to target-model-only "
        "modules, with monolithic-kernel opt-in support."
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
    # Order matters: the base class must gain the routing-replay buffer
    # methods before TrtLlmFp8ExpertsMonolithic calls them, and both must be
    # in place before GPUModelRunner starts dispatching capture to them.
    _patch_vllm_monolithic_routing_replay_base(patch_logger)
    _patch_vllm_trtllm_fp8_routing_replay(patch_logger)
    _patch_vllm_bind_routed_experts_capturer_monolithic(patch_logger)
