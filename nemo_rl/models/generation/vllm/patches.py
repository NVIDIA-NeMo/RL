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
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.util import find_spec
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

# Keep this module free of NeMo RL imports, including lazy ones inside
# functions: tools/external_gym_vllm/serve_vllm_on_ray.py loads it by path and
# calls _apply_vllm_patches in serving containers that do not install NeMo RL.

VLLM_NEMOTRON_H_FP32_LM_HEAD_ENV_VAR = "NRL_VLLM_FP32_LM_HEAD"
VLLM_DSA_TOPK_CAPTURE_ENV_VAR = "NRL_VLLM_DSA_TOPK_CAPTURE"
VLLM_DSA_TOPK_LAYER_IDS_ENV_VAR = "NRL_VLLM_DSA_TOPK_LAYER_IDS"


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

    vLLM 0.29 fixes the race upstream (vllm-project/vllm#53666, #50969): the
    rank-0 actor now binds the TCPStore itself, on a kernel-assigned port, and
    *holds* that socket (``self._dist_init_store = store``) until
    ``init_process_group`` reuses it, so there is no probe/bind window for the
    MessageQueue to land in. That is not the TOCTOU pattern the reserved band
    guards against (the port is never released between selection and use), and
    ``_select_tcpstore_port`` no longer exists to patch. When that upstream
    marker is present this function logs at info level and leaves the file
    alone.

    Returns without raising when neither form is found, but logs at warning
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

    # vLLM >= 0.29: RayWorkerProc.create_dist_init_method binds and keeps the
    # TCPStore before publishing its port (vllm-project/vllm#50969).
    upstream_fix_marker = "self._dist_init_store = store"
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
        if upstream_fix_marker in content:
            logger.info(
                "vLLM binds the RayExecutorV2 TCPStore before publishing its port "
                "(vllm-project/vllm#50969); NeMo-RL TCPStore port patch not needed."
            )
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
    """Keep MessageQueue's remote socket in the reserved band with bind retries.

    vLLM 0.28 binds port zero directly, avoiding the old probe/bind race but
    ignoring ``VLLM_PORT``. Restore reserved-band selection with retries so
    engine sockets do not consume the ephemeral ports used by other services.
    A probe alone releases its socket before ZMQ binds, leaving a TOCTOU race.

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
        '            self.remote_socket.bind(f"tcp://{connect_ip}:0")\n'
        "            last_endpoint = self.remote_socket.getsockopt(zmq.LAST_ENDPOINT)\n"
        '            remote_subscribe_port = last_endpoint.decode().rsplit(":", 1)[1]\n'
    )
    new_snippet = (
        "            from vllm.utils.network_utils import get_open_port, _get_open_port\n"
        "\n"
        "            remote_subscribe_port = get_open_port()\n"
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


def _patch_vllm_dsa_topk_capturer(logger, *, required: bool = False) -> bool:
    """Reuse vLLM's routed-experts transport for GLM DSA top-k indices.

    vLLM already has the hard part of replay capture: a per-forward device
    buffer plus a scheduler-side buffer keyed by physical KV-cache slots. In
    DSA mode this patch changes that channel's shape and binder so it carries
    one ``index_topk`` vector for each selected DSA *compute* layer instead of
    MoE expert ids. The selected layer ids are compacted to contiguous slots,
    which avoids allocating entries for DSA skip layers.

    The patch is runtime-gated by ``NRL_VLLM_DSA_TOPK_CAPTURE`` inside the
    installed vLLM source. Applying it is therefore harmless for ordinary MoE
    router replay. When DSA capture is requested all anchors are required: a
    partial patch would silently return plausible but incorrect routes.
    """
    try:
        file_to_patch = _get_vllm_file(
            "model_executor/layers/fused_moe/routed_experts_capturer.py"
        )
    except RuntimeError:
        message = (
            "Could not locate routed_experts_capturer.py for the DSA top-k "
            "capture patch."
        )
        if required:
            raise RuntimeError(message) from None
        logger.warning(message)
        return False

    marker = "NeMo-RL patch (DSA top-k capture transport)"
    import_old_snippet = "import logging\nfrom collections.abc import Callable\n"
    import_new_snippet = (
        "import logging\nimport os\nfrom collections.abc import Callable\n"
    )
    shape_old_snippet = """def _get_routed_experts_shape(vllm_config: VllmConfig) -> tuple[int, int, int]:
    model_config = vllm_config.model_config
    num_layers = model_config.get_total_num_hidden_layers()
    num_experts = model_config.get_num_experts()
    num_experts_per_tok = model_config.get_num_experts_per_tok()
    if num_layers <= 0 or num_experts <= 0 or num_experts_per_tok <= 0:
        raise ValueError(
            "Routed-experts capture requires positive layer, expert, and "
            "experts-per-token counts, got "
            f"{num_layers=}, {num_experts=}, {num_experts_per_tok=}."
        )
    return num_layers, num_experts, num_experts_per_tok
"""
    shape_new_snippet = '''# NeMo-RL patch (DSA top-k capture transport): use the routed-experts
# slot-indexed transport for per-layer DSA top-k indices when explicitly enabled.
_NRL_DSA_TOPK_CAPTURE_ENV_VAR = "NRL_VLLM_DSA_TOPK_CAPTURE"
_NRL_DSA_TOPK_LAYER_IDS_ENV_VAR = "NRL_VLLM_DSA_TOPK_LAYER_IDS"


def _nrl_dsa_topk_dtype(max_model_len: int):
    if max_model_len <= 0:
        raise ValueError(
            "DSA top-k capture requires a positive max_model_len, got "
            f"{max_model_len}."
        )
    # Key ids are in [0, max_model_len - 1], plus the -1 missing-key sentinel.
    return np.int16 if max_model_len <= 32768 else np.int32


def _nrl_dsa_topk_torch_dtype(max_model_len: int) -> torch.dtype:
    return (
        torch.int16
        if _nrl_dsa_topk_dtype(max_model_len) is np.int16
        else torch.int32
    )


class _NRLDSASparseSlotBuffer:
    """Lazily allocate DSA replay rows one physical KV block at a time."""

    def __init__(
        self,
        *,
        max_num_slots: int,
        block_size: int,
        num_layers: int,
        top_k: int,
        dtype,
    ) -> None:
        if (
            max_num_slots <= 0
            or block_size <= 0
            or max_num_slots % block_size != 0
            or num_layers <= 0
            or top_k <= 0
        ):
            raise ValueError(
                "Invalid DSA sparse slot-buffer shape: "
                f"{max_num_slots=}, {block_size=}, {num_layers=}, {top_k=}."
            )
        self.shape = (max_num_slots, num_layers, top_k)
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype(np.int16), np.dtype(np.int32)):
            raise TypeError(f"DSA sparse slot-buffer dtype must be signed, got {dtype}.")
        self._block_size = block_size
        self._row_shape = (num_layers, top_k)
        self._blocks: dict[int, np.ndarray] = {}

    @property
    def nbytes(self) -> int:
        block_elements = self._block_size * self._row_shape[0] * self._row_shape[1]
        return len(self._blocks) * block_elements * self.dtype.itemsize

    def _normalize_slots(self, slot_mapping) -> np.ndarray:
        slots = np.asarray(slot_mapping)
        if not np.issubdtype(slots.dtype, np.integer):
            raise TypeError(
                "DSA sparse slot mappings must contain integers, got "
                f"dtype={slots.dtype}."
            )
        slots = slots.astype(np.int64, copy=False)
        if slots.size:
            min_slot = int(slots.min())
            max_slot = int(slots.max())
            if min_slot < 0 or max_slot >= self.shape[0]:
                raise IndexError(
                    "DSA sparse slot mapping is out of range: expected slots in "
                    f"[0, {self.shape[0]}), got min={min_slot}, max={max_slot}."
                )
        return slots

    def __getitem__(self, slot_mapping) -> np.ndarray:
        slots = self._normalize_slots(slot_mapping)
        result = np.full(slots.shape + self._row_shape, -1, dtype=self.dtype)
        flat_slots = slots.reshape(-1)
        flat_result = result.reshape((-1, *self._row_shape))
        block_ids = flat_slots // self._block_size
        for block_id in np.unique(block_ids):
            block = self._blocks.get(int(block_id))
            if block is None:
                continue
            positions = np.flatnonzero(block_ids == block_id)
            offsets = flat_slots[positions] % self._block_size
            flat_result[positions] = block[offsets]
        return result

    def __setitem__(self, slot_mapping, data) -> None:
        slots = self._normalize_slots(slot_mapping)
        values = np.asarray(data)
        expected_shape = slots.shape + self._row_shape
        if values.shape != expected_shape:
            raise ValueError(
                "DSA sparse slot-buffer assignment shape mismatch: expected "
                f"{expected_shape}, got {values.shape}."
            )
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError(
                "DSA sparse slot-buffer values must be integers, got "
                f"dtype={values.dtype}."
            )
        if values.size:
            min_value = int(values.min())
            max_value = int(values.max())
            max_stored_value = int(np.iinfo(self.dtype).max)
            if min_value < -1 or max_value > max_stored_value:
                raise ValueError(
                    "DSA sparse slot-buffer values are out of range: expected "
                    f"[-1, {max_stored_value}], got min={min_value}, "
                    f"max={max_value}."
                )

        flat_slots = slots.reshape(-1)
        flat_values = values.reshape((-1, *self._row_shape))
        block_ids = flat_slots // self._block_size
        for block_id_value in np.unique(block_ids):
            block_id = int(block_id_value)
            positions = np.flatnonzero(block_ids == block_id_value)
            offsets = flat_slots[positions] % self._block_size
            # NumPy's repeated advanced-index assignment semantics should not
            # decide replay correctness. Select each slot's last input row.
            _, first_from_end = np.unique(offsets[::-1], return_index=True)
            last_positions = positions[offsets.size - 1 - first_from_end]
            block = self._blocks.get(block_id)
            if block is None:
                block = np.full(
                    (self._block_size, *self._row_shape), -1, dtype=self.dtype
                )
                self._blocks[block_id] = block
            block_offsets = flat_slots[last_positions] % self._block_size
            block[block_offsets] = flat_values[last_positions]


def _nrl_dsa_topk_capture_enabled() -> bool:
    return os.environ.get(_NRL_DSA_TOPK_CAPTURE_ENV_VAR) == "1"


def _nrl_dsa_topk_layer_ids(vllm_config: VllmConfig) -> list[int]:
    config = vllm_config.model_config.hf_text_config
    num_hidden_layers = int(getattr(config, "num_hidden_layers", 0))
    if num_hidden_layers <= 0:
        raise ValueError(
            "DSA top-k capture requires a positive num_hidden_layers, got "
            f"{num_hidden_layers}."
        )

    index_topk_freq = int(getattr(config, "index_topk_freq", 1))
    index_topk_pattern = getattr(config, "index_topk_pattern", None)
    index_skip_topk_offset = int(getattr(config, "index_skip_topk_offset", 2))
    if index_topk_freq <= 0:
        raise ValueError(
            "DSA top-k capture requires a positive index_topk_freq, got "
            f"{index_topk_freq}."
        )

    compute_layer_ids = []
    for layer_id in range(num_hidden_layers):
        if index_topk_pattern is None:
            skip_topk = (
                max(layer_id - index_skip_topk_offset + 1, 0) % index_topk_freq != 0
            )
        elif layer_id < len(index_topk_pattern):
            skip_topk = index_topk_pattern[layer_id] == "S"
        else:
            skip_topk = False
        if not skip_topk:
            compute_layer_ids.append(layer_id)

    raw_layer_ids = os.environ.get(_NRL_DSA_TOPK_LAYER_IDS_ENV_VAR, "").strip()
    if not raw_layer_ids:
        selected_layer_ids = compute_layer_ids
    else:
        fields = raw_layer_ids.split(",")
        if any(not field.strip() for field in fields):
            raise ValueError(
                "NRL_VLLM_DSA_TOPK_LAYER_IDS must be a comma-separated list "
                f"of integers, got {raw_layer_ids!r}."
            )
        try:
            selected_layer_ids = [int(field) for field in fields]
        except ValueError as exc:
            raise ValueError(
                "NRL_VLLM_DSA_TOPK_LAYER_IDS must be a comma-separated list "
                f"of integers, got {raw_layer_ids!r}."
            ) from exc
        if len(set(selected_layer_ids)) != len(selected_layer_ids):
            raise ValueError(
                "NRL_VLLM_DSA_TOPK_LAYER_IDS contains duplicate layer ids: "
                f"{selected_layer_ids}."
            )
        invalid_layer_ids = sorted(set(selected_layer_ids) - set(compute_layer_ids))
        if invalid_layer_ids:
            raise ValueError(
                "DSA top-k capture can only select non-MTP compute layers; "
                f"invalid layer ids are {invalid_layer_ids}, available compute "
                f"layers are {compute_layer_ids}."
            )

    if not selected_layer_ids:
        raise ValueError("DSA top-k capture selected no compute layers.")
    return selected_layer_ids


def _get_routed_experts_shape(vllm_config: VllmConfig) -> tuple[int, int, int]:
    model_config = vllm_config.model_config
    if _nrl_dsa_topk_capture_enabled():
        layer_ids = _nrl_dsa_topk_layer_ids(vllm_config)
        max_model_len = int(model_config.max_model_len)
        index_topk = int(getattr(model_config.hf_text_config, "index_topk", 0))
        if max_model_len <= 0 or index_topk <= 0:
            raise ValueError(
                "DSA top-k capture requires positive max_model_len and index_topk, "
                f"got {max_model_len=}, {index_topk=}."
            )
        return len(layer_ids), max_model_len, index_topk

    num_layers = model_config.get_total_num_hidden_layers()
    num_experts = model_config.get_num_experts()
    num_experts_per_tok = model_config.get_num_experts_per_tok()
    if num_layers <= 0 or num_experts <= 0 or num_experts_per_tok <= 0:
        raise ValueError(
            "Routed-experts capture requires positive layer, expert, and "
            "experts-per-token counts, got "
            f"{num_layers=}, {num_experts=}, {num_experts_per_tok=}."
        )
    return num_layers, num_experts, num_experts_per_tok
'''
    init_old_snippet = (
        "        num_layers, _, num_experts_per_tok = "
        "_get_routed_experts_shape(vllm_config)\n"
        "        logger.info(\n"
    )
    init_new_snippet = (
        "        num_layers, _, num_experts_per_tok = "
        "_get_routed_experts_shape(vllm_config)\n"
        "        nrl_dsa_topk_capture = _nrl_dsa_topk_capture_enabled()\n"
        "        self._nrl_dsa_topk_layer_ids = (\n"
        "            _nrl_dsa_topk_layer_ids(vllm_config)\n"
        "            if nrl_dsa_topk_capture\n"
        "            else None\n"
        "        )\n"
        "        nrl_dsa_topk_dtype = (\n"
        "            _nrl_dsa_topk_torch_dtype(\n"
        "                int(vllm_config.model_config.max_model_len)\n"
        "            )\n"
        "            if nrl_dsa_topk_capture\n"
        "            else None\n"
        "        )\n"
        "        logger.info(\n"
    )
    device_buffer_old_snippet = """        self.device_buffer = torch.zeros(
            (
                max_num_batched_tokens,
                num_layers,
                num_experts_per_tok,
            ),
            # Use int32 for the device / host transit buffers: it
            # matches the router's native topk_ids dtype, is universally
            # supported by NCCL (uint8/uint16 are version-dependent),
            # and the extra bytes are small (few MB per worker). The
            # big scheduler-side slot buffer stays narrow.
            dtype=torch.int32,
            device=current_platform.device_type,
        )
"""
    device_buffer_new_snippet = """        self.device_buffer = torch.full(
            (
                max_num_batched_tokens,
                num_layers,
                num_experts_per_tok,
            ),
            # Missing DSA rows must remain distinguishable from key 0. MoE
            # capture retains its historical zero initialization.
            fill_value=-1 if nrl_dsa_topk_capture else 0,
            # DSA narrows only this destination buffer. Any TP all-gather in
            # capture() has already run on the indexer's native int32 source,
            # while D2H and numpy conversion preserve signed int16 directly.
            dtype=nrl_dsa_topk_dtype or torch.int32,
            device=current_platform.device_type,
        )
"""
    binder_old_snippet = '''    """Attach capture callbacks to the target model's MoE routers."""
    from vllm.model_executor.layers.fused_moe.layer import MoERunner
'''
    binder_new_snippet = '''    """Attach capture callbacks to MoE routers or selected DSA layers."""
    if _nrl_dsa_topk_capture_enabled():
        from vllm.model_executor.models.utils import extract_layer_index
        from vllm.models.deepseek_v32.attention import DeepseekV32Attention

        selected_layer_ids = capturer._nrl_dsa_topk_layer_ids
        if selected_layer_ids is None:
            raise ValueError("DSA top-k capturer was not initialized in DSA mode.")
        compact_slot_by_layer = {
            layer_id: slot for slot, layer_id in enumerate(selected_layer_ids)
        }
        bound_layer_ids = []
        for module in model.modules():
            if not isinstance(module, DeepseekV32Attention):
                continue
            layer_id = extract_layer_index(module.layer_name)
            compact_slot = compact_slot_by_layer.get(layer_id)
            if compact_slot is None or module.indexer is None:
                continue
            if layer_id in bound_layer_ids:
                raise ValueError(
                    f"Found duplicate DeepseekV32Attention for DSA layer {layer_id}."
                )
            module._nrl_dsa_topk_capture_fn = partial(
                capturer.capture, compact_slot
            )
            bound_layer_ids.append(layer_id)

        missing_layer_ids = sorted(set(selected_layer_ids) - set(bound_layer_ids))
        if missing_layer_ids:
            raise ValueError(
                "Could not bind DSA top-k capture for selected compute layers "
                f"{missing_layer_ids}; bound layers were {sorted(bound_layer_ids)}."
            )
        return

    from vllm.model_executor.layers.fused_moe.layer import MoERunner
'''
    manager_old_snippet = """        expert_id_dtype = np.uint8 if num_experts <= 256 else np.uint16
        self.routed_experts_by_slot = np.zeros(
            (
                max_num_slots,
                num_layers,
                num_experts_per_tok,
            ),
            dtype=expert_id_dtype,
        )
"""
    manager_new_snippet = """        self._nrl_copy_step_outputs = _nrl_dsa_topk_capture_enabled()
        if self._nrl_copy_step_outputs:
            # DSA uses -1 for absent/padded key ids. Allocate only physical KV
            # blocks that receive rows; a full slot-pool ndarray is prohibitive
            # for many layers and index_topk=2048.
            expert_id_dtype = _nrl_dsa_topk_dtype(num_experts)
            self.routed_experts_by_slot = _NRLDSASparseSlotBuffer(
                max_num_slots=max_num_slots,
                block_size=self.block_size,
                num_layers=num_layers,
                top_k=num_experts_per_tok,
                dtype=expert_id_dtype,
            )
        else:
            expert_id_dtype = np.uint8 if num_experts <= 256 else np.uint16
            self.routed_experts_by_slot = np.zeros(
                (
                    max_num_slots,
                    num_layers,
                    num_experts_per_tok,
                ),
                dtype=expert_id_dtype,
            )
"""

    edits = (
        ("import", import_old_snippet, import_new_snippet),
        ("shape", shape_old_snippet, shape_new_snippet),
        ("capturer init", init_old_snippet, init_new_snippet),
        ("device buffer", device_buffer_old_snippet, device_buffer_new_snippet),
        ("binder", binder_old_snippet, binder_new_snippet),
        ("manager buffer", manager_old_snippet, manager_new_snippet),
    )
    with _locked_file_patch(file_to_patch) as (content, write_back):
        if marker in content:
            logger.info("vLLM DSA top-k capturer patch already applied.")
            return True
        invalid_anchors = [name for name, old, _ in edits if content.count(old) != 1]
        if invalid_anchors:
            message = (
                "Could not apply vLLM DSA top-k capturer patch: expected "
                f"exactly one of each source anchor {invalid_anchors} in "
                f"{file_to_patch}. The vLLM version may have changed."
            )
            if required:
                raise RuntimeError(message)
            logger.warning(message)
            return False
        for _, old, new in edits:
            content = content.replace(old, new, 1)
        write_back(content)

    logger.info("Successfully patched vLLM routed-experts transport for DSA top-k.")
    return True


def _patch_vllm_dsa_topk_scheduler(logger, *, required: bool = False) -> bool:
    """Keep legacy-runner DSA decode rows alive across scheduler steps.

    vLLM's synchronous legacy model runner exposes a NumPy view over a reused
    pinned CPU buffer.  The scheduler normally keeps that view for per-request
    decode output because routed-expert transit and storage dtypes differ.  DSA
    intentionally uses the same signed dtype on both sides, so ``astype`` would
    retain the reused view and a later D2H could overwrite earlier token rows.
    """
    try:
        file_to_patch = _get_vllm_file("v1/core/sched/scheduler.py")
    except RuntimeError:
        message = "Could not locate scheduler.py for the DSA top-k copy patch."
        if required:
            raise RuntimeError(message) from None
        logger.warning(message)
        return False

    marker = "NeMo-RL patch (retain DSA decode routes across steps)"
    old_snippet = """            routing_data = re.routing_data.astype(
                self.routed_experts_mgr.routed_experts_by_slot.dtype,
                copy=False,
            )
"""
    new_snippet = """            # NeMo-RL patch (retain DSA decode routes across steps): the legacy
            # synchronous model runner returns a view into a reused pinned CPU
            # buffer. DSA transit/storage dtypes can match, so force a private
            # scheduler copy before per-request slices outlive this step.
            routing_data = re.routing_data.astype(
                self.routed_experts_mgr.routed_experts_by_slot.dtype,
                copy=getattr(
                    self.routed_experts_mgr, "_nrl_copy_step_outputs", False
                ),
            )
"""

    with _locked_file_patch(file_to_patch) as (content, write_back):
        if marker in content:
            logger.info("vLLM DSA top-k scheduler copy patch already applied.")
            return True
        if content.count(old_snippet) != 1:
            message = (
                "Could not apply vLLM DSA top-k scheduler copy patch: expected "
                f"exactly one source anchor in {file_to_patch}. The vLLM "
                "version may have changed."
            )
            if required:
                raise RuntimeError(message)
            logger.warning(message)
            return False
        write_back(content.replace(old_snippet, new_snippet, 1))

    logger.info("Successfully patched vLLM scheduler for stable DSA decode rows.")
    return True


def _patch_vllm_dsa_topk_attention(logger, *, required: bool = False) -> bool:
    """Capture each DSA compute layer before its shared top-k buffer is reused."""
    try:
        file_to_patch = _get_vllm_file("models/deepseek_v32/attention.py")
    except RuntimeError:
        message = "Could not locate DeepSeek V3.2 attention.py for DSA top-k capture."
        if required:
            raise RuntimeError(message) from None
        logger.warning(message)
        return False

    marker = "NeMo-RL patch (capture DSA top-k before shared-buffer reuse)"
    import_old_snippet = "from vllm.config import CacheConfig, VllmConfig\n"
    import_new_snippet = (
        "from vllm.config import CacheConfig, CUDAGraphMode, VllmConfig\n"
    )
    capture_old_snippet = """        num_actual = attn_metadata.num_actual_tokens  # type: ignore[attr-defined]
        if num_actual == 0:
            output.zero_()
            return

        if self._use_sparse_mha(attn_metadata):
"""
    capture_new_snippet = """        num_actual = attn_metadata.num_actual_tokens  # type: ignore[attr-defined]
        if num_actual == 0:
            output.zero_()
            return

        use_mha = self._use_sparse_mha(attn_metadata)
        capture_fn = getattr(self, "_nrl_dsa_topk_capture_fn", None)
        if capture_fn is not None:
            # NeMo-RL patch (capture DSA top-k before shared-buffer reuse): every
            # compute layer writes the same model-level buffer, so copy it into
            # the slot-indexed capturer before the next layer overwrites it. For
            # dense prefill, capture the key set actually consumed by attention,
            # rather than stale or unused scorer output.
            assert self.indexer is not None
            assert self.topk_indices_buffer is not None
            prefill_metadata = getattr(attn_metadata, "prefill", None)
            num_decode_tokens = getattr(attn_metadata, "num_decode_tokens", -1)
            dense_prefill = use_mha and getattr(
                prefill_metadata, "use_dense_mha", False
            )
            scoring_was_skipped = (
                get_forward_context().cudagraph_runtime_mode != CUDAGraphMode.FULL
                and bool(self._dense_mha_metadata_layer_name)
                and getattr(prefill_metadata, "use_dense_mha", False)
                and num_decode_tokens == 0
                and not torch.cuda.is_current_stream_capturing()
            )
            if scoring_was_skipped and not dense_prefill:
                raise RuntimeError(
                    "DSA indexer scoring was skipped for a batch that did not "
                    "take the dense-prefill attention route."
                )

            topk = self.indexer.topk_tokens
            scored_topk = self.topk_indices_buffer[:num_actual, :topk]
            if dense_prefill:
                # forward_impl partitions a mixed batch as sparse decode rows
                # followed by dense-MHA prefill rows. Masked MHA is different:
                # it consumes the scorer top-k as a mask and therefore does not
                # enter this branch (use_dense_mha is false).
                if not 0 <= num_decode_tokens <= num_actual:
                    raise RuntimeError(
                        "Invalid DSA decode/prefill partition for top-k capture: "
                        f"num_decode_tokens={num_decode_tokens}, "
                        f"num_actual_tokens={num_actual}."
                    )
                dense_positions = positions[num_decode_tokens:num_actual].to(
                    device=self.topk_indices_buffer.device, dtype=torch.int64
                )
                if bool(torch.any((dense_positions < 0) | (dense_positions >= topk))):
                    raise RuntimeError(
                        "Dense DSA top-k capture cannot represent all causal keys: "
                        f"positions must be in [0, {topk}), got "
                        f"min={int(dense_positions.min().item())}, "
                        f"max={int(dense_positions.max().item())}."
                    )
                key_ids = torch.arange(
                    topk,
                    dtype=self.topk_indices_buffer.dtype,
                    device=self.topk_indices_buffer.device,
                ).unsqueeze(0)
                dense_topk = key_ids.expand(
                    num_actual - num_decode_tokens, -1
                ).clone()
                dense_topk.masked_fill_(key_ids > dense_positions.unsqueeze(1), -1)
                if num_decode_tokens:
                    effective_topk = scored_topk.clone()
                    effective_topk[num_decode_tokens:] = dense_topk
                else:
                    # Avoid reading the shared buffer when the pure-dense fast
                    # path deliberately skipped scoring and left it stale.
                    effective_topk = dense_topk
                capture_fn(effective_topk)
            else:
                capture_fn(scored_topk)

        if use_mha:
"""

    edits = (
        ("CUDAGraphMode import", import_old_snippet, import_new_snippet),
        ("capture", capture_old_snippet, capture_new_snippet),
    )
    with _locked_file_patch(file_to_patch) as (content, write_back):
        if marker in content:
            logger.info("vLLM DSA top-k attention capture patch already applied.")
            return True
        invalid_anchors = [name for name, old, _ in edits if content.count(old) != 1]
        if invalid_anchors:
            message = (
                "Could not apply vLLM DSA top-k attention capture patch: expected "
                f"exactly one of each source anchor {invalid_anchors} in "
                f"{file_to_patch}. The vLLM version may have changed."
            )
            if required:
                raise RuntimeError(message)
            logger.warning(message)
            return False
        for _, old, new in edits:
            content = content.replace(old, new, 1)
        write_back(content)

    logger.info("Successfully patched vLLM attention for DSA top-k capture.")
    return True


def _patch_vllm_moe_routed_experts_capture(logger, *, required: bool = False) -> bool:
    """Fire the routed-experts capture hook on the monolithic fused-MoE path.

    ``RoutedExpertsCapturer`` (used by router replay / R3) is driven by the
    ``capture_fn`` that only fires inside ``BaseRouter._select_experts``. But
    ``MoERunner._apply_quant_method`` calls ``select_experts`` only on the
    *modular* kernel branch; *monolithic* kernels (e.g. the FlashInfer TRT-LLM
    NVFP4-per-token fused MoE) compute top-k routing internally via
    ``forward_monolithic`` and never call it. The capture buffer therefore
    stays zero, and the returned ``routed_experts`` are all-zero -> Megatron's
    router replay sees duplicate expert ids and dies with "Split sizes doesn't
    match total dim 0" in the MoE all_to_all during get_logprobs.

    This inserts an explicit ``select_experts`` call on the monolithic branch,
    guarded by ``capture_fn is not None`` so it only runs during rollout when
    routing capture is active (no cost otherwise).
    """
    try:
        file_to_patch = _get_vllm_file(
            "model_executor/layers/fused_moe/runner/moe_runner.py"
        )
    except RuntimeError:
        message = "Could not locate moe_runner.py for routed-experts capture patch."
        if required:
            raise RuntimeError(message) from None
        logger.warning(message)
        return False

    marker = "NeMo-RL patch (routed-experts capture for router replay)"
    old_snippet = (
        "        if self.routed_experts.quant_method.is_monolithic:\n"
        "            # Monolithic kernels: pass router_logits to routed_experts\n"
        "            fused_out = self.routed_experts.forward_monolithic("
    )
    new_snippet = (
        "        if self.routed_experts.quant_method.is_monolithic:\n"
        "            # Monolithic kernels: pass router_logits to routed_experts\n"
        "            # NeMo-RL patch (routed-experts capture for router replay): "
        "monolithic MoE kernels compute top-k routing\n"
        "            # inside the fused kernel and never call router.select_experts,\n"
        "            # so the RoutedExpertsCapturer hook never fires and returned\n"
        "            # routes are all-zero. Fire it explicitly when capture is on.\n"
        '            if getattr(self.router, "capture_fn", None) is not None:\n'
        "                self.router.select_experts(\n"
        "                    hidden_states=hidden_states,\n"
        "                    router_logits=router_logits,\n"
        "                    topk_indices_dtype=self._quant_method.topk_indices_dtype,\n"
        "                    input_ids=input_ids,\n"
        "                )\n"
        "            fused_out = self.routed_experts.forward_monolithic("
    )

    with _locked_file_patch(file_to_patch) as (content, write_back):
        if marker in content:
            logger.info("MoE routed-experts capture patch already applied.")
            return True
        if old_snippet not in content:
            message = (
                "Could not apply MoE routed-experts capture patch: expected "
                f"code snippet not found in {file_to_patch}. The vLLM version "
                "may have changed."
            )
            if required:
                raise RuntimeError(message)
            logger.warning(message)
            return False
        content = content.replace(old_snippet, new_snippet, 1)
        write_back(content)

    logger.info("Successfully patched MoE routed-experts capture (monolithic path).")
    return True


def _patch_vllm_routed_experts_capture_router_fallback(
    logger, *, required: bool = False
) -> bool:
    """Let monolithic MoE kernels without in-kernel capture use the router hook.

    vLLM 0.29 moved the routed-experts binding into
    ``routed_experts_capturer.bind_routed_experts_capturer``. For a monolithic
    kernel it requires ``fused_experts.supports_routing_replay_capture()`` and
    binds the capture function to that experts *object* (the kernel then writes
    ``routing_replay_out`` itself); any other monolithic kernel is rejected with
    ``ValueError``. Two things make the object binding unusable for NeMo-RL's
    NVFP4 per-token method: the kernel is rebuilt on every refit, so the bound
    capture function is dropped after the first weight update and the returned
    routes go back to all-zero; and the per-token kernel is the one FlashInfer
    launch that has not been validated with a replay buffer attached. Kernels
    that report no in-kernel capture (see ``nvfp4_pertoken.host_captured_experts_cls``)
    therefore fall back to ``router.set_capture_fn`` — the hook that
    ``_patch_vllm_moe_routed_experts_capture`` fires on the monolithic branch
    and the path vLLM 0.26 used for every monolithic kernel.
    """
    try:
        file_to_patch = _get_vllm_file(
            "model_executor/layers/fused_moe/routed_experts_capturer.py"
        )
    except RuntimeError:
        message = (
            "Could not locate routed_experts_capturer.py for the routed-experts "
            "capture router-fallback patch."
        )
        if required:
            raise RuntimeError(message) from None
        logger.warning(message)
        return False

    marker = "NeMo-RL patch (router fallback for monolithic routed-experts capture)"
    old_snippet = (
        "        if quant_method.is_monolithic:\n"
        "            if not (\n"
        "                isinstance(fused_experts, FusedMoEExpertsMonolithic)\n"
        "                and fused_experts.supports_routing_replay_capture()\n"
        "            ):\n"
        "                raise ValueError(\n"
        '                    "Routed-experts capture is not supported with monolithic "\n'
        '                    f"MoE kernel {type(fused_experts).__name__}."\n'
        "                )\n"
        "            fused_experts.set_capture_fn(capture_fn)\n"
        "            num_bound += 1\n"
    )
    new_snippet = (
        "        if quant_method.is_monolithic:\n"
        "            # NeMo-RL patch (router fallback for monolithic routed-experts capture):\n"
        "            # a monolithic kernel that does not capture routing itself is\n"
        "            # captured through the router; NeMo-RL's moe_runner patch fires\n"
        "            # router.select_experts on the monolithic branch when capture_fn is set.\n"
        "            if (\n"
        "                isinstance(fused_experts, FusedMoEExpertsMonolithic)\n"
        "                and fused_experts.supports_routing_replay_capture()\n"
        "            ):\n"
        "                fused_experts.set_capture_fn(capture_fn)\n"
        "                num_bound += 1\n"
        "            elif isinstance(module.router, BaseRouter):\n"
        "                module.router.set_capture_fn(capture_fn)\n"
        "                num_bound += 1\n"
        "            else:\n"
        "                raise ValueError(\n"
        '                    "Routed-experts capture is not supported with monolithic "\n'
        '                    f"MoE kernel {type(fused_experts).__name__}."\n'
        "                )\n"
    )

    with _locked_file_patch(file_to_patch) as (content, write_back):
        if marker in content:
            logger.info("Routed-experts capture router-fallback patch already applied.")
            return True
        if old_snippet not in content:
            message = (
                "Could not apply the routed-experts capture router-fallback patch: "
                f"expected code snippet not found in {file_to_patch}. The vLLM "
                "version may have changed."
            )
            if required:
                raise RuntimeError(message)
            logger.warning(message)
            return False
        content = content.replace(old_snippet, new_snippet, 1)
        write_back(content)

    logger.info(
        "Successfully patched routed-experts capture (router fallback for "
        "monolithic kernels)."
    )
    return True


def _patch_vllm_nemotron_h_fp32_lm_head(logger) -> bool:
    """Compute NemotronH logits with an fp32 LM head (MiniMax-M1-style).

    bf16 rounding of the logits GEMM output is the dominant contributor to
    generation/training logprob mismatch (train/token_mult_prob_error). With
    this patch the sampled-token logprobs come from fp32 logits, matching a
    trainer that enables megatron_cfg.fp32_lm_head.

    This must be a source patch (not a monkeypatch): the model executes in
    vLLM's EngineCore worker subprocesses, which import vllm independently of
    this process. The patched code is opt-in at runtime via an internal
    NRL_VLLM_FP32_LM_HEAD=1 environment variable set from
    policy.generation.vllm_cfg.fp32_lm_head.
    When enabled, the live ParallelLMHead keeps its original parameter dtype
    and quantization config; only the projection path casts hidden states,
    weights, and optional bias to fp32 at runtime.
    """
    try:
        file_to_patch = _get_vllm_file("model_executor/models/nemotron_h.py")
    except RuntimeError:
        logger.warning("Could not locate nemotron_h.py for the fp32 LM head patch.")
        return False

    old_import_snippet = """import torch
from torch import nn"""
    old_fp32_import_snippet = """import os

import torch
from torch import nn"""
    new_import_snippet = old_fp32_import_snippet
    old_lm_head_snippet = """        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )"""
    new_lm_head_snippet = f"""        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self._nrl_fp32_lm_head = (
            os.environ.get("{VLLM_NEMOTRON_H_FP32_LM_HEAD_ENV_VAR}", "0") == "1"
        )"""
    old_logits_processor_snippet = (
        "        self.logits_processor = LogitsProcessor(config.vocab_size)"
    )
    new_logits_processor_snippet = """        self.logits_processor = LogitsProcessor(config.vocab_size)
        if self._nrl_fp32_lm_head:

            def _nrl_fp32_lm_head_forward(
                input_, embedding_bias=None, _lm_head=self.lm_head
            ):
                if not getattr(_lm_head, "_nrl_fp32_lm_head_forward_logged", False):
                    print(
                        "[fp32_lm_head] NemotronH vLLM lm_head.forward casts "
                        "input and weight to fp32",
                        flush=True,
                    )
                    _lm_head._nrl_fp32_lm_head_forward_logged = True
                logits = torch.matmul(
                    input_.to(dtype=torch.float32),
                    _lm_head.weight.to(dtype=torch.float32).t(),
                )
                if embedding_bias is not None:
                    logits = logits + embedding_bias.to(dtype=torch.float32)
                return logits

            self.lm_head.forward = _nrl_fp32_lm_head_forward
            _orig_quant_apply = self.lm_head.quant_method.apply

            def _nrl_fp32_lm_head_apply(
                layer,
                input_,
                bias=None,
                _lm_head=self.lm_head,
                _orig_apply=_orig_quant_apply,
                **kwargs,
            ):
                if layer is _lm_head:
                    return _lm_head(input_, bias)
                return _orig_apply(layer, input_, bias=bias, **kwargs)

            self.lm_head.quant_method.apply = _nrl_fp32_lm_head_apply"""
    old_snippet = """        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits"""

    with _locked_file_patch(file_to_patch) as (content, write_back):
        if (
            new_import_snippet in content
            and new_lm_head_snippet in content
            and new_logits_processor_snippet in content
        ):
            logger.info("NemotronH fp32 LM head patch already present.")
            return True

        if new_import_snippet not in content:
            if old_fp32_import_snippet in content:
                content = content.replace(
                    old_fp32_import_snippet, new_import_snippet, 1
                )
            elif content.count(old_import_snippet) == 1:
                content = content.replace(old_import_snippet, new_import_snippet, 1)
            else:
                logger.warning(
                    "NemotronH fp32 LM head import anchor not found exactly once "
                    "in %s; patch not applied.",
                    file_to_patch,
                )
                return False

        if new_lm_head_snippet not in content:
            if content.count(old_lm_head_snippet) != 1:
                logger.warning(
                    "NemotronH fp32 LM head constructor anchor not found exactly "
                    "once in %s; patch not applied.",
                    file_to_patch,
                )
                return False
            content = content.replace(old_lm_head_snippet, new_lm_head_snippet, 1)

        if new_logits_processor_snippet not in content:
            if content.count(old_logits_processor_snippet) != 1:
                logger.warning(
                    "NemotronH fp32 logits_processor anchor not found exactly once "
                    "in %s; patch not applied.",
                    file_to_patch,
                )
                return False
            content = content.replace(
                old_logits_processor_snippet, new_logits_processor_snippet, 1
            )

        if content.count(old_snippet) != 1:
            logger.warning(
                "NemotronH fp32 compute_logits anchor not found exactly once "
                "in %s; patch not applied.",
                file_to_patch,
            )
            return False
        write_back(content)

    logger.info("Applied NemotronH fp32 LM head source patch.")
    return True


@contextmanager
def modelopt_moe_amax_aliases(model: "torch.nn.Module") -> Iterator[None]:
    """Temporarily expose nested ModelOpt MoE amax buffers to vLLM's loader.

    Verified against vLLM 0.26.0: ``RoutedExperts.load_weights`` maps expert
    amax keys to names such as ``w13_input_quantizer._amax``, then resolves
    them with one ``getattr``. This refit path sends ModelOpt buffers through
    that loader, which offers no hook for resolving nested target names.

    Nemotron-H reached this loader after vLLM removed the inner
    ``NemotronHModel.load_weights`` in 0.26.0. Its old flat parameter lookup
    accepted dotted keys exposed by our buffer-to-parameter adapter:
    https://github.com/vllm-project/vllm/commit/c233d90aa826df072872df47b201450059be8e71

    Aliases reference the original tensors without registering additional
    buffers or state-dict entries. Existing attributes belong to the caller
    or an outer context and are preserved. Only aliases created here are
    removed, including when setup or weight loading raises.

    Args:
        model: ModelOpt model whose MoE quantizer buffers will be refitted.
    """
    aliased: list[tuple["torch.nn.Module", str]] = []
    try:
        for module in model.modules():
            for child_name, child in module.named_children():
                if not child_name.startswith(("w13_", "w2_")):
                    continue
                if not child_name.endswith("_quantizer"):
                    continue
                for buf_name, buf in child.named_buffers(recurse=False):
                    if not buf_name.endswith("_amax"):
                        continue
                    alias = f"{child_name}.{buf_name}"
                    if hasattr(module, alias):
                        continue
                    setattr(module, alias, buf)
                    aliased.append((module, alias))
        yield
    finally:
        for module, alias in reversed(aliased):
            delattr(module, alias)


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
    nemotron_h_fp32_lm_head: bool | None = None,
    require_moe_routed_experts_capture: bool = False,
    require_dsa_topk_capture: bool = False,
    dsa_topk_layer_ids: list[int] | None = None,
) -> None:
    # Import lazily so importing the worker module does not import vLLM.
    import vllm.envs as envs
    from vllm.logger import init_logger

    patch_logger = init_logger("vllm_patch")
    if require_moe_routed_experts_capture and require_dsa_topk_capture:
        raise ValueError(
            "MoE router replay and DSA top-k replay cannot both use vLLM's "
            "routed-experts capture channel."
        )
    if dsa_topk_layer_ids is not None and not require_dsa_topk_capture:
        raise ValueError(
            "dsa_topk_layer_ids was provided while DSA top-k capture is disabled."
        )

    if require_dsa_topk_capture:
        selected_layer_ids = dsa_topk_layer_ids or []
        if any(layer_id < 0 for layer_id in selected_layer_ids):
            raise ValueError(
                f"DSA top-k layer ids must be non-negative, got {selected_layer_ids}."
            )
        if len(set(selected_layer_ids)) != len(selected_layer_ids):
            raise ValueError(
                f"DSA top-k layer ids must be unique, got {selected_layer_ids}."
            )
        os.environ[VLLM_DSA_TOPK_CAPTURE_ENV_VAR] = "1"
        os.environ[VLLM_DSA_TOPK_LAYER_IDS_ENV_VAR] = ",".join(
            str(layer_id) for layer_id in selected_layer_ids
        )
        extra_env_vars = [
            *(extra_env_vars or []),
            VLLM_DSA_TOPK_CAPTURE_ENV_VAR,
            VLLM_DSA_TOPK_LAYER_IDS_ENV_VAR,
        ]
    else:
        os.environ.pop(VLLM_DSA_TOPK_CAPTURE_ENV_VAR, None)
        os.environ.pop(VLLM_DSA_TOPK_LAYER_IDS_ENV_VAR, None)

    nemotron_h_fp32_lm_head_enabled = bool(nemotron_h_fp32_lm_head)
    if nemotron_h_fp32_lm_head_enabled:
        os.environ[VLLM_NEMOTRON_H_FP32_LM_HEAD_ENV_VAR] = "1"
        extra_env_vars = [
            *(extra_env_vars or []),
            VLLM_NEMOTRON_H_FP32_LM_HEAD_ENV_VAR,
        ]
    else:
        os.environ.pop(VLLM_NEMOTRON_H_FP32_LM_HEAD_ENV_VAR, None)

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
    if nemotron_h_fp32_lm_head_enabled and not _patch_vllm_nemotron_h_fp32_lm_head(
        patch_logger
    ):
        raise RuntimeError(
            "vllm_cfg.fp32_lm_head is enabled, but that flag currently maps to "
            "the Nemotron-H-only vLLM fp32 LM head source patch, and the patch "
            "could not be applied. Disable the flag or update the patch anchors "
            "for this vLLM version."
        )
    if require_dsa_topk_capture:
        _patch_vllm_dsa_topk_capturer(patch_logger, required=True)
        _patch_vllm_dsa_topk_scheduler(patch_logger, required=True)
        _patch_vllm_dsa_topk_attention(patch_logger, required=True)
    else:
        _patch_vllm_moe_routed_experts_capture(
            patch_logger, required=require_moe_routed_experts_capture
        )
        _patch_vllm_routed_experts_capture_router_fallback(
            patch_logger, required=require_moe_routed_experts_capture
        )
