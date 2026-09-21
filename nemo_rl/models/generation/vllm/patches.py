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

from nemo_rl.models.generation.vllm.config import (
    VLLM_NEMOTRON_H_FP32_LM_HEAD_ENV_VAR,
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

    # vLLM >= 0.29: RayWorkerV2.create_dist_init_method binds and keeps the
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
) -> None:
    # Import lazily so importing the worker module does not import vLLM.
    import vllm.envs as envs
    from vllm.logger import init_logger

    patch_logger = init_logger("vllm_patch")
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
    _patch_vllm_moe_routed_experts_capture(
        patch_logger, required=require_moe_routed_experts_capture
    )
    _patch_vllm_routed_experts_capture_router_fallback(
        patch_logger, required=require_moe_routed_experts_capture
    )
