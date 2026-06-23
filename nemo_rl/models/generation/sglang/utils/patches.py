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

import logging
import os
from importlib.util import find_spec

logger = logging.getLogger(__name__)


def _get_sglang_file(relative_path: str) -> str:
    spec = find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            f"sglang package not found while attempting to patch '{relative_path}'. "
        )

    base_dir = next(iter(spec.submodule_search_locations))
    file_path = os.path.join(base_dir, *relative_path.split("/"))
    if not os.path.exists(file_path):
        raise RuntimeError(
            f"Expected sglang file '{relative_path}' not found at '{file_path}'. "
            "The sglang version may have moved this file; compat patch cannot be applied."
        )
    return file_path


def _write_and_verify(
    file_path: str, content: str, sentinel: str | tuple[str, ...]
) -> None:
    sentinels = (sentinel,) if isinstance(sentinel, str) else sentinel
    tmp_path = f"{file_path}.nemo_rl_compat.{os.getpid()}.tmp"
    with open(tmp_path, "w") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, file_path)

    with open(file_path, "r") as f:
        verify = f.read()
    missing = [item for item in sentinels if item not in verify]
    if missing:
        raise RuntimeError(
            f"Compat patch verification failed for {file_path}: "
            f"sentinel(s) {missing} not present after write. "
            "The write may have been silently dropped by the filesystem."
        )


def _neutralize_cutlass_dsl_experimental_stub() -> None:
    """Empty the cute.experimental NotImplementedError stub in nvidia-cutlass-dsl.

    The stub (shipped by wheels >=4.4.0.dev1) detonates under walk_packages
    in cute.compile and kills sglang workers.
    Related: https://github.com/NVIDIA/cutlass/issues/3132.
    """
    try:
        spec = find_spec("nvidia_cutlass_dsl")
    except (ImportError, ValueError):
        return
    if spec is None or not spec.submodule_search_locations:
        return

    base_dir = next(iter(spec.submodule_search_locations))
    stub_path = os.path.join(
        base_dir,
        "python_packages",
        "cutlass",
        "cute",
        "experimental",
        "__init__.py",
    )
    if not os.path.exists(stub_path):
        return

    try:
        with open(stub_path, "r") as f:
            content = f.read()
    except OSError as e:
        logger.warning("Could not read cute.experimental stub at %s: %s", stub_path, e)
        return

    sentinel = "CuTe Experimental module is only supported on Cuda toolkit"
    if sentinel not in content:
        return

    replacement = (
        "# Neutralized by nemo_rl: upstream nvidia-cutlass-dsl wheel ships a\n"
        "# stub that raises NotImplementedError on CTK<13.1, which detonates\n"
        "# under pkgutil.walk_packages during every cute.compile. Empty body\n"
        "# is safe because the wheel ships no real cute.experimental subpackages.\n"
    )
    try:
        _write_and_verify(stub_path, replacement, "Neutralized by nemo_rl")
    except (OSError, RuntimeError) as e:
        logger.warning(
            "Failed to neutralize cute.experimental stub at %s: %s. "
            "SGLang workers may hit NotImplementedError under cute.compile.",
            stub_path,
            e,
        )
        return
    logger.info(
        "Neutralized nvidia-cutlass-dsl cute.experimental stub at %s.", stub_path
    )


def _patch_sglang_file_replacements(
    relative_path: str,
    replacements: tuple[tuple[str, str, str], ...],
    description: str,
) -> None:
    file_to_patch = _get_sglang_file(relative_path)

    with open(file_to_patch, "r") as f:
        content = f.read()

    missing_replacements = [
        (sentinel, anchor, replacement)
        for sentinel, anchor, replacement in replacements
        if sentinel not in content
    ]
    if not missing_replacements:
        return

    for sentinel, anchor, replacement in missing_replacements:
        if anchor not in content:
            raise RuntimeError(
                f"{description} anchor for sentinel '{sentinel}' not found in "
                f"{file_to_patch}."
            )
        content = content.replace(anchor, replacement, 1)

    _write_and_verify(
        file_to_patch, content, tuple(sentinel for sentinel, _, _ in replacements)
    )
    logger.info("Patched %s in %s.", description, file_to_patch)


def _patch_sglang_safe_unpickler() -> None:
    file_to_patch = _get_sglang_file("srt/utils/common.py")

    with open(file_to_patch, "r") as f:
        content = f.read()

    sentinel = '"nemo_rl.models.generation.sglang.utils.train_utils."'
    if sentinel in content:
        return

    anchor = '        "torch.nn.parameter.",\n'
    insertion = (
        anchor + '        "nemo_rl.models.generation.sglang.utils.train_utils.",\n'
    )
    if anchor not in content:
        raise RuntimeError(
            f"SafeUnpickler allowlist anchor '{anchor.strip()}' not found in "
            f"{file_to_patch}."
        )

    content = content.replace(anchor, insertion, 1)
    _write_and_verify(file_to_patch, content, sentinel)
    logger.info("Patched SafeUnpickler allowlist in %s.", file_to_patch)


def _override_sglang_imbalance_check_env() -> None:
    """Force-disable sglang's per-GPU memory imbalance check.

    Pop the legacy names so the shim has nothing to copy, then set
    ``ENABLE=false`` directly. Inherited env reaches the subprocesses
    cleaned, so the shim no longer overwrites our ENABLE on re-import.
    """
    for legacy in (
        "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK",
        "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK",
    ):
        os.environ.pop(legacy, None)
    os.environ["SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK"] = "false"


def _get_megatron_file(subpackage: str, relative_path: str) -> str | None:
    """Locate a file inside ``megatron.<subpackage>`` (e.g. ``core``, ``training``).

    Returns ``None`` if megatron isn't importable so callers can treat that
    as "nothing to patch". Raises if the package is present but the
    expected file is missing (signals a megatron version mismatch).
    """
    full_pkg = f"megatron.{subpackage}"
    try:
        spec = find_spec(full_pkg)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None

    base_dir = next(iter(spec.submodule_search_locations))
    file_path = os.path.join(base_dir, *relative_path.split("/"))
    if not os.path.exists(file_path):
        raise RuntimeError(
            f"Expected megatron file '{full_pkg}/{relative_path}' not found at "
            f"'{file_path}'. The megatron version may have moved this file; "
            "compat patch cannot be applied."
        )
    return file_path


def _patch_megatron_hook_mode_in(file_path: str) -> None:
    """Comment out ``torch_memory_saver.hook_mode = "torch"`` in a megatron file.

    Megatron sets ``tms.hook_mode = "torch"`` at module import time on the
    global ``torch_memory_saver`` singleton. That mutation breaks sglang's
    pauseable CUDA graph path, which asserts ``_hook_mode == "preload"``
    inside ``TorchMemorySaver.cuda_graph(...)``. Commenting the line out
    leaves the singleton at its default ``"preload"`` mode that sglang
    expects.
    """
    with open(file_path, "r") as f:
        content = f.read()

    sentinel = '# torch_memory_saver.hook_mode = "torch"'
    if sentinel in content:
        return

    anchor = '    torch_memory_saver.hook_mode = "torch"\n'
    if anchor not in content:
        raise RuntimeError(
            f"Megatron hook_mode anchor '{anchor.strip()}' not found in "
            f"{file_path}; the megatron version may have moved or removed it."
        )

    replacement = (
        '    # torch_memory_saver.hook_mode = "torch"  '
        "# patched by nemo_rl: conflicts with sglang pauseable CUDA Graph\n"
    )
    content = content.replace(anchor, replacement, 1)
    _write_and_verify(file_path, content, sentinel)
    logger.info("Patched megatron tms.hook_mode mutation in %s.", file_path)


def _patch_megatron_dynamic_context_hook_mode() -> None:
    file_path = _get_megatron_file("core", "inference/contexts/dynamic_context.py")
    if file_path is None:
        return
    _patch_megatron_hook_mode_in(file_path)


def _patch_megatron_training_hook_mode() -> None:
    file_path = _get_megatron_file("training", "training.py")
    if file_path is None:
        return
    _patch_megatron_hook_mode_in(file_path)


def _patch_sglang_custom_all_reduce_v2_tms_cudagraph() -> None:
    """Backport sglang#27948 for colocated TMS CUDA graph capture.

    With ``SGLANG_MEMORY_SAVER_CUDA_GRAPH=true``, custom all-reduce v2 must
    avoid registering captured IPC addresses. TMS will otherwise replace those
    addresses during capture and custom_all_reduce.cuh can fail at replay time.
    """
    _patch_sglang_file_replacements(
        "jit_kernel/all_reduce.py",
        (
            (
                "        def set_cuda_graph_register_inputs(self, register_inputs: bool) -> None: ...\n",
                "        def set_cuda_graph_capture(self, is_capturing: bool) -> None: ...\n",
                "        def set_cuda_graph_capture(self, is_capturing: bool) -> None: ...\n"
                "        def set_cuda_graph_register_inputs(self, register_inputs: bool) -> None: ...\n",
            ),
        ),
        "custom all-reduce type stub graph-input registration toggle",
    )
    _patch_sglang_file_replacements(
        "jit_kernel/csrc/distributed/custom_all_reduce_base.cuh",
        (
            (
                '      .def("set_cuda_graph_register_inputs", &Class::set_cuda_graph_register_inputs)\n',
                '      .def("set_cuda_graph_capture", &Class::set_cuda_graph_capture)\n',
                '      .def("set_cuda_graph_capture", &Class::set_cuda_graph_capture)\n'
                '      .def("set_cuda_graph_register_inputs", &Class::set_cuda_graph_register_inputs)\n',
            ),
        ),
        "custom all-reduce C++ binding graph-input registration toggle",
    )
    _patch_sglang_file_replacements(
        "jit_kernel/include/sgl_kernel/distributed/custom_all_reduce.cuh",
        (
            (
                "  void set_cuda_graph_register_inputs(bool enabled) {\n",
                "  void set_cuda_graph_capture(bool enabled) {\n"
                "    m_is_graph_capturing = enabled;\n"
                "  }\n\n",
                "  void set_cuda_graph_capture(bool enabled) {\n"
                "    m_is_graph_capturing = enabled;\n"
                "  }\n\n"
                "  void set_cuda_graph_register_inputs(bool enabled) {\n"
                "    m_register_graph_inputs = enabled;\n"
                "  }\n\n",
            ),
            (
                "  bool m_register_graph_inputs = true;\n",
                "  bool m_is_graph_capturing = false;\n"
                "  int64_t m_cum_registered_count = 0;\n",
                "  bool m_is_graph_capturing = false;\n"
                "  bool m_register_graph_inputs = true;\n"
                "  int64_t m_cum_registered_count = 0;\n",
            ),
        ),
        "custom all-reduce graph-input registration flag",
    )
    _patch_sglang_file_replacements(
        "jit_kernel/csrc/distributed/custom_all_reduce_pull.cuh",
        (
            (
                "    if (check_capturing() && m_register_graph_inputs) {\n",
                "    if (check_capturing()) {\n",
                "    if (check_capturing() && m_register_graph_inputs) {\n",
            ),
        ),
        "custom all-reduce pull graph-input registration gate",
    )
    _patch_sglang_file_replacements(
        "srt/distributed/device_communicators/custom_all_reduce_v2.py",
        (
            (
                "from sglang.srt.environ import envs\n",
                "from sglang.srt.utils import is_sm100_supported, log_info_on_rank0\n",
                "from sglang.srt.environ import envs\n"
                "from sglang.srt.utils import is_sm100_supported, log_info_on_rank0\n",
            ),
            (
                "        self.tms_cudagraph = envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.get()\n",
                "        self.override_shot(None)  # set default config based on world size\n"
                "        self.override_algo: Optional[AllReduceAlgo] = None\n"
                "        self.obj = get_custom_all_reduce_cls()(\n",
                "        self.override_shot(None)  # set default config based on world size\n"
                "        self.override_algo: Optional[AllReduceAlgo] = None\n"
                "        self.tms_cudagraph = envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.get()\n"
                "        self.obj = get_custom_all_reduce_cls()(\n",
            ),
            (
                '            log_info_on_rank0(logger, "Registering 0 cuda graph addresses because tms is used")\n',
                "        try:\n"
                "            self.obj.set_cuda_graph_capture(True)\n"
                "            yield\n"
                "        finally:\n"
                "            self.obj.set_cuda_graph_capture(False)\n"
                "        # cannot call when graph is capturing\n",
                "        try:\n"
                "            self.obj.set_cuda_graph_register_inputs(not self.tms_cudagraph)\n"
                "            self.obj.set_cuda_graph_capture(True)\n"
                "            yield\n"
                "        finally:\n"
                "            self.obj.set_cuda_graph_capture(False)\n"
                "            self.obj.set_cuda_graph_register_inputs(True)\n"
                "        if self.tms_cudagraph:\n"
                '            log_info_on_rank0(logger, "Registering 0 cuda graph addresses because tms is used")\n'
                "            return\n"
                "        # cannot call when graph is capturing\n",
            ),
        ),
        "custom all-reduce v2 TMS CUDA graph capture path",
    )


def _apply_sglang_compat_patches() -> None:
    _neutralize_cutlass_dsl_experimental_stub()
    _patch_sglang_safe_unpickler()
    _patch_sglang_custom_all_reduce_v2_tms_cudagraph()
    _override_sglang_imbalance_check_env()
    _patch_megatron_dynamic_context_hook_mode()
    _patch_megatron_training_hook_mode()
