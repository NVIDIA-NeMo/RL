# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import importlib
import os
from importlib.util import find_spec
from typing import Callable, Optional

# Original TE cuBLAS workspace sizer, saved by apply_te_gemm_cublas_pinned_patch().
_TE_CUBLAS_WS_SIZE_FN_ORIG: Optional[Callable[[], int]] = None

# Minimum workspace that satisfies TE's NVFP4 alpha-scratch guard in cublaslt_gemm.cu.
_TE_CUBLAS_WS_PINNED_BYTES: int = 4


def _get_transformer_engine_file(relative_path: str) -> str:
    """Return absolute path to a Transformer Engine file or raise if it cannot be found.

    The relative_path should be a POSIX-style path under the transformer_engine
    package root, e.g. "pytorch/triton/permutation.py".
    """
    spec = find_spec("transformer_engine")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            "Transformer Engine package not found while attempting to patch "
            f"'{relative_path}'. Ensure `transformer-engine` is installed and "
            "available in this environment."
        )

    base_dir = next(iter(spec.submodule_search_locations))
    file_path = os.path.join(base_dir, *relative_path.split("/"))

    if not os.path.exists(file_path):
        raise RuntimeError(
            "Failed to locate expected Transformer Engine file to patch. "
            f"Looked for '{relative_path}' at '{file_path}'. "
            "This likely indicates an unexpected Transformer Engine installation "
            "layout or version mismatch."
        )

    return file_path


def apply_transformer_engine_patch():
    """Apply patch from https://github.com/NVIDIA/TransformerEngine/pull/2286/files.

    This locates the target file via importlib metadata instead of importing
    `transformer_engine`, to avoid side effects during initialization. If the
    permutation module has already been imported, it will be reloaded so that
    the patched source takes effect.
    """
    try:
        perm_file = _get_transformer_engine_file("pytorch/triton/permutation.py")

        with open(perm_file, "r") as f:
            content = f.read()

        if "get_int_dtype = triton.constexpr_function(get_int_dtype)" not in content:
            print(f"Applying Triton fix to {perm_file}...")

            # 1. Replace the usage
            old_usage = "idtype = core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)"
            new_usage = "idtype = get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)"

            # 2. Insert the definition before the first @triton.jit
            jit_anchor = "@triton.jit"

            new_definition = (
                "\n\n"
                "get_int_dtype = core.get_int_dtype\n"
                "get_int_dtype = triton.constexpr_function(get_int_dtype)\n"
            )

            new_content = None
            if old_usage in content:
                temp_content = content.replace(old_usage, new_usage)

                if jit_anchor in temp_content:
                    new_content = temp_content.replace(
                        jit_anchor, new_definition + jit_anchor, 1
                    )

            if new_content:
                try:
                    with open(perm_file, "w") as f:
                        f.write(new_content)
                    print("Successfully patched transformer_engine permutation.py.")
                except OSError as e:
                    print(
                        f"Could not write patch to transformer_engine (permission denied?): {e}"
                    )

        # If the permutation module is already imported in this process,
        # reload it so that the patched source takes effect for subsequent use.
        import importlib
        import sys

        perm_module_name = "transformer_engine.pytorch.triton.permutation"
        if perm_module_name in sys.modules:
            importlib.reload(sys.modules[perm_module_name])

    except Exception as e:
        print(f"Error checking/patching transformer_engine: {e}")


def apply_te_gemm_cublas_pinned_patch(
    target_bytes: int = _TE_CUBLAS_WS_PINNED_BYTES,
) -> None:
    """Shrink TE's cuBLAS workspace so cuBLASLt picks workspace-free algorithms.

    Mirrors megatron.core.transformer.custom_layers.batch_invariant_kernels.
    ``_shrink_te_cublas_workspace_for_invariance``. Intended for zero-KL /
    ``zero_train_gen_mismatch`` only — call from ``_apply_zero_train_gen_mismatch``
    in setup.py, not from generic batch-invariant mode.
    """
    global _TE_CUBLAS_WS_SIZE_FN_ORIG
    if _TE_CUBLAS_WS_SIZE_FN_ORIG is not None:
        return
    try:
        te_gemm_mod = importlib.import_module(
            "transformer_engine.pytorch.cpp_extensions.gemm"
        )
    except ImportError:
        print(
            "te_gemm_cublas_pinned: transformer_engine.pytorch.cpp_extensions.gemm "
            "is not importable; skipping workspace shrink."
        )
        return
    if not hasattr(te_gemm_mod, "get_cublas_workspace_size_bytes"):
        print(
            "te_gemm_cublas_pinned: TE gemm module has no get_cublas_workspace_size_bytes "
            "(TE version mismatch?); skipping workspace shrink."
        )
        return

    _TE_CUBLAS_WS_SIZE_FN_ORIG = te_gemm_mod.get_cublas_workspace_size_bytes
    te_gemm_mod.get_cublas_workspace_size_bytes = lambda: int(target_bytes)
    ws_fn = getattr(te_gemm_mod, "get_cublas_workspace", None)
    if ws_fn is not None and hasattr(ws_fn, "cache_clear"):
        try:
            ws_fn.cache_clear()
        except Exception:  # pylint: disable=broad-except
            pass
    print(
        f"[zero_train_gen_mismatch] shrunk TE cuBLAS workspace to {target_bytes} bytes "
        "(te_gemm_cublas_pinned via patches.py). "
        "Set CUBLASLT_LOG_LEVEL=5 to verify cuBLASLt picks a stable algo across batch sizes."
    )


def restore_te_gemm_cublas_pinned_patch() -> None:
    """Restore TE's original cuBLAS workspace sizer (for tests)."""
    global _TE_CUBLAS_WS_SIZE_FN_ORIG
    if _TE_CUBLAS_WS_SIZE_FN_ORIG is None:
        return
    try:
        te_gemm_mod = importlib.import_module(
            "transformer_engine.pytorch.cpp_extensions.gemm"
        )
    except ImportError:
        _TE_CUBLAS_WS_SIZE_FN_ORIG = None
        return
    if hasattr(te_gemm_mod, "get_cublas_workspace_size_bytes"):
        te_gemm_mod.get_cublas_workspace_size_bytes = _TE_CUBLAS_WS_SIZE_FN_ORIG
    ws_fn = getattr(te_gemm_mod, "get_cublas_workspace", None)
    if ws_fn is not None and hasattr(ws_fn, "cache_clear"):
        try:
            ws_fn.cache_clear()
        except Exception:  # pylint: disable=broad-except
            pass
    _TE_CUBLAS_WS_SIZE_FN_ORIG = None
