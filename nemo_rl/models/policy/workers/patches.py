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

import os
from importlib.util import find_spec

import torch
import torch.nn.functional as F
from torch.distributed.tensor._ops._tensor_ops import propagate_single_input_strategy
from torch.distributed.tensor._ops.utils import register_op_strategy


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


def apply_torch_aten_alias_tensor_patch():
    """Register a sharding rule for `torch.ops.aten.alias.default`.

    Work around 'NotImplementedError: Operator aten.alias.default does not have a sharding strategy registered'
    in PyTorch 2.9. See https://github.com/pytorch/pytorch/pull/166867 for the upstream fix.
    We can remove this patch when we upgrade torch to include this fix.
    """
    assert torch.__version__.startswith("2.9.0"), (
        "This patch is needed for torch 2.9.0. Please retest if you upgrade torch to a newer version and remove this patch."
    )
    try:
        register_op_strategy(torch.ops.aten.alias.default)(
            propagate_single_input_strategy
        )
    except Exception as e:
        print(f"Error applying torch.ops.aten.alias.default patch: {e}")


def patched_lora_linear_forward(self, x):
    if self.dropout_position == "pre":
        x = F.dropout(x, p=self.dropout_p, training=self.training)

    lora_weight = torch.matmul(self.lora_B.weight, self.lora_A.weight) * self.scale

    if self.dropout_position == "post":
        lora_weight = F.dropout(lora_weight, p=self.dropout_p, training=self.training)

    temp_weight = self.weight + lora_weight

    res = F.linear(x, temp_weight, self.bias)

    return res


def apply_lora_linear_forward_patch():
    from nemo_automodel.components._peft.lora import LinearLoRA

    LinearLoRA.forward = patched_lora_linear_forward
