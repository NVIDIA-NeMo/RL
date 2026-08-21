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

"""Runtime patches for Megatron colocated zero train/gen KL alignment."""

from nemo_rl.models.generation.megatron.zero_train_gen_kl_patches.core_patches import (
    apply_log_softmax_determinism_patch,
    apply_te_bik_attention_assert_skip_patch,
    apply_te_gemm_cublas_pinned_patch,
    restore_log_softmax_determinism_patch,
    restore_te_bik_attention_assert_skip_patch,
    restore_te_gemm_cublas_pinned_patch,
)
from nemo_rl.models.generation.megatron.zero_train_gen_kl_patches.mamba_zero_kl_patches import (
    apply_mamba_alignment_patch,
    policy_uses_mamba_layers,
    restore_mamba_alignment_patch,
)
from nemo_rl.models.generation.megatron.zero_train_gen_kl_patches.moe_zero_kl_patches import (
    apply_moe_determinism_patches,
    restore_moe_determinism_patches,
)

__all__ = [
    "apply_log_softmax_determinism_patch",
    "apply_mamba_alignment_patch",
    "apply_moe_determinism_patches",
    "apply_te_bik_attention_assert_skip_patch",
    "apply_te_gemm_cublas_pinned_patch",
    "policy_uses_mamba_layers",
    "restore_log_softmax_determinism_patch",
    "restore_mamba_alignment_patch",
    "restore_moe_determinism_patches",
    "restore_te_bik_attention_assert_skip_patch",
    "restore_te_gemm_cublas_pinned_patch",
]
