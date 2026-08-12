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
"""Helpers shared across the projection-prep CLIs in ``tools.x_token``.

These are not on the training import path — they're only invoked by the
standalone CLI tools that build projection matrices for cross-tokenizer
distillation runs.
"""

import re

import torch


def sinkhorn_one_dim(A, n_iters=1):
    """One-sided Sinkhorn normalization: scale rows to sum to 1."""
    for _ in range(n_iters):
        row_sums = A.sum(dim=1, keepdim=True)
        safe_row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
        A = A / safe_row_sums
    return A


def clean_model_name_for_filename(name: str) -> str:
    """Strip parameter counts and common suffixes from a model name for filenames."""
    cleaned_name = re.sub(r"-?[0-9\.]+[bBmB]", "", name, flags=re.IGNORECASE)
    cleaned_name = (
        cleaned_name.replace("-Base", "").replace("-it", "").replace("-Instruct", "")
    )
    cleaned_name = cleaned_name.strip("-_")
    if "mini" in name:
        cleaned_name += "_mini"
    return cleaned_name


def project_token_likelihoods(
    input_likelihoods,
    projection_map_indices,
    projection_map_values,
    target_vocab_size,
    device,
):
    """Project token likelihoods from a source to a target vocabulary using a sparse map."""
    batch_size, seq_len, source_vocab_size = input_likelihoods.shape
    if source_vocab_size != projection_map_indices.shape[0]:
        raise ValueError(
            f"Source vocab size of input ({source_vocab_size}) mismatches "
            f"projection map size ({projection_map_indices.shape[0]})"
        )

    top_k = projection_map_indices.shape[1]
    input_likelihoods = input_likelihoods.to(device)
    projection_map_indices = projection_map_indices.to(device)
    projection_map_values = projection_map_values.to(device)

    crow_indices = torch.arange(
        0, (source_vocab_size + 1) * top_k, top_k, device=device, dtype=torch.long
    )
    col_indices = projection_map_indices.flatten()
    values = projection_map_values.flatten()

    sparse_projection_matrix = torch.sparse_csr_tensor(
        crow_indices,
        col_indices,
        values,
        size=(source_vocab_size, target_vocab_size),
        device=device,
    )

    reshaped_input = input_likelihoods.reshape(batch_size * seq_len, source_vocab_size)
    projected_likelihoods_reshaped = torch.matmul(
        reshaped_input, sparse_projection_matrix
    )
    return projected_likelihoods_reshaped.reshape(
        batch_size, seq_len, target_vocab_size
    )
