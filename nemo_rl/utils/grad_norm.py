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

import math
import sys

import torch

_INF_GRAD_NORM_WARNING = """
WARNING: Infinite gradient norm detected.

This commonly occurs when a model checkpoint contains near-zero token embeddings —
tokens whose embeddings were driven toward zero by weight decay during pretraining.
When fine-tuning on data that includes these tokens the near-zero embeddings produce
large loss signals and gradient spikes that can destabilize training.

To diagnose, inspect the checkpoint embeddings:
    uv run tools/model_diagnostics/3.check_and_reinit_hf_model_embeddings_untrained.py --input <checkpoint_dir> --stats-only

If near-zero embeddings are found, reinitialize them before fine-tuning:
    uv run tools/model_diagnostics/3.check_and_reinit_hf_model_embeddings_untrained.py --input <checkpoint_dir> --output <output_dir>
""".strip()


def warn_if_inf_grad_norm(grad_norm) -> None:
    """Print a diagnostic warning on rank 0 if *grad_norm* is infinite.

    Accepts a float, int, or scalar torch.Tensor.  Silently ignores None
    (e.g. when there are no trainable parameters on a rank).
    """
    if grad_norm is None:
        return
    if isinstance(grad_norm, torch.Tensor):
        is_inf = torch.isinf(grad_norm).any().item()
    else:
        is_inf = math.isinf(float(grad_norm))
    if not is_inf:
        return
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(_INF_GRAD_NORM_WARNING, file=sys.stderr, flush=True)
