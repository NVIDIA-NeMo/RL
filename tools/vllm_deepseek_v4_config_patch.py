# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Patched DeepseekV4Config for vLLM PR #40760 + transformers 5.5+.

transformers >=5.5 dataclass-ified PretrainedConfig: non-declared kwargs
are no longer auto-bound to self before __post_init__ runs, so
standardize_rope_params (yarn branch in modeling_rope_utils.py:757)
crashes with AttributeError when it dereferences self.max_position_embeddings.

This file is dropped in over
vllm/transformers_utils/configs/deepseek_v4.py inside the
VllmGenerationWorker venv. Track upstream:
https://github.com/vllm-project/vllm/pull/40760
"""
from transformers import PretrainedConfig


class DeepseekV4Config(PretrainedConfig):
    model_type = "deepseek_v4"

    def __init__(self, **kwargs):
        # transformers >=5.5 dataclass-ified PretrainedConfig: only declared
        # fields are auto-bound from kwargs before __post_init__ runs.
        # Pre-bind every kwarg as an attribute so:
        #   (a) standardize_rope_params (yarn branch) finds max_position_embeddings
        #   (b) vllm/model_executor/models/deepseek_v4.py can use config.rope_theta,
        #       config.compress_rope_theta, config.compress_ratios, etc. without
        #       AttributeError.
        # super().__init__ may set them again; the assignment is idempotent.
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError:
                pass
        super().__init__(**kwargs)
