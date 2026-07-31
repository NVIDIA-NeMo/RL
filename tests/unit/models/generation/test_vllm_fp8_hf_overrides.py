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

"""Regression tests for merging fp8 kwargs with user-supplied hf_overrides.

The fp8 path returns a nested ``hf_overrides`` (holding ``quantization_config``).
A naive ``vllm_kwargs.update(fp8_kwargs)`` shallow-merges and clobbers any
user-supplied ``hf_overrides``. This exact bug was introduced, fixed (#1413),
silently reverted (#2188), and re-fixed (#2904). These tests pin the merge
behavior so it cannot regress a third time.
"""

from nemo_rl.models.generation.vllm.vllm_worker import _merge_fp8_kwargs


def test_fp8_and_user_hf_overrides_coexist():
    """Both fp8's quantization_config and a user override survive the merge."""
    vllm_kwargs = {"hf_overrides": {"max_position_embeddings": 8192}}
    fp8_kwargs = {
        "quantization": "fp8",
        "kv_cache_dtype": "auto",
        "hf_overrides": {"quantization_config": {"weight_block_size": [128, 128]}},
    }

    _merge_fp8_kwargs(vllm_kwargs, fp8_kwargs)

    # fp8 quantization settings applied
    assert vllm_kwargs["quantization"] == "fp8"
    assert vllm_kwargs["kv_cache_dtype"] == "auto"
    # fp8's quantization_config survives ...
    assert vllm_kwargs["hf_overrides"]["quantization_config"] == {
        "weight_block_size": [128, 128]
    }
    # ... and so does the user-supplied override
    assert vllm_kwargs["hf_overrides"]["max_position_embeddings"] == 8192


def test_user_hf_overrides_take_precedence():
    """On key collision, the user-supplied hf_overrides value wins."""
    vllm_kwargs = {"hf_overrides": {"quantization_config": {"user": "wins"}}}
    fp8_kwargs = {
        "hf_overrides": {"quantization_config": {"fp8": "base"}},
    }

    _merge_fp8_kwargs(vllm_kwargs, fp8_kwargs)

    assert vllm_kwargs["hf_overrides"]["quantization_config"] == {"user": "wins"}


def test_no_existing_hf_overrides():
    """fp8's hf_overrides apply cleanly when the user supplied none."""
    vllm_kwargs = {}
    fp8_kwargs = {
        "quantization": "fp8",
        "hf_overrides": {"quantization_config": {"weight_block_size": [128, 128]}},
    }

    _merge_fp8_kwargs(vllm_kwargs, fp8_kwargs)

    assert vllm_kwargs["hf_overrides"] == {
        "quantization_config": {"weight_block_size": [128, 128]}
    }


def test_none_hf_overrides_treated_as_empty():
    """A ``None`` hf_overrides (e.g. from config defaults) is handled as empty."""
    vllm_kwargs = {"hf_overrides": None}
    fp8_kwargs = {
        "hf_overrides": {"quantization_config": {"weight_block_size": [128, 128]}},
    }

    _merge_fp8_kwargs(vllm_kwargs, fp8_kwargs)

    assert vllm_kwargs["hf_overrides"] == {
        "quantization_config": {"weight_block_size": [128, 128]}
    }


def test_source_fp8_kwargs_not_mutated():
    """The merge must not mutate the caller's fp8_kwargs dict."""
    vllm_kwargs = {}
    fp8_kwargs = {
        "quantization": "fp8",
        "hf_overrides": {"quantization_config": {"weight_block_size": [128, 128]}},
    }

    _merge_fp8_kwargs(vllm_kwargs, fp8_kwargs)

    assert "hf_overrides" in fp8_kwargs
