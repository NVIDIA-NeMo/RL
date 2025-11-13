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

from itertools import product
from unittest.mock import MagicMock

import pytest
import torch
from torch.distributed.tensor.parallel import ParallelStyle, parallelize_module
from transformers import AutoModelForCausalLM

from nemo_rl.models.dtensor.parallelize import (
    _parallelize_gemma3,
    _parallelize_llama,
    _parallelize_qwen,
    get_grad_norm,
)


@pytest.mark.hf_gated
@pytest.mark.parametrize(
    "model_name, parallelize_func, sequence_parallel",
    [
        (model_name, parallelize_func, sp)
        for (model_name, parallelize_func), sp in product(
            [
                ("google/gemma-3-1b-it", _parallelize_gemma3),
                ("google/gemma-3-4b-it", _parallelize_gemma3),
                # ("Qwen/Qwen2.5-1.5B", _parallelize_qwen), # TODO: qwen2 doesn't have q_norm and k_norm, which will cause this test to fail
                ("Qwen/Qwen3-0.6B", _parallelize_qwen),
                ("meta-llama/Llama-3.2-1B-Instruct", _parallelize_llama),
            ],
            [True, False],
        )
    ],
)
def test_parallelize_plan_keys(model_name, parallelize_func, sequence_parallel):
    """Tests that the keys in the parallelization plans are valid by mocking parallel styles."""
    model = AutoModelForCausalLM.from_pretrained(model_name)
    parallel_plan = parallelize_func(model, sequence_parallel=sequence_parallel)

    applied_keys = set()

    class MockParallelStyle(ParallelStyle):
        def __init__(self, key, collector):
            self.key = key
            self.collector = collector

        def _apply(self, module, device_mesh):
            self.collector.add(self.key)

    mock_plan = {key: MockParallelStyle(key, applied_keys) for key in parallel_plan}
    dummy_device_mesh = MagicMock()
    dummy_device_mesh.ndim = 1

    parallelize_module(model, dummy_device_mesh, mock_plan)

    assert set(parallel_plan.keys()) == applied_keys, (
        f"Missing keys: {set(parallel_plan.keys()) - applied_keys}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "grad_dtype, norm_dtype, norm_order",
    [
        (torch.float32, torch.float32, 1),
        (torch.float32, torch.float32, 2),
        (torch.float32, torch.float32, torch.inf),
        (torch.bfloat16, torch.float32, 1),
        (torch.bfloat16, torch.float32, 2),
        (torch.bfloat16, torch.float32, torch.inf),
    ],
)
def test_get_grad_norm_precision(monkeypatch, grad_dtype, norm_dtype, norm_order):
    """Checks numerical precision of get_grad_norm."""

    def noop_all_reduce(tensor, op=None, group=None):
        return None

    monkeypatch.setattr(torch.distributed, "all_reduce", noop_all_reduce, raising=False)

    n = 65536
    vals = torch.logspace(-2, 2, steps=n, device="cuda", dtype=grad_dtype)
    signs = (torch.rand(n, device="cuda") > 0.5).to(grad_dtype) * 2 - 1
    grads_full = vals * signs

    p1 = torch.zeros(n // 2, device="cuda", dtype=grad_dtype, requires_grad=True)
    p2 = torch.zeros(n - n // 2, device="cuda", dtype=grad_dtype, requires_grad=True)
    p1.grad = grads_full[: n // 2].clone()
    p2.grad = grads_full[n // 2 :].clone()

    expected = torch.linalg.vector_norm(
        grads_full.to(torch.float64), ord=norm_order
    ).item()
    norm = get_grad_norm(
        [p1, p2],
        dp_cp_group=None,
        tp_group=None,
        norm_type=norm_order,
        dtype=norm_dtype,
    )
    assert norm == pytest.approx(expected)
