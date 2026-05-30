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

from nemo_rl.modelopt.models.generation.vllm_quant_worker import (
    _configure_quant_vllm_kwargs,
)


def test_quant_vllm_worker_defaults_moe_backend_to_triton():
    llm_kwargs = {}

    _configure_quant_vllm_kwargs(llm_kwargs)

    assert (
        llm_kwargs["worker_cls"]
        == "nemo_rl.modelopt.models.generation.vllm_quant_patch.FakeQuantWorker"
    )
    assert (
        llm_kwargs["worker_extension_cls"]
        == "nemo_rl.modelopt.models.generation.vllm_quant_backend.VllmQuantInternalWorkerExtension"
    )
    assert llm_kwargs["moe_backend"] == "triton"


def test_quant_vllm_worker_keeps_explicit_moe_backend():
    llm_kwargs = {"moe_backend": "custom_backend"}

    _configure_quant_vllm_kwargs(llm_kwargs)

    assert llm_kwargs["moe_backend"] == "custom_backend"
