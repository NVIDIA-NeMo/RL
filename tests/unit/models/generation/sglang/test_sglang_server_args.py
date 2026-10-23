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

from types import SimpleNamespace

import pytest

from nemo_rl.models.generation.sglang import sglang_worker
from nemo_rl.models.generation.sglang.sglang_worker import SGLangGenerationWorker


def test_quantized_runner_backends_are_forwarded(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    worker_cls = SGLangGenerationWorker.__ray_metadata__.modified_class
    worker = worker_cls.__new__(worker_cls)
    worker.gpus_per_node = 4
    worker.rank = 0
    worker.base_gpu_id = 0
    worker.num_gpus_per_engine = 4
    worker.sglang_cfg = {
        "sglang_cfg": {
            "model_path": "/model",
            "random_seed": 42,
            "tp_size": 4,
            "dp_size": 1,
            "pp_size": 1,
            "ep_size": 1,
            "skip_server_warmup": True,
            "quantization": {"scheme": "nvfp4"},
            "moe_runner_backend": "flashinfer_cutedsl",
            "fp4_gemm_runner_backend": "flashinfer_cutedsl",
            "sglang_server_config": {
                "num_gpus_per_engine": 4,
                "needs_offload": False,
                "cpu_weight_backup": False,
            },
        }
    }

    server_args = worker._compute_server_args("host:1234", 1235, "host", 1236)

    assert server_args["moe_runner_backend"] == "flashinfer_cutedsl"
    assert server_args["fp4_gemm_runner_backend"] == "flashinfer_cutedsl"


def test_make_request_rejects_semantic_failure(monkeypatch):
    worker_cls = SGLangGenerationWorker.__ray_metadata__.modified_class
    worker = worker_cls.__new__(worker_cls)
    worker.node_rank = 0
    worker.server_base_url = "http://sglang"

    response = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"success": False, "message": "partial weight update"},
    )
    monkeypatch.setattr(
        sglang_worker.requests, "post", lambda *args, **kwargs: response
    )

    with pytest.raises(RuntimeError, match="partial weight update"):
        worker._make_request("update_weights_from_distributed")
