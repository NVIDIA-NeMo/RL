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

"""Ray actors used by the real SGLang weight-update tests."""

import gc
import os

import ray
import torch
import torch.distributed as dist


def _hf_params_generator(model, target_dtype):
    """Yield regular HF state_dict tensors in the format expected by weight streaming."""
    for name, tensor in model.state_dict().items():
        yield name, tensor.to(target_dtype, non_blocking=True).contiguous()


@ray.remote(num_cpus=0.1)
class MockFSDPWorker:
    """Simulates one FSDP rank for weight streaming."""

    def init(self, rank, world_size, master_addr, master_port, model_path, gpu_index):
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(gpu_index)
        os.environ["WORLD_SIZE"] = str(world_size)

        self.rank = rank
        self.gpu_index = gpu_index
        self.dtype = torch.bfloat16

        from nemo_rl.models.generation.sglang.utils.train_utils import (
            monkey_patch_torch_reductions,
        )

        monkey_patch_torch_reductions()
        torch.cuda.set_device(gpu_index)
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

        from transformers import AutoModelForCausalLM

        device = torch.device(f"cuda:{gpu_index}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(device)

        from nemo_rl.utils.nvml import get_device_uuid

        self.device_uuid = get_device_uuid(gpu_index)

    def get_device_uuid(self):
        return self.device_uuid

    def stream_weights(self, rollout_engines, num_gpus_per_engine):
        from nemo_rl.models.policy.utils import stream_weights_via_http_impl

        if not hasattr(self, "_ipc_worker_state"):
            self._ipc_worker_state = {}

        rollout_engine_urls = ray.get(
            [e.get_base_url.remote() for e in rollout_engines]
        )

        stream_weights_via_http_impl(
            params_generator=_hf_params_generator(self.model, self.dtype),
            rollout_engine_urls=rollout_engine_urls,
            num_gpus_per_engine=num_gpus_per_engine,
            rank=self.rank,
            world_size=dist.get_world_size(),
            worker_name=f"MockFSDPWorker-{self.rank}",
            buffer_size_bytes=512 * 1024 * 1024,
            worker_state=self._ipc_worker_state,
        )

    def shutdown(self):
        if dist.is_initialized():
            dist.destroy_process_group()
        if hasattr(self, "model"):
            del self.model
        gc.collect()
        torch.cuda.empty_cache()
