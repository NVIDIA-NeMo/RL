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

"""Two-GPU checks for ``setup_distributed`` with real process groups.

``automodel_cfg.distributed_timeout_minutes`` must reach the default group *and* the
mesh sub-groups without changing their backends: with CPU offload the default group
is ``cuda:nccl,cpu:gloo``, and the CPU-offloaded FSDP parameters are gathered through
the DP mesh on the CPU (``prepare_refit_info`` -> ``DTensor.full_tensor``).
"""

import datetime
import inspect
import json
import os
import random
import traceback
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    import nemo_automodel  # noqa: F401

    NEMO_AUTOMODEL_AVAILABLE = True
except ImportError:
    NEMO_AUTOMODEL_AVAILABLE = False

_WORLD_SIZE = 2
_TIMEOUT_MINUTES = 45


def _automodel_keeps_default_backend_under_mesh_timeout() -> bool:
    """Whether Automodel carries NVIDIA-NeMo/Automodel#4602.

    Before that fix a mesh timeout replaced the default group's composite backend with
    NCCL-only sub-groups (NVIDIA-NeMo/Automodel#4601), so CPU collectives on the mesh
    failed. Drop this guard once the Automodel pin includes the fix.
    """
    if not NEMO_AUTOMODEL_AVAILABLE:
        return False
    from nemo_automodel.components.distributed import mesh_utils

    return "dist.get_backend()" in inspect.getsource(mesh_utils._nccl_backend_override)


def _run_rank(rank: int, world_size: int, port: int, results_dir: str) -> None:
    """One rank: set up distributed state through ``setup_distributed`` and check it."""
    from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard

    from nemo_rl.models.automodel.config import RuntimeConfig
    from nemo_rl.models.automodel.setup import setup_distributed

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    torch.cuda.set_device(rank)
    outcome = {"error": None}
    try:
        config = {
            "automodel_cfg": {
                "cpu_offload": True,
                "tensor_parallel_size": 1,
                "context_parallel_size": 1,
                "expert_parallel_size": 1,
                "dp_replicate_size": 1,
                "sequence_parallel": False,
                "activation_checkpointing": False,
                "distributed_timeout_minutes": _TIMEOUT_MINUTES,
            }
        }
        runtime_config = RuntimeConfig(
            model_class=None,
            model_config=None,
            hf_config_overrides={},
            allow_flash_attn_args=False,
            attn_impl=None,
            dtype=torch.bfloat16,
            enable_seq_packing=False,
            max_grad_norm=1.0,
            cpu_offload=True,
            offload_optimizer_for_logprob=False,
            is_generation_colocated=None,
            sampling_params=None,
            is_reward_model=False,
        )
        context = setup_distributed(config, runtime_config)
        expected_timeout = datetime.timedelta(minutes=_TIMEOUT_MINUTES)

        # The default group keeps both halves of cuda:nccl,cpu:gloo, each with the timeout.
        for device_type in ("cpu", "cuda"):
            backend = dist.group.WORLD._get_backend(torch.device(device_type))
            assert backend.options._timeout == expected_timeout, device_type

        # Every mesh axis supports CPU and CUDA collectives and carries the timeout.
        for axis in context.device_mesh.mesh_dim_names:
            group = context.device_mesh[axis].get_group()
            for device_type in ("cpu", "cuda"):
                tensor = torch.ones(1, device=device_type)
                dist.all_reduce(tensor, group=group)
                assert tensor.item() == dist.get_world_size(group), (axis, device_type)
                backend = group._get_backend(torch.device(device_type))
                assert backend.options._timeout == expected_timeout, (axis, device_type)

        # The NVIDIA-NeMo/Automodel#4601 failure path: a CPU-offloaded FSDP parameter is
        # all-gathered on the CPU through the DP mesh, as prepare_refit_info() does.
        torch.manual_seed(17)
        model = torch.nn.Linear(4, 4, bias=False, device="cuda")
        expected_weight = model.weight.detach().cpu().clone()
        fully_shard(
            model,
            mesh=context.device_mesh["dp"],
            offload_policy=CPUOffloadPolicy(pin_memory=False),
        )
        model.to("cpu")
        full_weight = model.state_dict()["weight"].full_tensor()
        torch.testing.assert_close(full_weight.cpu(), expected_weight)
    except Exception:  # Reported by the test on the parent process.
        outcome["error"] = traceback.format_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    Path(results_dir, f"rank{rank}.json").write_text(json.dumps(outcome))


@pytest.mark.automodel
@pytest.mark.skipif(not NEMO_AUTOMODEL_AVAILABLE, reason="nemo_automodel not available")
@pytest.mark.skipif(torch.cuda.device_count() < _WORLD_SIZE, reason="needs two GPUs")
@pytest.mark.skipif(
    not _automodel_keeps_default_backend_under_mesh_timeout(),
    reason="Automodel pin predates NVIDIA-NeMo/Automodel#4602 (mesh timeout drops Gloo, #4601)",
)
@pytest.mark.timeout(300)
def test_setup_distributed_cpu_offload_with_explicit_timeout_on_two_gpus(tmp_path):
    """An explicit timeout with CPU offload must keep CPU collectives on every mesh group."""
    port = random.randint(20000, 30000)
    mp.spawn(
        _run_rank,
        args=(_WORLD_SIZE, port, str(tmp_path)),
        nprocs=_WORLD_SIZE,
        join=True,
    )
    for rank in range(_WORLD_SIZE):
        outcome = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert outcome["error"] is None, f"rank {rank}:\n{outcome['error']}"
