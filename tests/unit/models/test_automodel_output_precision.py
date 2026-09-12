"""Check the mixed-precision policy passed to native AutoModel mesh setup."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.models.automodel import setup


@pytest.mark.parametrize("output_dtype", [None, "bfloat16", "float32"])
def test_distributed_setup_preserves_compute_and_reduction_precision(monkeypatch, output_dtype):
    """Changing output precision must preserve BF16 compute and FP32 reductions."""
    monkeypatch.setattr(torch.distributed, "init_process_group", lambda **kwargs: None)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 8)
    mesh = MagicMock()
    mesh.__getitem__.return_value.size.return_value = 1
    monkeypatch.setattr(setup.MeshContext, "build", lambda *args, **kwargs: SimpleNamespace(device_mesh=mesh, moe_mesh=mesh))
    dtensor = {"sequence_parallel": False, "activation_checkpointing": False, "expert_parallel_size": 8}
    if output_dtype is not None:
        dtensor["fsdp_output_dtype"] = output_dtype
    result = setup.setup_distributed(
        {"dtensor_cfg": dtensor},
        SimpleNamespace(dtype=torch.bfloat16, cpu_offload=False),
    )
    policy = result.fsdp2_config.mp_policy
    assert policy.param_dtype == torch.bfloat16
    assert policy.reduce_dtype == torch.float32
    assert policy.output_dtype == (torch.bfloat16 if output_dtype == "bfloat16" else torch.float32)
