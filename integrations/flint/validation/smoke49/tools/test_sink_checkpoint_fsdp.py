#!/usr/bin/env python3
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
"""Two-GPU regression for actual Flint attention and FSDP checkpoint recompute.

Run with the training interpreter's torch.distributed.run --standalone
--nproc-per-node=2. Candidate mode must match checkpoint-on/off outputs and
every parameter/input gradient exactly. Baseline mode must reproduce the
original no-argument FSDP IndexError on both ranks.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from nemo_automodel.components.distributed.parallelizer_utils import (
    fully_shard_by_dtype,
)
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.flint.config import FlintConfig
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor


def full_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def implementation(mode: str):
    if mode == "candidate":
        from nemo_automodel.components.models.flint.layers import FlintAttention

        return FlintAttention
    source = (
        Path(__file__).resolve().parents[1] / "evidence/runtime-source/flint_layers.py"
    )
    spec = importlib.util.spec_from_file_location("flint_sink_baseline_layers", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.FlintAttention


def run_case(attention_class, mesh, device, *, checkpoint: bool, sliding: bool):
    torch.manual_seed(8193)
    config = FlintConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        layer_types=["sliding_attention" if sliding else "full_attention"],
        no_rope_layer_indices=[] if sliding else [0],
        mlp_only_layers=[0],
        sliding_window=17,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        dtype="float32",
    )
    backend = BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch",
        rope="torch",
        rope_fusion=False,
        experts="torch",
        dispatcher="torch",
        gate_precision=None,
    )
    attention = attention_class(config, layer_idx=0, backend=backend).to(device)
    with torch.no_grad():
        attention._fp32_params.softmax_offset.copy_(
            torch.linspace(-0.4, 0.3, 4, device=device)
        )
    sink = attention._fp32_params
    model = checkpoint_wrapper(attention) if checkpoint else attention
    fully_shard_by_dtype(
        model,
        mesh=mesh,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=None,
            cast_forward_inputs=False,
        ),
        offload_policy=None,
        fp32_compute_module_names=("_fp32_params",),
    )
    assert isinstance(sink, FSDPModule), (
        "Regression must exercise independently sharded FP32 sinks"
    )
    inputs = torch.linspace(
        -0.9, 0.8, 2 * 37 * 64, device=device, dtype=torch.bfloat16
    ).reshape(2, 37, 64)
    inputs.requires_grad_()
    probe = torch.linspace(0.6, -0.7, inputs.numel(), device=device).reshape_as(inputs)
    output = model(inputs)
    (output.float() * probe).sum().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    gradients = {}
    for name, parameter in model.named_parameters():
        name = name.replace("_checkpoint_wrapped_module.", "")
        assert parameter.grad is not None, f"Missing parameter gradient: {name}"
        gradient = full_tensor(parameter.grad).detach()
        assert torch.isfinite(gradient).all(), f"Nonfinite parameter gradient: {name}"
        gradients[name] = gradient.cpu()
    sink_gradient = gradients["_fp32_params.softmax_offset"]
    assert sink_gradient.dtype == torch.float32
    assert sink_gradient.count_nonzero() == 4, (
        "Each learned sink must receive a nonzero gradient"
    )
    dist.barrier()
    return {
        "output": output.detach().cpu(),
        "input_gradient": inputs.grad.detach().cpu(),
        "gradients": gradients,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--implementation", choices=("candidate", "baseline"), default="candidate"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    assert int(os.environ["WORLD_SIZE"]) == 2, (
        "This regression requires exactly two ranks"
    )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    receipts = []
    try:
        mesh = init_device_mesh("cuda", (2,))
        attention_class = implementation(args.implementation)
        for sliding in (False, True):
            plain = run_case(
                attention_class, mesh, device, checkpoint=False, sliding=sliding
            )
            try:
                recomputed = run_case(
                    attention_class, mesh, device, checkpoint=True, sliding=sliding
                )
            except IndexError:
                if (
                    args.implementation != "baseline"
                    or "_root_pre_forward" not in traceback.format_exc()
                ):
                    raise
                receipts.append(
                    {
                        "sliding": sliding,
                        "status": "baseline_no_argument_fsdp_failure_reproduced",
                    }
                )
                # The FSDP state from a failed backward is intentionally discarded.
                break
            if args.implementation == "baseline":
                raise AssertionError(
                    "Original no-argument implementation unexpectedly passed recomputation"
                )
            assert plain["gradients"].keys() == recomputed["gradients"].keys()
            torch.testing.assert_close(
                plain["output"], recomputed["output"], rtol=0, atol=0
            )
            torch.testing.assert_close(
                plain["input_gradient"], recomputed["input_gradient"], rtol=0, atol=0
            )
            for name in plain["gradients"]:
                torch.testing.assert_close(
                    plain["gradients"][name],
                    recomputed["gradients"][name],
                    rtol=0,
                    atol=0,
                    msg=name,
                )
            receipts.append(
                {
                    "sliding": sliding,
                    "status": "pass",
                    "parameter_gradients_compared": len(plain["gradients"]),
                    "nonzero_fp32_sink_gradients": 4,
                    "output_and_gradient_atol": 0,
                    "output_and_gradient_rtol": 0,
                }
            )
        gathered = [None, None]
        dist.all_gather_object(gathered, {"rank": rank, "cases": receipts})
        if rank == 0:
            result = {
                "status": "pass",
                "implementation": args.implementation,
                "world_size": 2,
                "ranks": gathered,
            }
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
