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

"""Native response projection against real FSDP2 storage and a TE-style bias sentinel."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor


def _native_projection():
    root = Path(__file__).parents[4]
    source = root / "nemo_rl/models/automodel/speechlm.py"
    tree = ast.parse(source.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SpeechLMPolicy"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "next_token_logprobs"
    )
    spec = importlib.util.spec_from_file_location(
        "native_response_projection",
        root / "nemo_rl/models/automodel/speechlm_logprobs.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    namespace = {
        "torch": torch,
        "DTensor": DTensor,
        "need_top_k_or_top_p_filtering": lambda _: False,
        "response_logprobs": module.response_logprobs,
    }
    # Execute the exact production method without importing unrelated actor/framework dependencies.
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(
                    body=[
                        ast.ImportFrom(
                            module="__future__",
                            names=[ast.alias(name="annotations")],
                            level=0,
                        ),
                        method,
                    ],
                    type_ignores=[],
                )
            ),
            str(source),
            "exec",
        ),
        namespace,
    )
    return namespace["next_token_logprobs"]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Real CUDA FSDP2 storage required"
)
@pytest.mark.parametrize("bias_enabled", [False, True])
def test_frozen_fsdp_head_projection_after_offload_matches_dense_and_gradients(
    tmp_path, bias_enabled
):
    torch.cuda.set_device(0)
    dist.init_process_group(
        "nccl", init_method="file://" + str(tmp_path / "pg"), rank=0, world_size=1
    )
    try:
        torch.manual_seed(42)
        mesh = init_device_mesh("cuda", (1,))
        head = torch.nn.Linear(
            4, 6, bias=bias_enabled, device="cuda", dtype=torch.bfloat16
        )
        head.requires_grad_(False)
        weight = head.weight.detach().clone()
        bias = head.bias.detach().clone() if bias_enabled else None
        if not bias_enabled:
            # TE bias=False stores a non-parameter/non-buffer empty bias attribute.
            del head.bias
            head.bias = torch.empty(0, device="cpu", dtype=torch.bfloat16)
        fully_shard(head, mesh=mesh, reshard_after_forward=False)
        head.to("cpu")
        head.to("cuda")
        assert isinstance(head.weight, DTensor)
        assert head.weight.device.type == "cuda"
        if not bias_enabled:
            assert head.bias.device.type == "cpu" and head.bias.numel() == 0
        hidden = torch.randn(
            1, 3, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        ids = torch.tensor([[0, 1, 2, 3]], device="cuda")
        targets = ids[:, 1:]
        mask = torch.ones_like(ids)
        native = SimpleNamespace(
            native_model={
                "max_expanded_sequence_length": 16,
                "projection_chunk_size": 2,
            },
            prepare_inputs=lambda *_args, **_kwargs: {
                "input_embeds": hidden,
                "attention_mask": None,
                "llm_kwargs": {},
                "target_ids": targets,
            },
            llm=lambda **_kwargs: SimpleNamespace(hidden_states=[hidden]),
        )
        native.llm = type(
            "Backbone",
            (),
            {
                "__call__": lambda self, **_kwargs: SimpleNamespace(
                    hidden_states=[hidden]
                ),
                "get_output_embeddings": lambda self: head,
            },
        )()
        batch = {
            "input_ids": ids,
            "audio_samples": torch.zeros(1, 4, device="cuda"),
            "audio_lengths": torch.tensor([[4]], device="cuda"),
        }
        actual = _native_projection()(native, batch, mask)
        expected = (
            torch.log_softmax(
                torch.nn.functional.linear(hidden, weight, bias).float(), -1
            )
            .gather(-1, targets[..., None])
            .squeeze(-1)
        )
        torch.testing.assert_close(actual, expected)
        expected_gradient = torch.autograd.grad(
            expected.sum(), hidden, retain_graph=True
        )[0]
        actual.sum().backward()
        torch.testing.assert_close(hidden.grad, expected_gradient)
        assert all(parameter.grad is None for parameter in head.parameters())
    finally:
        dist.destroy_process_group()
