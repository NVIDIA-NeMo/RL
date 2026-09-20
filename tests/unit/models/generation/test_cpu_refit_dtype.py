# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import ast
from pathlib import Path
from types import SimpleNamespace
import torch


def test_export_staging_preserves_forced_and_router_precision():
    path = Path(__file__).resolve().parents[4] / 'nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py'
    tree = ast.parse(path.read_text())
    names = {'_refit_staging_dtype', '_refit_tensor_dtype'}
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for n in body:
        n.returns = None
        for a in n.args.args: a.annotation = None
    class Adapter:
        pass
    Adapter.__module__ = 'nemo_automodel.components.models.deepseek_v41.adapter'
    model = SimpleNamespace(state_dict_adapter=Adapter(), modules=lambda: [])
    ns = {'torch': torch, 'LinearLoRA': type('LinearLoRA', (), {}),
          '_maybe_adapt_tensor_to_hf': lambda m, n, t: [(n, t)]}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), ns)
    fn=ns['_refit_staging_dtype']
    tensor=torch.zeros(3, dtype=torch.float32)
    assert fn(model,'experts',tensor,torch.bfloat16,{}) == torch.bfloat16
    assert fn(model,'e_score_correction_bias',tensor,torch.bfloat16,{}) == torch.float32
    # Mixed export dtypes must preserve the highest precision in the shared input.
    ns['_maybe_adapt_tensor_to_hf'] = lambda m,n,t: [('expert',t),('e_score_correction_bias',t)]
    assert fn(model,'mixed',tensor,torch.bfloat16,{}) == torch.float32
    model.state_dict_adapter=None
    assert fn(model,'experts',tensor,torch.bfloat16,{}) == torch.float32
