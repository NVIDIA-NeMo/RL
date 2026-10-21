# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for value equality, dtype precision and fused tensor mapping."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from safetensors import safe_open
from step0_audit import compare, resident_view, Dump

class FakeModel:
    def __init__(self, params, switch=False):
        self.params=params
        self.module=SimpleNamespace(quant_method=SimpleNamespace(load_up_proj_weight_first=switch),use_triton_kernels=False)
    def named_parameters(self): return self.params.items()
    def get_submodule(self,name): return self.module

class Tests(unittest.TestCase):
    def test_lossless_cast(self):
        x=torch.tensor([1.25,2.5],dtype=torch.bfloat16)
        self.assertTrue(compare(x.float(),x)['value_equal'])
    def test_fp32_fraction_not_hidden_by_bf16(self):
        self.assertFalse(compare(torch.tensor([1.0001]),torch.tensor([1.],dtype=torch.bfloat16))['value_equal'])
    def test_nonfinite(self):
        for v in [float('nan'),float('inf')]: self.assertFalse(compare(torch.tensor([v]),torch.tensor([v]))['value_equal'])
    def test_shape(self): self.assertFalse(compare(torch.ones(2),torch.ones(1,2))['shape_match'])
    def test_expert_gate_order(self):
        for switch in (False,True):
            a=torch.arange(24.).reshape(2,6,2)
            m=FakeModel({'model.layers.2.mlp.experts.w13_weight':a},switch)
            for kind,second in [('gate_proj',switch),('up_proj',not switch)]:
                _,got=resident_view(m,f'model.layers.2.mlp.experts.1.{kind}.weight',None)
                self.assertTrue(torch.equal(got,a[1,3:6] if second else a[1,:3]))
    def test_expert_down(self):
        a=torch.arange(24.).reshape(2,3,4);m=FakeModel({'model.layers.2.mlp.experts.w2_weight':a})
        self.assertTrue(torch.equal(resident_view(m,'model.layers.2.mlp.experts.1.down_proj.weight',None)[1],a[1]))
    def test_qkv(self):
        a=torch.arange(32.).reshape(8,4)
        tensors={f'model.layers.0.self_attn.{p}.weight':torch.zeros(n,4) for p,n in [('q_proj',4),('k_proj',2),('v_proj',2)]}
        source=SimpleNamespace(get=lambda n:tensors[n]);m=FakeModel({'model.layers.0.self_attn.qkv_proj.weight':a})
        for p,start,end in [('q_proj',0,4),('k_proj',4,6),('v_proj',6,8)]:
            self.assertTrue(torch.equal(resident_view(m,f'model.layers.0.self_attn.{p}.weight',source)[1],a[start:end]))
    def test_dense_gate_up(self):
        a=torch.arange(24.).reshape(6,4);m=FakeModel({'model.layers.0.mlp.gate_up_proj.weight':a})
        self.assertTrue(torch.equal(resident_view(m,'model.layers.0.mlp.up_proj.weight',None)[1],a[3:]))
    def test_bad_name(self):
        with self.assertRaises(RuntimeError): resident_view(FakeModel({}),'unknown.weight',None)
    def test_dump_roundtrip(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);d=Dump(p);x=torch.tensor([1.25],dtype=torch.bfloat16);d.add('head',x);d.finish()
            with safe_open(str(p/'weights-0000.safetensors'),framework='pt') as h: self.assertTrue(torch.equal(h.get_tensor('head'),x))

if __name__=='__main__': unittest.main()
