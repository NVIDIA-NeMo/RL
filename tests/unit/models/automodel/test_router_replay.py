"""Exercise real AutoModel gates, nonzero gradients, and checkpoint replay."""
import copy
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.models.deepseek_v4.model import DeepseekV4VisionGate
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.moe.experts import GroupedExperts
from nemo_rl.models.automodel.router_replay import configure_router_replay, router_replay_context


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = MoEConfig(dim=8,inter_dim=16,moe_inter_dim=16,n_routed_experts=4,
            n_shared_experts=0,n_activated_experts=2,n_expert_groups=0,n_limited_groups=0,
            train_gate=True,gate_bias_update_factor=0.,aux_loss_coeff=0.,score_func='sqrtsoftplus',
            route_scale=1.5,norm_topk_prob=True,swiglu_limit=10.,dtype=torch.float32,
            router_weights_fp32=True,apply_router_weight_after_down=True)
        self.gate = DeepseekV4VisionGate(DeepseekV4Config(vocab_size=16), cfg,
            gate_precision=torch.float32, hash_routing=False)
        self.experts = GroupedExperts(cfg)
        self.seen = []
        for p in self.parameters():
            nn.init.normal_(p, std=.2)

    def forward(self,x):
        mask = torch.ones(x.shape[0],dtype=torch.bool)
        w, ids, _ = self.gate(x,mask,None)
        self.seen.append(ids.detach().clone())
        return self.experts(x,mask,w,ids)


class ReplayTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.model = Block()
        cfg = dict(router_replay={'enabled':True},generation={'backend':'vllm'},
            sequence_packing={'enabled':False},dtensor_cfg={'tensor_parallel_size':1,'context_parallel_size':1})
        configure_router_replay(self.model,cfg)
        self.routes = torch.tensor([[[[3,1]],[[0,2]],[[2,1]],[[1,0]], [[-1,-1]]]])
        self.mb = SimpleNamespace(data_dict={'routed_experts':self.routes,'token_mask':torch.tensor([[0,0,1,1,1]])},
            processed_inputs=SimpleNamespace(input_ids=torch.zeros(1,5,dtype=torch.long)))

    def test_checkpoint_matches_gradients_and_replays_twice(self):
        reference = copy.deepcopy(self.model)
        x = torch.randn(5,8,requires_grad=True)
        xr = x.detach().clone().requires_grad_()
        with router_replay_context(reference,self.mb):
            expected = reference(xr)
            expected[:4].square().sum().backward()
        with router_replay_context(self.model,self.mb), set_checkpoint_early_stop(False):
            actual = checkpoint(self.model,x,use_reentrant=False)
            actual[:4].square().sum().backward()
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        torch.testing.assert_close(x.grad,xr.grad,rtol=0,atol=0)
        for p,q in zip(self.model.parameters(),reference.parameters()):
            torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.abs().max().item(),0)
        self.assertEqual(len(self.model.seen),2)
        for ids in self.model.seen:
            torch.testing.assert_close(ids[:4],self.routes[0,:4,0],rtol=0,atol=0)
        self.assertIsNone(self.model.gate.router_replay.mode)
        self.assertIsNone(self.model.gate.router_replay.target_indices)

    def test_missing_required_route_rejected(self):
        self.routes[0,0,0] = -1
        with self.assertRaisesRegex(ValueError,'missing, duplicate'):
            with router_replay_context(self.model,self.mb):
                pass

    def test_duplicate_required_route_rejected(self):
        self.routes[0,1,0] = 2
        with self.assertRaisesRegex(ValueError,'missing, duplicate'):
            with router_replay_context(self.model,self.mb):
                pass

    def test_cleanup_on_failure(self):
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            with router_replay_context(self.model,self.mb):
                raise RuntimeError('deliberate')
        self.assertIsNone(self.model.gate.router_replay.mode)
        self.assertIsNone(self.model.gate.router_replay.target_indices)


if __name__ == '__main__':
    unittest.main(verbosity=2)
