# SPDX-License-Identifier: Apache-2.0
"""Runtime smoke for MDL incremental/partial-update (today's change).

Fake dense model with a fused QKV + fused gate_up + two direct params, exercised
through MdlLoader:
  1. full cold cycle  -> stock load + build map
  2. full warm cycle  -> all direct/fused, 0 fallback, byte-identical slots
  3. subset warm cycle -> only some params, 0 fallback
  4. incremental      -> params never seen in cold cycle get mapped on-the-fly
No transport/NIXL — pure load-side logic.
"""
import os
os.environ["MX_LOAD_MODE"] = "direct"
import torch
from modelexpress.engines.vllm.mdl import MdlLoader

D, Q, K, V, I = 8, 8, 4, 4, 6
P = "model.layers.0.self_attn."
M = "model.layers.0.mlp."


class FakeModel:
    stacked_params_mapping = [
        (".qkv_proj", ".q_proj", "q"), (".qkv_proj", ".k_proj", "k"),
        (".qkv_proj", ".v_proj", "v"),
        (".gate_up_proj", ".gate_proj", 0), (".gate_up_proj", ".up_proj", 1),
    ]
    _mx_layout_version = 1

    def __init__(self):
        self._p = {
            P + "qkv_proj.weight": torch.zeros(Q + K + V, D),
            M + "gate_up_proj.weight": torch.zeros(2 * I, D),
            "model.embed_tokens.weight": torch.zeros(10, D),
            "model.norm.weight": torch.zeros(D),
        }
        for t in self._p.values():
            t.requires_grad_(False)

    def named_parameters(self):
        return iter(self._p.items())

    def load_weights(self, weights):  # stock loader stand-in (cold + fallback)
        for _ in weights:
            pass


def hf(scale):
    return [
        (P + "q_proj.weight", torch.full((Q, D), 1.0 * scale)),
        (P + "k_proj.weight", torch.full((K, D), 2.0 * scale)),
        (P + "v_proj.weight", torch.full((V, D), 3.0 * scale)),
        (M + "gate_proj.weight", torch.full((I, D), 4.0 * scale)),
        (M + "up_proj.weight", torch.full((I, D), 5.0 * scale)),
        ("model.embed_tokens.weight", torch.full((10, D), 6.0 * scale)),
        ("model.norm.weight", torch.full((D,), 7.0 * scale)),
    ]


m = FakeModel()
mdl = MdlLoader(m)

# 1. cold
mdl.load_weights(hf(1.0))
assert mdl._param_cache is not None, "cold cycle did not build map"
print("cold: direct=%d fused=%d expert=%d" % (len(mdl._direct), len(mdl._fused), len(mdl._expert)))
assert len(mdl._fused) == 5, mdl._fused  # q,k,v,gate,up
assert len(mdl._direct) == 2, mdl._direct  # embed, norm

# 2. full warm + byte-identity of fused slots
mdl.load_weights(hf(2.0))
qkv = m._p[P + "qkv_proj.weight"]
assert torch.equal(qkv[0:Q], torch.full((Q, D), 2.0)), "q slot wrong"
assert torch.equal(qkv[Q:Q + K], torch.full((K, D), 4.0)), "k slot wrong"
assert torch.equal(qkv[Q + K:Q + K + V], torch.full((V, D), 6.0)), "v slot wrong"
gu = m._p[M + "gate_up_proj.weight"]
assert torch.equal(gu[0:I], torch.full((I, D), 8.0)), "gate slot wrong"
assert torch.equal(gu[I:2 * I], torch.full((I, D), 10.0)), "up slot wrong"
print("full warm: byte-identical fused slots OK")

# 3. subset warm (only q + gate) -> 0 fallback, values updated
before_k = qkv[Q:Q + K].clone()
mdl.load_weights([hf(3.0)[0], hf(3.0)[3]])  # q_proj, gate_proj only
assert torch.equal(qkv[0:Q], torch.full((Q, D), 3.0)), "subset q not updated"
assert torch.equal(qkv[Q:Q + K], before_k), "subset should not touch k"
assert torch.equal(gu[0:I], torch.full((I, D), 12.0)), "subset gate not updated"
print("subset warm: scoped update OK (q+gate updated, k untouched)")

# 4. incremental: fresh loader, cold cycle sees ONLY direct params;
#    a later cycle brings q/k/v -> must be mapped on-the-fly, not permanent fallback
m2 = FakeModel()
mdl2 = MdlLoader(m2)
mdl2.load_weights([("model.embed_tokens.weight", torch.full((10, D), 1.0)),
                   ("model.norm.weight", torch.full((D,), 1.0))])
assert len(mdl2._fused) == 0, "fused should be empty after direct-only cold"
mdl2.load_weights(hf(2.0))  # now includes q/k/v/gate/up
assert len(mdl2._fused) == 5, "incremental: fused not mapped on-the-fly: %s" % mdl2._fused
qkv2 = m2._p[P + "qkv_proj.weight"]
assert torch.equal(qkv2[0:Q], torch.full((Q, D), 2.0)), "incremental q slot wrong"
print("incremental: previously-unseen fused group mapped on-the-fly OK")

print("\nALL MDL PARTIAL-UPDATE SMOKE CHECKS PASSED")
