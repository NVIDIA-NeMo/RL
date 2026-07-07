# SPDX-License-Identifier: Apache-2.0
"""Quantified EP>1 byte-pruning proof using the real planner + expert adapter.

Models a 30B-style stacked expert tensor (128 experts) published by an EP=8
trainer (16 experts/rank). An inference receiver at EP=8 requests only its local
experts; we assert the plan pulls exactly 1/8 of the expert bytes (not all).
Also checks EP=4 (32/rank) and mixed trainer-EP8 -> inference-EP4.
Pure planner logic (no transport).
"""
from modelexpress.rl_expert_layout import (
    compute_local_expert_ids, expert_ids_to_contiguous_ranges,
)
from modelexpress.rl_reshard_planner import plan_coverage, collect_byte_savings_vs_allgather
from modelexpress.rl_slice_descriptors import SliceOwnership, SliceRequest

NE = 128
TENSOR = "model.layers.0.experts.w13_weight"
IN, OUT = 4096, 2048
EXPERT_BYTES = IN * OUT * 2  # bf16 per expert


def sources_ep(world):
    """One SHARD source per trainer EP rank (contiguous expert block)."""
    src = []
    per = NE // world
    for r in range(world):
        lo, hi = r * per, (r + 1) * per
        src.append(SliceOwnership(
            model_name="m", tensor_name=TENSOR, global_shape=(NE, IN, OUT),
            dtype="torch.bfloat16", placement_kind="SHARD", shard_axis=0,
            local_shard_range=(lo, hi), worker_rank=r,
            nixl_addr=0x1000 + r, byte_size=per * EXPERT_BYTES,
        ))
    return src


def requests_for(local_ids):
    reqs = []
    for lo, hi in expert_ids_to_contiguous_ranges(local_ids):
        reqs.append(SliceRequest(
            tensor_name=TENSOR, global_range=(lo, hi), shard_axis=0,
            dtype="torch.bfloat16", receiver_rank=0, target_addr=0xF000,
        ))
    return reqs


def check(trainer_world, infer_world, infer_rank):
    src = sources_ep(trainer_world)
    local = compute_local_expert_ids(infer_rank, infer_world, NE, "linear")
    plan = plan_coverage(src, requests_for(local))
    plan.raise_if_incomplete()
    pulled = sum(s.byte_count for s in plan.segments)
    total_expert_bytes = NE * EXPERT_BYTES
    frac = pulled / total_expert_bytes
    expected = len(local) / NE
    print(f"trainer EP={trainer_world}, infer EP={infer_world}, rank={infer_rank}: "
          f"pulled {pulled/1e6:.0f} MB = {frac*100:.1f}% of experts "
          f"(expected {expected*100:.1f}%), from ranks "
          f"{sorted({s.source.worker_rank for s in plan.segments})}")
    assert abs(frac - expected) < 1e-6, f"byte fraction {frac} != expected {expected}"
    return frac


print("=== EP>1 byte-pruning (real planner) ===")
f8 = check(8, 8, 0)                       # matched EP=8
assert abs(f8 - 1 / 8) < 1e-6
check(8, 8, 3)                            # a middle rank
f4 = check(4, 4, 1)                       # matched EP=4
assert abs(f4 - 1 / 4) < 1e-6
check(8, 4, 0)                            # mixed: trainer EP8 -> inference EP4 (pulls 32/128 = 25%)

# byte-savings vs the naive "pull all experts" baseline at EP=8
src = sources_ep(8)
plan = plan_coverage(src, requests_for(compute_local_expert_ids(0, 8, NE, "linear")))
sv = collect_byte_savings_vs_allgather(plan, src)
print(f"\nEP=8 savings vs all-experts baseline: {sv['savings_factor']:.1f}x "
      f"({sv['rank_to_rank_actual_bytes']/1e6:.0f} MB vs {sv['allgather_per_receiver_bytes']/1e6:.0f} MB)")
assert sv["savings_factor"] >= 7.9, sv

print("\nEP>1 BYTE-PRUNING PROOF PASSED (planner pulls 1/EP of expert bytes)")
