# megatron-ep-overlaps-but-spans-wider

> Megatron EP is carved out of the TP*CP*DP rank space (adds no GPUs), but the ep_group's SPAN is ETP*EP and can far exceed TP*PP*CP — so any NVLink/domain sizing that uses TP*PP*CP undersizes MoE inference

When reviewing anything that sizes a placement group / NVLink domain / "GPUs per instance" for
Megatron MoE **inference**, `TP*PP*CP` is the wrong quantity. Expect the objection *"EP is just
overlapped with TP/DP, so it's already covered"* — it is half right and the conclusion is wrong.

**Overlapped, yes.** `initialize_model_parallel` builds TWO `RankGenerator`s over ONE `world_size`:
the dense one with `ep=1`, the expert one with `cp=1` and
`expert_data_parallel_size = world_size // (ETP*EP*PP*egtp)`. `RankGenerator.__init__` asserts
`ep == 1 or cp == 1` ("CP is only included in default RankGenerator, and EP only in expert
RankGenerator"). So `ETP*EP*EDP == TP*CP*DP` — EP consumes no extra GPUs.

**Wider anyway.** EP borrows from **DP**, which is exactly the axis `TP*PP*CP` excludes. With
default `order="tp-cp-ep-dp-pp"`, one ep_group is `{base + k*ETP : k in [0,EP)}` — a contiguous
block of `ETP*EP` ranks. For TP=2, PP=1, CP=1, ETP=1, EP=8 on 8 GPUs:

```
TP groups: [0,1] [2,3] [4,5] [6,7]     <- TP*PP*CP = 2
EP group:  [0,1,2,3,4,5,6,7]           <- 8   (one TP block swept across all of DP)
```

Correct span: **`max(TP*CP, ETP*EP)` per pipeline stage.** Do NOT multiply by PP — `pp` is the
outermost axis, so no ep_group crosses a stage and PP traffic is p2p, not a collective. (Multiplying
by PP is still defensible if the quantity means "whole-instance footprint", e.g. for parity with
vLLM's `TP*PP`; say which you mean.) Use `max`, never a product — the grids overlap.

Why it matters: `NVLSAllGatherVDispatcher` allocates every symmetric-memory buffer with
`process_group=ep_group` and raises *"requires Hopper+ GPUs fully connected via NVLink … Use
inference_moe_token_dispatcher_type='nccl' on non-NVLS systems"*. Not NVLS-only either — the NCCL
dispatcher's CUDA-graph path collectives run on `tp_ep_group` (span `ETP*EP`).

Failure mode is silent: undersized `gpus_per_instance` -> `nodes_per_instance == 1` -> the `> 1`
gate skips constraint building -> no domain pinning, and the "skipping constraints" warning never
prints because it too is gated on `> 1`.

**Verify, don't reason.** Extract Megatron's own `generate_masked_orthogonal_rank_groups` with `ast`
and run it on the config's real numbers — it is pure Python, needs no GPU, and settles span
questions in seconds. Related: `megatron-expert-rank-layout`.
