# Qualified NIXL lease verification backport

Upstream: https://github.com/vllm-project/vllm/pull/62065 at
`03b2648165d24128611cd15818f4ecf51c0a0205` (an explicitly pinned PR revision).
Related issue: https://github.com/vllm-project/vllm/issues/61229.

This is an adapted backport to `0.30.0+g116da885561b.cu132`, rather than an
unmodified cherry-pick onto NeMo RL's default vLLM version. The patch retains
that build's inline lease expiry cleanup and worker APIs, adds lease slot
invalidation before expiry cleanup frees the KV blocks, and integrates consumer
nonce verification into the existing pull completion path. `lease_slots.py`
matches the pinned upstream revision. Newer upstream packed/PLE paths and
unrelated notification metrics are outside this backport.

The manifest contains the exact input and output source hashes. Wire version
`15062065` is reserved here for this adapted backport: deploy the same build on
all producers and consumers. Do not use the manifest as proof of compatibility
with a differently compiled wheel, CUDA stack, transfer topology, or model.

Use `tools/apply_vllm_nixl_lease_patch.py` before starting workers; see
`docs/guides/use-custom-vllm.md`. The installer deliberately refuses unknown
source and does not change dependency pins or existing runtime monkey patches.
The patch preserves upstream file copyright and licensing notices.
