# Qualified NIXL lease verification backport

Upstream: https://github.com/vllm-project/vllm/pull/62065 at
`03b2648165d24128611cd15818f4ecf51c0a0205` (an explicitly pinned PR revision).
Related issue: https://github.com/vllm-project/vllm/issues/61229.

This is an adapted backport to `0.30.0+g116da885561b.cu132`, rather than an
unmodified cherry-pick onto NeMo RL's default vLLM version. The patch retains
that build's inline lease expiry cleanup and worker APIs, adds lease slot
invalidation before expiry cleanup frees the KV blocks, and integrates consumer
nonce verification into the existing pull completion path. `lease_slots.py` is
based on the pinned upstream revision, with a local correction for permanent
landing-cell exhaustion after failed transfer-handle release. Quarantined cells
remain unavailable; queued checks fail when no active read or prepared batch
can return capacity. Temporary contention still waits for completion. Newer
upstream packed/PLE paths and unrelated notification metrics are outside this
backport.

The manifest contains the exact input and output source hashes. Wire version
`15062066` is reserved here for this adapted backport: deploy the same build on
all producers and consumers. Do not use the manifest as proof of compatibility
with a differently compiled wheel, CUDA stack, transfer topology, or model.

Use `tools/apply_vllm_nixl_lease_patch.py` before starting workers; see
`docs/guides/use-custom-vllm.md`. The installer deliberately refuses unknown
source and does not change dependency pins or existing runtime monkey patches.
The patch preserves upstream file copyright and licensing notices.

The local `job_failure.py` adds a default `kv_failure_action="stop_job"` policy:
KV transfer errors and lease failures log node/engine/rank/request context,
request cancellation of the current Slurm job, and raise before request-level
recovery. Startup requires a numeric current `SLURM_JOB_ID`, an available
`scancel` client, lease verification, and `kv_load_failure_policy="fail"`.
The log asks operators to "please report to AGA support team"; no Slack
messages are sent. Explicit `kv_failure_action="fail_request"` retains the
historically qualified fail/recompute path for controlled diagnostics.
Peer node names are advertised in handshake metadata, requiring wire version
`15062066` and the same bundle on every P/D/TP process. The seven output files
include two new modules. Existing pre-fail-stop installations are rejected by
the source guards; rebuild from the qualified original package before applying.
New stop/report behavior has CPU regression coverage, not live Slurm/GPU
qualification. See the custom vLLM guide for deployment requirements and limits.
