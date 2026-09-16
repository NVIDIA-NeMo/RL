# Qwen3-30B-A3B cumulative refit ablation

This experiment measures three ordered refit configurations on the same source,
container, model recipe, seed, and hardware:

| Arm | Transport | Batched expert shuffle | Loader-route cache |
|---|---|---|---|
| `legacy` | Legacy collective | Disabled | Disabled |
| `nccl_only` | NCCL Reshard | Disabled | Disabled |
| `nccl_full` | NCCL Reshard | Enabled with reusable scratch | Enabled |

All arms use
`examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-async-1off-mxfp8-rollout.yaml`
for 20 steps. The inherited global batch size, policy and generation node split,
parallelism, logprob work, and seed are unchanged.

Trainer-side prequantization is explicitly disabled in all arms because NCCL
Reshard does not support that path. Persistent IPC source buffers are also
disabled because this is a non-colocated NCCL experiment. This source does not
reuse NCCL receive buffers across refits, so the full arm measures every
additional optimization currently applicable to this Async path: batched
expert shuffle, reusable shuffle scratch, and loader-route caching.

The Sep 16 nightly contains prebuilt main and actor virtual environments whose
Python symlinks refer to uv-managed interpreters omitted from the squashfs.
`repair_container_python.sh` installs the matching Python 3.13 interpreter and
repoints those existing environments without reinstalling their packages. The
launcher stores the exact committed source as one tar artifact on Lustre, then
extracts it to node-local `/raid/scratch` and repairs Python once per node
through `SETUP_COMMAND`. It does not bind-mount the login node's `/home`, which
is not shared with OCI-HSG compute nodes.

NCCL M2N is distributed separately from `nvidia-nccl-cu13` in the
`nccl-extensions` wheel. `ensure_nccl_m2n.sh` installs version 0.1.0 into the
main, Megatron policy, and synchronous/asynchronous vLLM environments using a
node-local uv cache. It then imports `nccl.m2n.reshard` in every environment.
All three arms run this preflight so their package environments are identical;
an unavailable native M2N binding stops the job instead of silently measuring
the Python fallback.

The experiment reports means over steps 2-19:

- refit transfer and update time
- total refit bubble
- end-to-end step time and throughput
- generation time and throughput
- policy training and logprob time
- `gen_kl_error` and reward ranges

The ordered contributions are:

```text
NCCL Reshard              = (legacy - nccl_only) / legacy
Additional optimizations  = (nccl_only - nccl_full) / nccl_only
Combined                  = (legacy - nccl_full) / legacy
```

Submit one arm, or all three:

```bash
./research/qwen30_refit_cumulative_ablation/submit_oci.sh legacy
./research/qwen30_refit_cumulative_ablation/submit_oci.sh nccl_only
./research/qwen30_refit_cumulative_ablation/submit_oci.sh nccl_full
./research/qwen30_refit_cumulative_ablation/submit_oci.sh all
```

Set `DRY_RUN=1` to run `sbatch --test-only` without submitting.

Validate the staged container or run the focused vLLM tests with:

```bash
./research/qwen30_refit_cumulative_ablation/smoke_container.sh
./research/qwen30_refit_cumulative_ablation/run_vllm_pytest.sh \
  tests/unit/models/generation/test_vllm_fp8_quantization.py
```
