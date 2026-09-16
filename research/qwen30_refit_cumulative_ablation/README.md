# Qwen3-30B-A3B cumulative refit ablation

This experiment measures three ordered refit configurations on the same source,
container, model recipe, seed, and hardware:

| Arm | Transport | MXFP8 MoE conversion |
|---|---|---|
| `legacy` | Legacy collective | Per expert, reusable scratch disabled |
| `nccl_only` | NCCL Reshard | Per expert, reusable scratch disabled |
| `nccl_full` | NCCL Reshard | Batched conversion with reusable scratch |

All arms use
`examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-async-1off-mxfp8-rollout.yaml`
for 20 steps. The inherited global batch size, policy and generation node split,
parallelism, logprob work, and seed are unchanged.

The Sep 16 nightly contains prebuilt main and actor virtual environments whose
Python symlinks refer to uv-managed interpreters omitted from the squashfs.
`repair_container_python.sh` installs the matching Python 3.13 interpreter and
repoints those existing environments without reinstalling their packages. The
launcher runs it once on every node through `SETUP_COMMAND`.

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
