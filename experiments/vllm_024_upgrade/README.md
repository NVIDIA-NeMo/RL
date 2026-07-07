# NeMo-RL vLLM 0.24 Upgrade Validation

This experiment validates the vLLM 0.24 upgrade with unchanged upstream
NeMo-RL performance recipes on Pre-Tyche. Cluster-specific overrides only set
the run length, disable checkpoint writes, preserve CUDA Graph execution, and
separate logs and W&B runs.

| Label | Upstream recipe | Nodes | GPUs/node | Segment | Max sequence |
|---|---|---:|---:|---:|---:|
| qwen30ba3b | `grpo-qwen3-30ba3b-4n4g.yaml` | 4 | 4 | 4 | 4,096 |
| qwen32b | `grpo-qwen3-32b-4n4g.yaml` | 4 | 4 | 4 | 4,096 |
| qwen235b | `grpo-qwen3-235b-16n4g.yaml` | 16 | 4 | 16 | 8,192 |

Run scheduler validation before submission:

```bash
experiments/vllm_024_upgrade/submit_performance_step10.sh test-only all
experiments/vllm_024_upgrade/submit_performance_step10.sh submit all
```

The launcher reads the W&B key from `WANDB_API_KEY` or from the file named by
`WANDB_API_KEY_FILE`. It can also read the private `.netrc` created by the
cluster-setup skill when `WANDB_NETRC_HOME` is set. It never stores the key in
the repository or job logs. Set `USE_GRES=true` on OCI-HSG and AWS-DFW; keep it
false on Pre-Tyche.
