# GPT-OSS GRPO Reproduction Guide

**Author**: Seonjin Na
**Branch**: `sj/pip-cudnn-test` (NVIDIA-NeMo/RL)
**Last updated**: 2026-03-07

This guide describes every change needed relative to `main` to run GPT-OSS 20B and 120B
GRPO training on H100 clusters, with FusedAttention sub-backend 1 (cuDNN ≥ 9.18 + THD layout).

---

## Overview of Changes vs `main`

| File | What changed | Why |
|------|-------------|-----|
| `pyproject.toml` | `ray==2.49.2` → `ray==2.54.0` | Newer cluster GCS binary requires 2.54 |
| `cluster_config.sh` | Container updated to terryk's `tk-vllm-v5-c19f6d84f` | GPT-OSS model support in vLLM |
| `ray.sub` | Ray node-count via log grep (not `ray status`); job auto-rename; `RAY_ADDRESS` propagated to attach scripts | Fix for version-mismatched clusters |
| `nemo_rl/distributed/virtual_cluster.py` | Ray client mode support; version-mismatch RuntimeError suppressed | Allows `ray://` attach + Ray 2.54 GCS |
| `nemo_rl/distributed/worker_groups.py` | Fallback: propagate driver `LD_LIBRARY_PATH` to Ray actors when no pip cuDNN in worker venv | cuDNN 9.19 reaches all workers |
| `nemo_rl/utils/venvs.py` | `_fix_cudnn_in_venv()`: enforce `nvidia-cudnn-cu12==9.19.0.56` + symlink in worker venv | Ensures FusedAttention works cluster-wide |
| `nemo_rl/models/generation/vllm/vllm_worker.py` | `attention/layer.py` patch made optional | vLLM 0.11.2+ ships the fix upstream |
| `nemo_rl/models/policy/workers/patches.py` | Guard `ImportError` on torch DTensor ops; version gate on torch 2.9 patch | Compatibility with torch ≥ 2.10 |

Additionally, the following run scripts are included in this branch (not in `main`):
- `run_20b_final.sh` — GPT-OSS 20B, 2-node, TP=2 EP=8, alltoall (**validated**)
- `run_120b_8n_tp8.sh` — GPT-OSS 120B, 8-node, TP=8 PP=2 EP=4, alltoall (testing)
- `run_120b_16n.sh` — GPT-OSS 120B, 16-node, TP=8 PP=2 EP=8, alltoall (**step 1 validated**)

---

## Prerequisites

### Container

```
/lustre/fsw/portfolios/coreai/users/terryk/enroot-images/gitlab-master.nvidia.com/terryk/images/nemo-rl:tk-vllm-v5-c19f6d84f.squashfs
```

This container ships cuDNN 9.10.1 (insufficient). The run scripts install cuDNN 9.19 at startup.

### Model weights

Both models are loaded from HuggingFace in dummy mode (weights are random):
- GPT-OSS 20B: `unsloth/gpt-oss-20b-BF16`
- GPT-OSS 120B: `unsloth/gpt-oss-120b-BF16`

Ensure `HF_HOME` and `HF_TOKEN` are set, or the model cache is pre-populated.

### cuDNN 9.19 (installed automatically by run scripts)

FusedAttention sub-backend 1 requires cuDNN ≥ 9.18.  The run scripts handle this:

```bash
# Prepend pip cuDNN 9.19 lib to LD_LIBRARY_PATH (driver process)
PIP_CUDNN_LIB=$(uv run python3 -c "import nvidia.cudnn, pathlib; \
  print(pathlib.Path(list(nvidia.cudnn.__path__)[0]) / 'lib')")
export LD_LIBRARY_PATH="${PIP_CUDNN_LIB}:${LD_LIBRARY_PATH:-}"
ln -sf libcudnn.so.9 "${PIP_CUDNN_LIB}/libcudnn.so" 2>/dev/null || true

# Install into system venv so Ray workers also pick it up
/opt/nemo_rl_venv/bin/pip install "nvidia-cudnn-cu12==9.19.0.56" --no-deps -q
```

`worker_groups.py` propagates `LD_LIBRARY_PATH` to all Ray actors automatically.

---

## Submission

All jobs are submitted from the `RL-pip-cudnn-test/` working directory using `ray.sub`.

```bash
cd /path/to/RL-pip-cudnn-test
source cluster_config.sh && setup_cluster_config "batch" && export_cluster_config
```

### GPT-OSS 20B (2 nodes, FULLY VALIDATED)

```bash
COMMAND='bash run_20b_final.sh' sbatch \
  --nodes=2 --account=coreai_dlalgo_nemorl \
  --partition=batch --time=2:00:00 ${GRES_FLAG} ray.sub
```

**Config** (`grpo-gptoss-20b-8n8g-megatron.yaml` overrides):
```
cluster.num_nodes=2 / gpus_per_node=8
Megatron: TP=2, PP=1, EP=8
vLLM: TP=4, gpu_memory_utilization=0.5
moe_token_dispatcher_type=alltoall   # Bug 8 fix: eliminates allgather deadlock
moe_permute_fusion=true
sequence_packing.enabled=true
train_global_batch_size=128 (16 prompts x 8 generations)
max_num_steps=20
```

**Expected outcome**: FusedAttention sub-backend 1 active on all 16 Megatron workers
(logged at logprob and training backward). 11+ GRPO steps in 2 hours.

### GPT-OSS 120B — Option A: 16 nodes (STEP 1 VALIDATED)

```bash
COMMAND='bash run_120b_16n.sh' sbatch \
  --nodes=16 --account=coreai_dlalgo_nemorl \
  --partition=batch --time=2:00:00 ${GRES_FLAG} ray.sub
```

**Config**:
```
cluster.num_nodes=16 / gpus_per_node=8  (128 GPUs total)
Megatron: TP=8, PP=2, EP=8
vLLM: TP=8, gpu_memory_utilization=0.5
moe_token_dispatcher_type=alltoall
moe_permute_fusion=true
sequence_packing.enabled=true
train_global_batch_size=128
```

**Key constraint (Bug 11 fix)**: Megatron TP=8 must equal vLLM TP=8.
If Megatron TP < vLLM TP, `prepare_refit_info()` all_gathers weights across TP ranks,
causing GPU OOM on 120B. Matching TP makes it a 1:1 copy with no gather.

### GPT-OSS 120B — Option B: 8 nodes (TESTING)

```bash
COMMAND='bash run_120b_8n_tp8.sh' sbatch \
  --nodes=8 --account=coreai_dlalgo_nemorl \
  --partition=batch --time=2:00:00 ${GRES_FLAG} ray.sub
```

**Config**:
```
cluster.num_nodes=8 / gpus_per_node=8  (64 GPUs total)
Megatron: TP=8, PP=2, EP=4   # EP=4 because TP8 x PP2 x EP4 = 64 = 8x8
vLLM: TP=8, gpu_memory_utilization=0.5
moe_token_dispatcher_type=alltoall
moe_permute_fusion=true
train_global_batch_size=64   # 8 prompts x 8 gen (halved vs 16n to fit KV cache)
```

**Key constraint (Bug 12 fix)**: PP=2 required. PP=1 on 8 nodes with TP=8 causes OOM:
54.73 GiB (existing) + 26.71 GiB (new alloc) = 81.44 GiB > 79.11 GiB GPU memory.

---

## Bugs Fixed / Workarounds

| Bug | Symptom | Fix |
|-----|---------|-----|
| **Bug 8** | NCCL watchdog timeout at ~43 min. `WorkNCCL(SeqNum=193, OpType=_ALLGATHER_BASE)` on `EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP`. | `moe_token_dispatcher_type=alltoall` — eliminates the allgather in MoE dispatch entirely. |
| **Bug 10** | 120B `MegatronPolicyWorker.__init__()` OOM on 2 nodes: vLLM sleep holds ~59 GiB as non-PyTorch CUDA IPC handles regardless of `gpu_memory_utilization`. | Use 8+ nodes. On 8+ nodes, each GPU holds a smaller model shard and colocated vLLM+Megatron memory pressure is lower. |
| **Bug 11** | OOM at `prepare_refit_info()` when Megatron TP < vLLM TP. all_gather reshapes the full 120B param shard, exhausting GPU memory. | Set Megatron `tensor_model_parallel_size` = vLLM `tensor_parallel_size` (both = 8). Becomes a 1:1 copy. |
| **Bug 12** | OOM at `MegatronPolicyWorker.__init__()` with PP=1 on 8 nodes, TP=8: 81.44 GiB > 79.11 GiB. | Use PP=2. Each PP stage holds half the layers → ~27 GiB vs 54 GiB. |

---

## Monitoring

After submission, check progress in the job log:

```bash
# List today's logs
ls -lht *.log | head -5

# Follow the latest log
tail -f gpt-oss-*.log | grep -E "▶|FusedAttention|sub-backend|Error|OOM|NCCL"
```

Key milestones in log order:
1. `[cuDNN] libcudnn.so.9 version: 9.19.x` — cuDNN confirmed
2. `vLLM ... workers initialized` (all N/N) — vLLM ready
3. `CUDA graph capture 83/83` — vLLM graph done
4. `All X Megatron policy workers initialized` — Megatron ready
5. `▶ Preparing batch...` — first GRPO step starting
6. `Selected backend = FusedAttention (sub-backend 1)` — FusedAttention confirmed
7. `▶ Training policy` — training backward running

---

## WandB

Runs log to `sync-grpo-h100-gptoss-exp` (entity `nvidia`):
```
https://wandb.ai/nvidia/sync-grpo-h100-gptoss-exp
```

Enabled with `logger.wandb_enabled=True` in the run scripts.
