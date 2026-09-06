# Precision Matrix Refresh

This experiment compares three precision arms on one pinned NeMo-RL source
revision. Each model and execution mode keeps its workload, GPU count, and
parallelism fixed across the three arms.

| Arm | Policy training | Rollout |
|---|---|---|
| `bf16-bf16` | BF16 | BF16 FlashInfer TRTLLM, except Qwen3-235B TP8 uses Triton |
| `bf16-mxfp8` | BF16 | MXFP8 FlashInfer TRTLLM |
| `mxfp8-mxfp8` | MXFP8 with `fp8_param=true` | MXFP8 FlashInfer TRTLLM |

`sync` uses colocated CUDA IPC refit. The Qwen3-30B-A3B, Qwen3.5-35B-A3B,
and Nemotron 3.5 Lightning Sync recipes use eight nodes and EP32 so BF16
policy and reference initialization fit in host memory. `async` uses
disaggregated NCCL Reshard refit and keeps the smaller training topology because
generation has separate workers. Qwen3.5 Async reserves one four-node segment
for training and one four-node segment for generation, which keeps the EP16
training group inside one NVLink domain. All runs execute 20 steps; reports use
steps 2-19. Qwen3-235B has a
1536-wide expert dimension. TP8 produces a 192-wide local BF16 expert shard,
which the FlashInfer TRTLLM BF16 kernel rejects because it is not a multiple of
128. The Triton baseline matches the upstream Qwen3-235B performance recipe;
both MXFP8 arms continue to use FlashInfer TRTLLM.

Use the cluster-specific launcher so its scheduler arguments match the target
cluster. OCI requests `batch` and four GPU GRES per node. Ptyche requests
`batch`, and Lyris requests `gb200`; both allocate whole nodes without a GRES
request. Scheduler preflight rejects both clusters when no partition is given.
Lyris Qwen3-235B jobs read the immutable model snapshot from Lustre instead of
copying hundreds of GB into each job's node-local cache. Their dataset, venv,
Ray, and compiler caches still use `/raid/scratch`. Every exclusive allocation
clears only its own node-local run directory before creating the new cache.
Concurrent runs therefore cannot delete each other's source or environment;
durable Lustre logs and results are not removed.

The launcher keeps GPU-local CPU affinity but disables hard NUMA memory binding.
Large policy and reference workers can then use memory from the whole node
instead of exhausting one NUMA node while other host memory remains free. Set
`NRL_DISABLE_NUMA_MEMBIND=0` only for a controlled locality comparison.

Before submission, the launcher packs the clean source tree and all pinned
submodules into one immutable tar file under `/home`. Each allocated node
extracts that file into its local scratch directory and builds there. Parallel
jobs therefore cannot race while building editable Megatron-Core extensions
from one shared source checkout.

The original Hugging Face weights, venvs, and compiler caches stay node-local,
but `NRL_MEGATRON_CHECKPOINT_DIR` points to the shared converted-checkpoint
cache. All policy ranks can therefore read the `run_config.yaml` and weight
shards produced by the one-time Hugging Face-to-Megatron conversion.

Run one arm on OCI:

```bash
MODEL=qwen30 MODE=async ARM=mxfp8-mxfp8 ACTION=test-only \
  ./experiments/precision_matrix_refresh_20260905/submit_oci.sh

MODEL=qwen30 MODE=async ARM=mxfp8-mxfp8 ACTION=submit \
  ./experiments/precision_matrix_refresh_20260905/submit_oci.sh
```

Qwen3.5 EP32 host-memory smoke tests use eight 4-GPU nodes. Run the all-to-all
arm first to isolate the memory effect of EP32, then enable HybridEP with the
same topology to measure dispatcher performance:

```bash
MODEL=qwen35 MODE=sync ARM=bf16-bf16 TOPOLOGY=ep32-alltoall MAX_STEPS=2 \
  ACTION=test-only ./experiments/precision_matrix_refresh_20260905/submit_oci.sh

MODEL=qwen35 MODE=sync ARM=bf16-bf16 TOPOLOGY=ep32-hybridep MAX_STEPS=2 \
  ACTION=test-only ./experiments/precision_matrix_refresh_20260905/submit_oci.sh
```

Submit a matrix by invoking the launcher once per model, mode, and arm. Run
`ACTION=test-only` first. OCI-HSG has all four model caches. Ptyche currently
has Qwen3-30B-A3B, Nemotron 3.5 Lightning, and Qwen3.5-35B-A3B caches. Verify
the requested model cache before using the Lyris launcher.
