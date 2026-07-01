# GLM-5.1 cuDNN/tilelang GRPO — working H100 configs (DFW, dummy models)

Base recipe: `examples/configs/recipes/llm/grpo-glm5.1-dummy-4l-megatron-8K-full-onpolicy-4n-noncoloc-tp4cp2-new.yaml`
Container: `nvcr.io/nvidian/nemo-rl:nightly` (imported 2026-06-29). Models (songlinj):
`glm5.1-dummy-model-lite` (4L), `glm5.1-dummy-model-large` (20L, ~90 GB, 64 experts).
All configs below reached 5/5 steps unless noted. Backend toggle: `policy.megatron_cfg.dsa_kernel_backend`
(default is `tilelang` — set `cudnn` explicitly for cuDNN).

## Two required workarounds

1. **cuDNN only — cutlass-dsl provenance** (NVIDIA/cutlass#3170, #3259): the mcore venv must be single-provenance
   `-libs-cu13`, else cuDNN-frontend's cutedsl DSA path crashes (normalize_field_to_ir_name / atom_tma_partition /
   nvvm.atomicrmw). Set `NRL_FORCE_CUTLASS_CU13=1` (venvs.py hook on this branch) OR run
   `examples/slurm/reconcile_cutlass_cu13.sh <mcore-venv>` after the venv builds. Not needed for tilelang.
2. **Both backends — 131k generation hang**: vLLM chunked prefill hangs the DSA sparse indexer at long context
   (related upstream: vLLM #40969, #46715, #40926; general chunked-prefill hang #31641). WAR: disable chunked
   prefill and size `max_num_batched_tokens >= max_model_len`. Generation rollouts are short (prompt + max_new_tokens),
   so a modest `max_model_len` is fine while training stays at 131k.

## cuDNN 4L @8k — 5/5 (job 13292523, ~58.8s/step)
```
EXTRA_OVERRIDES="policy.megatron_cfg.dsa_kernel_backend=cudnn"
```
(base recipe defaults otherwise; 4 nodes)

## cuDNN 4L @131k — 5/5 (job 13294842, ~22s/step steady)
```
policy.megatron_cfg.dsa_kernel_backend=cudnn
policy.max_total_sequence_length=131072
policy.generation.vllm_cfg.max_model_len=16384
policy.generation.vllm_cfg.gpu_memory_utilization=0.7
policy.generation.vllm_kwargs.max_num_batched_tokens=16384
policy.generation.vllm_kwargs.enable_chunked_prefill=false
policy.generation.max_new_tokens=2048
```

## tilelang 4L @131k — 5/5 (job 13294238)
Same as cuDNN 4L @131k but `policy.megatron_cfg.dsa_kernel_backend=tilelang` (and no cutlass reconcile needed).

## 20L @131k — tilelang 5/5 on 5 nodes (job 13313007, ~8.6s/step train)
```
NUM_NODES=5   # non-colocated: 1 gen node + 4 train nodes (32 train GPUs)
policy.megatron_cfg.dsa_kernel_backend=tilelang
policy.megatron_cfg.context_parallel_size=4          # TP4 x CP4 = 16 over 32 train GPUs, DP2
policy.max_total_sequence_length=131072
policy.generation.vllm_cfg.tensor_parallel_size=8    # shard the ~90GB gen model
policy.generation.vllm_cfg.max_model_len=16384
policy.generation.vllm_cfg.gpu_memory_utilization=0.7
policy.generation.vllm_kwargs.max_num_batched_tokens=16384
policy.generation.vllm_kwargs.enable_chunked_prefill=false
policy.generation.max_new_tokens=2048
```
- **cuDNN 20L**: reproducible CUDA illegal-memory-access at step 2 (DSA-specific; tilelang runs fine). Under
  investigation — possibly a silent OOM (logs showed ~20–28 GB free at snapshots, so unconfirmed); try higher
  CP/TP as a discriminator.
- **20L @131k needs > 4 nodes** (does not fit non-colocated on 4).

## Perf @131k (4L): cuDNN ~6.5s/step steady (stable) vs tilelang ~10.5s (bimodal — per-shape JIT recompile spikes).
