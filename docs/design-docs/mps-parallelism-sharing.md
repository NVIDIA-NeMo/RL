# MPS parallelism sharing

This experimental path uses one learner and one Megatron inference process per
GPU. Training and inference can have different tensor and expert layouts while
sharing compatible local parameter storage. The inference backend still requires
CP1 and PP1; `inference_optimized` experts require ETP1. Training CP and PP
are represented by native shard metadata, with one model chunk per worker. Virtual
pipeline chunks, GTP rematerialization, quantized refit tensors, differing tensor
dtypes, and unequal worker worlds are rejected.

Use the existing MPS configuration with `refit_transport: ipc` and
`engine_weights: shared`. Set inference parallelism in
`policy.generation.mcore_generation_config`. With `inference_optimized` and TP>1,
set `sequence_parallel: true`. Changing inference TP changes its data-parallel
degree and aggregate request capacity; account for both when comparing runs.

| Destination storage | Publication behavior |
|---|---|
| Complete contiguous local parameter view, matching dtype | Aliases learner storage; no weight copy |
| Local strided view or packed fragments | Copies into existing inference storage |
| Any required fragment on another learner GPU | Native learner-world reshard writes into existing inference storage |
| Compatible local expert stack | Retains shared packed expert storage |
| Persistent buffer | Remains private and is refreshed during publication |

One planner classifies destinations using Megatron's resolved global tensor names
and logical shard coordinates. Equal names and shapes alone do not establish
compatibility. A destination requiring remote fragments remains wholly private;
the current backend does not execute a GEMM from pieces resident on several GPUs.
Aligned TP/ETP/EP increases use a deterministic permutation when the joint
dense/expert ownership signatures admit a complete local assignment. CP replica
ranks participate in the assignment. Other layouts use normal placement and
residual transfers; partial-coverage placements are not yet optimized globally.

Residual transfers run only among learner ranks, with one NCCL participant per
GPU. Each paired learner maps the inference-owned receive tensors through CUDA
IPC; it does not allocate another full model. All inference readers are paused
and drained before publication. If inference repacks a private tensor, its new
IPC handle is installed before the next transfer. Imported receive buffers are
released before their inference owners shut down. Source storage changes fail
closed. Every rank finishes copying and cache refresh before inference resumes.

The native planner needs [Megatron-LM #9465](https://github.com/NVIDIA/Megatron-LM/pull/9465):
optional whole-mesh transfer descriptions must reject an incomplete destination
mesh before indexing its first rank. The generic transfer schedule then handles
residual-only destinations normally. This is a companion Megatron fix, not a
runtime patch in NeMo RL.

Validation is staged. CPU tests cover native shard planning, both TP directions,
packed projections, expert/pipeline naming, rank permutations and publication
failure ordering. The earlier GPU IPC test covered TP1-to-TP2 local views only.
The general distributed TP and reverse-IPC tests are pending; full GPT/Hybrid
model execution, CP attention, pipeline training and end-to-end performance are
not yet established for these new combinations. No speedup is implied by sharing
coverage or a small projection test.
