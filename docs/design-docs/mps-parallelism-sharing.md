# MPS parallelism sharing

This experimental path places a Megatron inference process on each selected
learner GPU. The default places both roles on every GPU. Training and inference can have different tensor and expert layouts while
sharing compatible local parameter storage. The inference backend still requires
CP1 and PP1; `inference_optimized` experts require ETP1. Training CP and PP
are represented by native shard metadata, with one model chunk per worker. Virtual
pipeline chunks, GTP rematerialization, quantized refit tensors, differing tensor
dtypes, and inference GPUs outside the learner pool are rejected.

The default `sm_partition_backend: mps_gemm` gives both processes full SM access. `engine_sm_fraction` controls only the learner Transformer Engine grouped-GEMM margin: `0.42` targets approximately 58% of the SMs for those GEMMs, while other kernels can use the whole GPU. This hint does not reserve SMs. Use `mps_percentage` explicitly to cap the learner process instead. A zero fraction leaves both processes and learner GEMMs uncapped. Startup verifies that the uncapped learner is registered with the MPS daemon. The Nano recipe gives the learner lower MPS client priority (`1`) than inference (`0`), matching the benchmarked scheduling setup.

To run the eight-GPU Nano recipe with eight learner workers and four inference workers, keep its TP1/EP4 settings and choose a complete four-GPU inference group:

```bash
bash examples/run_nemotron_nano_mps.sh 'policy.generation.colocated.engine_gpu_indices=[4,5,6,7]'
```

`learner_gpu_indices` and `engine_gpu_indices` are ordered indices in the same topology-sorted physical GPU pool. Each index identifies a GPU across the full allocation. Each list orders that role’s distributed ranks. A null list uses the full pool. The engine list must be contained in the learner list, and each role must fill complete dense and expert model-parallel groups. For a 16-GPU pool, choosing engine indices 8 through 15 and leaving the learner list null places eight inference workers alongside 16 learner workers. With `mps_gemm`, only shared GPUs receive the learner GEMM margin. Learner-only GPUs can still send remote shards during publication. The conventional disjoint setup uses its existing resource split with `engine_sm_fraction` unset.

Streaming GRPO aligns prompt-group dispatches to the learner data-parallel and static microbatch sizes so every learner shard receives complete work. Forest training retains its execution-row padding rules.

Use the existing MPS configuration with `refit_transport: ipc` and
`engine_weights: shared`. Set inference parallelism in
`policy.generation.mcore_generation_config`. With `inference_optimized` and TP>1,
set `sequence_parallel: true`. Changing inference TP changes its data-parallel
degree and aggregate request capacity; account for both when comparing runs.
Training CP greater than one requires `policy.sequence_packing.enabled: true` and `policy.megatron_cfg.use_fused_linear_logprobs: false`, as enforced by native Megatron setup. Inference remains CP1.

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

Residual transfers use Megatron's native reshard planner and NCCL copy service. Every learner rank participates in communicator setup, including ranks without transfers. All workers must use the same Megatron-Core build.

The public Nano recipe uses matching TP1/EP4 layouts on eight GPUs. For mixed layouts, the co-resident planner builds point-to-point sends and receives from Megatron’s native transfer schedule. It retains the original TP coordinates while omitting destinations already bound locally. Both ends use the same native task and batch IDs.

The Megatron-Bridge revision `ae0a1b02` pins Megatron-Core `e0a4bfed`. This pin supports residual point-to-point planning, including TP2-to-TP4 partial meshes. Its NCCL copy service still uses private eager initialization on the learner’s existing process group. For GPU runs with remote shards, use a Core build containing the upstream communicator reuse fix to avoid leaking a communicator on affected PyTorch versions. [Megatron-LM #9485](https://github.com/NVIDIA/Megatron-LM/pull/9485)

Validation is staged. CPU tests cover native shard planning, both TP directions,
packed projections, expert/pipeline naming, rank permutations and publication
failure ordering. The earlier GPU IPC test covered TP1-to-TP2 local views only.
The native distributed projection gate passed 12 layouts on four ranks with two
publications each, including both TP directions and packed projections. The
separate-process reverse-IPC gate passed TP2-to-TP4 and TP4-to-TP2 numerical and
storage-lifetime checks. The corrected copy service also passed owned-process
cleanup: all consumer imports were released, all native process groups were
destroyed, and preserved logs contained no CUDA IPC producer-lifetime or NCCL
communicator teardown warnings. These small gates serialize the two worker
worlds; they do not exercise concurrent DynamicInferenceEngine, CP2 or GRPO
lifecycle behavior.

An eight-GPU Nano full-model qualification passed with training TP4/CP2/EP4/ETP1 and inference TP2/CP1/EP4/ETP1, with PP1 in both roles.
Native SingleController completed three Math GRPO updates with 192 accepted sequences, finite nonzero gradients and mixed-sign advantages, using sequence packing and a GPU-only optimizer configuration.
The run exercised concurrent DynamicInferenceEngine graphs, generation, learner forward/backward and optimizer updates, and four native refits combining local aliases with remote residual transfers.
Both nodes completed owned-process cleanup without CUDA IPC producer-lifetime or NCCL communicator teardown warnings, and the allocation completed successfully.

That GPU qualification ran NeMo RL revision `8af83078` with Megatron-Core base `3e95097b` plus five reviewed native file corrections: the two MTP prefill helpers, per-forward MoE metadata, partial-destination planning, and NCCL communicator initialization.
The full-model GPU result applies to revision `8af83078`. With the residual point-to-point planner, 1,213 focused CPU tests passed on the unmodified Core pin, and seven GPU tests skipped. The shard-planning and TP suite also passed on Core `a463e854` (52 passed, one GPU test skipped).
The test establishes functional mixed-TP/training-CP behavior for the stated layout, not full-model per-tensor numerical parity, trajectory equivalence, convergence or performance.

Existing Ultra training CP8 to inference CP1 evidence uses matching TP and covers a different layout.
Inference CP is not a supported target; inference remains CP1 and PP1, and inference-optimized experts remain ETP1.
Training ETP greater than one, pipeline training, arbitrary PP changes and changed-EP full-model layouts remain unqualified by these GPU tests.
Native metadata and CPU planning coverage do not establish those runtime cases.
No speedup is implied by sharing coverage or functional qualification.
