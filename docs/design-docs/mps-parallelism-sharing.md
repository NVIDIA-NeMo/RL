# MPS parallelism sharing

Megatron training and inference run in separate processes under CUDA MPS. Each inference GPU also hosts a learner. Training and inference can use different model-parallel layouts and share compatible weights through CUDA IPC.

## GPU placement and scheduling

The eight-GPU Nano recipe uses TP1/EP4 for both roles. By default, all eight GPUs run both processes. To train on all eight and run one four-GPU inference group, use

```bash
bash examples/run_nemotron_nano_mps.sh 'policy.generation.colocated.engine_gpu_indices=[4,5,6,7]'
```

`learner_gpu_indices` and `engine_gpu_indices` specify each role’s logical GPU order within the same topology-sorted parent pool. A null list selects the entire parent pool independently for that role. Every engine GPU must also be a learner GPU, and each role must fill complete dense and expert model-parallel groups. For 16 learners and eight inference workers on the last eight GPUs, leave the learner list null and select engine indices 8 through 15. Reversing those selections fails validation. The conventional disjoint setup uses its existing resource split with `engine_sm_fraction` unset.

The default `mps_gemm` backend gives both processes full SM access. The Nano recipe sets `engine_sm_fraction: 0.0`, giving learner GEMMs full access as well. With an explicit override of `0.42`, learner Transformer Engine grouped GEMMs on shared GPUs target approximately 58% of the SMs. Other kernels and learner-only GPUs retain full access. The `mps_percentage` backend applies a process-wide learner cap. Startup verifies uncapped learner registration with the MPS daemon. The Nano recipe sets the learner’s MPS client priority to low (`1`) and inference to normal (`0`).

Set inference parallelism in `policy.generation.mcore_generation_config`. Inference requires CP1 and PP1. The `inference_optimized` experts require ETP1, and TP greater than one requires sequence parallelism with that implementation. Training CP greater than one requires `policy.sequence_packing.enabled: true` and `policy.megatron_cfg.use_fused_linear_logprobs: false`. Changing inference TP also changes data parallelism and aggregate rollout capacity. Streaming GRPO fits dispatches to the learner’s data-parallel and microbatch sizes. Forest training pads execution rows within each prompt group.

## Publishing learner weights to inference

With `refit_transport: ipc` and `engine_weights: shared`, each inference worker opens its paired learner’s weight handles before CUDA graph capture. Megatron’s global tensor names and logical shard coordinates determine how each inference tensor gets its weights.

| Inference tensor | How it receives updated weights |
|---|---|
| Complete contiguous local parameter with matching dtype | Reads learner memory directly |
| Compatible packed expert stack | Reads the learner’s packed experts directly |
| Local strided view or packed fragments | Copies into existing inference storage |
| Tensor requiring a remote shard | Learner ranks assemble it in the engine’s private storage using NCCL and CUDA IPC |
| Persistent buffer | Refreshes the engine’s private copy |

Inference pauses and drains active GPU work before the optimizer writes shared parameters. It resumes after every engine rank completes its copies and refreshes derived caches. If inference repacks a private tensor, its learner imports the replacement handle before the next transfer. A change to exported learner storage stops publication. Learners release imported buffers before their inference owners shut down.

Where dense and expert ownership permit it, the planner reorders ranks to keep aligned TP/ETP/EP increases local, including across CP replicas. Otherwise it retains the normal placement. Remote shards use Megatron’s point-to-point transfer schedule with the original TP coordinates and task and batch IDs. Every learner joins the communicator, including learner-only GPUs and ranks with no transfers.

All workers must use the same Megatron-Core build. NeMo RL asks Core’s `build_plan_from_rosters` to plan the private copies. On Bridge’s pinned Core (`e0a4bfed`), NeMo RL falls back to Core’s point-to-point schedule if the planner fails because the first TP shard is already shared. Core `a463e854` handles that case directly. GPU runs with remote shards require the NCCL communicator reuse fix on affected PyTorch versions. [Megatron-LM #9485](https://github.com/NVIDIA/Megatron-LM/pull/9485)

## Validation and limits

| Coverage | Result |
|---|---|
| CPU at NeMo RL `93770197` | 1,449 tests passed on stock Core `e0a4bfed`. Seven GPU cases and one optional NeMo Gym case skipped. Sixty focused placement, scheduling, port and refit tests passed on both that pin and Core `a463e854`, including coverage of the 16-learner/eight-inference placement. |
| Distributed component tests | Twelve layouts on four ranks completed two publications each. Separate-process IPC tests checked values and storage lifetime in both TP2-to-TP4 and TP4-to-TP2 directions, including clean teardown. |
| Eight-GPU Nano Math GRPO at NeMo RL `8af83078` | Training TP4/CP2/EP4/ETP1, inference TP2/CP1/EP4/ETP1, PP1 for both. Concurrent inference CUDA graphs, sequence packing and a GPU-only optimizer completed three updates with 192 accepted sequences, finite nonzero gradients, mixed-sign advantages and four weight publications. Both nodes completed process cleanup without IPC or NCCL teardown warnings. |

The full-model run used Core `3e95097b` with fixes for two MTP prefill helpers, per-forward MoE metadata, partial-destination planning and NCCL communicator initialization. It qualified functional execution for that layout. Full-model per-tensor parity, convergence and throughput require separate measurements. Earlier Ultra training CP8 to inference CP1 evidence uses matching TP. Training ETP greater than one, training PP and changed-EP full-model layouts still need GPU qualification.

The implementation supports one model chunk per worker and matching refit dtypes. It rejects virtual pipeline chunks, GTP rematerialization and quantized refit storage.
