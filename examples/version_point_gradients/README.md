# Version-point gradients: precision and lag comparisons

These scripts reproduce the setup of the Qwen3-30B-A3B-Base async GRPO comparison on DAPOMath17K, with AIME 2024 validation. Ordinary BF16 and NVFP4 arms use 32 Blackwell GPUs (16 trainer and 16 rollout). BF16 VPG controls use 20 GPUs (4 trainer and 16 rollout), with full activation recomputation to fit the smaller trainer group. They contain no scheduler accounts, internal hosts, or shared filesystem paths.

| Script | Expert forward | Maximum trajectory age | VPG | Batch invariant |
| --- | --- | ---: | --- | --- |
| `bf16_lag1.sh` | BF16 | 1 | off | off |
| `bf16_lag4.sh` | BF16 | 4 | off | off |
| `bf16_vpg_lag1.sh` | BF16 | 1 | on | on |
| `bf16_vpg_lag4.sh` | BF16 | 4 | on | on |
| `nvfp4_vpg_lag1.sh` | Per-token NVFP4 | 1 | on | on |
| `nvfp4_vpg_lag4.sh` | Per-token NVFP4 | 4 | on | on |

Ordinary BF16 uses token importance sampling with TIS cap 2. NVFP4 applies to routed-expert forward computation; its surrogate backward recomputes in BF16 through TransformerEngine. Parameters remain BF16 and gradient reductions use FP32. BF16 VPG enables fused SwiGLU to match inference rounding. All GEMMs use TransformerEngine or FlashInfer; neither DeepGEMM nor a Triton GEMM backend is selected.

The BF16 VPG controls keep the same global batch size, sampling, optimizer settings, and loss while reserving fewer trainer GPUs for this generation-bound workload. Their data-parallel gradient-reduction order differs from the 16-trainer arms. Full activation recomputation is a memory/compute tradeoff.

The original BF16 versus NVFP4 comparison changes precision, VPG, and batch invariance together. Compare BF16 VPG against NVFP4 VPG at the same lag to separate those effects, and compare matched steps and multiple seeds. This current-main port has not completed a convergence reproduction; its startup gates must pass in the prepared runtime before a long run starts.

## Prepare the runtime

`provenance.json` records the original experiment commits and dependency versions. The Megatron implementation of batch invariance and matched NVFP4 experts is an external prerequisite, separate from this NeMo-RL feature. Check out `runtime.megatron_lm_base_commit` from the recorded Megatron repository and apply `patches/megatron-optimizer.patch` with `git apply`; this reconstructs the local optimizer fixes used by every study arm. Prepare the recorded Bridge version, TransformerEngine 2.19.0, FlashInfer 0.7.2.post1, and CuTe DSL 4.6.2 on all nodes. The repository's dependency lock alone does not reproduce this experimental runtime. Make the checkout, Python environment, model/data caches, and output directory available at the same paths on every node.

For BF16 VPG, apply `patches/megatron-native-te-bf16.patch` after the optimizer patch, using a separate Megatron checkout. The native TE grouped-GEMM adapter is qualified on SM100 with cuBLASLt 13.5.1 and rejects other hardware/library combinations. Add the selected Megatron checkout and Bridge `src/` to `PYTHONPATH`. Runtime fields without a first-class NeMo-RL key use `model_overrides`, with the matched-expert training mode explicitly disabled on generation. The high-level inference selector is named `torch`; `batch_invariant_backend=te_native` directs its GEMMs through TE.

Allocate five nodes for a BF16 VPG control or eight nodes for another arm, with four GPUs per node. Start a separate Ray cluster for each experiment:

```bash
# Head node:
ray start --head --port=6379 --num-gpus=4 --disable-usage-stats
# Each remaining node:
ray start --address="$HEAD_ADDRESS:6379" --num-gpus=4 --disable-usage-stats
```

## Run an arm

Invoke one script on the head node from the prepared runtime:

```bash
export OUTPUT_ROOT="$PWD/results/version_point_gradients"
export PYTHON="$(command -v python3)"
bash examples/version_point_gradients/bf16_vpg_lag1.sh
# Use another cluster/output arm for the matched NVFP4 run:
bash examples/version_point_gradients/nvfp4_vpg_lag1.sh
```

Each script first runs seed 901 for a startup gate, then seed 1 for 1,500 training steps. Every gate checks finite loss and gradient norm. BF16 baselines require five steps with VPG off. VPG gates require independently computed first-step rollout/trainer log-probs to match exactly, VPG applied on every step, zero parity error, and no missing snapshots. Lag 1 uses five steps; lag 4 uses six and verifies that age reaches four, including a post-refit batch. The extra steps validate startup behavior and do not change the long-run step budget.

```bash
DRY_RUN=1 bash examples/version_point_gradients/nvfp4_vpg_lag4.sh
PHASE=smoke bash examples/version_point_gradients/bf16_vpg_lag4.sh
PHASE=train bash examples/version_point_gradients/bf16_vpg_lag4.sh
```

`PHASE=train` requires a successful gate marker. Reuse the same output directory to resume from checkpoints; use a new directory after changing configuration or runtime. Extra arguments are forwarded as config overrides to both phases. Set `SEED` for another training seed.

Checkpoints are saved every 25 steps and at a 3h40m runtime budget. For four-hour allocations, submit the same script again on a fresh Ray cluster while preserving outputs. Scheduler dependencies and cleanup belong to the caller. Study-only rho probes and paper-specific diagnostics are excluded from this port; the recipes preserve the training and sampling settings. See `provenance.json` for the mapping from the study's configuration names.
