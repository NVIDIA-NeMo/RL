# CuTeDSL and Refit-Safe GLU Implementation Plan

**Required execution skill:** Use `superpowers:test-driven-development` for each implementation task and `superpowers:verification-before-completion` before claiming any gate passed.

**Goal:** Enable the Transformer Engine CuTeDSL fused grouped MLP path for NeMo-RL Megatron policy training while preserving correct Qwen3-MoE expert FC1 weights across HF import, Megatron training, checkpointing, and rollout refit.

**Architecture:** Megatron-Bridge owns conversion between contiguous HF `[gate; up]` tensors and Megatron's block-interleaved expert FC1 layout. NeMo-RL exposes and validates the two required MCore configuration fields, supplies Cutlass DSL in the `mcore` dependency group, and enables the kernel through policy-worker environment variables. The public `Policy.train()` and refit APIs remain unchanged.

**Stack:** Python 3.12, PyTorch, Megatron Core, Megatron-Bridge, Transformer Engine, Cutlass DSL, pytest, uv, Ray, SLURM/Pyxis, GB200.

## Supported first slice

- Model: `Qwen/Qwen3-30B-A3B`
- Policy backend: Megatron
- Precision: MXFP8
- Topology: TP=1, PP=1, CP=1, ETP=1, EP=4
- Training: synchronous GRPO policy update, fixed sequence length, sequence packing disabled
- Refit: Megatron policy weights exported through `AutoBridge.export_hf_weights()` to vLLM
- Hardware: GB200/SM100; OCI-HSG is the first GPU gate
- Kernel constraints: grouped GEMM enabled, TE op fuser enabled, expert GLU interleave size 32, cuDNN frontend 1.23+, and Cutlass DSL 4.5.2+

Full-iteration CUDA Graph and A2A overlap are separate follow-up plans. This change must not claim host-free full-iteration execution by itself.

## Task 1: Add a tested reusable GLU layout conversion utility in Megatron-Bridge

**Files:**

- Create: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src/megatron/bridge/models/conversion/glu_interleave.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/tests/unit_tests/models/test_param_mapping.py`

### Step 1: Write failing utility tests

Add parametrized tests for 1D bias and 2D weight tensors that prove:

- contiguous `[gate; up]` becomes `[gate_0, up_0, gate_1, up_1, ...]` in blocks of 32;
- de-interleave is the exact inverse;
- the input tensor is not mutated;
- a scalar tensor raises `ValueError`;
- a non-positive interleave size raises `ValueError`;
- a dim-0 size not divisible by `2 * interleave_size` raises `ValueError` with the actual sizes.

Use deterministic `torch.arange` tensors rather than random tensors for layout assertions.

### Step 2: Run the focused tests and confirm RED

Run from the Bridge submodule:

```bash
uv run pytest tests/unit_tests/models/test_param_mapping.py -k 'glu_interleave' -q
```

Expected: collection or import failure because `glu_interleave.py` and its functions do not exist.

### Step 3: Implement the minimal typed utility

Implement:

```python
def interleave_glu_tensor(
    tensor: torch.Tensor,
    interleave_size: int,
) -> torch.Tensor:
    ...


def deinterleave_glu_tensor(
    tensor: torch.Tensor,
    interleave_size: int,
) -> torch.Tensor:
    ...
```

Validate rank is at least one, `interleave_size > 0`, and dim 0 is divisible by `2 * interleave_size`. Preserve trailing dimensions, dtype, and device.

### Step 4: Run the focused tests and confirm GREEN

```bash
uv run pytest tests/unit_tests/models/test_param_mapping.py -k 'glu_interleave' -q
```

Expected: all new utility tests pass.

## Task 2: Make Bridge `GatedMLPMapping` refit-safe for interleaved routed experts

**Files:**

- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src/megatron/bridge/models/conversion/param_mapping.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/tests/unit_tests/models/test_param_mapping.py`

### Step 1: Write failing TP=1 mapping tests

Add tests using an expert parameter name and a mock module whose config has `moe_mlp_glu_interleave_size = 32`:

- HF→Megatron returns an interleaved local FC1 tensor;
- Megatron→HF de-interleaves before splitting and reconstructs the original gate/up tensors;
- weight and bias round trips are exact;
- a non-expert mapping ignores the MoE interleave setting;
- absent or `None` configuration preserves the current contiguous behavior.

Patch EP gather only where needed so the test targets layout conversion rather than distributed collectives.

### Step 2: Write failing TP=2 mapping tests

Prove that interleave/de-interleave happens per TP-local shard:

1. Split gate and up independently across TP.
2. Concatenate the matching local gate/up shard.
3. Interleave that local fused tensor.
4. On export, de-interleave each gathered local shard before reconstructing global gate/up tensors.

This prevents block boundaries from crossing TP shards.

### Step 3: Run the mapping tests and confirm RED

```bash
uv run pytest tests/unit_tests/models/test_param_mapping.py -k 'GatedMLPMapping and interleave' -q
```

Expected: assertions fail because `GatedMLPMapping` currently concatenates and chunks contiguous tensors directly.

### Step 4: Implement interleave-aware mapping

Add one private typed helper that returns the active interleave size only when all of these are true:

- the resolved Megatron parameter is a routed expert parameter;
- `megatron_module` is available;
- `megatron_module.config.moe_mlp_glu_interleave_size` is present and not `None`.

If a configured value is not a positive integer, raise `ValueError`; do not treat an invalid request as feature-off.

Apply conversion at these exact points:

- TP=1 HF→Megatron: after concatenating gate/up;
- TP>1 HF→Megatron: after constructing each local `[gate_shard; up_shard]` tensor and before scatter;
- TP=1 Megatron→HF: after dequantization and before `torch.chunk`;
- TP>1 Megatron→HF: independently on each gathered shard and before `torch.chunk`.

Do not mutate source tensors. Do not alter dense or shared-expert mapping semantics in this PR. Keep the quantized-export method outside the initial path unless a failing test proves it is used by NeMo-RL's normal MXFP8 refit.

### Step 5: Run focused and full mapping tests

```bash
uv run pytest tests/unit_tests/models/test_param_mapping.py -q
```

Expected: existing contiguous mapping tests and all new interleaved mapping tests pass.

## Task 3: Reuse the GLU utility in Bridge checkpointing without behavior drift

**Files:**

- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src/megatron/bridge/training/checkpointing.py`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/tests/unit_tests/training/test_checkpointing.py`

### Step 1: Locate or add checkpoint regression coverage

Add a focused test that passes an expert FC1 weight and bias through `_process_state_dict_for_glu_interleaving()` in both directions and proves exact round-trip equality. Retain the private checkpoint helper names because existing callers and tests may import them.

### Step 2: Run the checkpoint test before refactoring

```bash
uv run pytest tests/unit_tests/training/test_checkpointing.py -k 'glu_interleave' -q
```

Expected: existing behavior passes. This is a characterization test, not a RED test.

### Step 3: Replace duplicate implementations with imported aliases

Import the reusable functions as:

```python
from megatron.bridge.models.conversion.glu_interleave import (
    deinterleave_glu_tensor as _deinterleave_glu_tensor,
    interleave_glu_tensor as _interleave_glu_tensor,
)
```

Remove only the duplicated function bodies. Keep checkpoint call sites unchanged.

### Step 4: Run checkpoint and mapping regression tests

```bash
uv run pytest \
  tests/unit_tests/models/test_param_mapping.py \
  tests/unit_tests/training/test_checkpointing.py \
  -q
```

Expected: all selected Bridge tests pass.

### Step 5: Commit the Bridge change

Create a Bridge branch from pinned `554c7b9324225aa863eee52e8b8fdde7abced2b1`, then commit only Bridge files:

```bash
git switch -c sna/glu-interleave-auto-bridge-20260710
git add src/megatron/bridge/models/conversion/glu_interleave.py \
  src/megatron/bridge/models/conversion/param_mapping.py \
  src/megatron/bridge/training/checkpointing.py \
  tests/unit_tests/models/test_param_mapping.py \
  tests/unit_tests/training/test_checkpointing.py
git commit -s -m "fix: preserve interleaved GLU layout in model conversion"
```

Expected: one signed, independently reviewable Bridge commit. Record its SHA for the parent submodule update.

## Task 4: Expose and validate CuTeDSL configuration in NeMo-RL

**Files:**

- Modify: `nemo_rl/models/policy/__init__.py`
- Modify: `nemo_rl/models/megatron/setup.py`
- Modify: `tests/unit/models/megatron/test_megatron_setup.py`
- Modify: `examples/configs/grpo_math_1B.yaml`

### Step 1: Write failing config pass-through tests

Extend the MoE setup tests to prove:

- `use_transformer_engine_op_fuser=True/False` is copied exactly when present;
- `moe_mlp_glu_interleave_size=32` is copied exactly when present;
- absent optional keys do not overwrite upstream model-config defaults.

### Step 2: Write failing requested-feature validation tests

When `NVTE_CUTEDSL_FUSED_GROUPED_MLP` is exactly `"1"`, setup must fail early with a field-specific `ValueError` if any of these are missing or incompatible:

- `moe_grouped_gemm is True`;
- `use_transformer_engine_op_fuser is True`;
- `moe_mlp_glu_interleave_size == 32`;
- `expert_tensor_parallel_size == 1`;
- FP8 is enabled with the MXFP8 recipe.

Also prove no CuTeDSL-specific validation runs when the env var is absent or `"0"`. Hardware capability, cuDNN version, and tensor dimensions remain runtime checks because they are not fully known at config parsing time.

### Step 3: Run tests and confirm RED

```bash
uv run --group test pytest \
  tests/unit/models/megatron/test_megatron_setup.py \
  -k 'cutedsl or op_fuser or glu_interleave' \
  -q
```

Expected: missing TypedDict/setup behavior and validation cause failures.

### Step 4: Implement schema, mapping, and fail-fast validation

Add these optional fields to `MegatronConfig`:

```python
use_transformer_engine_op_fuser: NotRequired[bool]
moe_mlp_glu_interleave_size: NotRequired[int | None]
```

Map only present keys in `_apply_moe_config()`. Add a typed validation helper called during model config setup before expensive model construction. The helper must report every invalid requested precondition in one actionable message and must not silently disable the feature.

In `examples/configs/grpo_math_1B.yaml`, document legacy-safe defaults:

```yaml
use_transformer_engine_op_fuser: false
moe_mlp_glu_interleave_size: null
```

### Step 5: Run focused and setup regression tests

```bash
uv run --group test pytest tests/unit/models/megatron/test_megatron_setup.py -q
```

Expected: all setup tests pass with feature-off behavior unchanged.

## Task 5: Put Cutlass DSL in the MCore runtime dependency set

**Files:**

- Modify: `pyproject.toml`
- Modify: `uv.lock`

### Step 1: Add the dependency through uv

Run:

```bash
uv add --optional mcore 'nvidia-cutlass-dsl[cu13]>=4.5.2'
uv lock
```

Do not use pip and do not hand-edit lockfile package records.

### Step 2: Verify dependency resolution and import

```bash
uv lock --check
uv tree --extra mcore | rg 'nvidia-cutlass-dsl'
```

Expected: lock consistency succeeds and Cutlass DSL is reachable from the MCore extra. Perform the actual `import cutlass` check inside the Linux GB200 container in Task 8 rather than attempting to install CUDA-only runtime packages on the local macOS host.

### Step 3: Run packaging checks

```bash
uv run --group test pytest tests/unit/models/megatron/test_community_import.py -q
```

Expected: community/import isolation tests pass.

## Task 6: Add a reproducible Qwen3-30B-A3B CuTeDSL smoke recipe

**Files:**

- Create: `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-1n4g-megatron-mxfp8-cutedsl.yaml`
- Create: `tests/test_cutedsl_policy_recipe.py`

### Step 1: Write a failing recipe contract test

Load the recipe through the same OmegaConf/default resolution path used by examples and assert:

- model is `Qwen/Qwen3-30B-A3B`;
- cluster is one node with four GPUs;
- TP=1, PP=1, CP=1, ETP=1, EP=4;
- MXFP8, grouped GEMM, op fuser, interleave 32, and the CuTeDSL env var are enabled;
- sequence packing and dynamic batching are disabled;
- policy train microbatch is one and the configured policy global batch produces at least four training microbatches;
- the run is short and checkpoint output/log paths are recipe-specific.

### Step 2: Run and confirm RED

```bash
uv run --group test pytest tests/test_cutedsl_policy_recipe.py -q
```

Expected: failure because the recipe does not exist.

### Step 3: Add the minimal fixed-shape smoke recipe

Inherit from `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-mxfp8-rollout.yaml`. Override only the settings required for a one-node four-GPU functional/refit smoke. Keep generation/refit enabled so this validates the conversion path, not only a standalone training forward.

Use exact settings:

```yaml
grpo:
  num_prompts_per_step: 2
  num_generations_per_prompt: 2
  max_num_steps: 3
policy:
  model_name: Qwen/Qwen3-30B-A3B
  train_global_batch_size: 4
  train_micro_batch_size: 1
  max_total_sequence_length: 1024
  dynamic_batching:
    enabled: false
  sequence_packing:
    enabled: false
  megatron_cfg:
    tensor_model_parallel_size: 1
    pipeline_model_parallel_size: 1
    context_parallel_size: 1
    expert_tensor_parallel_size: 1
    expert_model_parallel_size: 4
    moe_grouped_gemm: true
    use_transformer_engine_op_fuser: true
    moe_mlp_glu_interleave_size: 32
    fp8_cfg:
      enabled: true
      fp8: e4m3
      fp8_recipe: mxfp8
      fp8_param: false
    env_vars:
      NVTE_CUTEDSL_FUSED_GROUPED_MLP: "1"
cluster:
  num_nodes: 1
  gpus_per_node: 4
```

Use the repository's existing `fp8_cfg` schema shown above. The inherited generation `vllm_cfg.precision: fp8` and `is_mx: true` settings remain enabled for rollout refit.

### Step 4: Run recipe and config regression tests

```bash
uv run --group test pytest \
  tests/test_cutedsl_policy_recipe.py \
  tests/unit/models/megatron/test_megatron_setup.py \
  -q
```

Expected: recipe resolves and the CuTeDSL contract passes.

## Task 7: Pin Bridge and finish local verification

**Files:**

- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge` (submodule pointer)
- Modify: all NeMo-RL files from Tasks 4–6

### Step 1: Confirm only intended changes exist

From the NeMo-RL worktree:

```bash
git status --short
git diff --submodule=log --check
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge status --short --branch
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge log -1 --show-signature --format=fuller
```

Expected: the Bridge worktree is clean on its feature commit, and the parent shows only the intended submodule pointer plus NeMo-RL files.

### Step 2: Run the focused local suite

```bash
uv run --group test pytest \
  tests/unit/models/megatron/test_megatron_setup.py \
  tests/unit/models/megatron/test_community_import.py \
  tests/test_cutedsl_policy_recipe.py \
  -q
uv run pre-commit run --all-files
```

Expected: all tests and hooks pass. If `pre-commit --all-files` reports unrelated baseline failures, rerun hooks on the changed file list and record both outputs rather than modifying unrelated files.

### Step 3: Commit the NeMo-RL change

```bash
git add pyproject.toml uv.lock \
  nemo_rl/models/policy/__init__.py \
  nemo_rl/models/megatron/setup.py \
  tests/unit/models/megatron/test_megatron_setup.py \
  tests/test_cutedsl_policy_recipe.py \
  examples/configs/grpo_math_1B.yaml \
  examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-1n4g-megatron-mxfp8-cutedsl.yaml \
  3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
git commit -s -m "perf: enable CuTeDSL grouped MLP for policy training"
```

Expected: one signed NeMo-RL feature commit referring to the independently reviewable Bridge commit.

### Step 4: Push both feature branches

Push Bridge first, then NeMo-RL. Verify remote branch tips match local SHAs. Do not submit a cluster job from an unpushed commit.

## Task 8: Validate the first GPU gate on OCI-HSG

**Files:**

- Create on the branch before submission: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/README.md`
- Create on the branch before submission: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/submit_oci_hsg.sh`
- Generated remotely: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/results/$RUN_ID/metadata.json`
- Generated remotely: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/results/$RUN_ID/slurm.out`
- Generated remotely: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/results/$RUN_ID/metrics.json`

### Step 1: Add and review the submission wrapper

The wrapper must:

- request one node, four GB200 GPUs, account `nemotron_n3_post`, partition `batch`, and the required `--gres=gpu:4`;
- use `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260707.sqsh` first;
- bind the pushed repository checkout and a run-specific results directory;
- record git/submodule SHAs, image path and SHA256, Python/PyTorch/CUDA/cuDNN/NCCL/TE/Cutlass versions, topology, and effective config;
- run `python -c 'import cutlass; from cutlass import cute; print(cutlass.__file__)'` inside the container, followed by a four-GPU PyTorch/TE import and device smoke before launching GRPO;
- run only 2–5 policy updates;
- preserve logs under the experiment result directory.

Commit and push the wrapper before any submission.

### Step 2: Check scheduling without consuming GPUs

On OCI-HSG, check out the pushed branch, run `git pull --ff-only`, synchronize recursive submodules, and then run:

```bash
sbatch --test-only experiments/cutedsl_qwen3_30ba3b_oci_1n4g/submit_oci_hsg.sh
```

Expected: SLURM accepts the request and reports a feasible start estimate. Resolve account, partition, or resource errors before submission.

### Step 3: Submit and monitor for at least five minutes

```bash
sbatch experiments/cutedsl_qwen3_30ba3b_oci_1n4g/submit_oci_hsg.sh
squeue -j "$JOB_ID" -o '%.18i %.2t %.10M %.6D %R'
```

Poll SLURM state and tail the run-specific log frequently for at least five minutes. Cancel only for a confirmed unrecoverable error, not normal model download or compilation time.

### Step 4: Classify failures before changing the image

Treat these as image/runtime failures:

- missing Cutlass DSL, TE, cuDNN frontend, CUDA, or incompatible binary symbols inside the container;
- container import/mount failure;
- runtime version below the documented CuTeDSL requirement.

Treat config validation, model shape, refit tensor mismatch, loss divergence, and Python exceptions in changed code as code failures. Fix code locally using the systematic debugging workflow, commit, push, pull once on the cluster, and resubmit.

### Step 5: Stage a new nightly only for an image/runtime failure

If and only if Step 4 identifies the image as the cause, use `e2etrain:stage-training-containers` to stage current `nvcr.io/nvidian/nemo-rl:nightly` on OCI-HSG under an immutable dated filename. Generate provenance metadata and SHA256, run the skill's GPU smoke, update the experiment metadata, and retry with the verified image. Do not overwrite `nemo_rl_nightly_20260707.sqsh`.

### Step 6: Validate correctness and kernel use

The first GPU gate passes only when all are true:

- at least two successful policy optimizer updates complete;
- Megatron→HF export and rollout refit complete without shape/layout error;
- post-refit rollout logits match an eager/non-interleaved reference within an agreed MXFP8 tolerance;
- loss and gradients are finite;
- TE debug output or an Nsight Systems profile identifies the CuTeDSL fused grouped-MLP kernel rather than only the generic op-fuser path;
- the run records median post-warmup policy step time, tokens/s, and peak memory.

If kernel evidence is absent, report the run as functionally passing but performance feature activation unverified.

### Step 7: Write the experiment result and commit it

Update the experiment README with:

- job IDs and direct log paths;
- exact code and image SHAs;
- baseline vs CuTeDSL configuration;
- correctness results;
- kernel evidence;
- measured timing/memory;
- failures and resolutions.

Commit the report and small text/JSON artifacts with a signed conventional commit. Do not commit model checkpoints, full Nsight captures, or large logs; link their cluster paths instead.

## Final verification gate

Run from the NeMo-RL worktree immediately before declaring the CuTeDSL slice complete:

```bash
git status --short --branch
git diff HEAD~1 --check
git submodule status --recursive
uv lock --check
uv run --group test pytest \
  tests/unit/models/megatron/test_megatron_setup.py \
  tests/unit/models/megatron/test_community_import.py \
  tests/test_cutedsl_policy_recipe.py \
  -q
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge status --short --branch
```

Completion requires fresh command output showing a clean Bridge submodule, consistent lockfile, green focused suites, a pushed NeMo-RL commit, and the OCI correctness/kernel evidence described above. Cross-cluster replication to Lyris, Pre-Tyche, and AWS-DFW starts only after this OCI gate passes.
