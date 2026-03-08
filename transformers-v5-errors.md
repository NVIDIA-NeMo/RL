# Transformers v5 Unit Test Errors

This file tracks all unit test failures caused by the transformers v4 → v5 upgrade.
Each error category is numbered and includes a description, stack trace summary, reproduction command, and list of affected tests.

## Plan / Resumption Guide

**Branch:** `hemil/automodel-transformers-v5`

**Goal:** Run all 3 L0 unit test suites, categorize every failure, mark each failing test with `pytest.mark.skip(reason="<error category>")`, and re-run until all suites pass (with skips).

### Test suites (run in order)
1. `FAST=1 bash tests/unit/L0_Unit_Tests_Other.sh` — `unit/` (ignoring `generation/` and `policy/`)
2. `bash tests/unit/L0_Unit_Tests_Generation.sh` — `unit/models/generation/`
3. `bash tests/unit/L0_Unit_Tests_Policy.sh` — `unit/models/policy/`

### Each suite runs 5 passes (in order)
1. Default (no extra flag) — `uv run --no-sync bash -x ./tests/run_unit.sh ... --hf-gated`
2. `--mcore-only` — `uv run --extra mcore bash -x ./tests/run_unit.sh ... --hf-gated --mcore-only`
3. `--automodel-only` — `uv run --extra automodel bash -x ./tests/run_unit.sh ... --hf-gated --automodel-only`
4. `--vllm-only` — `uv run --extra vllm bash -x ./tests/run_unit.sh ... --hf-gated --vllm-only`
5. `--sglang-only` — `uv run --extra sglang bash -x ./tests/run_unit.sh ... --hf-gated --sglang-only`

### Reproduction commands
```bash
# For default pass:
cd tests && uv run --group test pytest unit/path/test.py::test_name --hf-gated -x -s

# For mcore pass:
cd tests && uv run --extra mcore pytest unit/path/test.py::test_name --hf-gated --mcore-only -x -s

# For automodel pass:
cd tests && uv run --extra automodel pytest unit/path/test.py::test_name --hf-gated --automodel-only -x -s

# For vllm pass:
cd tests && uv run --extra vllm pytest unit/path/test.py::test_name --hf-gated --vllm-only -x -s

# For sglang pass:
cd tests && uv run --extra sglang pytest unit/path/test.py::test_name --hf-gated --sglang-only -x -s
```

### Process (iterative loop for each suite)
1. Run the suite script
2. When it fails (`-x` stops at first failure):
   - Identify the failing test and error from output
   - Check if this is a new error category or matches an existing one below
   - If new: add a new `## Err N.` section with description, stack trace summary, and reproduction command
   - If existing: add the test to the existing section's list
   - Add `@pytest.mark.skip(reason="transformers-v5: <error category>")` to the failing test
   - Re-run the same script
3. Repeat until the script passes fully (all 5 passes complete)
4. Move on to the next suite

### Important notes
- **`-x` is in pytest config** (`pyproject.toml addopts`): stops at first failure
- **`--testmon` is in pytest config**: already-passed tests won't re-run, so re-runs are fast
- **`FAST=1`** on L0_Other skips `research/*/tests/unit` (use it)
- Mark tests with skip **one at a time** unless confident multiple will fail the same way
- The skip reason should match the error section name for traceability
- **Run tests outside the sandbox** (`dangerouslyDisableSandbox: true`) — tests need GPUs, Ray, and network
- **NEVER bump test tolerance/baselines without understanding the root cause.** A failing tolerance could indicate a real bug. Always investigate why the numbers changed before adjusting thresholds. Run multiple times to check variance. Consider whether the test was ever actually running on this GPU type (CI uses A100s, so FP8/Hopper tests requiring compute capability ≥ 9.0 would have been skipped on CI).
- **FP8 tests require H100+ GPUs** (compute capability ≥ 9.0). Unit test CI runs on A100s, so FP8 tests are skipped there via `pytest.skip()` in the test body. Any FP8 tolerance values in the tests may never have been validated on CI.
- **Commit each fix** as you go (with the test unskipped), so each fix is separately reviewable. Include the root cause finding in the commit message.
- **`NRL_FORCE_REBUILD_VENVS=true`**: If you change code in vLLM workers or DTensor policy workers, run with this env var to ensure remote Ray venvs have the correct code. Without it, stale cached venvs may be used.

### Progress tracker
- [x] L0_Unit_Tests_Other.sh — PASSED (no transformers-v5 failures)
- [x] L0_Unit_Tests_Generation.sh — PASSED (with skips for Err 1-4)
- [x] L0_Unit_Tests_Policy.sh — PASSED (with skips for Err 3, 5, 6, 7)
- [x] Final verification — PASSED (all 3 suites pass)

---

## Err 1. vLLM FP8 QKVParallelLinear missing `input_scale`

**Description:** When loading a model with FP8 precision via vLLM, `QKVParallelLinear` object has no attribute `input_scale`. This is a vLLM/FP8 quantization compatibility issue triggered by the transformers v5 model weight format changes.

**Error:**
```
AttributeError: 'QKVParallelLinear' object has no attribute 'input_scale'. Did you mean: 'input_size'?
torch._dynamo.exc.ObservedAttributeError: 'QKVParallelLinear' object has no attribute 'input_scale'
```

**Reproduction:**
```bash
cd tests && uv run --no-sync pytest unit/models/generation/test_vllm_generation.py::test_vllm_generation_with_hf_training_colocated[True-False-fp8-False] --hf-gated -x -s
```

**Affected tests:**
- `test_vllm_generation.py::test_vllm_generation_with_hf_training_colocated[True-False-fp8-False]`
- `test_vllm_generation.py::test_vllm_generation_with_hf_training_colocated[False-True-fp8-False]`
- `test_vllm_generation.py::test_vllm_generation_with_hf_training_non_colocated[True-False-fp8-False]`
- `test_vllm_generation.py::test_vllm_generation_with_hf_training_non_colocated[False-True-fp8-False]`
- `test_vllm_generation.py::test_vllm_weight_update_and_prefix_cache_reset[fp8-1]`
- `test_vllm_generation.py::test_vllm_weight_update_and_prefix_cache_reset[fp8-2]`

**Root cause:** Our patched `process_weights_after_loading` in `nemo_rl/models/generation/vllm/quantization/fp8.py` didn't set `layer.input_scale = None` for the block_quant dynamic activation path. vLLM's original does this (line 440 of vLLM's fp8.py), and the `apply()` forward pass accesses `layer.input_scale` at lines 461 and 519 when `block_quant=True`.

**Fix:** Added `if not hasattr(layer, "input_scale"): layer.input_scale = None` after `maybe_post_process_fp8_weight_block(layer)` in our patched function.

**Status:** FIXED — colocated FP8 tests pass (4/4). Non-colocated FP8 tests (2 tests) still fail with a separate logprob tolerance issue (avg_prob_mult_error=1.1293 > threshold 1.08, deterministic). This is a pre-existing bug: `update_weights_from_collective` in `vllm_backend.py` does NOT call `process_weights_after_loading` after loading (unlike the IPC/colocated path which does). The `weight_update_and_prefix_cache_reset` FP8 tests (2 tests) still need verification.

## Err 2. vLLM HTTP server response format mismatch

**Description:** The `test_vllm_http_server` test compares expected vs actual HTTP response JSON from the vLLM OpenAI-compatible server. The response format has changed (likely new/different fields in the chat completion response) causing assertion mismatch.

**Error:**
```
AssertionError: assert _standardize(expected_result) == _standardize(actual_result)
Differing items in choices[0] — truncated diff, likely a new field or changed structure in chat completion response.
```

**Reproduction:**
```bash
cd tests && uv run --no-sync pytest unit/models/generation/test_vllm_generation.py::test_vllm_http_server --hf-gated -x -s
```

**Affected tests:**
- `test_vllm_generation.py::test_vllm_http_server`

**Root cause:** vLLM 0.17's chat completion response dropped the `reasoning_content` field from the message object. The expected response hardcoded `"reasoning_content": None` but the actual response doesn't include it. The `_standardize` function already stripped `reasoning` but not `reasoning_content`.

**Fix:** Updated `_standardize` to pop both `reasoning` and `reasoning_content` from the message, since these are version-dependent fields.

**Status:** FIXED

## Err 3. Ray ActorAlreadyExistsError (megatron actor cleanup issue)

**Description:** When running multiple megatron-parametrized test variants in sequence, the Ray actor `lm_policy-0-0` from the first test isn't cleaned up before the second starts, causing `ActorAlreadyExistsError`. This is a test isolation issue that surfaces in the mcore pass.

**Error:**
```
ray.exceptions.ActorAlreadyExistsError: The name lm_policy-0-0 (namespace=None) is already taken.
```

**Reproduction:**
```bash
cd tests && uv run --extra mcore pytest unit/models/generation/test_vllm_generation.py::test_vllm_refit_non_colocated_update_weights --hf-gated --mcore-only -x -s
```

**Affected tests:**
- `test_vllm_generation.py::test_vllm_refit_non_colocated_update_weights[megatron-*]` (all megatron variants)
- `test_vllm_generation.py::test_vllm_generation_with_megatron_training[*]` (all variants — actor cleanup collision)
- `test_vllm_generation.py::test_vllm_megatron_pipeline_parallel`
- `test_vllm_generation.py::test_vllm_generation_with_megatron_training_moe_model[*]`
- `test_vllm_generation.py::test_vllm_megatron_weight_update_memory`
- `test_vllm_generation.py::test_vllm_megatron_weight_update_with_packing`
- `test_megatron_worker.py::test_megatron_loss_independent_of_microbatch_size`
- `test_megatron_worker.py::test_megatron_grad_norm_invariant_to_number_of_microbatches`
- `test_megatron_worker.py::test_megatron_reference_policy_functionality`
- `test_megatron_worker.py::test_megatron_checkpoint_save_kill_and_restore[*]`
- `test_megatron_worker.py::test_megatron_dpo_training`
- `test_megatron_worker.py::test_megatron_context_parallel_topk_agreement`
- `test_megatron_worker.py::test_megatron_sft_training`
- `test_megatron_worker.py::test_megatron_context_parallel_logprob_agreement`
- `test_megatron_worker.py::test_megatron_context_parallel_training_agreement`

**Root cause:** `RayWorkerGroup.shutdown(cleanup_method="shutdown")` calls the worker's `shutdown()` method via RPC but does NOT call `ray.kill()` on the actors when graceful cleanup succeeds. The named actors (e.g., `lm_policy-0-0`) remain registered in Ray's actor registry. When the next test creates actors with the same name, it fails with `ActorAlreadyExistsError`.

**Fix:** Changed `worker_groups.py` `shutdown()` to always kill actors after graceful cleanup, not just on failure. The graceful cleanup is for giving workers time to release resources, but after that, actors must be killed to release their name registrations. Changed the conditional `if force or cleanup_method is None:` to always execute the kill block.

**Status:** FIXED — all 16 skip markers removed (5 in test_vllm_generation.py, 11 in test_megatron_worker.py). Verified megatron tests pass in sequence.
- `test_megatron_worker.py::test_megatron_gradient_norm_consistency_across_parallelism`
- `test_megatron_worker.py::test_megatron_policy_flops_range_check`

## Err 4. SGLang CUDA graph CUBLAS_STATUS_EXECUTION_FAILED

**Description:** SGLang server process crashes during startup when building piecewise CUDA graphs. The error is `CUBLAS_STATUS_EXECUTION_FAILED` in cublasGemmEx. This may be caused by GPU memory/state contamination from prior tests or SGLang incompatibility with the new transformers model loading.

**Error:**
```
RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling cublasGemmEx(...)
RuntimeError: [SGLang Server] Rank 0 Server process terminated unexpectedly.
```

**Reproduction:**
```bash
cd tests && uv run --extra sglang pytest unit/models/generation/test_sglang_generation.py::test_sglang_policy_generation --hf-gated --sglang-only -x -s
```

**Affected tests:**
- `test_sglang_generation.py::test_sglang_policy_generation`
- `test_sglang_generation.py::test_sglang_worker_seed_behavior`
- `test_sglang_generation.py::test_sglang_policy_tensor_parallel`
- `test_sglang_generation.py::test_sglang_generate_text`
- `test_sglang_generation.py::test_sglang_http_server`
- `test_sglang_generation.py::test_sglang_non_divisible_batch_handling`
- `test_sglang_generation.py::test_sglang_generation_with_hf_training_colocated`
- `test_sglang_generation.py::test_sglang_weight_update_and_prefix_cache_reset`

## Err 5. SDPA attention mask expand error with TP=2 SP=True

**Description:** When using tensor parallelism (TP=2) with sequence parallelism (SP=True), transformers v5's SDPA attention implementation tries to expand the attention mask to a size that doesn't match the sequence-parallel-split tensor dimensions. The mask has shape `[1, 1, 64, 128]` but SDPA tries to expand it to `[1, 1, 128, 128]`.

**Error:**
```
RuntimeError: The expanded size of the tensor (128) must match the existing size (64) at non-singleton dimension 2.  Target sizes: [1, 1, 128, 128].  Tensor sizes: [1, 1, 64, 128]
```

**Reproduction:**
```bash
cd tests && uv run --extra automodel pytest unit/models/policy/test_dtensor_worker.py::TestDTensorPolicyWorkerTraining::test_dtensor_worker_training[training_setup21-False] --hf-gated --automodel-only -x -s
```

**Affected tests:**
- `test_dtensor_worker.py::test_dtensor_worker_training[training_setup21-False]` (TP=2 SP=True llama)
- `test_dtensor_worker.py::test_dtensor_worker_training[training_setup22-False]` (TP=2 SP=True qwen2, already skipped by Hemil)

## Err 6. DTensor redistribute assertion error for gemma3 TP=2

**Description:** When running DTensor policy worker v2 with gemma3 model and TP=2, the DTensor redistribute operation fails with an assertion error about `src_shard_order` and `dst_shard_order` being None. This is a PyTorch DTensor compatibility issue with gemma3's architecture under tensor parallelism.

**Error:**
```
assert src_shard_order is not None and dst_shard_order is not None
AssertionError
```

**Reproduction:**
```bash
cd tests && uv run --extra automodel pytest unit/models/policy/test_dtensor_worker_v2.py::test_dtensor_worker_v1_v2_model_config_equivalence[tiny_gemma3_model_path-2-1-False-False-False] --hf-gated --automodel-only -x -s
```

**Affected tests:**
- `test_dtensor_worker_v2.py::test_dtensor_worker_v1_v2_model_config_equivalence[tiny_gemma3_model_path-2-1-False-False-False]`

## Err 7. TP tied model fails with automodel v2

**Description:** DTensor TP=2 with a tied llama model and custom parallel plan fails in the automodel v2 code path. Error trace was truncated; needs investigation.

**Reproduction:**
```bash
cd tests && uv run --extra automodel pytest unit/models/policy/test_dtensor_worker.py::TestTwoGPUCluster::test_dtensor_tp_and_tied_model_with_custom_parallel_plan[True] --hf-gated --automodel-only -x -s
```

**Affected tests:**
- `test_dtensor_worker.py::test_dtensor_tp_and_tied_model_with_custom_parallel_plan[True]`

---

## Phase 2: Fix Plan

**Strategy:** Work through Err 1-7 top to bottom. For each:
1. Run one skipped test from the category to reproduce the error with full output
2. Root-cause the issue (read source, check transformers v5 changelog, etc.)
3. Implement a fix
4. Remove skip, verify the fix works
5. Enable all other tests in that category, verify they pass
6. Summarize the root cause, fix, and any resources below

### Fix progress
- [x] Err 1: vLLM FP8 QKVParallelLinear missing `input_scale` — FIXED (4/6 pass; 2 non-colocated have pre-existing logprob tolerance issue)
- [x] Err 2: vLLM HTTP server response format mismatch — FIXED
- [x] Err 3: Ray ActorAlreadyExistsError — FIXED (always kill actors after graceful shutdown in worker_groups.py)
- [ ] Err 4: SGLang CUDA graph CUBLAS_STATUS_EXECUTION_FAILED (8 tests)
- [ ] Err 5: SDPA attention mask expand error TP=2 SP=True (2 tests)
- [ ] Err 6: DTensor redistribute assertion gemma3 TP=2 (1 test)
- [ ] Err 7: TP tied model fails with automodel v2 (1 test)

---

## Resources about transformers v5 upgrade

(Add links, issues, PRs, and notes here as they come in)

