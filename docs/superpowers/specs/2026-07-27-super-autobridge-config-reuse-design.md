# Super AutoBridge Config Reuse Design

## Problem

Nemotron3 Super 120B job `5640899` loaded the full 32-node topology, all
DeepEP `f725d296` runtimes, all vLLM engines, and all Megatron policy and
reference-model ranks. Before the first training step, rank 41 failed while
building refit conversion metadata:

```text
ValueError: No architectures found in model config
```

The shared Hugging Face snapshot was intact and its `config.json` contained
`architectures=["NemotronHForCausalLM"]`. A one-node stress job subsequently
loaded the same config 128/128 times. The failure is therefore consistent with
a rare, actor-local second config load rather than deterministic snapshot
corruption or a HybridEP, NCCL, CUDA, or memory failure.

Pinned Megatron-Bridge commit `45e4e4be` currently:

1. loads and validates a config with `safe_load_config_with_retry`;
2. creates a lazy `PreTrainedCausalLM` wrapper without attaching that config;
3. reloads the config when refit dispatch first accesses
   `bridge.hf_pretrained.config`.

Upstream Megatron-Bridge commit
`327bcc73c29439abc69d62d0902163de90e35037` eliminates the second load by
assigning the validated config to the lazy wrapper before constructing
`AutoBridge`.

## Goals

- Backport only the upstream config-reuse behavior to the pinned Bridge line.
- Preserve all existing loader arguments and lazy model-loading behavior.
- Prove that refit architecture dispatch uses the exact config object already
  loaded and validated by `AutoBridge.from_hf_pretrained`.
- Re-run the exact Super 120B 32n4g HybridEP performance recipe for 20 steps.

## Non-goals

- Cherry-pick the complete upstream 73-file legacy-model deprecation commit.
- Change the Super recipe, node split, parallelism, checkpoint, DeepEP pin, or
  nightly image.
- Add retry loops around the 128-GPU training job.
- Claim a HybridEP performance result unless the run completes the 20-step
  success gate.

## Implementation

Create a dedicated branch from `seonjinn/Megatron-Bridge` commit `45e4e4be`.
In `src/megatron/bridge/models/conversion/auto_bridge.py`, replace the direct
return with the minimal upstream sequence:

```python
hf_pretrained = PreTrainedCausalLM.from_pretrained(path, **kwargs)
hf_pretrained.config = config
return cls(hf_pretrained)
```

The `config` setter stores the object in `_config`. Future config access and
lazy model loading therefore reuse the validated config and do not call
`safe_load_config_with_retry` again. No loader arguments or exception behavior
change.

Add a focused test in
`tests/unit_tests/models/test_auto_bridge.py` that:

- supplies one sentinel `PretrainedConfig`;
- verifies the lazy wrapper receives the original loader arguments;
- verifies the returned bridge holds that exact sentinel object by identity;
- makes config access fail if a second load is attempted.

Commit and push the Bridge branch, then update only the NeMo-RL Bridge
submodule pointer. Keep the existing recursively pinned Megatron-LM commit
`4d04e762`.

## Verification

Run these gates in order:

1. Focused Bridge unit test and the adjacent AutoBridge loader tests.
2. Existing NeMo-RL HybridEP launcher regression test, shell syntax check, and
   `uv lock --check`.
3. OCI-HSG one-node refit/config preflight in the same nightly and mcore
   environment.
4. FairShare query and `sbatch --test-only`.
5. Exact Super profile
   `grpo-nemotron3-super-120BA12B-32n4g-async-1off.yaml`, 32 nodes × 4 GPUs,
   segment 8, `NCCL_NVLS_ENABLE=0`, source-native DeepEP `f725d296`, and 20
   steps.

Monitor the full job for at least five minutes and through the previously
failing refit-metadata phase. Success requires `COMPLETED 0:0`, 20/20 training
steps, and no config, HybridEP, NCCL, CUDA, OOM, rank-loss, NaN, or Inf error.

## Reporting and Rollback

Record the NeMo-RL, Bridge, Megatron-LM, DeepEP, image, FairShare, allocation,
node, and job provenance in the run artifact directory. Update the existing
HTML report with the preflight and full-run result.

Rollback is a single NeMo-RL submodule-pointer revert to Bridge `45e4e4be`.
The backport branch remains available for audit and does not alter the original
fork branch.
