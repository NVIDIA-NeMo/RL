# Native MXFP8 NCCL Reshard Refit Design

- **Date:** 2026-08-27
- **Status:** Proposed
- **Target runtime:** NeMo-RL with vLLM 0.25.1
- **Compatibility goal:** Isolate later vLLM version changes behind one adapter

## Decision

Extend the NCCL Reshard data plane introduced by NeMo-RL PR #3477 with the
native vLLM reload lifecycle introduced by PR #3651. The transport moves a
version-neutral stream of checkpoint-format components. A vLLM adapter binds
those components to the installed vLLM version and owns the complete weight
update lifecycle.

This is not a choice between the two PRs:

- PR #3477 remains responsible for Megatron-to-vLLM resharding, TP/EP/PP
  placement, and collective ordering.
- PR #3651 remains responsible for restoring checkpoint-format storage,
  loading an update, processing weights for the selected kernel, and preserving
  CUDA Graph-visible storage.

The initial implementation must run with NeMo-RL's pinned vLLM 0.25.1. It must
not claim untested runtime support for later vLLM releases. Instead, future
version changes must be confined to the vLLM adapter and its compatibility
tests.

## Goals

- Support Megatron MXFP8 training with `fp8_param=true` and vLLM MXFP8 rollout
  through non-colocated NCCL Reshard refit.
- Transfer native E4M3 weight values and E8M0 scales without a BF16 round trip.
- Keep existing BF16, BF16-to-MXFP8, and blockwise-FP8 refit behavior unchanged.
- Support mixed parameter scopes, including routed-expert MXFP8 with BF16
  attention, embeddings, shared experts, router, and output head.
- Preserve vLLM CUDA Graph tensor addresses across repeated updates.
- Make vLLM version-specific imports, parameter discovery, and finalization
  replaceable without changing the Megatron source or NCCL Reshard planner.

## Non-Goals

- Running NeMo-RL end to end on vLLM 0.28 before NeMo-RL officially bumps its
  dependency.
- Native MXFP8 transfer to a BF16 vLLM target.
- Supporting GEMM-swizzled Transformer Engine source scales.
- Supporting ETP greater than one in the first implementation.
- Adding QKVO, Mamba, embeddings, shared experts, or output heads to the native
  MXFP8 transfer set in the first implementation. These remain BF16 components.
- Replacing `xferdtensor` with vLLM's dense NCCL broadcast implementation.

## Architecture

```text
Megatron policy
  BF16 parameter
  blockwise-FP8 parameter
  native MXFP8 parameter
          |
          v
MegatronRefitSourceAdapter
  emits logical checkpoint components
          |
          v
RefitPlan
  logical parameter
    - weight
    - optional weight_scale
  shape, dtype, source placement, destination placement
          |
          v
NCCL Reshard / xferdtensor
  transfers each component in a fixed plan order
          |
          v
VllmRefitAdapter
  resolves checkpoint components for the installed vLLM
          |
          v
VllmWeightUpdateLifecycle
  begin -> receive all components -> finalize or fail closed
          |
          v
Kernel-format vLLM model
  FlashInfer TRTLLM / CuTeDSL / CUTLASS / other supported backend
```

The NCCL Reshard core must not know vLLM parameter suffixes, kernel packing,
transpose rules, or scale swizzle rules. The vLLM adapter must not know how
Megatron stores optimizer state or how source TP/EP shards are assembled.

## Component Contract

Every logical HF parameter has one or more ordered components:

```text
RefitComponentMeta
  logical_name: str
  checkpoint_name: str
  role: "weight" | "weight_scale"
  dtype: torch.dtype
  global_shape: tuple[int, ...]
  src_placements: tuple[Placement, ...]
  dst_placements: tuple[Placement, ...]
```

Legacy parameters implicitly have one `weight` component. Native MXFP8
parameters have exactly two ordered components:

```text
weight
  dtype: torch.float8_e4m3fn
  shape: [..., K]

weight_scale
  dtype: torch.uint8
  shape: [..., K / 32]
```

The component representation is checkpoint format, not kernel format. The wire
never carries FlashInfer-swizzled scales, CuTeDSL transposed weights, TRTLLM
packed experts, or other backend-owned runtime layouts.

The refit plan is serialized once during setup. Both sides validate the same
ordered component list and plan digest before creating or entering a
collective. A missing component, duplicate role, unsupported dtype, invalid
shape, or placement mismatch must fail before the first NCCL operation.

## Megatron Source Adapter

The source adapter selects behavior per logical parameter rather than from one
global precision flag.

### BF16 storage

Emit one `weight` component using the existing live local shard or grouped
expert staging logic.

### Existing blockwise FP8 storage

Keep the existing Bridge export behavior. Do not change its names, values, or
scale handling as part of this work.

### Native MXFP8 storage

For `fp8_param=true` and `fp8_recipe=mxfp8`:

1. Read Transformer Engine `rowwise_data` as byte-preserving E4M3 values.
2. Read `rowwise_scale_inv` as canonical E8M0 bytes.
3. Remove only validated TE padding and reshape scales to `[..., K / 32]`.
4. Reject missing metadata, wrong dtypes, `K % 32 != 0`, and GEMM-swizzled
   source scales.
5. Split fused gate/up value and scale components along the same output axis.
6. Stack grouped expert value and scale components independently in identical
   numeric expert order at transfer time, not at plan-construction time.

The optimizer continues to own high-precision master parameters and state.
`fp8_param=true` changes the model parameter and parameter-gather storage; it
does not make optimizer state FP8. NeMo-RL must pass `fp8_recipe=mxfp8` into
Megatron's optimizer configuration so MCore constructs the correct masters and
gather buffers.

## NCCL Reshard Data Plane

The existing logical parameter planner remains the source of TP/EP/PP
placements. It is extended so each component has its own shape, dtype, and
placement.

- Column-parallel gate/up values and scales shard on their output dimension.
- Row-parallel down values shard on K; their scales shard on compressed K.
- Grouped expert values and scales prepend the same global expert dimension.
- Each train rank sends only parameters owned by its PP stage.
- Every rank follows one deterministic parameter and component order.
- Existing multi-stream scheduling remains a transport concern and must not
  change component meaning.

The data plane does not quantize, dequantize, transpose, swizzle, or repack.

## vLLM Adapter And Lifecycle

NeMo-RL defines a small internal protocol:

```text
VllmRefitAdapter
  validate_plan(plan)
  prepare(plan)
  begin_update()
  resolve_destination(component) -> LocalParamSpec
  finish_update()
  abort_update(error)
```

`LocalParamSpec` continues to describe the local receive buffer and optional
pre/post hooks. Its lookup key becomes `(logical_name, role)` while preserving
the legacy `weight` default.

### vLLM 0.25.1 adapter

- Use PR #3651's native layerwise reload lifecycle as the sole owner of restore
  and finalization.
- Bind canonical MXFP8 weight and scale components to checkpoint-format slots.
- Let the selected vLLM quantization method perform transpose, swizzle, packing,
  and kernel setup.
- Finalize once after all bulk and misc components arrive.
- Treat any exception after lifecycle initialization as fatal for that worker.
- Preserve CUDA Graph-visible kernel storage through vLLM's copy-back behavior.

Direct calls from the NCCL Reshard core to vLLM
`process_weights_after_loading()` are forbidden. Compatibility workarounds that
remain necessary for vLLM 0.25.1 live only in the 0.25.1 adapter.

### Later vLLM adapter

vLLM 0.28 exposes worker and trainer weight-transfer engines with an explicit
`start_weight_update`, `update_weights`, and `finish_weight_update` lifecycle
and supports custom engine registration. When NeMo-RL bumps vLLM, a new adapter
can register NCCL Reshard as a custom engine or delegate to the new lifecycle.

The core selects adapters by required capabilities, not by parsing a version
string. Capability checks include:

- layerwise reload entrypoints
- weight-transfer engine registry
- trainer-side transfer engine support
- checkpoint parameter loader behavior
- CUDA Graph storage-preservation support

The adapter must fail with one concise compatibility error when required
capabilities are absent. It must not silently fall back to a raw live-model
write.

## Mixed Precision Scope

Precision is recorded per logical parameter in the plan.

Supported initial routes are:

| Source component | vLLM target | Behavior |
| --- | --- | --- |
| BF16 weight | BF16 | Direct reshard |
| BF16 weight | MXFP8 | Existing receiver-side quantization |
| Native MXFP8 weight + scale | MXFP8 | Direct canonical component load |
| Blockwise FP8 value + scale | Matching blockwise FP8 | Existing path |

Native MXFP8 to BF16 and format-mismatched FP8 routes fail during setup. This
allows Nano's routed experts to use native MXFP8 while ignored layers remain
BF16 without introducing a global storage assumption.

## Error Handling

- Validate precision pairing, component roles, dtypes, shapes, placements,
  expert count, and adapter capabilities before communicator setup.
- Exchange and compare a deterministic plan digest before transfer.
- Never enter a collective after a local validation failure.
- Mark a vLLM worker unusable if an update fails after native reload begins.
- Do not retry a partially completed collective in the same process group.
- Emit the logical parameter and component role in every transfer error.
- Do not convert native MXFP8 to BF16 as an implicit fallback.

## Testing

### Unit tests

- Native TE value and scale extraction, including padding and byte preservation.
- Missing metadata, wrong dtype, K alignment, and swizzled-scale rejection.
- Legacy implicit `weight` component compatibility.
- Per-component placement for dense and grouped MoE FC1/FC2.
- Stable expert ordering and fresh source reads across repeated refits.
- vLLM dense gate/up/down and grouped W13/W2 destination binding.
- Mixed BF16 and native MXFP8 plans.
- Exactly one lifecycle finalization after all components.
- Failure-before-collective and fail-closed worker behavior.
- CUDA Graph storage pointer preservation with value-changing updates.

### Version contract tests

- Required test against the pinned vLLM 0.25.1 runtime.
- Adapter protocol tests independent of a real vLLM import.
- Non-blocking compatibility probe against vLLM 0.28 during development,
  checking lifecycle signatures and custom engine registration.
- The 0.28 probe becomes required when NeMo-RL starts its dependency bump.

### GB200 validation

1. Qwen3-30B-A3B two-step smoke with native MXFP8 experts.
2. Qwen3-30B-A3B repeated-refit correctness test with a value-changing update.
3. Qwen3-30B-A3B 20-step comparison against `fp8_param=false`.
4. Nemotron-3 Nano two-step routed-expert smoke with BF16 ignored layers.
5. Nemotron-3 Nano 20-step comparison after the smoke passes.
6. Checkpoint save, resume, optimizer master state, and another successful
   refit after resume.

The 20-step report averages steps 2 through 19 and records policy training,
policy/reference logprob, generation, refit, E2E time, tokens/s/GPU, loss,
reward, entropy, and generation KL error.

## Delivery Sequence

### Change 1: Stable component contract

- Add optimizer `fp8_recipe` plumbing.
- Add native TE component extraction.
- Generalize refit metadata and `HFToLocalParamMap` to ordered component roles.
- Preserve all existing refit behavior behind compatibility tests.

This change is behavior-gated and can be reviewed without enabling native
MXFP8 transfer in a recipe.

### Change 2: vLLM 0.25.1 native MXFP8 adapter

- Add source mapping for dense FFN and routed MoE components.
- Add vLLM checkpoint-component destination binding.
- Integrate only through the PR #3651 lifecycle.
- Enable the exact validated source/target pair in the validator.
- Add smoke recipe and GB200 evidence.

### Future version bump

- Implement or select the adapter for the bumped vLLM lifecycle.
- Keep the component plan, Megatron source, and NCCL Reshard data plane
  unchanged.
- Promote the corresponding version compatibility probe to required CI.

## Acceptance Criteria

- Existing BF16, BF16-to-MXFP8, and blockwise-FP8 focused tests pass unchanged.
- Native MXFP8 value and scale bytes arrive at the correct vLLM logical
  parameters for dense FFN and routed MoE layers.
- No BF16 weight-sized staging buffer or receiver quantization appears on the
  native path.
- Repeated updates preserve CUDA Graph-visible storage and change model output
  consistently with a cold load of the same weights.
- Qwen3-30B-A3B and Nemotron-3 Nano complete the specified GB200 validation.
- vLLM-specific imports and runtime parameter discovery exist only in the
  adapter layer.
- A later vLLM bump does not require changes to source extraction, component
  metadata, placement planning, or `xferdtensor`.
