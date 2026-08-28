# Task 7 Report: Native MXFP8 vLLM Destination Binding

## Commit Range

Task 7 starts at `cbf1303931716377d59ee748e715860abb93b850`.

- `3fdcc396 test(refit): define native MXFP8 destination contract`
- `bcc0c243 feat(refit): bind native MXFP8 vLLM destinations`

Both commits contain a `Signed-off-by` trailer. The final documentation commit
is recorded after this report is committed.

## Changed Files

- `nemo_rl/models/generation/vllm/refit_adapter.py`
  - Binds ordered Task 2 `weight` and `weight_scale` components to dense merged
    and routed-expert vLLM 0.25.1 checkpoint storage.
  - Validates runtime and checkpoint scale aliases, role, local shape, dtype,
    and wrapped `weight_loader` availability before receive starts.
  - Bridges local TP/EP payloads through vLLM's wrapped loader, including merged
    shard and per-local-expert routing arguments and exact loader accounting.
  - Owns received payload storage until finalization and verifies that runtime
    value and scale pointers remain identical across repeated refits.
- `nemo_rl/models/generation/vllm/vllm_backend.py`
  - Detects native component metadata and runs
    `prepare -> receive -> loader/accounting -> finalize` through the adapter.
  - Resolves every native destination before the first collective and aborts
    fail-closed on preparation, receive, loader, or finalizer failure.
  - Keeps native MXFP8 bulk parameters disjoint from BF16 attention, embedding,
    shared-expert, router, output, and MTP miscellaneous parameters.
  - Preserves the legacy BF16 direct, BF16-to-MXFP8, routed alias, and blockwise
    paths when native component metadata is absent.
- `tests/unit/models/generation/test_vllm_refit_adapter.py`
  - Covers dense and routed value/scale binding, checkpoint and runtime scale
    aliases, validation failures, lifecycle order, wrapped-loader accounting,
    payload lifetime, repeated changed-byte refits, and pointer identity.
- `tests/unit/models/generation/test_nccl_reshard_backend.py`
  - Covers ordered native receive/load/finalize, pre-collective failure,
    mixed native/BF16 scope, and the legacy conversion regressions.

No launcher, configuration, or dependency files changed. Runtime support is
claimed only for the repository-pinned vLLM 0.25.1 lifecycle.

## RED Evidence

Tests were committed first in `3fdcc396`. In the isolated adapter suite, all
eight pre-existing Task 6 lifecycle tests passed and the first new destination
contract test failed at the missing API:

```text
AttributeError: 'Vllm0251RefitAdapter' object has no attribute 'resolve_destination'
```

The native repository backend command could not collect on this macOS host
because the checked-in lockfile supports Linux only:

```text
error: The current Python platform is not compatible with the lockfile's supported environments
```

That environment limitation is not represented as a backend behavioral RED.

## GREEN Evidence

The final focused adapter suite used an isolated dependency environment:

```text
32 passed in 4.17s
```

The final focused backend suite used the same isolated environment plus minimal
in-process vLLM import stubs, because vLLM is unavailable on macOS:

```text
18 passed, 1 warning in 0.07s
```

The backend suite includes native ordering/failure/mixed-scope cases and all
existing tests in `test_nccl_reshard_backend.py`, including BF16-to-MXFP8,
routed runtime scale aliases, dense gate/up conversion, and blockwise storage.
The warning is the expected missing `nccl.m2n.reshard` fallback in this CPU
harness.

Final static verification over all four owned Python files reported:

```text
ruff check: All checks passed!
ruff format --check: 4 files already formatted
python3 -m py_compile: success
pyrefly refit_adapter.py: INFO 0 errors
pyrefly vllm_backend.py: INFO 0 errors (6 suppressed, 1 warning not shown)
```

Pyrefly replaced unavailable `torch.*`, `vllm.*`, `zmq`, and `safetensors.*`
imports with `Any` as applicable. This is a focused source check, not a complete
repository type-check claim.

## Linux Native Gate

The required native gate remains a Linux GPU/container run with the exact
repository-pinned vLLM 0.25.1 build. It must execute these focused suites
without import stubs and validate real layerwise reload, NCCL TP/EP transfer,
wrapped-loader accounting, two changed-byte refits, and CUDA Graph-visible
value/scale pointer identity. Full repository integration and distributed GPU
coverage remain downstream Task 9/10 validation; no such runtime result is
claimed from this macOS host.

## Remaining Concerns

- The macOS backend evidence exercises the complete owned CPU test file but
  substitutes only vLLM imports; real vLLM layerwise behavior remains gated on
  Linux.
- Destination name mapping mirrors the existing backend mapping to keep this
  task scoped. Future vLLM support should centralize or version that contract
  rather than extending the 0.25.1 adapter implicitly.
- CUDA Graph pointer preservation is asserted with fake finalization locally;
  the native GPU gate must confirm the same behavior with actual kernels and
  graph capture.
