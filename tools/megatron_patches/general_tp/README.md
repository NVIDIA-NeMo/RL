# General TP resharding dependency fixes

This build-time patch supports co-resident Megatron refit across different tensor-parallel layouts. It targets exactly Megatron-LM commit `3e95097b4a9ab6ea6b2a39eb2b2b24e4a6dfa54a`; it does not upgrade the dependency to current upstream main.

The patch makes two changes:

- `planner.py` validates destination-mesh completeness before looking up its first rank. Residual-only destinations can omit that rank, so the optional whole-mesh description must decline while the existing generic transfer schedule remains usable. This is the narrow correction in [Megatron-LM #9465](https://github.com/NVIDIA/Megatron-LM/pull/9465), companion commit `22690d3996b7981efc8647d1fdc1405dbdb80419`.
- `nccl_copy_service.py` initializes its communicator once per service with a public all-reduce of one CUDA int32 element. Every process-group rank participates, including ranks without local transfer operations. Transfer planning and execution are unchanged.

`SOURCE.json`, `BASE.sha256` and `PATCHED.sha256` record the exact dependency revision and source hashes. The patch contains only these two corrections, without complete vendored modules or runtime monkeypatches.

## Apply and install

Use a separate checkout before starting model workers. Set `NRL_ROOT` to this NeMo RL checkout, `MCORE_ROOT` to the clean Megatron-LM checkout, and `POLICY_PYTHON` to the interpreter used by the actual policy and inference workers.

```bash
PATCH_ROOT="$NRL_ROOT/tools/megatron_patches/general_tp"
test "$(git -C "$MCORE_ROOT" rev-parse HEAD)" = 3e95097b4a9ab6ea6b2a39eb2b2b24e4a6dfa54a
(cd "$MCORE_ROOT" && sha256sum --check "$PATCH_ROOT/BASE.sha256")
git -C "$MCORE_ROOT" apply --unidiff-zero --check "$PATCH_ROOT/general-tp-resharding.patch"
git -C "$MCORE_ROOT" apply --unidiff-zero "$PATCH_ROOT/general-tp-resharding.patch"
(cd "$MCORE_ROOT" && sha256sum --check "$PATCH_ROOT/PATCHED.sha256")
```

Exact base-file checks are required for this zero-context patch. Reject a mismatched revision or hash; repeated application must fail. Do not force-apply the patch to another dependency revision.

Build the policy environment with the patched checkout through the existing Megatron-Bridge workspace dependency. Alternatively, install it into an already prepared policy environment, retaining that environment's dependency pins and build dependencies:

```bash
uv pip install --python "$POLICY_PYTHON" --no-deps --no-build-isolation "$MCORE_ROOT"
```

If the environment is packaged into a container, perform this installation during the image build and use that image on every worker. Updating a host checkout alone does not change an installed container package. Verify the imported source in the actual worker interpreter:

```bash
"$POLICY_PYTHON" - "$PATCH_ROOT/SOURCE.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import megatron.core.resharding.copy_services.nccl_copy_service as copy_service
import megatron.core.resharding.planner as planner

source = json.loads(Path(sys.argv[1]).read_text())
for module in (planner, copy_service):
    relative = module.__name__.replace(".", "/") + ".py"
    path = Path(module.__file__)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source["files"][relative]["patched_sha256"], path
    print(path)
PY
```

## Compose with the Ultra inference fixes

The [Ultra inference patch](../ultra_inference/README.md) changes three different files. Apply both patches to the same exact-base checkout before building or importing the policy environment, using each patch's base and output hash checks. `COMPOSITION_CHECK.json` records successful application in both orders and verification of all five final source hashes; source composition is not a full-model qualification result.

## Runtime contract and validation limits

All ranks of a copy service's process group must enter `run()` in matching order, including ranks with no local operations. Each learner must select its assigned CUDA device before using the service. Native rank mapping and shard metadata remain authoritative; incomplete source coverage is still an error.

Engine-owned receive allocations must remain alive through reverse-IPC writes and completion acknowledgment. Learner receive mappings must be released before their inference owners exit. NeMo RL pauses and drains inference, completes residual and local copies, refreshes caches, and resumes only after publication.

Inference requires CP1 and PP1; `inference_optimized` experts require ETP1. Training CP greater than one requires `policy.sequence_packing.enabled: true` and `policy.megatron_cfg.use_fused_linear_logprobs: false`. Virtual pipeline chunks and GTP rematerialization are unsupported by this sharing path.

Native distributed projection, reverse-IPC numerical/storage-lifetime and owned-process cleanup checks have passed. **Full-model mixed TP/CP qualification remains pending.** Training ETP greater than one, pipeline training, arbitrary PP changes and mixed-EP full-model layouts are not established by these checks. There is no speedup or convergence claim from the small gates. See the [sharing design and validation scope](../../../docs/design-docs/mps-parallelism-sharing.md).
