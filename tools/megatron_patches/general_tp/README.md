# General TP resharding dependency fixes — held staging

This is a build-time patch for exactly Megatron-LM
`3e95097b4a9ab6ea6b2a39eb2b2b24e4a6dfa54a`. It is **not promoted** into the
clean NeMo RL branch. Promotion requires the candidate native reverse-IPC cleanup
result (now passed) and the full-model Nano TP/CP qualification (still pending),
followed by source review.
File-patch composition alone is not either of those results.

The two changes are:

- `planner.py`: validate destination-mesh completeness before looking up its first
  rank. Partial residual destination rosters can omit that rank. The optional
  tensor-mesh specification then returns its existing unsupported reason, while
  the existing generic transfer schedule remains usable. This is the narrow fix
  in [PR 9465](https://github.com/NVIDIA/Megatron-LM/pull/9465), companion commit
  `22690d3996b7981efc8647d1fdc1405dbdb80419`.
- `nccl_copy_service.py`: replace private eager communicator connection with one
  public all-reduce of one CUDA int32 element per service, including ranks with
  empty local transfer queues. Transfer methods are unchanged. This candidate
  addresses duplicated, untracked communicators on the tested PyTorch revision;
  native owned-process cleanup passed; full-model TP/CP evidence is still a
  promotion requirement.

`SOURCE.json` records exact source and output hashes. The patch contains only
these changes, not full vendored files. The companion planner file and the native
copy-service base are otherwise byte-identical to this pinned dependency, so no
unrelated current-main upgrade is required.

## Exact application

Set `PATCH_ROOT` to this staged directory and `MCORE_ROOT` to a separate clean
checkout of the exact base. Review the held status before using this in a future
qualification environment; do not alter an active model checkout.

```bash
test "$(git -C "$MCORE_ROOT" rev-parse HEAD)" = 3e95097b4a9ab6ea6b2a39eb2b2b24e4a6dfa54a
(cd "$MCORE_ROOT" && sha256sum --check "$PATCH_ROOT/BASE.sha256")
git -C "$MCORE_ROOT" apply --unidiff-zero --check "$PATCH_ROOT/general-tp-resharding.patch"
git -C "$MCORE_ROOT" apply --unidiff-zero "$PATCH_ROOT/general-tp-resharding.patch"
(cd "$MCORE_ROOT" && sha256sum --check "$PATCH_ROOT/PATCHED.sha256")
```

Exact base-file hashes are mandatory for this zero-context patch. It is safe
against whitespace/EOF fixers; do not force-apply it to a different revision.
Repeated application must fail rather than silently add another modification.

## Composition with Ultra inference fixes

This patch touches two resharding files. The clean branch's
`tools/megatron_patches/ultra_inference` patch touches three different inference
files. `COMPOSITION_CHECK.json` records successful application in **both orders**
against fresh copies of the exact base files and verification of all five final
hashes. Each patch retains its own base/output checks. Apply both before building
or importing the policy environment; inspect the actual normal module paths.
There is no runtime monkeypatch, container worker overlay or private import path.

## Integration hazards and remaining scope

1. All ranks of a copy service's process group must enter `run()` in matching
   order, even zero-operation ranks. The public collective is once per service,
   not once globally. Cached services must be created/replaced symmetrically.
2. The connection token uses the current CUDA device. Each learner must select
   its assigned device before constructing/executing the service. Native
   subgroup rank mapping and source/destination metadata remain authoritative.
3. A partial destination roster deliberately lacks the optional full-mesh tensor
   specification. Generic scheduling still must validate all required coverage;
   this fix does not accept missing source data or invent remote owners.
4. Engine-owned residual allocations must stay alive through reverse-IPC writes
   and acknowledgment. Learner receive mappings must be released before engine
   owners exit. These are NeMo RL lifecycle responsibilities, not patch effects.
5. The serialized small native gate does not establish concurrent model pause,
   refit, TP/CP, optimizer or publication correctness. Require the full native
   model qualification before promotion; no throughput improvement is claimed.
6. The portable transfer source checker currently accepts exactly the three
   Ultra inference files. A future reviewed integration must explicitly extend
   that allowlist and exact hashes to these two files (five total), refresh the
   native NeMo RL worker pin after the TP merge, and requalify the new layout.
   Do not relax arbitrary dirty-source rejection.

No scheduler actions, GPU executions, source-checkout edits, clean-branch
modifications, freeze or push were performed to prepare this staging artifact.

## Proposed documentation update

`proposal/docs/design-docs/mps-parallelism-sharing.md` and `DOC_UPDATE.patch`
stage the corresponding source-document correction without editing an active
branch. They record the primitive cleanup PASS and preserve the full-model
qualification hold. Inference CP1/PP1 and inference-optimized ETP1 remain required;
training ETP>1, pipeline training/arbitrary PP changes and changed-EP full-model
cases remain unqualified by the current gates. Apply this docs change only with
the reviewed dependency inclusion; its link targets the intended
`tools/megatron_patches/general_tp` installation path.
