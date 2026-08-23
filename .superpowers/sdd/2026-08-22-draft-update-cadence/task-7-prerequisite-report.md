# Task 7 receipt and rollout-science prerequisite report

## Scope completed

- Added an opt-in, typed `DraftApplyRequest` that binds a nonnegative serving
  version to an existing absolute snapshot path and its SHA256 digest.
- IPC and collective synchronizers now return `{"successful": true}` after a
  verified transfer and add the existing nested `draft_apply_receipt` schema
  only when a valid request is supplied. They revalidate the artifact after
  transfer and remain stale on every failure path.
- Unsupported transports accept the uniform interface but reject receipt
  requests before transfer. Legacy calls retain their prior transport behavior.
- The common GRPO refit adapter preserves the nested receipt and rejects a
  receipt request when no capable synchronizer is attached.
- `TQPolicy` advertises apply-receipt support only when worker update receipts
  exist and the configured backend selects the default vLLM IPC/collective
  transport. CP1, CP2, and CP4 configurations exercise the same truthful gate.
- vLLM exposes a per-rollout speculative-counter interval independent of the
  existing whole-step metric snapshot.
- `SyncRolloutActor` can opt into one counter interval per rollout batch,
  validates accepted/draft counts, stamps the selected serving-draft version,
  rejects absent, mixed, or stale versions, and cancels an open interval if
  generation fails. Serving-version publication is monotonic and idempotent.
- Corrected the two stale collective vLLM test doubles (external drafter and
  MTP) to accept the production keyword contract `group=` and `src=`. The
  production keyword call was not changed.

This branch does not enable fixed/adaptive scheduling, edit the Task 7
controller helpers, or add experiment configuration/submission code.

## Receipt contract

The caller must create the immutable snapshot before refit and pass:

```python
DraftApplyRequest(
    version=decision.decision_id,
    snapshot_path=str(snapshot_path.resolve()),
    sha256=snapshot_sha256,
)
```

A capable synchronizer returns the following only after sender and receiver
complete successfully and the snapshot still matches its digest:

```python
{
    "successful": True,
    "draft_apply_receipt": {
        "successful": True,
        "version": decision.decision_id,
        "snapshot_path": str(snapshot_path.resolve()),
        "sha256": snapshot_sha256,
    },
}
```

Target-only sync rejects a draft request. A failed transfer, changed or missing
snapshot, unsupported transport, or false receiver result cannot produce a
successful apply receipt.

## Strict TDD evidence

The dependency-free harness first failed because the producer contracts were
absent:

```text
AssertionError: RED: receipt/science producer contract is absent
```

After the initial implementation, a fresh synchronizer remained incorrectly
fresh when a later transfer failed:

```text
AssertionError: failed transfer left synchronizer falsely fresh
```

An additional pre-transfer mutation regression exposed the same state problem
before endpoint invocation:

```text
AssertionError: invalid apply request left synchronizer falsely fresh
```

The prior vacuous `all()` check could also accept a receiver set containing no
explicit success:

```text
AssertionError: receiver without explicit success was accepted
```

The Linux-gate-compatible vLLM fake contract failed before correction:

```text
TypeError: missing a required argument: '_group'
```

The generic refit adapter initially rejected the new producer identity:

```text
TypeError: refit_policy_generation() got an unexpected keyword argument
'draft_apply_request'
```

All five contracts now report:

```text
GREEN: draft apply identity is typed and digest-bound
GREEN: rollout science binds counts to a monotonic serving version
GREEN: transfer failure restores stale state
GREEN: vLLM collective test doubles accept production keywords
GREEN: refit adapter preserves typed apply receipt payload
```

## Local validation

The following checks pass on the macOS arm64 workstation:

```sh
python3 .superpowers/sdd/2026-08-22-draft-update-cadence/task-7-prerequisite-harness.py
ruff check <all changed Python files>
ruff format --check <all changed Python files>
python3 -m compileall -q <all changed Python files>
uv tool run --from pyrefly==0.24.2 pyrefly check \
  nemo_rl/weight_sync/__init__.py \
  nemo_rl/weight_sync/*_weight_synchronizer.py \
  nemo_rl/weight_sync/interfaces.py --config pyrefly.toml
git diff --check
```

Focused weight-sync Pyrefly output:

```text
INFO errors shown: 0, errors ignored: 3, modules: 9
```

Full-file Pyrefly probes on `grpo.py`, `sync_rollout_actor.py`, and
`vllm_generation.py` retain their pre-existing baseline findings, but report no
finding in the new receipt, serving-version, or rollout-science code. The
adapter return-type correction removed three prior `refit_metrics` assignment
findings.

The repository pytest environment cannot run locally because the frozen lock
supports Linux only. The focused command exits before collection with:

```text
The current Python platform is not compatible with the lockfile's supported
environments: platform_machine == 'x86_64' and sys_platform == 'linux',
platform_machine == 'aarch64' and sys_platform == 'linux'
```

## Linux gate handoff

Job `6465032` evidence supplied by the integrator showed the MCore slice at 13
passes and the vLLM slice at 21 passes with one stale fake-signature failure.
The external-drafter fake and the analogous MTP fake now bind the production
keyword signature, so the next Linux gate can reach product assertions.

Run the following focused Linux test before integration:

```sh
uv run --frozen --group test pytest -q \
  tests/unit/weight_sync/test_weight_synchronizer.py \
  tests/unit/experience/test_sync_rollout_science.py \
  tests/unit/models/generation/test_vllm_rollout_science.py \
  tests/unit/models/generation/test_vllm_backend.py \
  tests/unit/models/policy/test_split_api_wrappers.py \
  tests/unit/algorithms/test_grpo.py \
  -k 'draft_apply or rollout_science or serving_version or apply_receipt or \
      collective_target_only_receiver or collective_mtp_selection or \
      refit_policy_generation_preserves_digest_bound or \
      refit_policy_generation_rejects_receipt'
```

## Task 7 integration requirements

The controller wiring must remain explicit:

1. Create and durably write the immutable snapshot before calling the refit
   adapter; pass its `DraftApplyRequest` only for selected draft refits.
2. Close the apply receipt, transaction, scheduler outcome, decision ledger,
   and checkpoint bundle before invoking
   `SyncRolloutActor.publish_applied_draft_version`.
3. For every selected generation or dynamic-sampling batch, call
   `rollout_to_tq(capture_draft_science=True,
   expected_applied_draft_version=reserved_version)` and retain every returned
   metric batch for count-weighted aggregation.
4. Do not infer success from capability flags, a sender-only result, or a
   version publication. Each is a separate fail-closed boundary.

No experiment is ready to submit until this snapshot passes independent review,
the focused Linux gate, and the subsequent Task 7 controller wiring review.
