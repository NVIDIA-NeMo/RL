# Task 7 synchronous GRPO report

## Scope completed

- Added count-weighted acceptance reconstruction across all supplied rollout
  metric batches. The implementation sums accepted and drafted token counts and
  never averages per-batch rates.
- Added opt-in selected-serving-version stamping and strict experiment-mode
  rejection for absent, nonintegral, mixed, and stale version tags before the
  scheduler decision mutates state.
- Added the synchronous decision preparation contract. Adaptive mode consumes
  the reconstructed acceptance; fixed and `always` keep scheduler acceptance
  input `None`, while experiment mode can still record Step+1 science evidence.
- Added the transaction-bound synchronous refit finalizer. It requires a real
  update receipt before transfer, synchronizes target weights every successful
  step, selects draft bytes only for the immutable decision, closes scheduler,
  ledger, snapshot, and atomic bundle state before publishing either version,
  and closes failed/skipped outcomes without fabricating success.
- Added controller forwarding for opt-in worker update-receipt capture on both
  CP1 monolithic training and CP>1 split training.
- Added the new observation module to the Pyrefly project allow-list.

## RED evidence

The workstation is macOS arm64 while the repository lock supports Linux x86_64
and Linux aarch64 only. The requested pytest command therefore stops before
collection. After initializing recursive submodules, the exact error is:

```text
The current Python platform is not compatible with the lockfile's supported
environments: platform_machine == 'x86_64' and sys_platform == 'linux',
platform_machine == 'aarch64' and sys_platform == 'linux'
```

Dependency-independent RED checks against the exact base source produced:

```text
ModuleNotFoundError: No module named
'nemo_rl.algorithms.draft_update_observation'
```

```text
AssertionError: RED: sync training helper cannot request worker update receipts
```

The pre-transfer accounting regression also failed before its fix:

```text
AssertionError: RED: pre-transfer receipt failure claimed refit attempt
```

## GREEN evidence

Dependency-independent harnesses execute the real new observation module and
AST-extracted real controller helpers and report:

```text
GREEN: count-weighted acceptance, opt-in stamp, and invalid-count handling
GREEN: sync CP1 and split helper can request worker update receipts
GREEN: target-every-step selection, selective draft, close-before-publish, and update-failure stop
GREEN: receipt failure before transfer does not claim a refit attempt
```

The following local checks pass:

```sh
ruff check nemo_rl/algorithms/draft_update_observation.py \
  nemo_rl/algorithms/grpo_sync.py \
  tests/unit/algorithms/test_draft_update_observation.py \
  tests/unit/algorithms/test_grpo_sync_draft_schedule.py \
  tests/unit/algorithms/test_grpo.py
ruff format --check nemo_rl/algorithms/draft_update_observation.py \
  nemo_rl/algorithms/grpo_sync.py \
  tests/unit/algorithms/test_draft_update_observation.py \
  tests/unit/algorithms/test_grpo_sync_draft_schedule.py \
  tests/unit/algorithms/test_grpo.py
uv run --no-project python -m compileall -q \
  nemo_rl/algorithms/draft_update_observation.py \
  nemo_rl/algorithms/grpo_sync.py \
  tests/unit/algorithms/test_draft_update_observation.py \
  tests/unit/algorithms/test_grpo_sync_draft_schedule.py \
  tests/unit/algorithms/test_grpo.py
git diff --check
```

## Blocking dependencies for real controller wiring

The exact Task 6 base explicitly bounded itself to selective payload transfer
and states that controller decisions, sender/apply receipts, cadence science,
and fixed/adaptive enablement were out of scope. Consequently this Task 7
snapshot does not connect the new helpers to `grpo_train_sync`; doing so now
would make every real cadence run fail before it could establish truthful
evidence:

1. IPC and collective `WeightSynchronizer.sync_weights` still return `None`,
   not a mapping with `successful` and a digest-bound `draft_apply_receipt`.
2. Real `TQPolicy` does not expose `supports_draft_apply_receipts`, so the
   existing cadence preflight rejects it.
3. The sync rollout actor returns rollout metrics without the canonical vLLM
   accepted/draft count mapping and without selected-serving-version provenance;
   the current generation counter API exposes only a whole-step delta after all
   dynamic-sampling calls.
4. No serving-version publication API exists on `SyncRolloutActor`; the planned
   DataPlane provenance producer is a later task.
5. Current scheduler, ledger, and transaction-store construction in
   `grpo_train_sync` is still conditional on experiment instrumentation. That
   cannot satisfy adaptive scheduling when `cadence_runtime.enabled=false` or
   strict non-experiment resume recovery.

No receipt, timing, selected-version tag, or successful controller outcome was
fabricated to bypass these dependencies. The permanent Linux tests should run
after the missing producer contracts are implemented, before connecting the
new helpers to the training loop.
