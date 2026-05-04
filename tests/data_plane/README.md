# Data-plane test environment

Layout follows the test plan in
[`research/data_plane_test_plan.md`](../../research/data_plane_test_plan.md).
Two tiers, two directories:

```
tests/data_plane/
├── conftest.py                 # shared (just repo_root fixture)
├── unit/                       # Tier 1 — no Ray, no GPU, no transfer_queue
│   ├── conftest.py
│   ├── test_architecture_invariants.py
│   ├── test_dispatch.py
│   ├── test_factory.py
│   ├── test_import_isolation.py
│   ├── test_interface_contract.py
│   ├── test_kvbatchmeta.py
│   └── test_shard_parity.py
└── functional/                 # Tier 2 — Ray + transfer_queue, single-node
    ├── conftest.py
    ├── test_tq_lifecycle.py
    └── test_tq_multinode.py
```

## Why a separate test root

Per the plan §11: the project's `tests/unit/conftest.py` drags in
`mlflow`, `torch.distributed`, `init_ray`, etc. None of that is needed
for data-plane Tier 1 tests. Keeping our suite under
`tests/data_plane/` with a *local* `conftest.py` lets unit tests run in
a slim venv (torch + tensordict + pytest only).

## Running

```bash
# Tier 1 — fast, no extras required
uv run --group test pytest tests/data_plane/unit/ -v

# Tier 2 — needs a Ray cluster (transfer_queue is now a base dep)
uv run --group test pytest tests/data_plane/functional/ -v
```

The functional `conftest.py` auto-skips every test in that directory
with a clear reason if `transfer_queue` is missing — no silent skips.

## Quick run without pytest installed

The architecture invariants depend only on `pathlib` + `re`, so they
can be exercised with plain Python during development:

```bash
python3 -c "
import sys, types
sys.modules['pytest'] = types.ModuleType('pytest')
sys.modules['pytest'].mark = types.SimpleNamespace(parametrize=lambda *a, **k: (lambda f: f))
sys.path.insert(0, 'tests/data_plane/unit')
import test_architecture_invariants as ti
ti.test_legacy_grpo_has_zero_dataplane_refs()
ti.test_no_data_plane_in_master_config()
ti.test_grpo_sync_constructs_kvbatchmeta()
ti.test_factory_does_not_construct_noop()
print('arch invariants ok')
"
```

This is what we run pre-commit. It catches the highest-leverage class
of regression — the kind where a future PR silently couples files that
should stay decoupled.

## Coverage status

| Plan section | Status |
|---|---|
| §4.1 Interface contract | implemented (`test_interface_contract.py`) — runs against NoOp |
| §4.2 Codec | not yet implemented (Stage 2 work) |
| §4.3 Factory | implemented (`test_factory.py`) — production path rejects disabled / noop |
| §4.4 KVBatchMeta | implemented (`test_kvbatchmeta.py`) — incl. pickle survival |
| §4.5 Shard parity | partial (`test_shard_parity.py`) — sort+stride only; vanilla `shard_by_batch_size` parity is Stage 4 follow-up |
| §4.6 Schema | not yet implemented (Stage 2 work) |
| §4.7 Import isolation | implemented (`test_import_isolation.py`) |
| §4.8 Architecture invariants | implemented (`test_architecture_invariants.py`) — adapted for the decorator design (see notes in that file) |
| §5.1 TQ lifecycle | smoke test only (`test_tq_lifecycle.py`); full plan items pending |
| §5.6 Multinode | smoke test only (`test_tq_multinode.py`) |

## Notes — decorator-design adaptation

The plan's §4.8 was written assuming we'd ship `policy.train_from_dp_meta`
as a separate method. We chose `@dp_dispatch` for polymorphism — same
method name (`policy.train`), different argument types. The architecture
invariants are adjusted:

  * **Plan check** "grpo_sync.py must NOT contain `policy.train(`" — dropped.
    With the decorator, `policy.train(meta)` IS the TQ-mediated dispatch.
  * **Replacement check** `test_grpo_sync_constructs_kvbatchmeta` —
    asserts that `grpo_sync.py` constructs `KVBatchMeta` objects, which
    is what makes the decorator's TQ branch fire instead of falling
    through to legacy.

The underlying invariant (sibling-trainer separation, no cross-trainer
gates, factory-as-bouncer) is the same.
