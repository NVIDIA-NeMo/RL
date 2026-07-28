# Implementation Log — Token Capture v3 (Gate Token Custody)

Tracks stage-by-stage progress of `tq-gym-gate-authoritative.md` (MVP: S1–S5,
one PR, sign-off gates between stages). Branch: `yukih/sc-entrypoint`.

Standing constraints (user-mandated):

- **Dormant by default.** `token_capture.enabled=false` is the default; every
  legacy codepath must behave exactly as before. Any change that affects
  behavior regardless of the flag (bug fixes, submodule pin bumps) is
  disclosed explicitly at the stage gate for sign-off.
- Wire shapes (§ 3.4) and the serving rule (§ 3.3) freeze at the S1 gate.

Environment: dev node with 8×H100 for stage-gate test runs.
Prototype donor checkout: `/lustre/fsw/portfolios/coreai/users/pthombre/gym/RL`
(branch `pranav/tq_gym_prototype` @ `05e0adfa0`, Gym pin `6ea5810`).

---

## S1 — primitives (Gym fork) + buffer surgery (RL repo)

Status: **SIGNED OFF (user review, 2026-07-28).** Wire shapes and the serving
rule are frozen. Post-review closeout below (subpackage restructure done;
regression evidence recorded as it lands).

### Gym fork (submodule branch `tq-gate-capture`)

Base: **upstream PR #2124 head `32b555f04`** (= upstream main @ `fa0c2da3`
+ the token-id-capture core), per § 9.2. S1 work is two commits on top:
`70e43b60` "feat(token-id-capture): S1 gate-authoritative capture primitives"
and `61fbb660` "refactor(token-id-capture): move S1 capture core into
staging/ subpackage" (the post-review restructure; see Open TODOs).

| Item | Status | Notes |
|---|---|---|
| records.py wire shapes | done | StagedCallRecord / StageResult / CommitCoords / CallRecord / RolloutReceipt / StagedCallSnapshot, `staging_key()`, `SCHEMA_VERSION=1`; reserved `chain_hash`/`cum_hash` fields for H2 |
| protocols.py | done | TokenSink / TokenSource / WeightVersionProvider / CaptureAdapter protocols; `install_capture` signature frozen (body lands in S2 `capture.py`) |
| digest.py | done | `compute_staging_digest`, encoders, `build_staging_delta` — **verified byte-identical to the prototype donor** (`rollout_writer.py`) on golden vectors incl. -0.0/NaN-adjacent cases |
| lineage.py | done | pure `RolloutLineage` (admit/commit/fail/seal → manifest) + create-only `LineageRegistry` with TTL sweep; no leases/hash-claims — the marker names the parent explicitly |
| rebuild.py | done | `snapshots_to_entries` (mask-driven, **verified equal to the prototype's `StagedSnapshotTokenSource.entries()`** on the worked-example forest incl. the c3 fork) + `linearize(main_chain_only, terminal_hint)` |
| conformance kit | done | 4 golden fixtures (worked example § 4.1, single-call, capture-failed, mixed-wv); `run_lineage_conformance` + `run_sink_source_conformance`; goldens frozen |
| purity rule | done | core modules import with no fastapi/ray/torch/TQ/aiohttp (subprocess-import test); `token_id_capture/__init__` resolves the #2124 reader/route exports lazily (PEP 562) so the core stays pure with the public API unchanged |
| tests | done | `tests/unit_tests/test_token_capture_gate_primitives.py` — 24 tests; **44/44 green including #2124's 20-test base suite** (the S3 base sanity check, already green at S1) |

### NeMo-RL repo

| Item | Status | Notes |
|---|---|---|
| `nemo_rl/data_plane/tq_token_sink.py` | done | TQTokenSink / TQTokenSource over put_samples/get_samples; 3 jagged + 2 scalar columns per staged row; parent pointers rejoined from the manifest by the finalizer |
| `TokenCaptureConfig` (`single_controller_utils/config.py`) | done | pydantic BaseModel on MasterConfig, `enabled=False` default; staging_partition / on_capture_failure / mixed_weight_version_policy / min_valid_fraction_per_group / TTLs |
| TQReplayBuffer surgery (`async_utils/replay_buffer.py`) | done | `commit_finalized` (slot's effective version = `group_min_wv`), `abort`, `rollout_ids` on slots, staging-aware `remove`; evicted-slot `commit` fix (pre-write check + un-write on mid-write eviction) |
| `_buffer_capacity` leak fix (`single_controller.py:319`) | done | release on dispatch exception, paired with `generate_and_push` aborting the reserved slot so eviction never double-releases |
| Partition pre-registration (`setup.py`) | done | registers `rollout_data` + `rollout_staging` from the driver thread, **gated on `token_capture.enabled`** (the TQ lazy-field controller race) |
| Tests | done | `test_tq_replay_buffer.py` 18/18 (new: token-capture mode + evicted-commit classes); `test_tq_token_sink.py` 7/7 — **conformance kit green against a live TQ simple backend** (byte-exact digests through float32 storage); `test_rollout_manager.py` 8/8 (2 new failure-path tests) |

### S1-gate checklist (§ 10)

- **Submodule fork logistics**: local branch `tq-gate-capture` on the vendored
  submodule; base = `refs/pull/2124/head` @ `32b555f04` fetched from
  `NVIDIA-NeMo/Gym` origin. **Open decision for sign-off**: where the gitlink
  should point for CI (fork remote vs. an NVIDIA-NeMo/Gym branch) — the rev
  currently exists only locally.
- **Leaf package importable in the worker venv**: `nemo_gym.token_id_capture`
  core modules import in the RL venv with zero serving deps (purity test);
  worker `py_executable` check to be repeated in S2 when `install_capture`
  is wired into the vLLM worker.
- **vLLM prefix-ids + splice validated** (scratchpad
  `validate_prefix_ids_vllm.py`, 1×H100):
  - Template-level splice (`_replace_prefix_tokens`) for the functional-test
    templates **Qwen/Qwen3-0.6B** (SC DP functional test) and
    **Qwen/Qwen2.5-1.5B-Instruct** (SC exemplar): exact model prefix
    preserved under retokenization drift (Qwen3's history-render strips
    `<think>` blocks — retokenization_differs=True — and the splice handles it).
  - Live vLLM token-in smoke (Qwen3-0.6B): turn-2 `TokensPrompt` of spliced
    exact ids → `out.prompt_token_ids == spliced` (exact prefix conditioning,
    47-token prefix), native generated-id + logprob extraction with no string
    parsing and no `/tokenize` round trip.

### Open TODOs (pre-sign-off)

- [x] **Group the S1 modules into a subpackage** — done post-sign-off
  (2026-07-28), Gym fork commit `61fbb660` (a follow-up commit, not an amend).
  `records.py` (staging shapes, split back out; #2124's `records.py` reverts
  to base-identical) + `protocols.py` / `digest.py` / `lineage.py` /
  `rebuild.py` / `conformance/` now live under
  `nemo_gym/token_id_capture/staging/`; `staging/__init__` re-exports the
  wire shapes + protocols (the disclosure-5 name collision dissolves into
  namespacing); the purity test now globs the subpackage instead of a
  hand-maintained list (46/46 green, incl. #2124's 20-test base suite).
  RL-side imports (`tq_token_sink.py`, `test_tq_token_sink.py` — 7/7 green
  vs live TQ after the move) and the design doc's § 3.0/§ 9.3 paths updated.
  S2's `capture.py` lands in `staging/`; `adapters/` and gate hosting stay
  outside the purity scope.

### Deviations / disclosures (for S1 sign-off)

1. **Gym submodule pin moves `610a08ab` → `61fbb660`** (= #2124 head + S1
   commits; ~50 upstream commits ahead of the old pin). Attempted alternative —
   cherry-picking #2124 onto the old pin — required hand-merging its
   prerequisite (#1715 observability capture, 34 files / 2.7k lines) into an
   upstream-untested combination; rejected in favor of the design's
   prescribed base. **This changes Gym behavior with the flag off**; the
   flag-off SC gym functional test is the regression evidence to run at the
   gate (queued; see below).
2. **`uv.lock` regenerated** for the new pin (aiohttp 3.13.3→3.14.1 floor,
   +simple-websocket/toml/websocket-client, starlette/typing-extensions
   bumps). Affects all environments regardless of the flag.
3. **Bug fixes active with the flag off** (all disclosed by design §§ 9.1):
   evicted-slot `commit` no longer orphans TQ rows; `_buffer_capacity` no
   longer leaks on failed dispatch (failed dispatch now also aborts its
   reserved slot instead of leaving a phantom entry until staleness eviction).
4. **`#2124` API preserved via lazy exports**: `token_id_capture/__init__`
   now resolves reader/route/source names through module `__getattr__`
   (required by the § 3.0 purity rule). No call-site changes.
5. **Protocol naming**: the design's `TokenSink`/`TokenSource` protocols
   collide with names #2124 already exports; they live in
   `token_id_capture/protocols.py` and are imported by module path, not
   re-exported from the package root.
6. **Purity enforcement** is a subprocess-import unit test rather than an
   import-linter dependency (same guarantee, no new Gym dependency).
7. **Pre-existing test breakage fixed**: `test_rollout_manager.py`'s
   `_FakeBuffer` lacked the `target_step` kwarg (broken before this work);
   fixed while adding the new failure-path tests.

### Regression evidence (flag off)

- `tests/unit/test_config_validation.py` + `test_config_v2.py`: **476 passed**.
- `tests/unit/experience/`: 8/8; `tests/unit/single_controller/test_tq_replay_buffer.py`: 18/18.
- Full `tests/unit/single_controller/` + `tests/unit/experience/` suite
  (2026-07-28, `NRL_FORCE_REBUILD_VENVS=true` venv rebuild first): **59
  passed; 10 failures, all pre-existing at branch HEAD** (each reproduced
  byte-identically with ALL working-tree changes stashed — committed test
  fixtures out of sync with committed branch code, not this work):
  - `test_rollout_pump.py::test_rollout_pump_writes_expected_tq_data` —
    SC actor init reads `master_config.logger` (`single_controller.py:116`,
    committed 2026-06-21) but the test's `MasterConfig.model_construct`
    never sets the required `logger` field → actor creation dies →
    0 rows in TQ. (The prior session's "Ray-version mismatch" diagnosis was
    the venv symptom; after rebuild this is the real, deterministic cause.)
  - 9 × `test_single_controller_setup.py::TestSetup::*` — `KeyError:
    'seed'`: `setup_single_controller` calls `set_seed(grpo_config["seed"])`
    (commit `888cb8eeb`) but the test's `_make_master_config` has no
    `grpo.seed`.
  Flagged for the user at the gate (fixable as a test-fixture patch, but
  left untouched to keep the stage diff clean).
- SC functional tests (`L1_Functional_Tests_SingleController.sh`, 8×H100)
  with the flag off: planned as pin-bump regression evidence at the gate.

## S2 — capture core + vLLM adapter + worker hosting

Status: not started (blocked on S1 sign-off)

## S3 — Gym gate

Status: not started (blocked on S2 sign-off)

## S4 — receipts, finalizer, SC integration

Status: not started (blocked on S3 sign-off)

## S5 — verification

Status: not started (blocked on S4 sign-off)
