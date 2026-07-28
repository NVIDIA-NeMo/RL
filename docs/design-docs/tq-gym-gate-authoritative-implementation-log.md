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
- SC functional tests (flag off, dev node GPUs) as pin-bump regression
  evidence:
  - `tests/functional/grpo_dp_single_controller.sh` (Qwen3-0.6B, 2 GPUs,
    Megatron + async vLLM + TQ): **PASS** (2026-07-28) — all 5 metric
    checks green (`gen_kl_error` max 6.0e-4 < 2e-3; probs-ratio clamps
    exactly 1.0). Caveat: required `NRL_FORCE_REBUILD_VENVS=true` — the
    container's prebaked `/opt/ray_venvs` are stale vs the branch lock
    (Ray 2.54.0 vs 2.55.1; `nvidia-resiliency-ext` 0.6.0.dev33 < 0.6.0's
    minimum). Both mismatches pre-date the S1 uv.lock regen (Ray 2.55.1
    was already pinned at branch HEAD); environmental, not this work.
  - `tests/functional/grpo_async_gym_single_controller.sh` (SC + NeMo-Gym
    workplace-assistant, Qwen3-0.6B, 2 GPUs, 10 steps): **PASS**
    (2026-07-28) — metric checks green (`median(gen_kl_error)`=0.041 < 1.3;
    `max(reward)`=0.5 > 0). Same caveat: needs `NRL_FORCE_REBUILD_VENVS=true`
    on this node (stale prebaked NemoGym-actor venv, Ray 2.54.0).
    **This is the flag-off pin-bump evidence for the Gym submodule move
    `610a08ab` → the tq-gate-capture branch** (disclosure 1): the legacy
    token-echo path through the new pin trains correctly.

## S2 — capture core + vLLM adapter + worker hosting

Status: **code + tests complete (2026-07-28); awaiting user sign-off at the
S2 gate.**

### Gym fork (submodule branch `tq-gate-capture`)

One commit on top of the S1 pair: `51b8092e` "feat(token-id-capture): S2
engine-blind capture core + vLLM adapter".

| Item | Status | Notes |
|---|---|---|
| `staging/capture.py` | done | `RolloutTokenCapture.begin_call/complete_call` — engine-blind record + digest build; **fail-closed ordering**: staged coords exist only after `sink.stage` reports bytes durable, every capture failure (bad delta, sink rejection/exception, extraction error) degrades to `capture_failed` coords without breaking the served completion; weight version stamped at `begin_call` (generation-start semantics); streaming rejected (`StreamingUnsupportedError`); double-complete is a loud caller bug; `complete_call_from_response` drives the adapter; `install_capture` working body via the `CaptureHost` one-method seam (instance also returned) |
| `staging/protocols.py` | done | S1-frozen `install_capture` signature now delegates to the capture core (was `NotImplementedError`); same callable re-exported from `staging/__init__` |
| `adapters/vllm.py` | done | engine-specific only, **no vllm imports** (duck-typed payloads): `enter_prefix` via the worker's existing `required_prefix_token_ids` field; `replace_prefix_tokens` relocated **verbatim** from `nemo_rl/models/generation/vllm/vllm_worker_async.py`; native extraction off the final chat payload — message token fields or `choice.logprobs.content` `token_id:` entries (in-process; no second `/tokenize`); one-choice guard; `extract_prompt_ids` reads the hookup-attached engine prompt (vLLM's OpenAI response doesn't carry it) |
| tests | done | `tests/unit_tests/test_token_capture_s2_worker.py` — 29 tests: mock adapter + mock sink ordering matrix, install wiring, extraction shapes, splice goldens incl. the § 4.1-style retokenization-drift example; **75/75 green** (S2 + S1 primitives + #2124 base suite; the purity glob picked up `capture.py` automatically) |

### NeMo-RL repo

| Item | Status | Notes |
|---|---|---|
| `vllm_worker_async.py` hosting | done | `install_token_capture` (CaptureHost seam), `setup_token_capture(dp_cfg, staging_partition)` fan-out target (in-worker DP client + `TQTokenSink` + the single `install_capture` call with `VLLMCaptureAdapter`; model-owner ranks only), `_rollout_weight_version` attribute + `set_rollout_weight_version`. All dormant until the fan-out runs. |
| `vllm_generation.py` fan-outs | done | `setup_token_capture` (asserts async engine) + `set_rollout_weight_version`, standard `run_all_workers_single_data` DP-leader pattern |
| SC `_sync_weights` rotation | done | flag-gated `set_rollout_weight_version(self._trainer_version)` fan-out beside the existing `RolloutManager.set_weight_version` (§ 9.1 lists this under S4; pulled forward as it completes the S2 version-stamping story — disclosed) |
| `PY_EXECUTABLES.VLLM_GYM` | done | `--extra vllm --extra nemo_gym` worker env for capture-enabled runs (constant only; the flag-gated registry override for `VllmAsyncGenerationWorker` is S4 setup wiring) |
| pyrefly | done | `tq_token_sink.py` added to `project-includes`; `nemo_gym.*` added to `replace-imports-with-any` (editable finder hook unresolvable, same pattern as vllm/megatron) |
| Tests | done | `tests/unit/models/generation/test_vllm_token_capture_hosting.py` — 6 tests (`--nemo-gym-only`): install wiring w/ vLLM adapter, non-model-owner skip, live version stamping through the install closure into staged records, both fan-outs incl. async-engine guard |

### S2 checks (carried from the S1 gate)

- **Leaf package importable in the worker venv**: `uv run --locked --extra
  vllm --extra nemo_gym` resolves and imports
  `nemo_gym.token_id_capture.staging` + `adapters.vllm` beside vllm 0.20.0
  (the prebaked `--extra vllm`-only venv does *not* contain `nemo_gym` —
  hence `VLLM_GYM`, switched in at setup only when capture is enabled).
- **Splice relocation equivalence** (scratchpad
  `validate_splice_relocation.py`): Gym's `replace_prefix_tokens` produces
  byte-identical output to the RL worker's `_replace_prefix_tokens` on the
  S1-gate templates — Qwen3-0.6B (retokenization_differs=True, the
  `<think>`-strip drift case) and Qwen2.5-1.5B-Instruct — plus the
  no-prefix path. The RL original stays in place until S3 hosts the Gym
  copy on the request path, so both existing callers are untouched.

### Deviations / disclosures (for S2 sign-off)

1. **`install_capture` returns the capture instance** (S1 signature said
   `-> None`): additive; the `CaptureHost` seam remains the primary wiring.
2. **SC `_sync_weights` fan-out pulled forward from S4** (flag-gated,
   dormant): completes per-call version stamping end-to-end in S2.
3. **`pyrefly.toml` edits** are active regardless of the flag (type-checker
   config only; no runtime effect).
4. **Worker request-path integration deferred to S3** by design: the gate
   owns the gate→worker context wire shape (serving rule § 3.3), so
   `begin_call`/`complete_call` are not yet called from the HTTP handler —
   S2 delivers the hosting seam, the capture core, and the adapter.

## S3 — Gym gate

Status: **code + tests complete (2026-07-28); awaiting user sign-off at the
S3 gate.** (S2 sign-off was given verbally — "In case S2 is done can you
start S3" — with the S2 gate summary below still standing for review.)

### Gym fork (submodule branch `tq-gate-capture`)

One commit: `05986b04` "feat(token-id-capture): S3 gate — lineage hosting,
prefix serving, marker plumbing, control plane". No rebase was needed — the
fork already sits on #2124's head (`32b555f04`), and its 20-test base suite
stays green.

| Item | Status | Notes |
|---|---|---|
| `token_id_capture/memory_store.py` | done | `MemoryRolloutTokenBuffer` — per-rollout **delta forest** (parent-linked deltas; forks share prefixes through the chain instead of duplicating cumulative sequences; O(total committed tokens)); create-only register, chain-walk `cumulative_ids`, `drop` at seal/fail/TTL. Ids only, never logprobs |
| `token_id_capture/gate.py` | done | `RolloutCaptureGate` hosting the S1 `lineage.py` machine — **no new lineage logic**: registration (create-only), `find_marker` (deepest `ng_call_id` names the parent; sub-agent forks resolve to interior nodes), `message_fingerprint` (normalized role / text content with `<think>` stripped / tool-call name+args / tool linkage; capture carriers and token fields excluded — pinned by conformance tests), the § 3.3 serving rule in `prepare_call` (token-in only for known marker + matching fingerprint; else text-mode new root with `fallback_reason`), `ingest_coords` = authoritative commit (buffer extend + fingerprint record + marker release; `capture_failed` poisons and releases nothing), `seal_rollout` → token-free receipt + full state drop, `fail_rollout` / `expire_stale` TTL, § 8 metrics counters. `RolloutGateConfig` (enabled=False) |
| `token_id_capture/control_routes.py` | done | `PUT /ng-control/rollouts/{id}` (create-only, 409 on dup), `POST .../seal` (receipt; 404 after drop), `POST .../fail` (idempotent), `GET /ng-control/metrics`; `RolloutControlClient` for the framework side (aiohttp via `server_utils.request`, deferred import) |
| `openai_utils.py` | done | `CallMarkerMixin` / `CallMarkerTypedDictMixin` + `WithMarker` variants of the five response items, the assistant chat param, and the chat response message, mirrored on the `ForTraining` pattern; unions extended |
| `responses_converter.py` | done | outbound: marker attaches to the last content-bearing output item (`RESPONSES_TO_MARKER`, same carrier as the token arrays); inbound: an echoed marker item survives conversion onto the flushed assistant chat message. Both directions are presence-driven — no config, dormant when no gate mints markers |
| `responses_api_models/vllm_model/app.py` | done | flag-gated gate hosting: `prepare_call` after `_preprocess_chat_completion_create_params` (both sides of the fingerprint see one normalization pipeline); exact prefix into the worker's existing `required_prefix_token_ids` splice seam + `ng_capture` call context; engine logprob/token-id request fields set gate-side (worker extracts natively; the gate strips `logprobs` from the response so they never reach the agent); coords popped off the response = the commit — a **missing-coords response is committed as `capture_failed`** (rollout poisoned loudly, completion still served); marker attach; sha256(rollout_id) → client **affinity** (stateless, no map to leak); control routes installed in `setup_webserver`; backend errors/context-length shortcuts fail the admitted call (no marker → no children); setup-time `ValueError` when gate + legacy token echo are both enabled |
| tests | done | `tests/unit_tests/test_token_capture_s3_gate.py` — 21 tests: **all 4 S1 golden call sequences replayed through the gate produce receipts byte-identical to the direct `RolloutLineage` drive** (fixture→gate call-id renaming only); fallback matrix (marker stripped / history edited / unknown marker / reasoning-strip fingerprint stability); duplicate + wrong-rollout coords rejection; capture-failed poisoning; seal-drops-state + TTL; buffer fork prefixes; control-route round trips; converter marker echo round trip. `responses_api_models/vllm_model/tests/test_token_capture_gate_app.py` — 5 server e2e tests over HTTP with a fake capture-enabled worker (register → 2-turn conversation with exact prefix service and prev_len chaining → seal receipt; missing-coords poisoning; edited-history two-root fallback; dormant server exposes nothing; config guard). **Full Gym suite 1507 passed / 30 skipped** incl. all prior capture suites and the 72 existing vllm_model app tests |

### Deviations / disclosures (for S3 sign-off)

1. **Buffer interface**: the in-memory buffer exposes a delta-chain interface
   rather than implementing #2124's `TokenCaptureStore` (JSONL `TokenEntry`
   append/read) — prefix serving needs parent-linked deltas, not per-call
   full snapshots. The JSONL store remains untouched as the debug/persistence
   backend (H1).
2. **Marker carriers are presence-driven, not config-gated**, in the
   converter and typed models: markers can only exist when a gate minted
   them, so the legacy path is byte-identical with the flag off (union
   additions verified against the full 1507-test suite).
3. **Rollout affinity is a stable hash** (sha256(rollout_id) mod clients),
   not the prototype's sticky map — stateless, so nothing leaks when
   rollouts outlive sessions. Design § 9.2 asked for "rollout affinity in
   `_resolve_client`"; the mechanism choice is disclosed here.
4. **Gate-side engine fields**: in gate mode the gate (not
   `return_token_id_information`) sets `logprobs=True, top_logprobs=0,
   return_tokens_as_token_ids=True` on the worker request — the S2 adapter's
   extraction shapes need them until native extraction is wired deeper
   (worker-side stripping + coords attach is the S4 RL hookup).
5. **`fail_rollout` control route is idempotent** (returns `failed: false`
   after seal/TTL instead of erroring) — a cancelled dispatch double-fail
   must not crash teardown (§ 7 cleanup).
6. Functional-test side effect: `ng_prepare_data` rewrote
   `workplace_assistant` metrics JSONs in the submodule working tree during
   the flag-off evidence run; reverted, not committed.

Not in S3 (lands in S4 with the RL wiring): NeMo-RL's use of
`RolloutControlClient` (register/seal/fail from `environments/nemo_gym.py`),
rollout-id minting + metadata plumbing from the SC, worker-side coords
attach/strip (the serving hookup that pairs with S2's `RolloutTokenCapture`),
receipts through `run_rollouts`, and the finalizer.

## S4 — receipts, finalizer, SC integration

Status: not started (blocked on S3 sign-off)

## S5 — verification

Status: not started (blocked on S4 sign-off)
