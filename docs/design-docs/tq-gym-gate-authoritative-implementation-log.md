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
   the flag-off evidence runs; reverted, not committed.

### Regression evidence (flag off, S3 pin)

- `tests/functional/grpo_async_gym_single_controller.sh` re-run against the
  S3 pin (`05986b04`, gate code present but `token_capture_gate.enabled`
  defaulting false): **PASS** (2026-07-28) — `median(gen_kl_error)`=0.038
  < 1.3, `max(reward)`=0.5 > 0. The legacy token-echo path through the
  edited `app.py`/converter is behaviorally unchanged.
- Full Gym unit suite at the S3 commit: 1507 passed / 30 skipped.

Not in S3 (lands in S4 with the RL wiring): NeMo-RL's use of
`RolloutControlClient` (register/seal/fail from `environments/nemo_gym.py`),
rollout-id minting + metadata plumbing from the SC, worker-side coords
attach/strip (the serving hookup that pairs with S2's `RolloutTokenCapture`),
receipts through `run_rollouts`, and the finalizer.

## S4 — receipts, finalizer, SC integration

Status: **complete (2026-07-28) — code, unit tests, and live capture-enabled
gate evidence (below) all green; awaiting user sign-off at the combined
S2–S4 gate.** (S3 sign-off pending — the user asked to "Finish S4"; S2–S4
are presented together at the gate.)

All NeMo-RL-side; no Gym fork changes in this stage.

| Item | Status | Notes |
|---|---|---|
| `nemo_rl/experience/blackbox_finalizer.py` (new) | done | orchestration only: `finalize_rollout` — receipt → `TQTokenSource.fetch` by manifest keys → § 5 verification (digest recompute over fetched float32 values, mask ∈ {0,1}, finite logprobs, `prev_len + delta_len == cum_len`, weight-version tag equality) → Gym `linearize(main_chain_only, terminal_hint)`; `finalize_group` — always N rows (`{group_id}_g{i}` == the gate-registered rollout ids), placeholders (`sample_mask=0`, `prompt_ids_for_adv` from a valid sibling), `group_min_wv`/`group_max_wv` (fallback wv for all-placeholder groups), `min_valid_fraction_per_group` group drop, `mixed_weight_version_policy` allow/reject, publish via `pack_payload` → `put_samples`, then clear the group's staged rows (finalizer is the staging partition's only reader) |
| worker request path (`vllm_worker_async.py`) | done | the S2/S3 pairing: `_begin_request_capture` at `preprocess_chat` (both render paths — post-splice ids in token-in mode, full render in text mode), `_finish_request_capture` in the chat endpoint (stage → coords ride `ng_commit_coords`; logprobs stripped so the worker→gate hop is token-light § 3.2), `_abort_request_capture` on every endpoint error path; `ng_capture` field on `NeMoRLChatCompletionRequest` |
| `environments/nemo_gym.py` | done | `token_capture` on `NemoGymConfig`; `_spinup` injects the gate config through the `policy_model.responses_api_models.vllm_model` global-config override block (`token_capture_gate.enabled=true`, `return_token_id_information=false`) and **hard-errors unless `rollout_max_attempts_to_avoid_lp_nan == 1`** (NaN-retry would re-register create-only ids); control-plane helpers (`register_rollouts`/`fail_rollouts`/`gate_metrics` + seal) via Gym's `ServerClient` resolving the `policy_model` server by name; receipt-mode `run_rollouts` registers before dispatch, seals per completed row, and returns token-free results (the legacy token walk + contiguity assert never runs — the gate owns that guarantee) |
| `experience/rollout_manager.py` | done | capture dispatch `_generate_and_finalize`: mints `{group_id}_g{i}` (sample ids == rollout ids), reserves the slot with them, threads them into row `metadata.ng_rollout_id`, finalizes via `asyncio.to_thread`, commits via `commit_finalized`; failure path aborts the slot **and** best-effort-fails the gate registrations; receipt-mode `Completion` (token-free; receipt in `env_extras`); receipt-derived token metrics; receipts excluded from the wandb table |
| `algorithms/advantage_estimator.py` + SC | done | `GRPOAdvantageEstimator.compute_advantage(valid_mask=...)` replaces the hardwired `torch.ones_like` in `calculate_baseline_and_std_per_prompt`; the SC advantage pump passes `sample_mask` (validity folded, no new train field); other estimators absorb the kwarg |
| `single_controller_utils/setup.py` | done | MVP-matrix validation (requires NeMo-Gym path + vllm + async engine, loud `ValueError`/`NotImplementedError`); `VLLM_GYM` registry override for `VllmAsyncGenerationWorker` **before** generation builds; `setup_token_capture` + `set_rollout_weight_version(0)` fan-outs after partition pre-registration; `BlackboxFinalizer` built and threaded into `RolloutManager`; gate config into `spinup_nemo_gym_actor` |
| exemplar YAML | done | documented `token_capture` block in `examples/configs/grpo_math_1B_single_controller.yaml` (defaults live on `TokenCaptureConfig`); config-validation suite green |
| pyrefly | done | `blackbox_finalizer.py` added to `project-includes`; no new errors |

### Tests (all green)

- `tests/unit/data_plane/test_blackbox_finalizer.py` — 5 tests vs a **live TQ
  simple backend**: the S1 worked-example golden row reproduces byte-exact
  through staging + finalize; the rejection matrix (missing receipt,
  poisoned, empty manifest, identity mismatch, missing rows, digest
  corruption); mixed-wv allow/reject; N-row publish with sibling-prompt
  placeholder + staging cleanup; `min_valid_fraction_per_group` drop.
- `tests/unit/models/generation/test_vllm_token_capture_hosting.py` — +4
  request-path tests (10 total): begin→finish round trip (stage before
  coords, logprobs stripped, state drained), token-in `prev_len` chaining,
  no-op off the capture path, abort semantics.
- `tests/unit/experience/test_rollout_manager.py` — +3 receipt-mode tests
  (13 total): id minting/threading end-to-end, `commit_finalized` carry,
  dropped-group abort, failed-dispatch abort + gate `fail_rollouts`.
- `tests/unit/algorithms/test_advantage_validity.py` — 2 tests: invalid rows
  excluded from the per-prompt baseline; `None` keeps legacy behavior.
- Flag-off regression: `tests/unit/single_controller/` +
  `tests/unit/experience/` + `test_config_validation.py`: **522 passed**
  (same two pre-existing branch-HEAD failures excluded, documented at S1).

### Deviations / disclosures (for S4 sign-off)

1. **Finalizer runs in the dispatch task** via `asyncio.to_thread` (design's
   MVP placement); finalize latency rides the rollout dispatch, not the
   train pump.
2. **Gate config injection** uses the `policy_model` global-config override
   block (the mechanism env yamls already use) instead of new
   `global_config.py` keys — no Gym-side config change needed.
3. **Legacy test fixtures updated**: `_make_manager`/hand-built managers in
   `test_rollout_manager.py` gained the new `_finalizer`/`_env_handles`
   attributes (the legacy `run_rollout` impl call stays byte-identical —
   verified by the pre-existing flow tests).
4. **`gate_metrics` control endpoint** is exposed via the NemoGym actor but
   not yet logged per train step (receipt-derived rollout metrics +
   `finalize/*` metrics land in `rollout_metrics`); wiring
   `token_in_rate` into the SC logger is S5 work with the § 8 metrics pass.
5. **`commit_finalized` staging_keys are empty** by design: the finalizer
   clears staged rows right after publish, so eviction has nothing extra to
   clear (the design's staging-aware `remove` remains for abnormal paths).

### Gate evidence (2-GPU capture-enabled run)

- `grpo_async_gym_single_controller.sh ++token_capture.enabled=true`
  (2026-07-28, dev node, 2×H100, 10 steps): **PASS** — both metric checks
  green: `median(gen_kl_error)`=0.0375 < 1.3 (statistically identical to
  the flag-off S3-pin run's 0.038 — the gate→worker→TQ→finalizer path
  reproduces legacy token fidelity), `max(reward)`=0.5 > 0. Every train
  step ran with `global_valid_seqs=8.0` — all rows finalizer-verified
  valid, zero placeholder rows trained.
- Observed failure-path exercise (loud, handled, § 7 semantics): 3 of the
  ~40 dispatched groups were cancelled by the SC mid-flight; their agents'
  in-flight calls then hit the gate after `fail_rollouts` dropped the
  create-only registrations → `UnknownRolloutError` from
  `gate.prepare_call` → HTTP 500 to the (already-dead) rollout, and one
  group's late `seal` correctly got 404 and was absorbed by the
  `seal(...) failed` warning path. No leaks into training: none of these
  groups produced rows, and all trained steps were full-valid.
- Environment caveat (unchanged from S1): first attempt failed on the
  node's stale prebaked `/opt/ray_venvs` (vllm 0.17.1/Ray 2.54.0 vs the
  lock's 0.20.0/2.55.1 → `ModuleNotFoundError:
  vllm.entrypoints.serve.render`); the recorded PASS is the rerun with
  `NRL_FORCE_REBUILD_VENVS=true`.

## S5 — verification

Status: **in progress (2026-07-28).** S4 signed off ("Commit S4 start S5")
and committed as RL `6b32665ca`. Work items (§ 10 S5): fixed-seed
legacy-vs-capture row diff, reward-curve comparison, per-call HTTP bytes
vs the echo path (+ `token_in_rate` into the SC logger, deferred from S4),
chaos smoke (kill gate mid-step), capture-enabled functional test in
`L1_Functional_Tests_SingleController.sh` (blocked on the gitlink/CI
decision). S5 sign-off = MVP acceptance.

### Row diff: legacy vs capture (2026-07-28)

Method: env-gated row dump (`nemo_rl/experience/row_dump.py`,
`NRL_SC_DUMP_TRAIN_ROWS=<dir>`, no-op unless set) hooked at both canonical
publish sites; identical direct-invocation runs (same node/placement/data)
differing only in `token_capture.enabled`; offline matcher keyed by
`(weight_version, prompt_ids_for_adv)`.

Constraints discovered (documented, not fixable in-scope):

- **No per-request sampling determinism exists** (engine seed is
  placement-derived; `SamplingParams` carries no seed; the Gym path
  rejects `top_k` overrides at `rollout_manager.py:452`; dataset-level
  `temperature` is overwritten by the generation config at
  `rollout_manager.py:474`).
- `policy.generation.temperature=0` (greedy) reaches the engine but NaNs
  the loss at iteration 1 on **both** paths identically (vLLM's degenerate
  greedy logprobs → NaN grad norm) — so the byte diff is a one-step
  (wv=0) comparison, and full-length curves run at temperature 1.
- vLLM is cross-run nondeterministic under continuous batching (logit
  jitter with batch composition), bounding what any cross-run diff can
  show.

Results:

- Temp-1 pair (10 steps, 80 rows/run): all 40 group keys matched 1:1,
  **prompt prefixes byte-identical in every row**; generated suffixes
  diverge (sampling, as expected).
- Greedy wv=0 pair (identical weights): **7/8 rows byte-identical in ids
  and masks end-to-end** across two entirely different pipelines
  (echo-splice-tensorize vs gate-TQ-finalizer). The 1/8 divergence is a
  mid-generation argmax flip at a near-tie (candidate logprobs −1.01 vs
  −0.97) — engine jitter, not pipeline drift. Logprob deltas on
  identical-token rows: exactly 0.0 on several rows, ≤ 0.147 max
  elsewhere (batch-composition numerics; a transformation bug would be
  systematic, not zero-on-some-rows).
- Step-1 rows are all single-call at this sequence budget; cross-run
  multi-turn splice fidelity is not separately re-proven here — it rests
  on the S1 live token-in smoke, S3 gate e2e prefix chaining, the per-row
  digest verification live in every capture run, and the capture run's
  gen_kl_error (0.0375) matching legacy (0.038).

### Reward-curve comparison (2026-07-28, temp-1 pair, 10 steps)

- `train/reward` per step: **identical 10/10** —
  `[0, 0, 0.25, 0, 0.25, 0.25, 0.25, 0.5, 0, 0]` on both paths (seeded
  dataset order; same prompts per step; same reward outcomes).
- `train/gen_kl_error`: same band — legacy median 0.0379 (max 0.068),
  capture median 0.0358 (max 0.046).
- `train/global_valid_seqs` = 8.0 every step on both paths.

### Wire metrics (2026-07-28)

- `gate/*` metrics wired into the SC logger (the S4 deferral): per-step
  fetch of the gate's cumulative § 8 counters through a new
  `RolloutManager.gate_metrics()` passthrough, logged with derived
  `gate/token_in_rate`; fetch failures are swallowed loudly (metrics never
  kill a step). Live reading recorded with the chaos run below.
- Per-call token-carrier bytes, computed from the temp-1 run's actual rows
  (JSON as on the wire): legacy echo attachment
  (`prompt_token_ids`+`generation_token_ids`+`generation_log_probs`) mean
  **2,540 B/call** (8.6 B/token at ~296 tok/call) vs capture marker
  **35 B/call** — **−98.6 %** on the gate→agent carrier for this
  single-turn workload; multi-turn re-echo compounds the legacy side per
  turn. The full HTTP-level perf report (bytes/token, step time, gate
  latency) remains post-MVP validation per § 10.

### Instrumented perf A/B (2026-07-28, user-requested; pulled forward from
### post-MVP validation)

Method: env-gated ASGI byte counters on every HTTP server (Gym
`HttpByteCounterMiddleware` in `server_utils.py`, `NG_HTTP_BYTES_DIR`; RL
mirror `nemo_rl/utils/http_byte_counter.py` on the vLLM worker app,
`NRL_HTTP_BYTES_DIR`); identical 10-step runs at 1024 seq budget differing
only in the flag; per-hop aggregation + timing from TB metrics.

| Metric | Legacy | Capture | Δ |
|---|---|---|---|
| HTTP bytes / trained token | 107.3 B | 69.1 B | **−35.6 %** |
| Total HTTP bytes (10 steps) | 2.52 MB | 1.64 MB | −35 % |
| `total_step_time` median | 4.48 s | 4.07 s | **−9.0 %** |
| `exposed_generation` median | 1.99 s | 1.65 s | **−16.8 %** |
| `valid_tokens/s/GPU` median | 30.8 | 34.7 | **+12.5 %** |

Per-hop highlights: the worker's `/tokenize` route disappears entirely
(77 calls → 0); worker `/v1/chat/completions` response bytes 0.34 → 0.19 MB
(logprob echo gone); gate `/v1/responses` responses 0.28 → 0.13 MB (token
arrays → markers); even the verifier hop shrinks (0.37 → 0.23 MB in) since
agent histories no longer carry token arrays. Health parity: same KL band
(0.0376 vs 0.0413 median), identical `max_reward` 0.5, all rows valid.

Caveats: this workload remains single-call per rollout even at 1024
(gate counters: 80 registered/sealed, 0 token_in, 0 fallbacks — all roots),
so the echo's per-turn compounding and `token_in_rate` are not exercised —
the prototype's −46.9 % bytes/token multi-turn number remains the
reference; 10-step timing on a 0.6B model is directional, not a benchmark.

Reading of the numbers (assessment):

- **The bytes reduction is structural, not statistical** — it comes from
  routes/payloads that categorically no longer exist (`/tokenize` gone,
  logprob echo gone, token arrays → 35 B markers), and this single-call
  workload is capture's *worst case*: legacy never paid its per-turn
  history re-echo (roughly quadratic in turns) while capture stays
  O(generated tokens). −35.6 % is a floor that grows with agentic depth
  and context length.
- **The timing wins are plausibly real but not yet bankable**: one run
  pair, n=10 medians, 0.6B model — variance on this stack can be
  ±5–10 %. In their favor: a concrete mechanism (one fewer HTTP
  round-trip per call + ~40 % smaller payloads on the generation critical
  path) and the gain concentrating in `exposed_generation` (−16.8 %)
  exactly where the mechanism predicts. Expect the relative timing win to
  compress on large models (GPU decode dominates) while the bytes win
  grows.
- **Capture is at minimum perf-neutral while buying provenance**: the
  gate, staging writes, digest verification, and finalizer all sit in the
  measured path and the capture run is faster, with identical training
  health — the design's custody guarantees carry no perf tax.
- To make the timing claim quotable: 3–5 repeated pairs (variance bars)
  and a genuinely multi-turn workload (untrimmed tools, larger budget),
  which is also the run that measures a real `token_in_rate` — earmarked
  as the post-MVP perf report (§ 10).

### Chaos smoke: gate killed mid-step (2026-07-28)

Method: capture run, `SIGKILL` to the verified `policy_model` (gate)
process after step 3, 3-minute observation, then teardown. (Four earlier
attempts were invalidated by harness bugs — orphaned wrapper teardown,
GPU-squatting orphan EngineCore, a pgrep pattern that couldn't match the
gate, a task timeout — all documented in the session log; none were
capture-path defects.)

Verdict vs the § 7 failure model:

- **No corruption, no crash**: the SC actor and run stayed alive; step
  count froze at 3 — nothing trained on bad data after the kill, and the
  three completed steps were healthy.
- **FINDING (S5 → H1): gate death is a silent stall, not the promised
  loud failure.** The NemoGym actor's control-plane `register_rollouts`
  call to the dead gate sat in Gym's `server_utils.request` retry loop —
  observed at **retry=375+ over the full window** (`ClientOSError`,
  unbounded for connection errors) — so the dispatch never failed, the § 7
  fail-path (abort slot → `fail_rollouts` → placeholders) never engaged,
  and the run would stall indefinitely. § 7's "SC dispatch timeout" row
  presumes a timeout that is not wired for control-plane calls.
  Recommended fix (H1 scope, where the design already places the failure
  sweep + kill-gate CI test): bound control-plane retries/time
  (`RolloutControlClient` request timeout), so gate death surfaces as
  failed dispatches + placeholders + staging TTL, per the § 7 table.
- Staging-TTL sweep not observable in a 3-minute window (TTL 3600 s);
  covered by unit tests.

S5 finding fixed en route: agent `/run` 500s raise
`aiohttp.ClientResponseError` out of `run_rollouts`, and Ray cannot pickle
its `CIMultiDictProxy` headers — the SC saw a masking `TypeError` instead
of the real error. Fixed in `environments/nemo_gym.py` (catch + re-raise
picklable `RuntimeError`). **Active with the flag off** (any legacy agent
500 hit the same masking); disclosed for the S5 gate.
