# Token Capture v2 in the Async SingleController Pipeline

Design and implementation plan for running NeMo-Gym rollouts through the
**Token Capture v2** architecture — Gym-owned capture, NeMo-RL as the byte
mover — inside the async SingleController (SC) GRPO pipeline on this branch.

**Target architecture:** the v2 ownership split (`tq_gym_v2.md` /
`tq_gym_v2_rollout.html` in the prototype checkout). **Reference
implementation / donor code:** the sync-only prototype at
`/lustre/fsw/portfolios/coreai/users/pthombre/gym/RL/`
(branch `pranav/tq_gym_prototype`), which proved the dataflow and the perf
numbers but predates the v2 ownership split — it is quarry, not foundation.

**Sources reconciled:**

- This branch (`yukih/sc-entrypoint`): async SC pipeline —
  `nemo_rl/algorithms/single_controller.py`, `single_controller_utils/`,
  `experience/rollout_manager.py`, `async_utils/replay_buffer.py`
  (`TQReplayBuffer`), `async_utils/staleness_sampler.py`, `models/policy/tq_policy.py`.
- v2 design docs: `tq_gym_v2.md` + `tq_gym_v2_rollout.html` (ownership split,
  `parent_hint`, stage acks, enriched seal receipts, terminal-hint linearize).
- Prototype (donor): `experience/rollout_writer.py`,
  `experience/blackbox_finalizer.py`, `experience/staged_token_source.py`,
  `models/generation/vllm/vllm_worker_async.py`, `algorithms/grpo_sync.py`;
  Gym pinned as an editable workspace member with the ingress-gate stack.
- Gym upstream: PR [#1967](https://github.com/NVIDIA-NeMo/Gym/pull/1967)
  (closed) and the live [#2124](https://github.com/NVIDIA-NeMo/Gym/pull/2124)–#2128
  stack (open; a reduced file-backed capture core — not yet the v2 surface).

---

## 1. Goal

Replace the async SC's Gym rollout path — where every generated token transits
Gym HTTP responses, a Ray return, and an SC-side tensorize before reaching the
TransferQueue — with the v2 capture pipeline:

- **Gym owns every decision** about tokens and lineage: envelope parsing, hint
  confirmation, parent fallback, delta/mask construction, hash chain, digest,
  stage→commit ordering, the registry state machine, rebuild, and
  linearization.
- **NeMo-RL implements exactly two protocols and hosts two processes**: a
  `TokenSink` that lands bytes in TQ, a `ForestCursor` that transports
  registry calls to a Ray actor shell hosting Gym's state machine, plus the
  finalizer orchestration (fetch, reconcile policy, publish) and the
  `weight_version` value.
- The SC pipeline consumes the result through its existing seams: a buffer
  slot per prompt group, canonical rows in the existing `rollout_data`
  partition, unchanged train pump.

Proven upside from the prototype (sync loop, same dataflow): **−46.9 % HTTP
bytes/generated token, −55.7 % HTTP exchanges, −45.3 % terminal Gym→RL
bytes/sample, −4.3 % total step time (p50)**. v2 additionally converts the
`ambiguous_forest` rejection class (sub-agent branches) into trainable rows
via terminal-hint linearization.

## 2. The v2 contract

Everything above the line ships in Gym's `nemo_gym/token_capture/` leaf
package (no fastapi, no Ray, no TQ imports); everything below is the entire
surface NeMo-RL writes.

```
GYM  records.py    StagedCallRecord, StageAck, DelegationEnvelope, ParentHint,
                   ForestCandidate, Reservation, ParentClaim, CommitCoords
GYM  hashing.py    hash_token_ids, compute_staging_digest, EMPTY_PREFIX_HASH, SCHEMA_VERSION
GYM  protocols.py  TokenSink.stage(rec) -> StageAck          # data plane
                   ForestCursor.get_candidates/reserve/commit/fail   # control plane
GYM  forest.py     ForestCursorStateMachine (+ CursorConflictError,
                   DuplicateRequestError, CursorFailedError) — retry-idempotent
                   reserve, parent validation, growth check, leases, manifest order
GYM  capture.py    RolloutTokenCapture.begin_call / complete_call — hint confirm,
                   candidate-scan fallback, delta build, fail-closed ordering
GYM  rebuild.py    staged deltas -> TokenEntrys -> linearize(policy, terminal_hint)
GYM  adapters/vllm.py  hook wiring + token extraction; install_capture(...)
GYM  gate          admission, call_id mint, trajectory tree + coordinate ledger,
                   parent_hint stamping, stage-ack backfill, seal -> enriched receipt
────────────────────────────────────────────────────────────────────────────
RL   TQTokenSink(dp_client, partition="rollout_staging")      # one method: stage()
RL   RayForestCursor(registry_actor)                          # ~20-line transport
RL   RolloutForestRegistry (Ray actor SHELL hosting Gym's state machine)
RL   finalizer orchestration: row fetch, reconcile policy, publish, placeholders
RL   worker setup (once): install_capture(worker, sink=…, cursor=…,
                          weight_version_fn=lambda: self._rollout_weight_version)
RL   weight_version value (trainer state)
```

Wire behavior (v2 additions over the prototype): the gate stamps a
`DelegationEnvelope` with a `parent_hint` (~120 B); the worker confirms the
hint with **one hash** and only falls back to the candidate scan on a miss;
the `StageAck` (~100 B) rides the response back so the gate's coordinate
ledger stays current; seal returns an **enriched receipt** with
`terminal_call_id` + a token-free tree summary, letting the finalizer run
`linearize(policy="main_chain_only", terminal_hint=…)`.

### 2.1 Gap between v2 and what exists today

| v2 element | Today |
|---|---|
| `nemo_gym/token_capture/` package | Does not exist. The prototype's Gym pin has the ingress gate + a flat manifest (no tree/ledger, no hints, no acks); upstream #2124 is a file-backed capture core without the gate/forest surface. |
| `ForestCursorStateMachine` in Gym | Implemented **in NeMo-RL** (`rollout_writer.py`), same semantics — the direct donor for Gym's `forest.py`. |
| `TQTokenSink` / `RayForestCursor` / `install_capture` | Staging is inlined in `vllm_worker_async.py`; workers call the registry actor directly; wiring is `configure_rollout_writer(...)` pushed from the driver. |
| `parent_hint` / stage acks / enriched receipts | Absent — worker always candidate-scans; `stage_disposition` is inferred at seal; branching rollouts reject as `ambiguous_forest`. |
| `rebuild.py` in Gym | Split across NeMo-RL (`staged_token_source.py`) and Gym (`trajectory/builder.py`). |

The migration is therefore **two coordinated tracks** (§ 7): Gym lands
`token_capture` (largely by relocating prototype-proven code per the v2 doc's
own 6-step migration), and NeMo-RL builds the framework half against the v2
protocol names from day one — so nothing is written twice.

## 3. Current async SC rollout path (what changes)

```
_rollout_pump ─► RolloutManager.generate_and_push(prompt)
                   ├─ tq_buffer.reserve(weight_version)          # slot, dispatch order
                   ├─ AsyncNemoGymRolloutImpl.run_rollout
                   │    └─ NemoGym.run_rollouts.remote(rows)     # tokens echoed back ✗
                   └─ tq_buffer.commit(group_id, record, …)
                        ├─ record_to_train_batch(record)          # tensorize on SC ✗
                        └─ dp_client.put_samples(N rows)          # rollout_data partition
_train_pump  ─► sampler.select → logprobs → _advantage_pump → train_microbatch_from_meta
_sync_weights ─► pause dispatches → WeightSynchronizer.sync_weights → resume
```

The two ✗ steps disappear: Gym's capture library, running inside the vLLM
worker, lands every token in TQ at generation time through NeMo-RL's
`TQTokenSink`. The SC keeps its reserve/commit slot discipline — `commit`
becomes *"finalize the group's staged rows and record the resulting meta."*

## 4. Target design

### 4.1 Data flow

```
SC _rollout_pump
  └─ generate_and_push(prompt, target_step)
       ├─ tq_buffer.reserve(weight_version=v_start)                    (unchanged)
       ├─ mint N rollout_ids, register at gate                         (new)
       ├─ NemoGym.run_rollouts.remote(rows + rollout_ids)              (token-free)
       │     agent ⇄ gate (tree + ledger, parent_hint) ⇄ vLLM worker
       │        Gym capture: begin_call (hint confirm | scan | new root)
       │                     → cursor.reserve            [RayForestCursor → registry shell]
       │                     complete_call: delta/hash/digest
       │                     → sink.stage(record)        [TQTokenSink → rollout_staging]
       │                     → cursor.commit(coords)     fail-closed
       │        StageAck rides the response → gate ledger backfill
       ├─ seal each rollout → enriched RolloutReceipt (+ reward)       (new)
       ├─ finalize group: receipt × manifest reconcile → fetch + Gym
       │     re-verify → rebuild → linearize(terminal_hint) →          (new)
       │     N canonical rows → put_samples(rollout_data, {group_id}_g{i})
       └─ tq_buffer.commit_finalized(group_id, meta, versions)         (changed)

SC _train_pump: unchanged, except advantage/validity handling (§ 4.5)
SC _sync_weights: additionally rotates the workers' weight_version (§ 4.4)
```

Two TQ partitions, distinct lifecycles:

- `rollout_staging` — one delta row per model call, keyed
  `{rollout_id}/{call_id}`, written by `TQTokenSink.stage` (the framework's
  entire hot-path contribution); cleared by the finalizer after publication,
  by eviction, and by TTL.
- `rollout_data` (existing SC partition) — canonical per-sample training rows;
  today's schema (`input_ids`, `input_lengths`, `generation_logprobs`,
  `token_mask`, `sample_mask`, `prompt_ids_for_adv`, `total_reward`)
  **plus `trajectory_valid_mask`**; lifecycle unchanged (select → train →
  `clear_samples`).

### 4.2 NeMo-RL components (the whole framework surface)

- **`nemo_rl/data_plane/tq_token_sink.py`** — `TQTokenSink`: maps a
  `StagedCallRecord` to a TensorDict row + tags and calls `put_samples`. TQ
  vocabulary appears nowhere else in the capture path.
- **`nemo_rl/experience/rollout_writer.py`** (slimmed) — `RayForestCursor`
  (each method awaits the corresponding registry-actor method) and
  `RolloutForestRegistry`, reduced to an actor **shell**:
  `self.state = ForestCursorStateMachine(...)` imported from Gym `forest.py`;
  every method delegates 1:1. Lease/TTL knobs come from config; semantics,
  idempotency, and the exception taxonomy are Gym's.
- **Worker setup** — one call at vLLM async-worker startup:
  `install_capture(worker, sink=TQTokenSink(...), cursor=RayForestCursor(...),
  weight_version_fn=lambda: self._rollout_weight_version)`. The
  driver-pushed `set_rollout_weight_version(int)` remains the mechanism that
  updates the value the closure reads.
- **Finalizer** (`nemo_rl/experience/blackbox_finalizer.py`, reworked) —
  orchestration only: get manifest from the registry shell, reconcile against
  the enriched receipt, fetch rows, call Gym's `rebuild` + verification
  functions (the same bytes the worker hashed), call
  `linearize(policy="main_chain_only", terminal_hint=receipt.terminal_call_id)`,
  publish canonical rows or masked placeholders, clear staging + registry.
- **`TQReplayBuffer.commit_finalized`** and the setup wiring in
  `single_controller_utils/setup.py` (§ 6).

### 4.3 Where finalization runs; group assembly

Finalization is CPU + TQ I/O work, run per prompt group inside the existing
`generate_and_push` dispatch task via `asyncio.to_thread` — a group's slot
flips ready only when its canonical rows exist, preserving today's ordering.
If finalize latency (~60 ms/rollout measured on sync) shows up in
`exposed_generation`, promote to a small finalizer actor pool behind the same
`finalize_group(group_id, rollout_ids, receipts) -> GroupFinalizeResult` seam.

Group semantics (GRPO needs N generations per prompt group):

1. Finalize each of the N rollouts; any rejection (manifest mismatch, digest
   mismatch, non-finite logprobs, residual ambiguity) becomes a masked
   placeholder row — `trajectory_valid_mask=0`, `sample_mask=0`, reward
   zeroed — so the group always publishes exactly N rows and
   `shard_meta_for_dp` invariants hold. With terminal-hint linearize, the
   dominant v1 rejection class (`ambiguous_forest`) becomes a trained main
   chain with verified-but-untrained side branches.
2. `prompt_ids_for_adv` for a placeholder is copied from a verified sibling;
   a fully rejected group keeps a constant fallback and zero advantage.
3. Rewards come from the seal receipts, never from token traffic.
4. Optional `min_valid_fraction_per_group`: below it, drop the group entirely
   (slot removed, staging cleared, capacity permit released — the SC's
   over-sampling machinery already tolerates disappearing groups).

### 4.4 Weight versions, refit, and staleness — the async-specific design

Nothing in the v2 docs covers async; these are the new decisions:

1. **Per-call tagging is the source of truth.** `weight_version_fn` reads
   worker state that `_sync_weights` rotates: after
   `weight_synchronizer.sync_weights()` and before `_rollout_permitted.set()`,
   the SC fans out `set_rollout_weight_version(trainer_version)`. Every
   `StagedCallRecord` then carries the true version of the weights that
   produced it — strictly better than SC-side start/end stamps.

   > **TODO (T1, deferred): atomic version rotation.** The driver fan-out
   > races with the weight swap per worker: calls completing inside the refit
   > window can be mis-tagged by ±1 version. Tolerable for the MVP matrix
   > (`staleness_window` + `allow`, which absorbs ±1 by design; no worse than
   > the legacy async path, which cannot see straddles at all). The fix —
   > thread `weight_version` through `WeightSynchronizer.sync_weights()` into
   > the worker's update handler so swap and stamp happen in one task, with
   > capture sampling the version at `begin_call` and `complete_call`
   > (mismatch → stamp older + `wv_straddled` flag) — is a **prerequisite**
   > for `strict_on_policy` / `mixed_weight_version_policy: reject`, which
   > must not ship on the racy mechanism.
2. **Group staleness = oldest call version.** `finalize_group` computes
   `group_min_wv`/`group_max_wv` across all calls of all N rollouts;
   `commit_finalized` stores `group_min_wv` as the slot's effective version.
   The `StalenessSampler` is unchanged — its window test now uses the
   conservative oldest-call version, so a refit-straddling rollout is evicted
   exactly when its oldest tokens age out. The reserve-time stamp remains for
   dispatch-order/quota accounting only.
3. **Mixed-version groups are a policy, not an error.** New
   `async_rl.mixed_weight_version_policy: allow | reject`:
   - `allow` (default under `staleness_window`): finalize normally; log
     `wv_spread = group_max_wv − group_min_wv`.
   - `reject` (forced by `strict_on_policy`): version-spanning rollouts become
     placeholders. `strict_on_policy` should also re-enable the drain
     (`_inflight_rollouts → 0`) before syncing, making spans impossible;
     `reject` is then a safety net.
4. **`generation_logprobs` are behavior-policy logprobs** for the version that
   generated each token; the per-token importance-sampling correction in
   `ClippedPGLossFn` handles mixed-version sequences, and the staleness window
   bounds how far off-policy they can drift.

### 4.5 Buffer, eviction, and cleanup

Every path where a group can die must clear **three** stores — the SC slot,
the canonical rows, and the staging rows + registry state:

- `TQReplayBuffer.remove(..., remove_in_dp=True)` (used by `sampler.evict`)
  additionally clears `rollout_staging` keys and calls
  `registry.clear_rollout(rid)` for the group's rollouts (slots record their
  rollout_ids at reserve time).
- Post-train `clear_samples` is unchanged — staging was already cleared at
  publication.
- SC shutdown / cancelled dispatch tasks must `fail_rollout` registered
  rollout_ids and clear staged rows (`try/finally`, mirroring the existing
  `sem.release()` discipline); otherwise staging leaks until `cursor_ttl_s`.
- Backpressure: `max_buffered_rollouts` bounds groups; staging is additionally
  bounded by `max_inflight_prompts × num_generations_per_prompt ×
  max_rollout_turns` delta rows — size the partition accordingly and keep the
  registry's `expire_stale` TTL as the backstop.

### 4.6 Constraints

- vLLM async backend first (Gym's `adapters/vllm.py`); an SGLang port is a new
  adapter file plus the same `install_capture` call — the SC-side design is
  engine-blind by construction.
- Streaming and `n>1` per request are rejected by the adapter (v1 behavior,
  unchanged until Gym's adapter supports them).
- `rollout_max_attempts_to_avoid_lp_nan` must be 1 (gate registration is
  create-only; sealing is terminal). Non-finite logprobs → placeholder, not
  retry.
- Router replay requires `routed_experts` in the staged-record `extras` —
  supported by the v2 record shape, deferred until needed.
- Gateway→worker identity rides the `DelegationEnvelope`; the vLLM endpoint
  must remain network-isolated (no auth on that hop).

## 5. Config surface

```yaml
data_plane:
  token_capture:                 # renamed from the prototype's rollout_writer
    enabled: false
    # no shadow mode: verification is offline row-equivalence (see § 7 M3)
    staging_partition: rollout_staging
    finalize_timeout_s: 30.0
    lease_ttl_s: 30.0            # registry shell → Gym state machine
    cursor_ttl_s: 3600.0         # also gate registration TTL
    linearize_policy: main_chain_only   # forwarded to Gym rebuild

async_rl:
  mixed_weight_version_policy: allow | reject   # § 4.4
  min_valid_fraction_per_group: 0.0             # § 4.3 (0 = always publish)
```

`strict_on_policy` auto-forces `mixed_weight_version_policy: reject` (same
pattern as its existing forcing of staleness/over-sampling). Misconfiguration
(`token_capture.enabled` without `env.should_use_nemo_gym`, or an unsupported
backend) raises at `setup_single_controller` time.

## 6. Component changes (this repo)

| Component | Change |
|---|---|
| `data_plane/tq_token_sink.py` (new) | `TQTokenSink` — `StagedCallRecord` → TensorDict row + tags → `put_samples`; the only file that knows TQ on the hot path |
| `experience/rollout_writer.py` (new, slim) | `RayForestCursor` transport; `RolloutForestRegistry` actor shell hosting Gym's `ForestCursorStateMachine` |
| `experience/blackbox_finalizer.py` (new) | Orchestration-only finalizer over Gym `rebuild`/`linearize`; per-group assembly, placeholders, `group_min_wv` |
| `single_controller_utils/setup.py` | Launch registry shell; register staging partition; pass gate config (registration/seal, TTLs, control token) into `spinup_nemo_gym_actor`; hand sink/cursor factories to generation setup |
| `models/generation/vllm/` | Worker startup calls Gym `install_capture(sink, cursor, weight_version_fn)`; keep `set_rollout_weight_version` fan-out; delete any inlined staging logic |
| `experience/rollout_manager.py` | Black-box mode: mint rollout_ids, register → dispatch (token-free) → seal → collect enriched receipts → `finalize_group` → `commit_finalized` |
| `async_utils/replay_buffer.py` | Slots record rollout_ids; `commit_finalized(group_id, meta, group_min_wv, group_max_wv)`; `remove()` clears staging + registry |
| `single_controller.py` | `_sync_weights` rotates worker weight_version; teardown fails/clears in-flight rollouts; metrics: `wv_spread`, hint hit rate, wasted hints, invalid-row rate, finalize latency |
| `environments/nemo_gym.py` | Register/seal control-plane helpers; gate config synthesis; receipts returned with token-free results |
| `data_plane/schema.py`, `interfaces.py`, `single_controller_utils/config.py` | Staging fields (from Gym `SCHEMA_VERSION`), `TokenCaptureConfig`, `AsyncRLConfig` additions; `AdvantageConfig` validity awareness |

**Gym dependency:** a build containing `nemo_gym/token_capture/` and the v2
gate (tree + ledger, hints, acks, enriched receipts). Until released, pin the
Gym branch where Track A lands (the prototype pin `6ea5810` is the starting
point; upstream #2124–#2128 should converge into it).

## 7. Implementation plan

**Bring-up first.** The MVP runs entirely out of this repo against the
vendored Gym pin: capture logic and the `ForestCursorStateMachine` are ported
from the prototype into NeMo-RL **behind the v2-named seams** (`TokenSink`,
`ForestCursor`, `install_capture`-shaped worker setup, defined locally).
Track A — relocating those internals into Gym's `token_capture` — is deferred
to hardening (T6) and, by construction, changes no SC-facing code when it
lands. The MVP config matrix is restricted to `staleness_window` +
`mixed_weight_version_policy: allow`; `strict_on_policy`, `reject`, and
`force_in_order` raise `NotImplementedError` at setup until T1/T4.

### MVP (get it running) — one PR, five sign-off-gated stages

The MVP is developed on a single branch and lands as **one PR**, built in five
stages. Each stage is its own signed-off commit (series), keeps the tree green
in isolation (S1–S3 are dormant behind `token_capture.enabled=false`), and
ends at a **review gate: the stage diff, test results, and any deviations
from this doc are presented for explicit user sign-off before the next stage
begins.** Protocol/record shapes freeze at the S1 gate.

- **S1 — token_capture primitives.** `nemo_rl/experience/token_capture/`
  (protocols, records + id grammar + `staging_key`, hashing, `forest.py`
  state machine — all ported from the prototype behind v2 names),
  `TQTokenSink`, `RayForestCursor` + registry shell, staging schema,
  `TokenCaptureConfig`. Forest/sink unit tests against the protocol boundary.
- **S2 — vLLM worker capture.** `capture.py` (candidate-scan only),
  `adapters/vllm.py` `install_capture`, generation fan-outs
  (`configure_token_capture`, `set_rollout_weight_version`),
  `prepare_token_capture` partition registration. Worker unit tests:
  stage→commit ordering, fail-closed, version stamping.
- **S3 — gate control plane + finalizer.** Register/seal helpers, receipts
  through `run_rollouts`, NaN-retry hard error; `blackbox_finalizer` +
  `staged_token_source`; `finalize_group` (always N rows, placeholders with
  sibling `prompt_ids_for_adv`, `group_min_wv`/`group_max_wv`,
  `min_valid_fraction_per_group`).
- **S4 — SC integration, direct mode.** Config validation (MVP matrix only —
  strict modes raise), setup wiring (§ 6), rollout-manager capture mode,
  `commit_finalized` + cleanup in the buffer, `_sync_weights` version
  fan-out, teardown sweep, validity-aware advantages (**baseline mean/std
  over valid rows only; invalid rows get advantage 0** — unit-tested in this
  stage), metrics (finalize p50/p99, invalid rate, `wv_spread`, registry RPC
  latency). Gate evidence includes a 2-GPU manual run.
- **S5 — verification, no shadow mode.** Fixed-seed job on legacy and direct
  paths, training rows dumped from TQ and diffed offline; reward-curve
  comparison; capture-enabled functional test wired into
  `L1_Functional_Tests_SingleController.sh`. S5 sign-off = MVP acceptance;
  the PR opens after it (or earlier as a draft, decided at the S1 gate).

### Hardening TODOs (ordered)

- **T1 — atomic weight-version rotation** (§ 4.4 TODO). Prerequisite for T4.
- **T2 — failure sweep + chaos test.** Registry/gate actor death → fail
  affected dispatch tasks, release permits, clear orphaned staging keys;
  kill-registry-mid-step test. First, because bring-up debugging kills actors.
- **T3 — finalizer isolation.** Dedicated bounded executor + own DP client so
  a TQ stall cannot starve the trainer's `to_thread` calls.
- **T4 — strict modes.** `strict_on_policy` (with drain re-enabled),
  `mixed_weight_version_policy: reject`, and `force_in_order` (which matches
  on the reserve-time `target_step` only; `group_min_wv` governs only the
  window modes).
- **T5 — calls-per-rollout cap.** Gate-side admission limit with a
  registry-side backstop — the design's only unbounded resource.
- **T6 — Track A: relocate into Gym** (below) + the rollout-id contract: one
  id grammar/validator, gate as sole `call_id` minter (unifying with
  upstream's `model_call_id`), `staging_key()` defined in Gym `records.py`,
  `DelegationEnvelope` as the only gate→worker carrier.
- **T7 — scale.** Registry sharding by `hash(rollout_id)` (driven by the M2
  latency counter), cross-repo metrics channel (Gym counters riding the seal
  receipt), finalizer actor pool, consistent-hash rollout→worker affinity.

### Track A — Gym: land `token_capture` (deferred to T6; mostly relocation)
1. Move hashing + record dataclasses + `build_staging_delta` from the
   prototype into `nemo_gym/token_capture/` (digests byte-identical — the
   prototype's shadow/digest tests are the check); move
   `ForestCursorStateMachine` + exception taxonomy into `forest.py` with its
   unit tests (the NeMo-RL implementation in the prototype's
   `rollout_writer.py` is the donor).
2. Define `TokenSink`/`ForestCursor` protocols; wire capture through them.
3. Move the capture algorithm (`_prepare_rollout_request` /
   `_stage_rollout_response` bodies) into `RolloutTokenCapture`.
4. Move the engine adapter into `adapters/vllm.py`; expose `install_capture`.
5. Move rebuild (`StagedSnapshotTokenSource` core) into `rebuild.py` +
   `linearize(policy, terminal_hint)`.
6. Behavior additions: gate tree + ledger, `parent_hint`, stage acks, enriched
   receipts. Ship the adapter conformance suite (golden captures →
   byte-identical records/digests).
   Steps 1–5 change no behavior and are exactly the package to upstream
   (reconciling with the #2124 stack).

Steps 1–5 change no behavior; the MVP's locally-hosted state machine and
capture logic are the donors, so the relocation is a swap behind the seams
from M1 — no SC-facing code changes. Until step 6 lands, the gate produces
v1 receipts (no hints/acks): capture always takes the candidate-scan fallback
and `linearize` runs without a terminal hint — the seams tolerate the v1 gate,
so nothing downstream waits on step 6; its features (hint hit rate, lower
`ambiguous_forest` rate) simply light up in existing metrics when it ships.

**Post-MVP validation milestone (with T1–T4 landed):** 1-off async nightly
(`grpo-llama3.1-8b …-async-1off-single-controller`) in direct mode, plus a
perf report reproducing the sync measurements (HTTP bytes/token, step time)
on the async path. Unit tests accompanying T1/T4: sampler evicts by
oldest-call version; refit-straddling rollout finalizes under `allow`,
placeholders under `reject`; fully-rejected group releases capacity.

## 8. Risks and open questions

- **Gym release coupling.** `token_capture` becomes hot-path code in every
  engine worker on Gym's release cadence; lease/TTL knobs and manifest shape
  become Gym API. Mitigations from the v2 doc: `SCHEMA_VERSION` pinning, the
  core package stays dependency-free, `adapters/vllm.py` lazily imports vLLM.
  Until upstreamed, we carry the Gym branch pin.
- **Upstream convergence.** The open #2124–#2128 stack is a different
  (file-backed) shape; Track A must reconcile with it or supersede it —
  coordination with the Gym team is the schedule risk — which is why the MVP
  runs entirely out of this repo and Track A is deferred to T6, insulated by
  the protocol boundary.
- **Finalize latency on the rollout critical path.** ~60 ms/rollout hides
  behind generation seconds, but async moves it out from behind training —
  measured from M2 via the finalize-latency counter.
- **Placeholder-heavy groups** shift the GRPO baseline; the M2 advantage rule
  (baseline over valid rows only) plus `min_valid_fraction_per_group` are the
  mitigations. Needs a post-MVP experiment.
- **Checkpointing.** SC checkpointing is itself TODO; staging rows and
  registry state are deliberately not checkpointed — in-flight rollouts are
  abandoned on restore (registry TTL + create-only registration make this
  safe), matching the buffer's restore semantics.
- **Multi-row rollouts.** GRPO grouping and reward attribution for
  `all_leaves` linearization is explicitly undesigned in v2; out of scope —
  `main_chain_only` trains the main chain and verifies side branches only.
