# Token Capture v4: TQ-Backed Capture on the Upstream Tokidcap Stack

Consolidated migration plan for rebuilding the token-in/token-out pipeline **on
top of the upstream Gym token-id-capture PR stack** instead of beside it, with
NeMo-RL contributing the TransferQueue (TQ) storage layer and the training
assembly. This supersedes the *implementation strategy* of the v3 doc
(`tq-gym-gate-authoritative.md`); the v3 doc's goals, invariants, wire budget,
and § 3.4 wire shapes remain the reference and are preserved by this plan.

## 1. Context

The upstream stack (NVIDIA-NeMo/Gym, author ananthsub; reviewed 2026-07-31):

| PR | Adds | Relationship to us |
|---|---|---|
| #2190 | server-side `sampling_overrides` pin (all three engine paths) | adopt verbatim; fixes the S5 sampling finding for harness-owned requests |
| #2124 | capture core: `TokenEntry`, `TokenSink`/`TokenSource` protocols, `install_token_sink`, file store, `.incomplete` sentinel, capture-in-dispatch | replaces our re-created core verbatim; 4 asks posted |
| #2125 | trajectory builder (`run_builder(prefix_merging)`, retry/quarantine, `BuildNotes`) | replaces most of our `rebuild.py` |
| #2126 | delivery + retirement in Gym rollout collection | not on our hot path (RL calls `run_examples`, which has no delivery); 2 asks posted |
| #2180 | request-time parent resolution: `LineageIndex`, assistant-only dialect-normalized fingerprints, token-bounded eviction | replaces our marker as the Stage-1 lineage mechanism; asks drafted |
| #2181 | prefix supply (`required_prefix_token_ids` on chat + tokenize), supplied/eligible metric | replaces our gate's prefix-serving trigger; asks to draft |
| #2182 | bearer token on the read route | pattern adopted for **our control routes** |
| #2189 | docs: external-harness training contract + metrics table | alignment reference |

Verified seam facts this plan depends on:

- `install_token_sink` is exported and documented but **never consulted** by
  `capture_tokens` at the top of the stack — the sink comes exclusively from
  `CaptureContext.store`, hard-built by `make_token_store` inside
  `install_model_call_capture` (`base_responses_api_model.py:1286-1298`).
  Ask #2124-c1 covers this; the workaround (monkeypatch/subclass) is contained.
- #2126 delivery runs only in `run_from_config`
  (`rollout_collection.py:556-644`); RL's `run_examples` path bypasses it
  entirely, so RL invokes the builder itself.
- The Gym model server is a Popen subprocess with a nemo-gym-only venv; the
  heavy TQ write therefore stays **in the RL vLLM worker** (worker-locus),
  which already hosts the in-process DP client (S2 work).
- `LineageIndex` nodes store the **full cumulative token sequence per call**
  (~290 MiB default budget, oldest-rollout-first eviction on access, can evict
  a live rollout); it is per-process.
- The receipt manifest carries exact staging keys, so `TQTokenSource` needs no
  scan API — `kv_batch_get` by manifest keys suffices on the hot path.
- Upstream `TokenEntry` has no `weight_version`; `extra="allow"` lets the
  worker stamp it.

## 2. Ownership after migration

```
[stack]  capture middleware + CaptureContext + model_call_id mint
         sampling pin; LineageIndex + assistant-turn fingerprint resolution
         prefix supply (required_prefix_token_ids); run_builder + BuildNotes
[ask]    install_token_sink honored (#2124-c1); mark_incomplete on the
         protocol (#2124-c2); commit_entry split (#2124-c3); schema_version
         (#2124-c4); external delivery mode (#2126-c1); opaque rollout_id
         key (#2126-c2); set_lineage_index + capacity config (#2180);
         ng_capture hook + required_prefix_token_ids contract (#2181)
[ours]   Gym fork additions: staging wire shapes (records/digest), worker
         capture core + vLLM adapter, gate = thin registry hosting
         (register/seal/fail, receipts, TTLs, bearer auth) rewired over the
         stack's LineageIndex + sink observer
         NeMo-RL: TQTokenSink/TQTokenSource, finalizer over run_builder,
         SC/RolloutManager wiring, weight-version fan-out, S5 tooling
```

Module fate on the new Gym branch (cut from pinned stack-top revs;
cherry-pick, never rebase the old branch):

| Old module | Fate |
|---|---|
| re-created capture core (records/sink/store/config/routes/reader @ `32b555f0`) | delete — stack replaces verbatim |
| `staging/records.py` (StagedCallRecord, CommitCoords, RolloutReceipt, SCHEMA_VERSION) | keep — worker↔TQ and gate↔SC wire schema; no stack equivalent |
| `staging/digest.py` (row-integrity digest over ids+mask+logprob bits) | keep — complementary to the stack's parent-identity digest |
| `staging/capture.py` + `adapters/vllm.py` | keep — worker-locus core; rekey identity onto the stack's `model_call_id` |
| `gate.py` + `control_routes.py` | keep, rewire — thin hosting over `LineageIndex` + sink observer; **add bearer auth** |
| `staging/lineage.py` (pure state machine) | keep as the gate registry; inputs become stack capture events |
| `staging/rebuild.py` | mostly delete — `run_builder(prefix_merging)` is the rebuild engine; keep a thin terminal-aware chain-selection wrapper (day 1, see § 3 finding 3) |
| `memory_store.py` | delete — `LineageIndex` nodes hold `cum_tokens`; no second buffer |
| `staging/conformance/` + byte-counter middleware | keep; retarget fixtures to stack identity |

## 3. Migration plan

### Stage 0 — land the asks, restructure the base

1. Post #2180's asks **including the resolver seam** (it is what keeps Stage 2
   fork-free); draft + post #2181's two (ng_capture hook,
   `required_prefix_token_ids` contract). Track outcomes — each ask has a
   priced workaround (§ 4 matrix) and the outcomes size the fork.
2. Cut the new Gym branch from **pinned exact revs** of the stack top; record
   the SHAs and the gitlink/CI answer at this gate (the stack has already been
   squash-rebased once).
3. Cherry-pick per the fate table; re-pin the RL submodule; run the flag-off
   L1 sweep to flush API drift in one pass.
4. Identity carrier switch: drop the `responses_create_params.metadata`
   side-channel. RolloutManager puts the rollout id into the **run body**
   (opaque key if #2126-c2 lands; else `task_index` = persistent global
   dispatch counter + `rollout_index`); agents stamp `/ng-rollout/<id>` via
   the stack's existing helpers; the middleware's `model_call_id` becomes our
   `call_id`. Gate registration stays create-only, pre-dispatch.

### Stage 1 — worker-locus MVP

Seam-by-seam wiring per the § 4 matrix, plus RL-side alignment: rekey
`TQTokenSink` rows and staging keys to `{rollout_id}/{model_call_id}`; point
`install_capture` at the identity in `ng_capture`; keep the weight-version
fan-out (worker stamps at generation start); finalizer fetches by receipt
manifest, verifies (staging digest, `prev_len + delta_len == cum_len`, wv
tags), rebuilds via `run_builder` over reconstructed entries, publishes
always-N with placeholders, `commit_finalized(group_min_wv)`.

Day-1 items folded in from the plan review (previously deferred or missing):

1. **`LineageIndex` capacity is load-bearing.** The gate constructs the index
   (we own construction either way) with capacity sized from the training
   config (in-flight rollouts × max context), `drop()` at seal, and an
   **eviction counter** in metrics. Eviction of a live rollout silently
   degrades prefix supply and token-in to new-root fallbacks; it must be loud.
2. **Fallback-by-cause metrics split ambiguity out** (`no_match` /
   `ambiguous` / `cross_worker` / `evicted`). Identical assistant turns are
   routine in agentic SWE loops and collide under fingerprint resolution;
   the Stage-2 marker trigger reads this counter.
3. **Terminal-aware chain selection ships day 1**, not "if S5 shows
   mispicks": the finalizer overrides `run_builder`'s token-mass main-chain
   pick with the registry's terminal call. Token-mass mispicks silently on
   sub-agent forks that out-generate the main conversation. The finalizer
   also consumes `BuildNotes.unresolved_retries` → placeholder.
4. **Control-route bearer auth, default-required** (#2182 pattern applied to
   `/ng-control/*`): without it a sandboxed harness can seal a sibling with a
   forged reward and the forged seal wins the state transition. Ids are
   guessable (`{group_id}_g{i}`) and now visible in the run body/URL prefix.
5. **Rejection scope**: the gate rejects only *prefixed-but-unknown* rollout
   ids; uncorrelated traffic (no `/ng-rollout/` prefix) passes through
   untouched, preserving the dual-mode fallback (no coords → stack's own
   extraction + JSONL path, unchanged).
6. **Evidence retention + empty-delta handling**: staging rows of *rejected*
   rows/groups are left for the TTL sweep instead of eager clearing (they are
   the only evidence of why verification failed); a committed call with
   `delta_len <= 0` is recorded as poisoned-with-counter, never an unhandled
   `LineageStateError`.
7. **Bounded control-plane retries** (`RolloutControlClient` request timeout):
   the S5 chaos run showed gate death is otherwise a silent stall (retry=375+
   on `ClientOSError`), never reaching the fail-path.

**Stage-1 acceptance** (S5 tooling retargeted): locus equivalence — same seed
through server-locus JSONL and worker-locus TQ produces byte-identical
canonical rows; fixed-seed A/B vs the legacy echo path with the offline row
diff; chaos smoke (kill gate mid-step → failed dispatches + placeholders +