# Tokidcap Migration — Implementation Log

Working log for executing [tq-gym-tokidcap-migration.md](tq-gym-tokidcap-migration.md)
(re-basing the gate-authoritative token-capture work onto the upstream Gym
token-id-capture stack). Companion to the MVP-era
[tq-gym-gate-authoritative-implementation-log.md](tq-gym-gate-authoritative-implementation-log.md),
which remains the record for the fork-based S1–S5 work.

Conventions: every change lands here with its **status**
(`planned` → `in-progress` → `done (<commit>)` / `dropped (<why>)`), the seam
or finding it serves, and any divergence from the plan doc. Un-gated behavior
changes (active with `token_capture.enabled=false`) get an explicit
**DISCLOSURE** marker.

## Base facts (pinned)

- Upstream stack: linear chain #2190→#2124→#2125→#2126→#2180→#2181→#2182,
  verified 2026-07-31 in the submodule clone; **stack top = `81ac2736`**
  (#2182, "require a bearer token on the token read route").
- New Gym branch: `tq-tokidcap-capture`, cut from `81ac2736` (exact rev,
  §6 finding G). Old fork branch `tq-gate-capture` (`e3b3eac6`, base
  `fa0c2da3`) is left untouched as the cherry-pick donor.
- RL branch: work proceeds on `yukih/sc-entrypoint`; the §9b.1a companion-base
  question (nano-SWE branch `3fcc69666`, owner Zhiyu Li) is an open
  coordination point — capture commits are kept clean for later cherry-pick.
- Upstream asks: none landed as of 2026-07-31; every seam uses its §4
  workaround, contained in our modules. Ask outcomes shrink the diff later.

## Workaround ledger (asks not landed)

| Seam | Workaround in effect | Where |
|---|---|---|
| #2124-c1 install_token_sink honored | gate installs its manifest-observer sink by construction (we own the gate module); no base patch needed for worker-locus MVP | gate.py |
| #2124-c2 mark_incomplete on protocol | capture-poison flows through our CommitCoords/lineage, not the base sink protocol | staging/lineage.py |
| #2124-c3 commit_entry split | coords ingestion implemented in our gate module, feeding `RolloutLineage.record` on the base `LineageIndex` directly | gate.py |
| #2124-c4 schema_version | version carried in our staging records/receipts only | staging/records.py |
| #2126-c2 opaque rollout_id key | contained fork edit: accept an opaque `_ng_rollout_id` run-body key in `rollout_correlation.py` (shape-identical to the ask; collapses when it lands) — done (`1bdd7cdc`) | rollout_correlation.py |
| (new, to post as an ask) run-wide agent opt-in | `token_id_capture_all_agents` global key treats every agent as `token_id_capture=true` — the SC cannot enumerate agent servers configured via `config_paths` — done (`b6051536`) | base_responses_api_agent.py |
| #2180-c1 set_lineage_index | gate constructs a capacity-sized `LineageIndex` and replaces `sink._LINEAGE` before first request; attr name pinned in the bump checklist below | gate.py |
| #2180-c2 capacity config + eviction metric | comes free with the above (we construct the instance); eviction counter polled into gate metrics | gate.py |
| #2180-c3 resolver seam | not needed for MVP (no marker); Stage-2 exit kept in mind | — |
| #2181-c1 ng_capture hook | attached in `vllm_model/app.py` chat path (file we already edit for the gate); the highest-churn-risk edit, kept minimal | vllm_model/app.py |
| #2181-c2 required_prefix_token_ids contract | consumed as-is by the worker splice (proven code); rename risk noted in bump checklist | adapters/vllm.py |

**Bump checklist** (things a stack rebase can silently break): `sink._LINEAGE`
attr name; `required_prefix_token_ids` field name; `model_call_id` mint site
(`base_responses_api_model.py` `_CaptureMiddleware`); `run_builder` /
`BuildNotes.unresolved_retries` shape; `/ng-rollout/` prefix regex.

## Commit series (Gym branch `tq-tokidcap-capture`)

Per plan §9b.2; each commit = a future stack PR. Status updated as they land.

| # | Commit | Status | Notes |
|---|---|---|---|
| 1 | staging wire schema + digest (`staging/records.py`, `digest.py`, `protocols.py`; sink protocols renamed `StagingSink`/`StagingSource`) | done (`79540d2e`) | digest golden vectors unchanged; purity + records/digest tests in `test_token_capture_staging_core.py` (10 green) |
| 2 | terminal-aware linearize over `run_builder` (thin `staging/rebuild.py`) | done (`3d813fee`) | `snapshots_to_entries` + `LinearizedRow` kept; manifest walk deleted; terminal hint overrides token-mass pick (test proves the fork case); `unresolved_retries` → `RebuildError` → placeholder; 18 tests green |
| 3 | engine-blind capture core + vLLM adapter (`staging/capture.py`, `adapters/vllm.py`) | done (`232c7a43`) | module is identity-agnostic; rekey lands at call sites; 47 tests green |
| 4 | gate hosting, prefix serving, control plane (`gate.py`, `staging/lineage.py`, `control_routes.py` + bearer auth; seam edits in `vllm_model/app.py`) | done (`e1a9b4e7`) | identity switch complete (URL-prefix + `model_call_id`, marker plumbing gone); gate hosts capacity-sized eviction-counting `LineageIndex` via `sink._LINEAGE` replacement; coords ingestion feeds `LineageIndex.record`; bearer auth default-required; bounded client deadlines; conformance kit byte-exact through the `run_builder` linearize; e2e suite rehosted (ambiguity, auth, unknown-id rejection, flag-off dormancy); 199 tests green |
| 5 | observability (byte-counter middleware cherry-pick, unattributed-call counter; fallback-by-cause + eviction counter landed in commit 4) | done (`04597a3a`) | one adaptation: fork's `Dict` annotation → builtin `dict` (base dropped the import) |

Deleted relative to the fork (base owns them): flat capture core
(`records`/`sink`/`store`/`config`/`routes`/`reader`/`source` + lazy
`__init__`), `memory_store.py`, most of `rebuild.py`, all marker plumbing
(`openai_utils.py` / `responses_converter.py` `*WithMarker` classes,
`find_marker`, `NG_CALL_ID_FIELD` message stamping), every fork edit to
`base_responses_api_model.py`.

## RL-side changes (branch `yukih/sc-entrypoint`)

| Change | Status | Notes |
|---|---|---|
| Re-pin submodule to `tq-tokidcap-capture` | in-progress | gitlink commits with the RL alignment; NRL_FORCE_REBUILD_VENVS on first run after |
| Identity carrier switch (drop `metadata["ng_rollout_id"]`; `_ng_rollout_id` run-body key → `/ng-rollout/<id>`) | done (uncommitted) | rollout_manager `_build_inputs`, nemo_gym.py; verified `run_examples` posts rows verbatim so the key reaches the agent |
| Per-agent correlation opt-in | done (uncommitted) | SC sets `token_id_capture_all_agents=true` (Gym `b6051536`) |
| Gate hosting config: lineage capacity (derived: rollouts = 2×in-flight, tokens = rollouts×max seq len), bearer token (minted per run), capture dir (`<log_dir>/gym_token_capture`, the #2124-c1 workaround) | done (uncommitted) | TokenCaptureConfig + setup.py derivation + nemo_gym.py injection + exemplar YAML |
| Bounded control-plane deadline (`control_timeout_s`, default 60 s) | done (uncommitted) | `asyncio.wait_for` around every `_control` call (S5 silent-stall finding, H1 pulled into Stage 1) |
| Control-plane bearer auth on the client | done (uncommitted) | Authorization header on every `_control` call |
| Rekey staging keys to `{rollout_id}/{model_call_id}` | done (no code change) | keys minted Gym-side; worker/finalizer identity-agnostic |
| Finalizer: rebuild via `run_builder` wrapper, terminal hint + `unresolved_retries` → placeholder | done (no code change) | new Gym `linearize` keeps the signature; `unresolved_retries` surfaces as `RebuildError` → existing `rebuild_failed` placeholder path |
| Fix dead `staging_ttl_s` config | deferred | still unread (as in the fork MVP); H1 scope with the failure sweep |
| `uv.lock` regeneration for the new pin | done (uncommitted) | the stack top dropped Gym's `docs` dependency-group → lock update required; regenerated with uv 0.11.6 (the version family that wrote revision 3) for a minimal 934-line-deletion diff — uv 0.12 rewrites the whole lock to revision 4, avoid it for this repo |

Note: the SC seals without an explicit `terminal_call_id` (it cannot know it);
the receipt's terminal defaults to the last-committed call (chronological),
which the finalizer's terminal-aware selection consumes. A background
sub-agent that commits after the main conversation's final call would win
the hint — accepted for Stage 1, revisit if the A/B row diff surfaces it.

## Test gates

- [ ] Gym: base capture suite + full unit suite green at every commit
- [ ] Gym: `staging/` purity test (subprocess import, no fastapi/ray/torch)
- [ ] Gym: conformance kit + gate e2e rehosted on stack request path
- [ ] Gym: flag-off byte-identity (capture disabled ⇒ legacy path unchanged)
- [x] RL: capture unit tests (`--nemo-gym-only`) green on new pin —
      23 passed (tq_token_sink / blackbox_finalizer / vllm hosting) on
      2026-07-31; adjacent default suites (replay buffer, rollout manager,
      SC setup) 31 passed + 1 PRE-EXISTING failure
      (`test_returns_bundle`, grpo.seed fixture — documented branch-HEAD
      breakage predating the migration, see the fork-era log)
- [ ] RL: flag-off 2-GPU functional (`grpo_async_gym_single_controller.sh`) —
      pin-bump regression
- [ ] RL: capture-enabled functional; fixed-seed A/B row diff; chaos smoke

## Disclosures (active with the flag off)

(none yet — all Gym-side behavior changes are behind `token_capture_gate.enabled`
or `NG_HTTP_BYTES_DIR`; the `_resolve_client` signature gained an optional
`rollout_id=None` parameter, a no-op when unset)

## Divergences from the plan doc

- **Gate activation requires the base token capture enabled with a real
  capture dir** (`token_id_capture_enabled=true` + `token_id_capture_dir`) —
  the #2124-c1 "activation without a capture dir" ask is not landed, and the
  capture middleware only mints `model_call_id`/sets the capture context when
  a token store exists. Cost: the base `capture_tokens` no-ops on the gate
  path (worker strips token fields before the response), so the dir stays
  ~empty; the RL launcher points it at the run's log dir.
- **Editing a user turn no longer breaks lineage.** The fork's fingerprint
  covered the full history; the base's covers model-authored turns only, so
  a harness that rewrites user/tool content keeps its chain (by design
  upstream). The e2e "edited history" test now edits the assistant turn.
- **Fallback cause names** are `no_history` / `no_match` / `ambiguous`
  (plan §8 sketched `no_prefix`/`no_match`/`ambiguity`/`multi_worker`;
  `multi_worker` is not distinguishable at the gate and is deferred).

## Rebase onto the nano-SWE branch (2026-07-31, user-directed)

The RL companion was rebuilt on Zhiyu Li's nano-SWE recipe branch
(`3fcc69666` = current `main` + the TQ-honoured SWE recipe), per §9b.1a of
the plan doc, replacing the `yukih/sc-entrypoint`-based series. Method: the
capture-era delta (`fe8ac47b7..624bb277d`) applied as a single squashed
commit (granular history preserved on the prior branch head `624bb277d`).
Port adaptations onto main's evolved SC structures, all test-verified:

- `run_rollouts` is a streaming Ray generator on main — receipt-mode
  postprocess branches inside main's loop; the picklable aiohttp error
  re-raise precedes main's generic handler.
- `TQReplayBuffer.commit` keeps main's semantics (pre-write "no live slot"
  guard, evicted-during-write clears rows, BaseException rollback) with
  `maybe_dump_train_rows` inserted before the write; the legacy dispatch
  failure path keeps main's `remove_group` (capture path keeps `abort`).
- Exemplar `token_capture` block moved to
  `examples/nemo_gym/grpo_qwen3_30ba3b_instruct.yaml` (the SC functional's
  config; the old SC exemplar yaml no longer exists on main).
- Main's `_reference_logprobs_required` mechanism supersedes the fork's
  ref-KL advantage-field tweak (dropped).
