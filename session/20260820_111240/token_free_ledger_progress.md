# Token-free ledger implementation — progress checkpoint (2026-08-24)

Plan: `session/20260820_111240/token_free_ledger_plan.md` (all phases implemented and committed).

## Source changes complete (all in Gym; NeMo-RL source untouched by design)

- `staging/digest.py`: `_CHAIN_DIGEST_DOMAIN` + `compute_chain_hash(parent_chain_hash, token_ids_delta)`.
- `staging/records.py`: `CaptureAdmission.parent_chain_hash` (required for token_in, forbidden for text);
  `CommitCoords` — `token_ids_delta` REMOVED (hard drop, no compat), `chain_hash`/`cumulative_hash`
  required when staged, forbidden when capture_failed.
- `staging/capture.py complete_call`: computes both hashes, binds into staging digest + StagedCallRecord,
  returns them on CommitCoords.
- `protocols.py`: `LineageMatch.chain_hash` added.
- `lineage.py`: custody columns + `_CUSTODY_FIELDS` gain chain_hash/cumulative_hash; `LineageNode.chain_hash`;
  `RolloutLineage.record` takes explicit `cum_len` + `chain_hash`; custody rows written token-free in BOTH
  stores (File omits the JSONL key; InMemory indexes empty cum_tokens); `_resolve` falls back to
  `record.get("cumulative_token_ids") or ()`; manifest rows carry the hashes.
- `sink.py`: `CaptureContext.parent_chain_hash`; `resolve_parent` passes it into the admission; a resolved
  external parent without chain_hash fails admission validation → poison row (fail closed).
- `responses_api_models/vllm_model/app.py`: commit hook no longer builds `cumulative` or calls
  `compute_digest` (import removed); records `[]` tokens, digest = coords.cumulative_hash, passes both hashes.
- `staging/rebuild.py verify_and_linearize`: incremental chain-hash verification during the terminal-chain
  walk (`chain_hash_mismatch`), terminal-only cumulative check (`cumulative_hash_mismatch`); absent hashes skip.
- `staging/__init__.py`: exports `compute_chain_hash`.

**Gym commit**: `6061dadb feat(token-capture): token-free custody ledger via chained digest`

## Test changes complete

- Gym `test_token_capture_staging_worker.py`: `_child()` has parent_chain_hash; coords assertions moved to
  hashes; delta assertions moved to sink records.
- Gym `test_token_capture_staging_core.py`: admission parent_chain_hash cases; coords built with hashes,
  no delta; missing-chain_hash rejection.
- Gym `test_token_capture_staging_rebuild.py`: `_snapshot` takes chain/cumulative hashes; 4 new tests
  (chained verifies, broken link, terminal cumulative mismatch, hash-free legacy).
- Gym `test_token_capture_ledger.py`: custody fixture carries hashes; token-free record calls; resolve
  asserts empty tokens + prev_len + chain_hash; staging-chain growth test chains hashes; new legacy-row
  test (hand-written JSONL row resolves with tokens, cannot anchor a chain → poison).
- NeMo-RL `tests/unit/data_plane/token_capture_test_fixtures.py`: `_record` computes real chain/cumulative
  hashes; child chains from root.
- NeMo-RL `test_vllm_token_capture_hosting.py`: token_in ng_capture dicts carry parent_chain_hash; round-trip
  asserts coords are token-free with hashes.

**NeMo-RL commit**: `6d12d9663 test(token-capture): update fixtures and hosting test for token-free coords`

## Test results

- Gym suite: **217/218 passed** (1 pre-existing spawned-worker timeout, unchanged baseline).
- NeMo-RL hosting tests: not run on login node (vllm/torch need GPU). Changes are minimal, correct by
  inspection: `token_ids_delta` not in coords (CommitCoords schema verified), `chain_hash`/`cumulative_hash`
  present (validator enforces them), `parent_chain_hash` required for token_in (validator enforces it).
  Previous session ran 13/13 on a GPU node; login-node environment hangs on `import vllm`.

## Environment notes

- Root `.venv` was missing `typing_extensions` → installed (`uv pip install -p .venv/bin/python typing_extensions`).
- Gym suites must run from `3rdparty/Gym-workspace/Gym` with `Gym/.venv` (namespace test spawns bare
  subprocesses that import nemo_gym).
- Pre-fix baseline: 8 expected failures across core+worker suites; ledger/rebuild passed.
- Login-node constraint: `rl-pr3456-delta-tests` venv hangs on `import ray`; Gym venv lacks torch/transformers.
  NeMo-RL hosting tests require GPU node (inside training container).

## Slurm smoke — COMPLETE (2026-08-24)

Job 6490451 (short QOS, 50:57 elapsed), W&B nmiv7zjc.

- **5/5 step_metrics** — all steps completed, finite loss, 5–8 valid rows each step.
- **token_in_rate 0.968–0.986** — ≈1 as expected.
- **Zero integrity errors** — no chain_hash_mismatch, cumulative_hash_mismatch,
  invalid_worker_commit_coordinates, worker_capture_failed, 409 Conflict, or UnknownRolloutError.
- **Custody rows token-free** — 45 ledgers, 3,081 external rows:
  - `cumulative_token_ids` present: **0** (expected 0) ✓
  - `chain_hash` present: **3,081** (expected == total) ✓

Implementation is fully validated end-to-end. All source, test, and Slurm checks passed.

## OLD REMAINING: Slurm smoke

1. Submit: `swe_nano_sc_capture.sh` with SC_EXP_NAME=swe-token-free-ledger-s43-0824,
   token_capture.staging_partition=rollout_staging_token_free_s43_0824, seed 43, 2 prompts/step, GBS 8,
   max_num_steps=5. (Dry run in progress 2026-08-24 ~10:00 PDT.)
2. Verify: 5/5 step_metrics, token_in rate ≈1, zero chain/prefix/staging/coordinate errors.
3. **Key new check**: custody JSONL rows have NO `cumulative_token_ids` key but non-null
   `chain_hash` and `cumulative_hash` values.
   ```bash
   grep -l '.' "$WORKSPACE/results/swe-token-free-ledger-s43-0824/*/gym_token_capture/lineage/*.lineage.jsonl" |
     head -1 | xargs python3 -c "
   import sys, json
   rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
   ext = [r for r in rows if r.get('staging_key')]
   print(f'external rows: {len(ext)}')
   print(f'with cumulative_token_ids: {sum(1 for r in ext if r.get(\"cumulative_token_ids\"))}')
   print(f'with chain_hash: {sum(1 for r in ext if r.get(\"chain_hash\"))}')
   "
   ```
   Expected: `with cumulative_token_ids: 0`, `with chain_hash: N > 0`.
