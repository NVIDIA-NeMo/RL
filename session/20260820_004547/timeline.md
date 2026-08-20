# Timeline

## 2026-08-20 00:45
- User asked: implement token-capture-lineage-ledger-approach.md, unit-test, then 5-step smoke A/B (capture on/off, R3 on, num_workers>1), compare dynamics+perf.
- Context gathered: plan doc; CAMPAIGN.md of prior swe-r3-capture campaign; Gym gate/lineage/sink/control_routes/records/rebuild/config/base_responses_api_model/vllm app; RL nemo_gym.py/blackbox_finalizer/rollout_manager/single_controller.
- Key findings:
  - Pre-existing uncommitted RL changes = pure gate→ledger doc renames (safe to carry).
  - `_validate_manifest_graph` (staging/rebuild.py:159) tolerates dead-branch rows → plan's step-8 open check resolved: no pruning needed in receipt assembly.
  - RL vLLM worker imports only `token_id_capture.staging.*` → worker path untouched by gate deletion.
  - Lineage JSONL files were never cleaned in gate mode either → skipping ledger-file drop is no regression (documented follow-up).
  - Gate metrics route dies with the gate → replace with finalizer-derived mode counters (manifest rows carry mode) logged as finalize/token_in_rate.
- Design decisions:
  - New Gym config flag `token_id_capture.external_staging: bool` replaces `gate:` block (extra="forbid" on TokenIdCaptureSettings makes leftover gate keys fail loudly). Requires enabled + rebuild_response=false; keeps control_auth_token_env for manifest-route bearer auth.
  - Ledger row: existing 6 lineage keys + parent_call_id/staging_key/weight_version/prev_len/delta_len/cum_len/extras_digest/mode/logical_request_id + staging_digest (coords.digest; distinct from lineage digest). Failure rows: model_call_id+failure_reason, no fingerprint (resolve uses .get so they can't match).
  - CallRecord gains optional logical_request_id; new RolloutManifest/ManifestFailure wire models in staging/records.py; manifest route GET /training-token-capture/rollouts/{id}/manifest.
  - CaptureContext: staging_gate/data_capability deleted; adds external_staging bool + request_items (stashed in resolve_parent for commit-time record()).
  - Data-capability plumbing fully removed (rollout_correlation, server_utils, base agents, swe sandbox mount); swe sandbox token-capture URL prefix re-keyed to agent-level token_id_capture enabled.
  - Abandoned-rollout staging rows: accept leak until partition teardown (plan option b), fail_rollouts deleted.
- Result: starting Gym implementation.
- 2026-08-20 01:39:46 Gym side committed (1e1a16cb): ledger implemented, gate deleted, 436 Gym tests green (ledger 23, capture 112, swe 158, vllm 143). Starting RL companion.

## 2026-08-20 06:30
- RL companion committed (1f9af7987 + launcher fixes 890a485c7, uv-cache fix, Gym pin bumps; Gym lint fix 74d4ef85).
- Smoke A/B submitted: capture=6358267, legacy=6358268 (first pair 6357295/6357320 failed/cancelled — UV_CACHE_DIR_OVERRIDE mount severed the container venv's hardlinked packages; removed from swe_nano.env).
- RL unit tests: receipt-assembly 6/6 green (--noconftest). Ray-fixture suites hang on the login node (conftest autouse init_ray + TQ actors); moved to compute-node job 6359381.

## 2026-08-20 07:40
- Unit-test scope closed: Gym capture suites all green (436); RL receipt-assembly (6) + rollout-manager (16) green. Ray-heavy RL suites (blackbox_finalizer/tq_token_sink/vllm hosting) cannot run on login node (conftest autouse Ray wedge) nor bare/pyxis compute shells (uv-run worker env drift; /opt/nemo_rl_venv is ray.sub-materialized). They are exercised end-to-end by the smoke and will run in CI.
- Ported NRL_TQ_SKIP_RUNTIME_ENV_PIN into transfer_queue.py (committed) while diagnosing.
- Smoke r2: capture 6359951 TRAINING (step 2/5 at ~37 min — venv rebuild-list fix confirmed). Legacy 6359952 still queued.

## 2026-08-20 09:10
- BOTH SMOKE ARMS PASSED 5/5 steps. Capture: tmpe 1.0137-1.0147, gen_kl ~0.001, token_in_rate 0.95-0.99, 2821 ledger rows / 0 unresolved / 0 worker failures; 16 rollouts fail-closed on aborted calls (request_finished_without_staged_coordinates — retry-idempotency follow-up). Legacy: tmpe 1.020-1.066, gen_kl ~0.004, all 8 samples valid every step. Capture strictly tighter — matches gate-era campaign bands.
- Review of legacy fix done via /review-pr (local): fix correct; added 2 regression params to test_nemo_gym_utils (7/7 pass); renamed _content -> item_content.
- Report: reports/auto_research/lineage-ledger-0820/ledger-ab-report.md.
