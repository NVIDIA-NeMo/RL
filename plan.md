# Draft Co-Training Migration Plan: RL_dspark → NeMo_RL (dspark / dflash / eagle3)

## Goal Description

Migrate speculative-decoding draft-model co-training (dspark, dflash, eagle3) from the RL_dspark fork (`/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/speculative_rl/RL_dspark`, branch `dflash` @ `4bc432fb`, fork point `2c5ee18f`) into NeMo_RL branch `draft-cotraining-migration` (off `bump-automodel-r0.6.0`). Vendor copies become `nemo_automodel.components.speculative.*` imports where byte-identical; RL-specific deltas live in a thin local extension layer under `nemo_rl/models/automodel/draft/`. The Automodel submodule stays pinned to official r0.6.0. The serving contract (vLLM 0.25.1 `draft.*` refit key naming, per-method skip lists, d2t int64 / t2d bool dtypes) stays byte-identical to RL_dspark's. dspark was validated end-to-end in the fork; eagle3/dflash were smoke-tested only — migration acceptance is smoke-level for all three.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Thin extension layer exists; byte-identical vendor files are imports, only delta code is local.
  - Positive Tests (expected to PASS):
    - `uv run --no-sync python -c "import nemo_rl.models.automodel.draft"` succeeds and resolves `sample_tokens`, `create_dflash_block_mask`, `create_dflash_sdpa_mask`, `build_draft_config` to `nemo_automodel` modules.
    - Local files are only: `common.py` (local `build_anchor_candidate_mask`, `build_eval_mask(..., supervised_from_slot)`, extended `DSparkForwardOutput` with `first_supervised_slot`, and a local `sample_anchor_positions` re-composition — the upstream one hard-binds upstream `build_anchor_candidate_mask` at `nemo_automodel/components/speculative/dspark/common.py:152`, so importing it would silently drop the RL anchor gating), `markov_head.py` (local `build_markov_head` wrapper adding `embed_vocab_size` reduced-vocab split), `draft_qwen3.py`, `loss.py`, `eagle3_qwen3.py`, `integration.py`, `__init__.py`. Each local file header states what it extends in `nemo_automodel` and why the delta exists.
    - A unit test asserts prompt-boundary anchors are admitted (upstream semantics would reject them) and one-token responses produce draft signal.
  - Negative Tests (expected to FAIL):
    - `nemo_rl/models/automodel/draft/_sampling.py`, `dflash_mask.py`, `config.py` do not exist.
    - Grep for a second definition of `extract_context_feature` / `create_position_ids` / `pin_rope_inv_freq_fp32` / `create_dflash_block_mask` in `nemo_rl/` returns nothing.
- AC-2: All unit tests pass under this repo's Transformers 5.12.1 pin.
  - Positive Tests:
    - `uv run --no-sync pytest tests/unit/models/automodel/ tests/unit/models/generation/test_vllm_backend.py tests/unit/algorithms/test_draft_runtime_loss_wrapper.py` is green: the 6 ported fork tests (`test_draft_config_adapters.py`, `test_draft_loss_decay.py`, `test_draft_vocab_maps.py`, `test_dspark_anchor_sampling.py`, `test_dspark_checkpoint_meta.py`, `test_eagle3_trainer_grads.py`), the ~17 appended `test_vllm_backend.py` tests (owner-manifest + alias-guard), and NEW `tests/unit/algorithms/test_draft_runtime_loss_wrapper.py` covering `DraftRuntimeLossWrapper` (combined-loss composition, `loss_weight=0` isolation, metrics merge). The pre-existing upstream `tests/unit/algorithms/test_draft_loss_wrapper.py` (Megatron `DraftLossWrapper`, identical in both repos) stays untouched and green.
  - Negative Tests:
    - Alias-guard tests assert RuntimeError on aliased drafter embed/lm_head; manifest tests assert RuntimeError on missing/unexpected `draft.*` keys; checkpoint-meta tests assert rejection of cross-algo resume and optimizer-layout mismatch.
- AC-3: Config surface follows repo conventions and validation gates work.
  - AC-3.1: `DSparkDraftOptions` is a pydantic `BaseModel(extra="allow")` with centralized field defaults (NOT TypedDict — `.agents/contributor-skills/config-conventions/SKILL.md` forbids new TypedDict config classes); `Eagle3DraftOptions` stays BaseModel; the existing `DraftConfig` TypedDict is extended with `algo`/`dspark`/`eagle3` keys (extending existing TypedDicts is permitted); options are validated at draft setup where consumed; call sites never use `.get(..., default)` patterns.
    - Positive: `tests/unit/test_config_v2.py` passes with `examples/configs/grpo_math_1B.yaml` and `tests/unit/reference_configs/grpo_math_1B.yaml` updated together.
    - Negative: constructing dspark options with an invalid type fails at setup validation.
  - AC-3.2: Validation gates in `lm_policy.Policy.__init__` (the one layer that sees both `policy.draft` and `policy.generation`).
    - Positive: each migrated recipe validates and starts.
    - Negative: dspark/dflash + Megatron backend raises; draft + sequence packing raises; draft + `sequence_parallel`/LoRA raises; draft co-training + refit transport in {`nccl_reshard`, `vllm_s3_sparse`, `vllm_zmq_sparse`} raises (`normalize_vllm_refit_config` alone cannot decide this — it has no view of `policy.draft`).
- AC-4: Training-side integration is correct and isolated.
  - Positive Tests:
    - One 1n8g dspark train step yields finite `draft_loss` and `draft_grad_norm > 0`.
    - Optimizer has named param groups `["policy", "draft"]` in stable order with the draft group carrying its configured LR; a unit assertion verifies torch schedulers scale both groups multiplicatively (LR ratio preserved).
    - Both v2 workers access `ModelAndOptimizerState` by named fields (tripwire test guards against tuple-unpack regressions).
  - Negative Tests:
    - With `loss_weight=0`, policy gradients are identical to a no-draft run (draft loss never backprops into the policy trunk — captures/teacher detached).
    - With draft enabled + `enable_seq_packing`, `LossPostProcessor.__init__` raises.
- AC-5: Refit contract intact end-to-end.
  - Positive Tests:
    - `tools/draft_verification/vllm_refit_preflight.py` passes for ALL THREE algos — dspark (`deepseek-ai/dspark_qwen3_8b_block7`; dflash is dspark-minus-markov/confidence, so dflash+eagle3 alone would not exercise dspark-only keys/skip-list entries), dflash (`RedHatAI/Qwen3-8B-speculator.dflash`), eagle3 (`RedHatAI/Qwen3-8B-speculator.eagle3`): exact key set, d2t stays int64 / t2d stays bool wherever the checkpoint carries them, no module aliasing.
    - `prepare_refit_info` advertises `draft.*` entries (float tensors as training dtype; int/bool buffers keep dtype).
    - Unit tests capture `_refit_params_generator()` output through each supported transport path — `stream_weights_via_ipc_zmq`, `broadcast_weights_for_collective`, and `_checkpoint_engine_params()` — asserting `draft.*` keys present and correctly typed.
    - MTP paths untouched: `draft.*` is never routed to `_maybe_refit_mtp_drafter`; `_weight_update_lifecycle.finalize()` post-processes only main model + MTP drafter (no double-finalize for cotrained methods).
  - Negative Tests:
    - `_validate_draft_refit_info` raises when `_draft_full_refit` is set but the manifest has no `draft.*` keys (enabled-but-empty is an error, distinct from static-drafter mode).
    - `_assert_drafter_owns_modules` raises when drafter embed/lm_head alias the target model's tensors.
- AC-6: Checkpoint/resume atomic contract.
  - Positive Tests:
    - Save produces the sibling `<parent>/draft/` DCP dir + `dspark_meta.json` inside the same `tmp_step_N` fence (before promotion; the draft DCP save is synchronous inside `save_checkpoint`, and `finalize_async_save` fences the async policy/optimizer writes).
    - Resume restores policy+draft weights and the composite-paired optimizer (`checkpointer.save_optimizer(..., model=composite_model)` — one-model-one-optimizer pairing) with named groups preserved (names, order, counts, per-group LR) and completes one further optimizer step.
  - Negative Tests:
    - Resume with mismatched algo / ttt_steps / optimizer layout raises; legacy no-`algo` meta resumes as dspark only.
    - A checkpoint interrupted before `finalize_async_save` is never promoted from `tmp_step_N`.
- AC-7: 1n8g smoke per algorithm (order dspark → dflash → eagle3), 2–3 training steps each with the migrated recipes (fixed recipe, fixed seed, default `num_speculative_tokens`), run in-container via `uv run`.
  - Positive Tests (HARD gates):
    - Run starts; every refit completes without draft-key errors or alias trips; `draft_loss` finite at every step; `draft_grad_norm > 0`; the dspark run additionally saves a checkpoint, resumes, and completes one more train/refit/generate cycle.
  - Telemetry (per DEC-1, resolved: trend/diagnostic):
    - `vllm/spec_acceptance_length` recorded each step and compared to RL_dspark reference ranges (dspark ≥ 2.0, dflash ≈ 3.0–3.5, eagle3 ≈ 2.0–2.5); only a gross anomaly (e.g. < 1.5 or a cliff after the first refit — the canary for embed/lm_head streaming bugs) fails the smoke.
  - Negative Tests:
    - The startup warning "Speculative decoding … without draft refit sync" does not appear (co-training refit is wired).
- AC-8: Hygiene and nightly wiring.
  - Positive: `uv run --no-sync ruff check` + `ruff format` clean on touched files; nightly drivers `tests/test_suites/llm/grpo-qwen3-8b-1n8g-automodel-{dflash,eagle3}.sh` + `nightly.txt` entries replayed (30-step gates unchanged; CP=1 only).
  - Negative: no CP>1-dependent recipe is wired into nightly coverage while the DEC-2 guard is in effect.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
Full parity with RL_dspark behavior on DTensor-v2 for all three algorithms, including CP>1 dspark co-training with the draft-loss gradient-scale derivation redone against the current `* dp_size * cp_size / cp_gradient_fanout` backward formula and the Automodel `ContextParallelSharder` layout, dynamic-batching support (`begin_global_batch` DP-max all-reduce), async-rollout compatibility (holds by construction: module-sharing disable lives in the shared `_load_model`, and async RPC mirrors reach the same extension methods), all migrated recipes (incl. 4n8g and 35B dspark), verification tools, unit tests, and nightly wiring.

### Lower Bound (Minimum Acceptable Scope)
All three algorithms work on 1n8g DTensor-v2 with CP=1/TP=1: thin extension layer, training glue, refit across ipc/collective/checkpoint-engine, checkpoint/resume, all unit tests green, three smoke runs passing their hard gates. Per DEC-2 (resolved), CP>1 draft co-training may land guarded (hard error) if the task7 derivation reveals non-equivalence with the fork's validated semantics — in that case CP-dependent recipes (35B dspark) are annotated follow-up, excluded from nightly coverage, and the derivation is recorded for auditability.

### Allowed Choices
- Can use: `nemo_automodel.components.speculative.dspark.*` and `nemo_automodel.components.attention.dflash_mask` imports; subclassing Automodel's `Qwen3DSparkModel` only if the override surface stays within `__init__`/`forward`/`compute_logits` (default: keep the fork-form standalone module); pydantic BaseModel for new config classes; `cp_sharder.gather_token_tensor` for CP teacher-logit gathering.
- Cannot use: modifying the Automodel submodule; changing the `draft.*` refit key naming / skip lists / dtypes (pinned vLLM 0.25.1 loaders and `_expected_draft_keys` inversion depend on them); Automodel's `DSparkTrainerModule`/`DFlashTrainerModule`/`Eagle3TrainerModule` as the training loss path (approach C — rejected with evidence in `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/speculative_rl/moving_deep.md`); YAML-configurable draft architecture fields (they come only from the draft checkpoint's `config.json`); activation checkpointing in the draft recipes (stays off, matching the validated configuration).

> Deterministic-design note: the refit key contract, checkpoint formats, and algorithm semantics are fixed by the fork's validated behavior and the pinned serving stack; boundaries are intentionally narrow there. Freedom exists mainly in code organization of the thin extension layer and test structure.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach
Replay-then-rewire. Files upstream never touched replay in final form from the fork; the `draft/` package is rewritten as imports + local deltas; the three files refactored upstream (CP/`cp_sharder`/`DistributedSetup`) get their fork hooks re-attached at code-verified anchors:
- `nemo_rl/models/automodel/setup.py`: draft build inserts between the CPU-offload block and optimizer construction (setup.py:744-746); `device_mesh["dp_cp"]` is available (Automodel registers the flattened axis; robust accessor `get_flat_mesh`); optimizer becomes two named param groups.
- `nemo_rl/models/automodel/train.py`: add `will_scale_temperature`; stash detached raw policy logits between `extract_logits` (train.py:341) and the temperature-scaling block (train.py:345); `LossPostProcessor` gains `draft_runtime` (append after `sampling_params`, keep `_cp_gradient_fanout` computation intact) and a `DraftRuntimeLossWrapper` branch between the seq-packing branch and the else-branch (the branch's `prepare_fn` must tolerate the new `context_parallel_group`/`cp_sharder` kwargs).
- `nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py`: capture ctx replaces the train-mode `nullcontext()` (worker:403); `begin_global_batch` between microbatch-iterator creation and `automodel_forward_backward` (worker:459-462); independent draft grad clip after the policy clip (worker:518) via a second `scale_grads_and_clip_grad_norm([self.draft_model], ...)`; `_refit_params_generator` swaps at the four verified call sites (worker:1077, 1116, 1129, 1157); `prepare_refit_info` draft entries before worker:1039; checkpoint save/load at worker:1297-1333 respecting the `finalize_async_save` fence.
Under CP the stashed teacher logits are the Automodel sharder's CP-local shards — the runtime's gather must match that layout (task7 decides equivalence or guard).

### Relevant References
- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/speculative_rl/RL_dspark/nemo_rl/models/automodel/draft/` — fork's vendored trainer + `integration.py` glue (the porting source of truth).
- `3rdparty/Automodel-workspace/Automodel/nemo_automodel/components/speculative/dspark/` and `components/attention/dflash_mask.py` — import targets (byte-identical to vendor base `6f423feb0`).
- `docs/superpowers/specs/2026-08-21-draft-cotraining-migration-design.md` — approved design spec (this plan's draft, appended below).
- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/speculative_rl/moving_deep.md` — approach-C rejection evidence (file:line).
- `nemo_rl/models/generation/vllm/vllm_backend.py` — current `_weight_update_lifecycle` / MTP gating to reconcile with the fork's draft validation.
- Pinned vLLM: `/root/.cache/uv/archive-v0/pKoLXX0MCRwJ24JW/vllm` (0.25.1); draft checkpoints in HF cache under `$HF_HOME/hub`; speculators oracle checkout at `/lustre/fsw/portfolios/coreai/users/yuekaiz/speculative/speculators`.
- `RL_dspark/tests/test_suites/llm/grpo-qwen3-8b-1n8g-automodel-{dflash,eagle3}.sh` — nightly gate values and smoke reference behavior.

## Dependencies and Sequence

### Milestones
1. M1 Foundations: convert both v2 workers (`nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py:303-315`, `nemo_rl/models/value/workers/dtensor_value_worker_v2.py:196-208`) from tuple-unpack to named-field access; add `draft_model`/`composite_model` NamedTuple fields; policy config types + constants + `lm_policy` gates (incl. transport guard); replay loss wrapper, generation `__init__`/`interfaces`, entrypoints, `vllm_worker.py` module-sharing disable, `.gitignore`.
2. M2 Thin extension layer: `draft/` package (imports + local deltas), then the 6 ported unit tests green.
3. M3 `integration.py` port (imports re-pointed; CP gather adapted or explicitly deferred to task7).
4. M4 Training glue re-port: `setup.py`, `train.py`, both v2 workers at the verified anchors.
5. M5 `vllm_backend.py` replay + `_weight_update_lifecycle`/MTP reconciliation + enabled-but-empty manifest hardening.
6. M6 Recipes/exemplars; full unit suite + ruff.
7. M7 Verification: refit preflight (all three algos) → smoke dspark (incl. save/resume cycle) → dflash → eagle3; nightly wiring.

The task7 CP>1 derivation gates whether M4 ships CP support or the DEC-2 guard.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Convert both v2 workers to named-field access of `ModelAndOptimizerState`; add `draft_model`/`composite_model` fields | AC-4 | coding | - |
| task2 | Policy config types (BaseModel `DSparkDraftOptions`, `DraftConfig` keys, constants) + `lm_policy` algo/backend/transport gates + exemplar/reference YAML sync | AC-3 | coding | - |
| task3 | Replay loss wrapper, generation `__init__`/`interfaces`, entrypoints, `vllm_worker.py` module-sharing disable, `.gitignore` | AC-3, AC-5 | coding | - |
| task4 | Build `draft/` thin extension layer (imports + local common/markov/draft_qwen3/loss/eagle3 files with delta headers) | AC-1 | coding | - |
| task5 | Port 6 draft unit tests, re-point imports, green under Transformers 5.12.1 | AC-2 | coding | task4 |
| task5b | NEW unit tests: `DraftRuntimeLossWrapper` (composition, `loss_weight=0` isolation) + optimizer named-groups/scheduler-ratio assertion + named-field-access tripwire for both v2 workers | AC-2, AC-4 | coding | task1, task3, task8 |
| task6 | Port `integration.py` with re-pointed imports | AC-4 | coding | task4 |
| task7 | Analyze: derive draft-loss gradient scale under the `* dp_size * cp_size / cp_gradient_fanout` backward formula for CP>1; verify fork `allgather_cp_sharded_tensor` vs Automodel `ContextParallelSharder` layout; decide port-CP vs DEC-2 guard; record the derivation in a design note or test comment | AC-4 | analyze | task6 |
| task8 | Re-port `setup.py` draft build + named param groups + optimizer layout record | AC-4, AC-6 | coding | task1, task6 |
| task9 | Re-port `train.py` hooks (`will_scale_temperature`, teacher stash, `LossPostProcessor` draft_runtime, packing rejection, wrapper branch) | AC-4 | coding | task6 |
| task10 | Re-port both v2 workers' hooks (capture ctx, `begin_global_batch`, dual grad clip, refit generators, `prepare_refit_info`, checkpoint save/load) | AC-4, AC-5, AC-6 | coding | task1, task8, task9 |
| task11 | Replay + reconcile `vllm_backend.py` (draft validation, alias guards, preflight RPC, lifecycle/MTP reconciliation, enabled-but-empty hardening) | AC-5 | coding | task3 |
| task12 | Port appended `test_vllm_backend.py` tests + NEW transport-path tests (`draft.*` stream captured through ipc/zmq, collective, checkpoint-engine generators); green | AC-2, AC-5 | coding | task10, task11 |
| task13 | Recipes + exemplar YAMLs replay (CP-dependent recipes annotated per task7/DEC-2 outcome) | AC-3, AC-7 | coding | task2 |
| task14 | Port `tools/draft_verification/`; run refit preflight for dspark, dflash AND eagle3 | AC-5 | coding | task4, task11 |
| task15 | Full unit suite + ruff on touched files | AC-2, AC-8 | coding | task5, task5b, task10, task11, task12 |
| task16 | Smoke dspark 1n8g incl. checkpoint save/resume cycle | AC-7 | coding | task13, task14, task15 |
| task17 | Smoke dflash 1n8g | AC-7 | coding | task16 |
| task18 | Smoke eagle3 1n8g | AC-7 | coding | task17 |
| task19 | Nightly drivers + `nightly.txt` wiring; final docs pass | AC-8 | coding | task16, task17, task18 |

## Claude-Codex Deliberation

### Agreements
- Migration boundary: reuse byte-identical Automodel speculative modules; keep RL-specific teacher/loss/refit/checkpoint deltas local (thin extension layer); approach C (Automodel Trainer modules) stays rejected.
- task1 (worker named-field conversion) must land first — both v2 workers still tuple-unpack `ModelAndOptimizerState`.
- The `train.py` teacher-stash point (between `extract_logits` and temperature scaling) and the new CP fanout formula are real; task7 is a necessary gate with the DEC-2 guarded fallback.
- Transport policy: ipc/collective/checkpoint-engine supported for `draft.*`; sparse + `nccl_reshard` rejected; guard lives in `lm_policy.Policy.__init__`.
- Lifecycle/MTP reconciliation scoped correctly (`draft.*` and MTP policy-stream refit are distinct paths; no double-finalize).
- Async rollouts covered by construction; draft LR follows the shared schedule multiplicatively (per-group base LRs preserved).

### Resolved Disagreements
- Anchor sampling semantics (round 1): Codex argued importing upstream `sample_anchor_positions` would lose RL anchor gating; code check confirmed it (upstream hard-binds its own `build_anchor_candidate_mask` internally). Resolution: keep a local `sample_anchor_positions` re-composition. Codex position adopted with evidence.
- `test_draft_loss_wrapper.py` (round 1): Codex asked to port it; verification showed it is a pre-fork upstream file (identical in both repos, covers only the Megatron `DraftLossWrapper`). Resolution: no port needed, but the underlying gap was real — a NEW `test_draft_runtime_loss_wrapper.py` is added (task5b). Mixed resolution, both positions partially correct.
- Smoke gate hardness (rounds 1–2): resolved via DEC-1 → acceptance-length is trend/diagnostic telemetry; hard gates are correctness-only.
- Preflight coverage (round 2): Codex required dspark in the live preflight (dflash = dspark minus markov/confidence heads does not cover dspark-only keys). Adopted (task14, AC-5).
- Transport-path tests (round 2): Codex required concrete `draft.*` streaming tests through each supported transport. Adopted (task12, AC-5).

### Convergence Status
- Final Status: `converged` (3 rounds; round 3 returned empty DISAGREE and empty REQUIRED_CHANGES).

## Pending User Decisions

- DEC-1: Smoke acceptance-length gates — hard requirement vs trend indicator?
  - Claude Position: trend/diagnostic; hard gates are correctness-only (refit success, finite loss, resume cycle).
  - Codex Position: same (hard thresholds over 2–3 steps are not a reasonable deterministic gate).
  - Tradeoff Summary: hard gates catch regressions automatically but are flaky over a 2–3-step window; trend gates require a human glance at telemetry.
  - Decision Status: **Resolved — trend/diagnostic.** Acceptance length is recorded per step and compared to RL_dspark reference ranges (dspark ≥ 2.0, dflash ≈ 3.0–3.5, eagle3 ≈ 2.0–2.5); only gross anomalies fail the smoke.
- DEC-2: If task7 finds CP>1 draft-loss scaling cannot match the fork's validated semantics: guard or block?
  - Claude Position: guard CP>1 + follow-up (1n8g smoke is CP=1; the 35B CP recipe was only fork-validated anyway).
  - Codex Position: guarded fallback is reasonable; if guarded, CP recipes/nightlies must be excluded or marked follow-up.
  - Tradeoff Summary: guarding unblocks the migration but defers the 35B dspark recipe; blocking guarantees full parity at the cost of significant derivation/testing work inside this migration.
  - Decision Status: **Resolved — guard + follow-up.** CP-dependent recipes annotated follow-up, excluded from nightly coverage; derivation recorded for auditability.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead

### Additional Notes
- Every kept-local `draft/` file carries a header stating what it extends in `nemo_automodel` and why the delta exists (no plan terminology — describe the technical reason).
- Commits follow Conventional Commits with `-s` sign-off (repo contributing convention).
- The `draft.*` refit key contract is byte-frozen; any deviation fails `_expected_draft_keys` validation against pinned vLLM 0.25.1.
- New user-facing config defaults live in exemplar YAML + BaseModel field defaults only (repo forbidden-default patterns rule).

## Output File Convention

Main output file: `plan.md` (this file). `alternative_plan_language` is explicitly empty in the merged Humanize config, so no translated variant is written.

> **Note on the appended draft:** the original design draft below is preserved verbatim as the plan's input. Where it differs from the structured plan above, the structured plan reflects the converged and corrected decisions — specifically: (a) `sample_anchor_positions` stays LOCAL (draft §4.1 listed it as an import; disproven — upstream hard-binds its own anchor mask); (b) `DSparkDraftOptions` is a pydantic BaseModel, not TypedDict (draft §4.2; repo config convention); (c) the transport guard also rejects `vllm_s3_sparse`/`vllm_zmq_sparse`, not just `nccl_reshard`, and lives in `lm_policy.Policy.__init__` (draft §4.3/§5); (d) refit preflight covers all three algos incl. dspark (draft §6.3 listed dflash+eagle3); (e) acceptance-length numbers are trend/diagnostic telemetry per DEC-1, not hard gates (draft §6.4).

--- Original Design Draft Start ---

# Draft Co-Training Migration: RL_dspark → NeMo_RL (dspark / dflash / eagle3)

Date: 2026-08-21
Status: approved design (brainstorming complete)
Target branch: `draft-cotraining-migration` (off `bump-automodel-r0.6.0`)

## 1. Background and goal

`RL_dspark` (`/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/speculative_rl/RL_dspark`, branch `dflash` @ `4bc432fb`) is a NeMo-RL fork (fork point `2c5ee18f`, an ancestor of this repo's HEAD) that implements speculative-decoding draft-model co-training with RL:

- **dspark** — validated end-to-end (commits up to `03ff5973`), 1n8g and 4n8g recipes.
- **eagle3** and **dflash** — smoke-tested only (1n8g, ~5–30 steps).

Because RL_dspark's pinned Automodel submodule (`24b47e85`) predates the speculative package, ~2,800 lines were vendored from Automodel `6f423feb0` into `nemo_rl/models/automodel/draft/`. This repo's Automodel submodule (**official** `NVIDIA-NeMo/Automodel` `r0.6.0` @ `7e9493d7`) now ships `nemo_automodel/components/speculative/{dspark,dflash,eagle}/` — and its `dspark/*` + `attention/dflash_mask.py` are **byte-identical** to the vendor base `6f423feb0`.

**Goal:** migrate all three algorithms into this repo, replacing vendor copies with `nemo_automodel` imports wherever the code is unchanged, keeping RL-specific deltas in a thin local extension layer. Acceptance: unit tests green + 1n8g smoke (2–3 training steps) per algorithm with normal acceptance-length levels.

## 2. Decisions (user-confirmed)

1. **Scope: all three algorithms in one migration.** The shared glue (integration runtime, setup/train hooks, vLLM refit) is one surface; splitting would touch the same files twice.
2. **Deltas live in a thin local extension layer** under `nemo_rl/models/automodel/draft/`. The Automodel submodule stays pinned to official r0.6.0; no upstream PRs block this work.
3. **Mechanics: hybrid.** Low/zero-conflict files are replayed in final form from RL_dspark; the `draft/` package is rewritten as the thin extension layer; three files refactored upstream (CP/`cp_sharder`/`DistributedSetup`) are re-ported by hand against the new APIs.
4. **Deep refactor onto Automodel's `DSparkTrainerModule`/`DFlashTrainerModule`/`Eagle3TrainerModule` ("approach C") was assessed and rejected — HIGH risk for all three algos.** Full evidence with file:line citations: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/speculative_rl/moving_deep.md`. Root cause: the serving contract is pinned by vLLM 0.25.1 + the existing checkpoints, while Automodel's speculative package targets offline distillation with from-scratch drafts and its own FSDP gradient conventions (world-group loss reduction, no external-teacher hook, no pretrained/speculators loader, no reduced-vocab dspark, opposite `norm_before_residual`, DFlash without owned embed/lm_head/d2t).

## 3. Verified facts the design relies on

- Fork point `2c5ee18f` is an ancestor of this branch → diffs replay cleanly where upstream churn is low.
- RL_dspark changed 56 files: 21 modified (all still exist here; none moved/deleted), 35 added.
- Vendor lineage: every `draft/` file declares "Vendored from NVIDIA NeMo Automodel @ 6f423feb0"; r0.6.0 is byte-identical to that base for `dspark/*` and `attention/dflash_mask.py`. Every semantic delta in RL_dspark is therefore a deliberate fork-local change, none are stale code.
- Both repos pin **vLLM 0.25.1** → the `_should_share` monkey-patch, `model_runner.{drafter,speculator}.model` probing, and the `draft.*` refit key contract remain valid.
- This repo pins **Transformers 5.12.1** (RL_dspark was older) → locally kept model files must be re-checked against 5.12.1 APIs; imported Automodel modules are already aligned by r0.6.0.
- Draft checkpoints (all in HF cache): `deepseek-ai/dspark_qwen3_8b_block7` (flat, full-vocab), `RedHatAI/Qwen3-8B-speculator.{dflash,eagle3}` and `RedHatAI/Qwen3.6-35B-A3B-speculator.dspark` (speculators format, reduced 32000 vocab, d2t/t2d).

## 4. Design

### 4.1 Thin extension layer — `nemo_rl/models/automodel/draft/`

| RL_dspark file | Action in NeMo_RL |
|---|---|
| `_sampling.py` | **Delete.** Import `sample_tokens` etc. from `nemo_automodel.components.speculative.dspark._sampling`. |
| `dflash_mask.py` | **Delete.** Import `create_dflash_block_mask`, `create_dflash_sdpa_mask` from `nemo_automodel.components.attention.dflash_mask`. |
| `config.py` | **Delete.** Import `build_draft_config` from `nemo_automodel.components.speculative.dspark.config`. |
| `common.py` | **Slim.** Import unchanged helpers from `nemo_automodel.components.speculative.dspark.common` (`extract_context_feature`, `validate_target_layer_ids`, `context_doc_ids`, `sample_anchor_positions`, `create_position_ids`, `create_noise_embed`, `pin_rope_inv_freq_fp32`, `AcceptRatePredictor`). Keep local: `build_anchor_candidate_mask` (RL gating: first-target-token loss mask only), `build_eval_mask(..., supervised_from_slot)` (generalized doc-boundary step), extended `DSparkForwardOutput` (adds `first_supervised_slot`). |
| `markov_head.py` | **Slim.** Import head classes from `nemo_automodel.components.speculative.dspark.markov_head`; keep a local `build_markov_head` wrapper adding the `embed_vocab_size` reduced-vocab split (full-vocab prev-token embedding, draft-vocab bias output). |
| `draft_qwen3.py` | **Keep local** (`Qwen3DSparkModel` with three RL deltas interleaved in `forward`: caller-supplied `teacher_logits`, reduced-vocab d2t/t2d buffers + CE-label mapping, `sample_from_anchor` at-position/bonus-anchor layout for dflash). Default: keep the fork-form standalone module (least risk, matches validated behavior); subclassing Automodel's `Qwen3DSparkModel` is acceptable only if the override surface stays confined to `__init__`/`forward`/`compute_logits`. |
| `loss.py` | **Keep local** (`compute_dspark_loss` with explicit DP `process_group` all-reduce, no `world_size` multiply — matches `automodel_forward_backward`'s `dp*cp` scaling — plus `first_supervised_slot`-aware decay/diagnostics). Import identical sub-helpers where practical. |
| `eagle3_qwen3.py` | **Keep verbatim** (`Qwen3Eagle3DraftModel`, vendored from vllm-project/speculators @ `0b08a89`; honors `norm_before_residual`, returns per-step loss num/den; bound to the RedHat checkpoint + vLLM serving contract). |
| `integration.py` | **Port wholesale**, re-pointing vendored imports to the paths above. Contents: speculators-config adapters (`load_draft_hf_config`, `_adapt_speculators_{dspark,eagle3}_config` incl. the vLLM→trainer −1 layer-id shift), `validate_dspark_draft_config`, `build_{dspark,eagle3}_draft_model`, `PolicyWithDraft`, `DSparkHiddenCapture`, `_DraftRuntimeBase`/`DSparkRuntime`/`Eagle3Runtime`, `draft_capture_ctx`, checkpoint save/load + versioned `draft_meta_record`. |
| `__init__.py` | Re-export shim over local + Automodel symbols. |

Every kept local file gets a header stating what it extends in `nemo_automodel` and why the delta exists.

### 4.2 Training-side glue

**Replayed as-is (zero upstream churn):** `nemo_rl/algorithms/loss/wrapper.py` (`DraftRuntimeLossWrapper`: `combined = policy_loss + loss_weight * draft_loss`; draft loss never backprops into the policy trunk — captures/teacher are detached), `loss/__init__.py`, `nemo_rl/models/automodel/config.py` (`ModelAndOptimizerState` gains optional `draft_model`, `composite_model`; workers switch to named-field access).

**Re-ported by hand against refactored upstream APIs:**

- `nemo_rl/models/automodel/setup.py` — draft build inserted after policy build, before optimizer, rewritten against the r0.6.0 `DistributedSetup`/`MeshContext` bundle: load + adapt HF config → `from_pretrained` in `runtime_config.dtype` → `set_embedding_head_trainable` → FSDP2 `fully_shard` per layer + root over `device_mesh["dp_cp"]` → wrap `PolicyWithDraft`. Single shared optimizer with named param groups `["policy", "draft"]` in stable order; draft group carries its own LR (`dspark.learning_rate` / `eagle3.learning_rate`).
- `nemo_rl/models/automodel/train.py` — `LossPostProcessor` gains `draft_runtime` param, fitted onto the new `cp_sharder` flow; `forward_with_post_processing_fn` stashes raw policy logits via `runtime.stash_teacher_logits(logits, will_scale_inplace=will_scale_temperature(...))` **before** in-place temperature scaling; sequence packing rejected with draft enabled.
- `nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py` — runtime construction + `attach_capture`; `draft_capture_ctx` re-fitted onto the refactored train context (upstream removed `train_context_fn`); `begin_global_batch(mb_slots)` (DP-max slot count; all-reduce under dynamic batching); policy and draft clipped **independently** with the same `max_grad_norm` (`draft_grad_norm` metric, CPU tensor); `_refit_params_generator` chains policy params + `draft.<name>` params across all existing transports; `prepare_refit_info` adds `draft.*` entries (float tensors advertised as training dtype; int/bool d2t/t2d keep dtype); checkpoint save/load via composite-model optimizer pairing + sibling `draft/` DCP dir + `dspark_meta.json` validation (legacy no-`algo` meta = dspark).
- `nemo_rl/models/value/workers/dtensor_value_worker_v2.py` — named-field access only.

**Config types (manual merge; upstream drifted):** `nemo_rl/models/policy/__init__.py` — `DraftConfig` gains `algo: Literal["eagle3","dspark","dflash"]`, `dspark: DSparkDraftOptions` (TypedDict), `eagle3: Eagle3DraftOptions` (pydantic BaseModel); module constants `DEFAULT_DRAFT_ALGO`, `BLOCK_DRAFT_ALGOS`, `DRAFT_ALGOS`. `lm_policy.py` validation gate: eagle3 on Megatron or DTensor-v2; dspark/dflash DTensor-v2 only; `model_name` mandatory; hard-error on `sequence_parallel`, LoRA, sequence packing. Megatron eagle3 path stays untouched.

### 4.3 Generation / refit side

- `nemo_rl/models/generation/__init__.py` + `interfaces.py` — `draft_full_refit_enabled(policy_cfg)` → `GenerationConfig["_draft_full_refit"]`; wired in `examples/run_grpo.py`, `run_grpo_single_controller.py`, `examples/nemo_gym/run_grpo_nemo_gym.py`.
- `nemo_rl/models/generation/vllm/vllm_worker.py` (zero upstream churn — clean replay) — before engine creation with `load_format=="dummy"` and method in {dspark, dflash} (or eagle3 with `_draft_full_refit`): set `NRL_DRAFT_DISABLE_MODULE_SHARING=1` and monkey-patch `_should_share` to always-False in `vllm.v1.worker.gpu.spec_decode.{eagle,dspark,dflash}.utils`, so a `draft.*` refit can never overwrite the policy's serving embed/lm_head through vLLM's module aliasing.
- `nemo_rl/models/generation/vllm/vllm_backend.py` — replay `COTRAINED_SPECULATIVE_METHODS`, `_expected_draft_keys` (inverts the pinned-vLLM loader naming: `model.` prefix, fused qkv/gate_up, `d2t`→`draft_id_to_target_id`, per-method skip lists), `_validate_draft_refit_info` (hard error on missing/unexpected keys; no `draft.*` at all = static-drafter mode), `_load_draft_weights` strict-cotraining hardening, `_assert_drafter_owns_modules` (data_ptr alias backstop), `_get_drafter_model` (probes `drafter.model` and `speculator.model`), `draft_refit_preflight_report` RPC. Reconcile with the upstream `_weight_update_lifecycle` refactor and MTP gating so draft weights are neither double-finalized nor filtered.
- **New guard:** draft co-training + `nccl_reshard` transport → hard error at setup with a message naming the unsupported combination. Rationale: the `draft.*` stream is wired into ipc/http/checkpoint-engine/collective only; `nccl_reshard` is currently a Megatron-only, non-colocated transport (sender exists only in `megatron_policy_worker.py`) while dspark/dflash require DTensor-v2 — the guard prevents a silently stale drafter if that ever changes. Wiring `draft.*` into the nccl_reshard misc packed-broadcast path is an explicit follow-up, not in scope.
- The `draft.*` key naming, skip lists, and d2t (int64) / t2d (bool) dtypes must stay **byte-identical** to RL_dspark's — `_expected_draft_keys` validation and the pinned vLLM loaders depend on them.

### 4.4 Config surface and recipes

Replay: exemplar `examples/configs/grpo_math_1B.yaml` (`policy.draft.algo` + `dspark:` options block, commented `eagle3:` block) and `tests/unit/reference_configs/grpo_math_1B.yaml`; recipes `grpo-qwen3-8b-1n8g-automodel{,-dspark,-dflash,-eagle3}.yaml`, `dapo-qwen3-8b-1n8g-automodel{,-dspark}.yaml`, `grpo-qwen3-8b-math-baseline-4n8g-automodel{,-dspark}.yaml`, `grpo-qwen3.6-35ba3b-math-baseline-4n8g-automodel{,-dspark}.yaml`; `.gitignore` additions. Draft architecture fields (block_size, target_layer_ids, mask_token_id, d2t/t2d, aux ids, ttt capability) are read **only** from the draft checkpoint's `config.json`, never YAML. `policy.generation.vllm_kwargs.speculative_config` passes through verbatim (`method`, `model: ${policy.draft.model_name}`, `num_speculative_tokens`, dspark/dflash also `attention_backend: FLASH_ATTN`, `draft_sample_method: probabilistic`). Activation checkpointing stays **off** in the draft recipes, matching the validated configuration (the capture hooks arm/disarm so AC backward replays would be no-ops, but AC-on was never acceptance-gated in RL_dspark).

### 4.5 Checkpointing

Draft weights: torch DCP in sibling `<parent>/draft/` dir (keeps `detect_checkpoint_format` unconfused) + versioned `dspark_meta.json` (`meta_version, algo, model_name, train_embed_and_head, optimizer_layout` + per-algo fields). Optimizer/scheduler saved via `checkpointer.save_optimizer(..., model=composite_model)` — one-model-one-optimizer pairing invariant. Load validates meta (cross-algo resume rejected).

### 4.6 Metrics

dspark/dflash: `draft_loss`, `draft_ce_loss`, `draft_tv_loss`, `draft_conf_loss`, `draft_tau`, `draft_accept_rate@{k}`. eagle3: `draft_loss`, `draft_ttt_loss@{step}`, `draft_full_acc@{step}`, `draft_cond_acc@{step}`. Worker: `draft_grad_norm`. Serving-side acceptance (`vllm/spec_acceptance_length` etc.) comes from the pre-existing Prometheus counter scraping — no changes needed.

## 5. Error handling and guards (consolidated)

- lm_policy: backend gate (dspark/dflash → DTensor-v2 only), reject sequence_parallel / LoRA / sequence packing with draft enabled.
- setup: `nccl_reshard` + draft co-training → hard error (new).
- Worker: warn when `speculative_config.num_speculative_tokens` mismatches drafter capability (dspark: block_size; dflash: block_size−1; eagle3: ttt_steps).
- Refit: manifest validation hard-errors on missing/unexpected `draft.*` keys; alias assertion on embed/lm_head data_ptr; module-sharing disabled under dummy load.
- Checkpoint: meta validation hard-errors on algo/field/optimizer-layout mismatch.
- Config adapters: refuse to guess (`norm_before_residual` must be explicit; dflash requires `confidence_loss_alpha == 0`, no markov/confidence heads).

## 6. Testing and acceptance criteria

1. **Unit tests** (replayed, imports re-pointed; run first and throughout): `test_draft_config_adapters.py`, `test_draft_loss_decay.py`, `test_draft_vocab_maps.py`, `test_dspark_anchor_sampling.py`, `test_dspark_checkpoint_meta.py`, `test_eagle3_trainer_grads.py`, plus the `test_vllm_backend.py` additions. All pass under this repo's Transformers 5.12.1 pin.
2. **Lint:** `uv run --no-sync ruff check` + `ruff format` clean on touched files.
3. **Refit canary:** `tools/draft_verification/vllm_refit_preflight.py` (ported with `_speculators_oracle.py`, `verify_eagle3_parity.py`, `verify_dflash_alignment.py`) passes for dflash + eagle3 before smoke runs.
4. **Smoke (1n8g, 2–3 training steps each, in-container `uv run`), order dspark → dflash → eagle3:**
   - run starts, refit completes with no draft-key errors, no module-alias trips;
   - `draft_loss` finite and trending down;
   - `vllm/spec_acceptance_length` at normal levels from RL_dspark experience: dspark ≥ 2.0, dflash ≈ 3.0–3.5, eagle3 ≈ 2.0–2.5;
   - checkpoint save + resume sanity for one algo (dspark).
5. **Nightly wiring:** replay `tests/test_suites/llm/grpo-qwen3-8b-1n8g-automodel-{dflash,eagle3}.sh` + `nightly.txt` entries (30-step gates unchanged).

## 7. Implementation order

1. Branch `draft-cotraining-migration` (done) + replay low-conflict files: config types (`policy/__init__.py`, manual merge), `lm_policy.py`, loss wrapper, `automodel/config.py`, generation `__init__`/`interfaces`, `vllm_worker.py`, entrypoints, `.gitignore`.
2. `draft/` thin extension layer + unit tests green.
3. `integration.py` port (imports re-pointed).
4. `setup.py` / `train.py` / `dtensor_policy_worker_v2.py` / `dtensor_value_worker_v2.py` re-wiring against r0.6.0 APIs.
5. `vllm_backend.py` replay + lifecycle reconciliation + `nccl_reshard` guard.
6. Recipes + exemplar YAMLs.
7. Full unit suite + ruff.
8. `vllm_refit_preflight` + smoke dspark → dflash → eagle3.
9. Nightly wiring; final docs pass.

## 8. Risks and mitigations

- **Silent semantic drift while re-porting the three refactored files** (capture ctx vs new train context, `cp_sharder`, `DistributedSetup`): mitigate with the ported unit tests + one-step training assertion that `draft_grad_norm > 0` and policy grads are unaffected when `loss_weight=0`.
- **Transformers 5.12.1 breakage in kept-local model files** (`draft_qwen3.py`, `eagle3_qwen3.py` were written against an older pin): mitigate by preferring imports of Automodel classes (already 5.12.1-aligned) and running `test_draft_vocab_maps` / `test_eagle3_trainer_grads` early.
- **`_weight_update_lifecycle` / MTP-gating interaction** in `vllm_backend.py`: mitigate with `vllm_refit_preflight.py` before any smoke run.
- **Accept-length regression vs RL_dspark numbers**: smoke gates above; dspark 4n8g re-validation is out of scope here (follow-up).

## 9. Out of scope / follow-ups

- Wiring `draft.*` into the `nccl_reshard` misc broadcast path.
- Upstreaming the RL deltas (teacher_logits, DP process_group loss, reduced-vocab dspark, `norm_before_residual` for eagle) to NVIDIA-NeMo/Automodel.
- Adopting Automodel extras (sequence packing for drafts, chunked linear-CE, FA2 eagle attention, LK loss metrics, z-lab DFlash-b16 checkpoint experiments).
- Full 4n8g dspark re-validation and dspark nightly driver.

--- Original Design Draft End ---
