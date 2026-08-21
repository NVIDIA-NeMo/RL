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
