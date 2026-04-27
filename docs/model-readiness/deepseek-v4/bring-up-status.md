# DeepSeek-V4-Flash bring-up on NeMo-RL — status tracker

Tracking progress of DSV4 (Flash + Flash-Base) model support on NeMo-RL. Partner issues: NVIDIA-NeMo/Automodel#2034, vllm-project/vllm#40760 (rebased into and superseded by **vllm-project/vllm#40860 — merged into main 2026-04-27** as commit `4d51588e2`).

## ✅ Done

### Model assets
- [x] Downloaded `deepseek-ai/DeepSeek-V4-Flash-Base` (275 GB, FP8 experts) to lustre
- [x] Downloaded `deepseek-ai/DeepSeek-V4-Flash` (149 GB, FP4+FP8 experts) to lustre — includes DeepSeek's `inference/` reference code

### Readiness analysis
- [x] Arch-check report (`docs/model-readiness/deepseek-v4/arch-check-report.json`): 0 must-handle / 0 verify / 3 inspect / 8 info; novel atoms captured (hyper-connections, per-layer compress_ratios, MTP head, MLA with o_lora/o_groups, hash routing on first 3 layers, dual RoPE)
- [x] Compatibility report (`docs/model-readiness/deepseek-v4/model-compat-report.md`): transformers has no `deepseek_v4` in any release / main / open PR; vLLM has it only in PR #40760; ~288B total / ~17.7B active params

### vLLM-side bring-up (Flash variant only)
- [x] Stage A — standalone inference sanity test via `vllm/vllm-openai:deepseekv4-cu129` sqsh + raw `vllm serve`; "Paris." returned from "The capital of France is"
- [x] Stage B1 — vLLM wheel extracted from the DSV4 image to lustre (`/lustre/fsw/.../wheels/vllm-0.1.dev15826+g306b63f67-cp38-abi3-linux_x86_64.whl`, 1.2 GB)
- [x] Stage B2 — `pyproject.toml` + `uv.lock` updated on branch `deepseek-v4-support` (commit `7f0dd4dc6` on `shuangy/RL`): torch 2.10→2.11, torchvision 0.25→0.26, torchaudio 2.10→2.11, `deep_gemm` pinned to `7f2a703ed51ac1f7af07f5e1453b2d3267d37d50` (matches vLLM-DSV4's vendored commit — the older `7b6b5563` lacks DSV4 kernels), vllm swapped to `file://` wheel
- [x] Stage B3 — env-refresh bakes `/lustre/fsw/.../images/nemo-rl-dsv4-dg7f2a-2026-04-24.sqsh` (98 GB); `import vllm._C` + `DeepseekV4ForCausalLM` in ModelRegistry both verified on Python 3.13
- [x] Default `grpo_math_1B` smoke step (Qwen2.5-1.5B) passes on the new sqsh — policy worker venv (torch 2.11, automodel OK) and vllm worker venv (torch 2.11, vllm 0.1.dev15826) both import clean
- [x] End-to-end `vllm.LLM(**llm_kwargs)` on DSV4-**Flash** via NeMo-RL's init pattern returns "Paris" — matches Stage A byte-for-byte; test harness at `tools/test_vllm_dsv4_inference.py`

### Automodel-side bring-up (Flash variant only)
- [x] Automodel pinned to PR #2039 (`khazic/Automodel_lao @ feat/deepseek-v4-flash`, commit `ab2d7a08`); transformers 5.3→5.5; sqsh baked at `nemo-rl-dsv4-pr2039-ab2d7a-tf55-2026-04-25.sqsh` (commit `fb8221102`)
- [x] Meta-load smoke (`tools/dsv4_automodel_meta_load.py`): 284B-param graph built on meta device, 43 layers / 256 experts / 1242 params, `state_dict_adapter` registered (`tools/dsv4_meta_load.log`)
- [x] **Upstream PR's canonical 16-node SFT smoke passes end-to-end** (job `11332097`, 2026-04-25): `examples/llm_finetune/deepseek_v4/deepseek_v4_flash_hellaswag.yaml` with `max_steps=5`, FSDP2 + PP=4 + EP=32 across 16 H200 nodes / 128 GPUs. Real-weight load via `state_dict_adapter` with `dequantize_base_checkpoint=true` — no shape mismatch on FP4-packed experts + FP8 attention. All 5 steps clean: loss 2.49→1.99 monotone↓, grad_norm 6.0→2.1 monotone↓, val loss @ step 4 = 1.94, peak mem 38.97 GiB/GPU, ~225 tok/s/GPU steady. Submission script: `submit_test_automodel_dsv4_flash_smoke.sh`.

### NeMo-RL GRPO integration smoke (in progress, Flash variant)
- [x] Recipe drafted: `examples/configs/recipes/llm/grpo-deepseek-v4-flash-16n8g-automodel-smoke.yaml`. PP=1, EP=64, FSDP_dim=128 (PP-free per NeMo-RL's Automodel constraint). vLLM TP=16 cross-node, EP=16, `kv_cache_dtype=fp8`, `load_format=auto`, `gpu_memory_utilization=0.5`, `enforce_eager=true`. `max_total_sequence_length=4096`, `max_num_steps=2`, no ckpt/val/wandb. Reward via `dapo_math_verify` over `BytedTsinghua-SIA/dapo-math-17k`.
- [x] Submit script: `submit_test_grpo_dsv4_flash_smoke.sh`. Sources `~/.env`, mounts both fs1 + fsw, exports `NRL_FORCE_REBUILD_VENVS=true` and `NRL_ALLOW_FP8_KV_DSV4=1`, prepends `${WORK_DIR}/3rdparty/Automodel-workspace/Automodel` to PYTHONPATH so the driver venv (which lacks `nemo_automodel`) can still import the registry.
- [x] **Local NeMo-RL patches** (all needed before the smoke can even *start*):
  - `examples/run_grpo.py:23` — added `import nemo_automodel._transformers.registry` (defensive try/except) so DSV4 model_type registers with transformers AutoConfig before AutoTokenizer fires.
  - `nemo_rl/models/huggingface/common.py:58 is_gemma_model()` — wrap AutoConfig.from_pretrained in try/except (ValueError, KeyError, OSError); unknown model_type can't be Gemma so return False.
  - `nemo_rl/models/generation/vllm/vllm_worker.py:509` — same try/except pattern around AutoConfig.from_pretrained guarding the GptOss/Gemma3/Qwen3.5 architecture branch.
  - `nemo_rl/models/generation/vllm/vllm_worker.py:497` — forward `kv_cache_dtype` from `vllm_cfg` to vLLM unconditionally (was silently dropped when `precision != "fp8"`).
  - `nemo_rl/algorithms/grpo.py:setup` — gate the fp8-KV-needs-fp8-weights and fp8-KV-blocks-DTensor asserts behind `NRL_ALLOW_FP8_KV_DSV4=1`. DSV4 vLLM hard-asserts `kv_cache_dtype.startswith("fp8")` regardless of weight precision.
- [x] **Refreshed vLLM wheel + sqsh** (2026-04-26):
  - Pulled current `vllm/vllm-openai:deepseekv4-cu129` (digest `sha256:3bc5d44a7dfc3...`, last pushed 2026-04-24 22:59 UTC) to `nemo-rl-dsv4-pr2039-ab2d7a-tf55-vllm15833-2026-04-26.sqsh`.
  - Re-extracted wheel: `vllm-0.1.dev15833+g62d441ee8-cp38-abi3-linux_x86_64.whl` (vLLM build #15833, internal commit `62d441ee8` — like the prior `g306b63f67`, this SHA is not in the public PR #40760 graph; image is built from a private/squashed branch). Code-level inference (file-content diff against the public PR graph): wheel sits roughly at PR #40760 state ~Apr 24 mid-day UTC, post-`4bab47b43` (FP4-indexer-cache fix) and post-`aa114601` (1024-topk add), but **pre-`6d244bdb4` (dummy-load fix)** and pre-`06e4b4f5b` ("Add model change", +600 lines in `deepseek_v4.py`). `finalize_weights()` is absent from this wheel's `deepseek_v4.py` (0 hits) vs 3 hits at public PR head — confirming the fix is missing. So `load_format: auto` recipe override remains required for this sqsh.
  - `pyproject.toml` + `uv.lock` updated to point at the new wheel.
  - `env_refresh_dsv4.sh` baked the new sqsh: policy worker venv has transformers 5.5 + automodel `ab2d7a08`; vLLM worker venv has vllm `0.1.dev15833+g62d441ee8` + DSV4 config patch (md5 `2071ef2b53d52869e44de910eceec3c8`).
- [x] **PR #40860 merged + new wheel from teammate's local build** (2026-04-27):
  - vllm-project/vllm PR #40860 "[Feat] DeepSeek V4 Rebased" merged into vllm main as commit `4d51588e2`. This is the rebased successor to PR #40760 (which is being abandoned) — full DSV4 model + the dummy-load fix (`finalize_weights()` 3× hits in main's `deepseek_v4.py`, 1437 lines vs our previous wheels' 849).
  - New wheel from larkz local build off vllm main @ `c0879d948` (post-#40860 merge): `/lustre/fsw/portfolios/coreai/users/larkz/nemorl-ds4/dist/vllm-0.19.2rc1.dev219+gc0879d948.cu129-cp313-cp313-linux_x86_64.whl` (683 MB, md5 `50571679730c1d26196ddb3107d25204`). Note: this wheel is `cp313-cp313` only (NOT `cp38-abi3` like the prior internal builds) — Python 3.13-specific.
  - `pyproject.toml` updates (commit `723ac2457` on `shuangy/RL:deepseek-v4-support`):
    - `deep_gemm`: `7f2a703e...` → `891d57b4db1071624b5c8fa0d1e51cb317fa709f` (matches vllm main's `cmake/external_projects/deepgemm.cmake` pin after #40860 bumped from `477618cd5` → `891d57b4`).
    - `vllm`: file-URL bumped to the larkz wheel above.
    - `flashinfer-python` / `flashinfer-cubin`: `0.6.4` → `0.6.8.post1`; `nvidia-cutlass-dsl`: `>=4.4.0.dev1` → `>=4.4.2`. All three match the wheel's `Requires-Dist` exactly.
    - `requires-python`: added upper bound `<3.14` (cp313-only wheel breaks 3.14 splits in uv lock).
    - `[tool.uv].override-dependencies` flashinfer pins tightened to exact `==0.6.8.post1` (uv overrides REPLACE constraints rather than merge; loose floor was letting uv resolve to 0.6.9 instead of the wheel's pin). sglang's 0.6.7.post2 is intentionally superseded since this branch is vllm-only.
- [x] **Torch ABI rollback 2.11 → 2.10** (2026-04-27, post-bake debug):
  - First bake attempt (job `11362567`) failed at venv rebuild with `ImportError: vllm/_C.abi3.so: undefined symbol: _ZN3c1013MessageLoggerC1EPKciib`. Diagnosis: larkz's local vllm wheel was built in a `torch==2.10.0` venv (`larkz/nemorl-ds4/pyproject.toml:torch==2.10.0`) even though the wheel's METADATA reports `Requires-Dist: torch==2.11.0` (string copied from upstream vllm setup; not derived from build link state). The wheel's `_C.abi3.so` calls the old 4-arg `MessageLogger(char const*, int, int, bool)` constructor. torch 2.11.0+cu129's libc10 only exports the new `MessageLogger(c10::SourceLocation, int, bool)` constructor.
  - Rolled `pyproject.toml` back to `torch==2.10.0+cu129` / `torchvision==0.25.0` / `torchaudio==2.10.0` in 5 sites (project deps, build group, override-dependencies). Added explicit `torchvision==0.25.0` to override-deps too. uv lock confirmed: `torch v2.11.0+cu129 → 2.10.0+cu129`, `torchvision 0.26.0+cu129 → 0.25.0+cu129`, `torchaudio 2.11.0+cu129 → 2.10.0+cu129`, `nvidia-nccl-cu12 2.28.9 → 2.27.5` (transitively), `cuda-toolkit v12.9.1` removed.
- [x] **DSV4-Base FP8 quick-patch updated for post-#40860 wheel** (2026-04-27):
  - Anchor #7 of `tools/patch_vllm_dsv4_base_fp8_quick.sh` rewritten: upstream's rewritten `load_weights` now captures the result before calling `self.model.finalize_mega_moe_weights()`, so `"return loader.load_weights(...)"` no longer matches; new anchor `"loaded_params = loader.load_weights(...)"`.
  - Added new 8th anchor: forces `use_mega_moe = False` when `VLLM_DSV4_BASE_FP8=1`. Without this, the new `finalize_mega_moe_weights()` post-load would invoke `experts.finalize_weights()` on a non-MegaMoE FusedMoE layer (Base routes through `Fp8MoEMethod`, not `DeepseekV4MegaMoEExperts`).
  - Verified end-to-end against the larkz wheel: 9 patch anchors apply cleanly; `py_compile` succeeds on patched files.
- [x] **DSV4-Base FP8 sqsh baked + Base inference end-to-end** (2026-04-27, jobs `11363285` + `11363286`):
  - Sqsh baked: `/lustre/fsw/portfolios/coreai/users/shuangy/images/nemo-rl-dsv4-vllm-c0879d-base-torch210-2026-04-27.sqsh` (~99.8 GiB). Both patches applied during bake — config patch md5 `2071ef2b53d52869e44de910eceec3c8`, Base FP8 quick-patch 9-anchor success. First Qwen2.5-1.5B grpo step landed clean (loss 0.0203, KL 0.0007, reward 0.0801) confirming venv health.
  - Standalone vLLM inference test on **`DeepSeek-V4-Flash-Base`** (test script: `tools/test_vllm_dsv4_base_inference.py`, env `VLLM_DSV4_BASE_FP8=1`, TP=8 EP=on): `vllm.LLM` ready in 784s (first-load incl. TileLang JIT for MLA kernels). All key DSV4 markers active: `FLASHINFER_CUTLASS` FP8 MoE backend, `MoEPrepareAndFinalizeNoDPEPModular`, `DEEPSEEK_SPARSE_SWA` KV cache, FP8 indexer cache. **No `_load_w13: size mismatch`** — Base mapper's `.scale → .weight_scale_inv` rename fired correctly.
  - Generated text (greedy, 32 tokens): `' Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of Portugal is Lisbon. The capital'`. Coherent pattern-completion → end-to-end DSV4-Flash-Base FP8 inference works on NeMo-RL's vLLM path.
- [ ] **GRPO smoke run does not yet pass end-to-end.** 12 attempts logged to date on the prior sqsh — all caught and fixed pre-flight or driver-init bugs (mount path, registry import, kv_cache_dtype config, AutoConfig calls in worker venv, dummy load FP8 div-by-zero, etc.). Next attempt blocks on the Automodel branch swap (teammate's WIP — Base FP8-block-quant state_dict_adapter). **Next failure mode to expect:** refit roundtrip during step 2's weight transfer (`state_dict_adapter.convert_single_tensor_to_hf` going bf16 → FP8 block layout). The dummy-load failure mode is now gone since main's `finalize_weights()` is in the new wheel.

## ⬜ To do (our side)

- [ ] **Architecture deep-dive to design the RL recipe.** Goal: define precision / quantization / GRPO config based on how the model was actually trained and served. Sources to cross-reference:
  - Technical report: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf (architecture details, training regime, post-training pipeline — two-stage domain-expert SFT+GRPO then on-policy distillation)
  - DeepSeek's reference inference code: `/lustre/fsw/portfolios/coreai/users/shuangy/models/deepseek-ai/DeepSeek-V4-Flash/inference/` (`model.py`, `kernel.py`, `convert.py`, `config.json`) — ground truth for layer semantics, weight dispatch (FP4/FP8), rotary + YaRN setup, compressor/indexer wiring, hyper-connections, MTP
  - vLLM implementation: vllm-project/vllm#40760 (branch `zyongye:dsv4`) and the `vllm/model_executor/models/deepseek_v4.py` in the baked sqsh — transformer-shape port, shows how to plumb the above into a production inference stack
  - Cross-reference: https://github.com/radixark/miles/pull/1045
  - Specific decisions to lock down: (1) policy precision for training (bf16 full, bf16 + FP8 experts, FP8 block-quant kept frozen, etc.); (2) whether to load Base as-is FP8 or dequantize to bf16 on load; (3) vLLM generation precision / `kv_cache_dtype` / TP + EP layout; (4) GRPO hyperparameters (rollout size, LR, KL coefficient, advantage normalization, sequence length cap inside 65K YaRN origin); (5) whether MTP head is kept / frozen / dropped during RL; (6) reasoning-mode handling (DSV4's three modes: non-think / think-high / think-max) and corresponding prompt templates / `reasoning-parser`.
- [ ] Draft DSV4 recipe YAML (via `/new-recipe`) once the above is settled
- [ ] Draft DSV4 test script (via `/test-script`) + register with nightly (via `/register-nightly`) — parked until automodel + vllm Base-loading unblock
- [ ] Inference-only validation on Flash while waiting: batched correctness sweep, MMLU/GSM8K via the OpenAI endpoint, long-context smoke at 65K tokens. Gives us a baseline for later RL regression checks.

## 🔗 External dependencies

- **NVIDIA-NeMo/Automodel#2034** — "Support DSV4 flash" (opened 2026-04-24 by HuiyingLi, no PR in flight yet). NeMo-RL's policy worker can't initialize without this. PR #2039 ships a custom automodel impl for **Flash** (FP4 experts) but not Base (FP8 block-quant). Teammate is preparing a new Automodel branch to add Base support — will replace the #2039 pin once available.
- **vllm-project/vllm PR #40760** — "[New Model] Support DeepseekV4" — **superseded by #40860** (rebased + cleaned). Track only for historical context now.
- **vllm-project/vllm PR #40860** — "[Feat] DeepSeek V4 Rebased" — **MERGED 2026-04-27** as commit `4d51588e2`. Full DSV4 model + dummy-load fix on main. Standard release path is now unblocked at the vllm side; the next public release should ship DSV4 OOTB.
- **huggingface/transformers** — no `deepseek_v4` in release/main/PRs and no remote code in the HF repo. Not strictly blocking (automodel can land its own impl), but would simplify path (b) if it lands.
- **`tools/patch_vllm_dsv4_base_fp8_quick.sh`** (in-tree, commit `0813759fb` by Zhaopeng Qiu, updated 2026-04-27 for post-#40860 wheel) — runtime monkey-patch on the vLLM worker venv that adds DSV4-Flash-**Base** FP8-block-quant support: routes Base experts through `Fp8MoEMethod` instead of `Mxfp4MoEMethod`, renames `.scale → .weight_scale_inv`, replaces `indexer.k_norm` with `nn.Identity()`, and forces `use_mega_moe=False` so `finalize_mega_moe_weights()` is a no-op (Base doesn't use MegaMoE). Gated by env `VLLM_DSV4_BASE_FP8=1`. **Verified end-to-end** against the larkz wheel: 9 anchors apply cleanly, `py_compile` clean, baked into the 2026-04-27 sqsh, generated coherent text in inference job `11363286`.

## 🚫 Blockers pending

1. **DSV4-Flash-Base support — partially unblocked, validation pending** (as of 2026-04-27). Layout diff:
   - Base: `w*.weight` `[2048, 4096]` F8_E4M3, `w*.scale` `[16, 32]` F32 (128×128 block)
   - Flash: `w*.weight` `[2048, 2048]` I8 (FP4 packed), `w*.scale` `[2048, 128]` F8_E8M0 (per-row)

   - **vLLM**: 🟢 **CLEARED** — Flash works OOTB; Base now works end-to-end via `tools/patch_vllm_dsv4_base_fp8_quick.sh` (env `VLLM_DSV4_BASE_FP8=1`) baked into the new sqsh. Original `_load_w13` shape mismatch resolved by `.scale → .weight_scale_inv` rename. Inference job `11363286` (2026-04-27) generated coherent text from "The capital of France is" → " Paris. The capital of Germany is Berlin..." confirming weight load + MoE forward + attention + KV cache all functional.
   - **Automodel**: 🟡 PR #2039 (`khazic/Automodel_lao @ feat/deepseek-v4-flash`) implements only the FP4-expert loader path; Base's FP8-block-quant expert layout is missing from the state_dict_adapter. **Teammate is preparing a new Automodel branch with Base support** — will replace #2039 in the submodule pin and unblock end-to-end Base load.

   **Critical for RL** — policy initialization starts from Base, not Flash. **vLLM half cleared as of 2026-04-27**; the Automodel half remains pending the teammate's new branch. Tracked via `tools/track_deps.py` against `dep-tracking.json` (Automodel issue #2034 meta-tracker, PR #2039, now-merged vLLM PR #40860).
2. **Ninja not in PATH for vLLM workers** (minor). deep_gemm JIT-compiles kernels and shells out to `ninja`, which NeMo-RL's multiproc executor doesn't expose from the worker venv. Workaround: prepend `$VENV/bin` to PATH in the srun/launcher wrapper; long-term fix belongs in NeMo-RL's worker entrypoint or the base container's apt layer.
3. **`get_mn_major_tma_aligned_tensor` not re-exported** from `deep_gemm` top-level at commit `7f2a703` (exists in C++ `layout.hpp` only). Soft — vLLM uses `getattr(..., None)` so import doesn't crash and the inference path works around it. Flag if any DSV4 codepath actually calls it (would surface as `TypeError: 'NoneType' object is not callable` from `vllm/utils/deep_gemm.py`). **2026-04-27 update**: deep_gemm pin bumped to `891d57b4` (matches vllm main); re-verify whether the symbol is now exported at this newer commit.
4. **DSV4 vLLM `load_format=dummy` breaks on FP8 block-quant weight requant** — **upstream fix is now in our pinned wheel** (as of 2026-04-27 wheel bump). NeMo-RL defaults `load_format=dummy` for training (the first refit overwrites weights anyway), but DSV4's FP8 block-quant post-processing (`requant_weight_ue8m0_inplace`, `vllm/model_executor/layers/quantization/utils/fp8_utils.py:1003`) divided by zero on dummy-initialized scales. Upstream fix: PR #40760's commit `6d244bdb4` "Support dummy loading" calls `self.finalize_weights()` lazily in the MoE forward; this commit was rebased into PR #40860 and merged to vllm main as `4d51588e2` (`finalize_weights()` confirmed at 3 hits in main's `deepseek_v4.py`). Our prior wheels `g306b63f67` (Apr 23) and `g62d441ee8` (Apr 24-26) both predated the fix; the new larkz wheel `0.19.2rc1.dev219+gc0879d948` (built off main @ `c0879d948`, post-#40860) **includes** the fix. **Action**: once the next sqsh is baked with the new wheel, run a dummy-load smoke (flip recipe to `load_format: dummy` on standalone `tools/test_vllm_dsv4_inference.py`); if init survives + first refit produces sane logits, drop the `load_format: auto` recipe override at `grpo-deepseek-v4-flash-16n8g-automodel-smoke.yaml:144`.

## 📌 Artifacts

- Branch: [`shuangy/RL:deepseek-v4-support`](https://github.com/sharonyu-115/RL/tree/deepseek-v4-support) (commit `723ac2457`, "feat: bump vLLM to post-#40860 main wheel; align deep_gemm + flashinfer", 2026-04-27)
- Fallback branch: `shuangy/RL:deepseek-v4-support-py312` (Python 3.12 workaround, not needed once the 7f2a703 path verified)
- vLLM upstream reference clone: `3rdparty/vllm-pr40760/` (working tree on `main`, includes branches `pr-40760` and `pr-40860` for archaeology; partial clone, ~144 MB).
- Sqsh history (newest at top):
  - `/lustre/fsw/portfolios/coreai/users/shuangy/images/nemo-rl-dsv4-vllm-c0879d-base-torch210-2026-04-27.sqsh` (~99.8 GiB) — **current**, baked by job `11363285` (2026-04-27). Pins: vllm wheel `0.19.2rc1.dev219+gc0879d948.cu129` (post-#40860 main) + automodel `ab2d7a08` + transformers 5.5 + deep_gemm `891d57b4` + flashinfer 0.6.8.post1 + cutlass-dsl 4.4.2 + **torch 2.10.0+cu129** (rolled back from 2.11 to match larkz wheel ABI). Both DSV4 patches baked: vllm config patch + Base FP8 quick-patch. **Validated**: Base inference job `11363286` produced coherent text. Awaiting Automodel branch swap before next rebake.
  - `/lustre/fsw/.../images/nemo-rl-dsv4-vllm-c0879d-base-2026-04-27.sqsh` (106 GiB) — **broken**, pyxis exported it on srun exit despite `set -e` aborting `env_refresh_dsv4.sh` mid-bake (torch 2.11 ABI mismatch with larkz wheel). Patches were never applied. Preserved for forensics; safe to delete.
  - `/lustre/fsw/.../images/nemo-rl-dsv4-pr2039-ab2d7a-tf55-vllm15833-2026-04-26.sqsh` (99 GB) — prior, vllm wheel `g62d441ee8` + automodel `ab2d7a08` + transformers 5.5 + DSV4 vllm config patch (md5 `2071ef2b53d52869e44de910eceec3c8`).
  - `/lustre/fsw/.../images/nemo-rl-dsv4-pr2039-ab2d7a-tf55-2026-04-25.sqsh` (98 GB) — older, vllm wheel `g306b63f67`. Used for SFT smoke `11332097`.
  - `/lustre/fsw/.../images/nemo-rl-dsv4-dg7f2a-2026-04-24.sqsh` (98 GB) — oldest, pre-PR-#2039-pin.
- Extracted / built wheels (newest at top):
  - `/lustre/fsw/portfolios/coreai/users/larkz/nemorl-ds4/dist/vllm-0.19.2rc1.dev219+gc0879d948.cu129-cp313-cp313-linux_x86_64.whl` (683 MB, md5 `50571679730c1d26196ddb3107d25204`, **cp313-only — not abi3**) — **current pin in pyproject.toml**, built off vllm main @ `c0879d948` post-#40860 merge. Includes dummy-load fix.
  - `/lustre/fsw/.../wheels/vllm-0.1.dev15833+g62d441ee8-cp38-abi3-linux_x86_64.whl` (431 MB, repacked via `wheel pack`) — prior pin.
  - `/lustre/fsw/.../wheels/vllm-0.1.dev15826+g306b63f67-cp38-abi3-linux_x86_64.whl` (1.2 GB, original pip-built) — earlier pin.
- Standalone test script: `tools/test_vllm_dsv4_inference.py` (local).
- DSV4-Flash-Base FP8 quick patch: `tools/patch_vllm_dsv4_base_fp8_quick.sh` (commit `0813759fb` by Zhaopeng Qiu, gated by `VLLM_DSV4_BASE_FP8=1`).
- GRPO smoke artifacts: `examples/configs/recipes/llm/grpo-deepseek-v4-flash-16n8g-automodel-smoke.yaml`, `submit_test_grpo_dsv4_flash_smoke.sh`.
- Readiness docs: `docs/model-readiness/deepseek-v4/` (local).
