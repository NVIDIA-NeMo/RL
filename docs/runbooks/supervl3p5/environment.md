# Environment used for SuperVL3.5

Recorded on 2026-10-04. These pins describe the active HSG checkouts; local patches are also required.

| Component | Commit / version |
|---|---|
| NeMo-RL | `009f763a90fefd5f2f38e998aaeed948e41965fb`, base commit; working branch `rohit/unified-teacher-supervl3p5` |
| Megatron-Bridge | `1f8873bb00a8ddf3af811649f0a7efdb2363570c`, detached |
| Megatron-LM | `6a3660905a2736b5670baed1ca5954372937918b`, detached |
| NeMo-Gym | `c004bce8068eaa35690be8705d3e263be10581e8`, branch `rohit/unified-teacher-image-tools-capture` |
| vLLM | `0.29.0` in the worker environment |
| NeMo Lens runtime | `b0f977d414b2f89938604a0b7eaa78ee08bc8700` |

- Container: `/home/rohitkumarj/data/enroot-containers/rl.nightly.sep30.2026.sqsh`. Its digest has not been recorded in this draft; the filename alone does not specify the installed vLLM version.
- Host NeMo-RL: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/rohitkumarj/rem/unified-teacher-supervl3p5/nemo-rl-nv-main`.
- Container NeMo-RL: `/opt/nemo-rl`. Bridge, Megatron-LM, and Gym resolve recursively under its `3rdparty/` tree. Use the existing project mounts.
- Model: `super-vl-35-rlvr-v43-falcon-r3-20260905/hf`; architecture `NemotronH_Omni_Reasoning_V3`, RADIO vision, Nemotron-H hybrid/MoE language model. This checkpoint has `sound_config: null`; audio support is outside the verified recipe.
- Full RL placement: 32 nodes × 4 GPUs, with 16 generation nodes. A smaller full-production placement and the exact GPU SKU are not qualified by this document.
- Driver Python: `/opt/nemo_rl_venv/bin/python`.
- Megatron worker Python: `/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python`.
- vLLM worker Python: `/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python`.

## Required local changes

| Change | File / location | Evidence and limit |
|---|---|---|
| Decode percent-encoded `file://` image paths once | `nemo_rl/data/multimodal_utils.py` | 22 focused tests passed; failing row 6080 passed with 16 generations; five encoded URIs found in the 16,000-row dataset |
| RADIO final LayerNorm FP32 backport | `nemo_rl/models/generation/vllm/patches.py`; `scripts/patch_vllm_super_omni_radio_layernorm_0_29.py` | 53 unit checks plus TP4 norm/refit checks; this patch was already active in earlier failing TMPE runs |
| Native engine token/media capture and image-tool URL propagation | Patched Gym checkout and `examples/nemo_gym/supervl3p5` | Required for the current Gym rollout path; keep both directories available |
| Image Python dependencies and MCore helpers | Existing per-node setup script | Lens, media dependencies, worker venvs, and the runtime vLLM patch must be ready before workers start |

These changes do not establish that the remaining raw TMPE outliers are fixed. The recipe fixes and SA-V verifier are committed on the working branch. Gym commit `c004bce8` is published in `NVIDIA-NeMo/Gym` on branch `rohit/unified-teacher-supervl3p5`; `.gitmodules` selects that repository and branch. The image still needs the per-node HSG setup. A container digest, Super conversion round-trip, and held-out evaluation remain release tasks.

## HSG paths

See [hsg.env.sh](nemo-rl/scripts/hsg.env.sh) for the exact model, dataset, cache, Gym extras, runtime, and experiment paths. Dataset and media files stay read-only; results, crops, logs, and caches use writable output directories.

## Source checks behind the drafts

- The production YAML was composed from [the authored config chain](evidence/source-configs/examples/configs/recipes/vlm/super_vl_35_mixed_teachers_v2_test_batch4h.yaml), using [the checkout's config loader](evidence/config_loader.py). Paths and run identity were parameterized.
- [The step-110 resolved config](evidence/in-order-step110-config.yaml) records the completed setup. Its temporary runtime control token is redacted; it is evidence, not a launch config.
- Bridge commands were checked against `scripts/conversion/README.md`, `scripts/conversion/convert.sh`, and `examples/models/nemotron/nemotron_3_omni/hf_to_megatron_generate_nemotron_omni.py` in the pinned checkout.
