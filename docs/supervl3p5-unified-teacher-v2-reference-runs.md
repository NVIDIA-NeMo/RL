# SuperVL3.5 unified vision GRPO: V2 reference runs

Updated: 2026-10-21. This runbook records the **V2 single-controller** runs.

## The runs

All runs use `<V2_ENTRYPOINT>` and the same initial SuperVL3.5 HF checkpoint at `<MODEL_DIR>`. Runs 1–6 are complete or stopped; their checkpoints were subsequently deleted at the user's request. Run 7's two five-step checkpoints are retained. Status and metrics below were checked against W&B and the filesystem on 2026-10-21.

| # | V2 run | Sampler / staleness | R3 | Completed steps | W&B state / result | W&B |
| --- | --- | --- | --- | ---: | --- | --- |
| 1 | Unified teachers, R3 on | `ready_first` / 2 | On | 23 / 150 | `crashed`; runtime limit reached | [Run 1](https://forge.coreweave.com/wandb/nvidia/rohit-unified-teacher-supervl3p5/runs/super-vl-35-mixed-teachers-v2-ready-first-s2-r3-20261019) |
| 2 | Unified teachers, R3 off | `ready_first` / 2 | Off | 274 / 300 | `finished`; target not reached | [Run 2](https://forge.coreweave.com/wandb/nvidia/rohit-unified-teacher-supervl3p5/runs/super-vl-35-mixed-teachers-v2-ready-first-s2-20261002) |
| 3 | CapRL video | `ready_first` / 1 | Off | 50 / 50 | `finished` | [Run 3](https://forge.coreweave.com/wandb/nvidia/rohit-unified-teacher-supervl3p5/runs/supervl3p5-caprl-v2-ready-first-s1-20261019) |
| 4 | SA-V visibility | `ready_first` / 1 | Off | 10 / 50 | 10 iterations completed; W&B `crashed` | [Run 4](https://forge.coreweave.com/wandb/nvidia/rohit-unified-teacher-supervl3p5/runs/supervl3p5-sav-v2-ready-first-s1-nor3-20261019) |
| 5 | RAVEN grouping | `ready_first` / 1 | Off | 50 / 50 | `finished` | [Run 5](https://forge.coreweave.com/wandb/nvidia/rohit-unified-teacher-supervl3p5/runs/supervl3p5-raven-v2-ready-first-s1-nor3-20261019) |
| 6 | Static-image control | `ready_first` / 1 | Off | 50 / 50 | `finished` | [Run 6](https://forge.coreweave.com/wandb/nvidia/rohit-unified-teacher-supervl3p5/runs/supervl3p5-static-control-v2-ready-first-s1-nor3-20261019) |
| 7 | Visual coding, OpenCode, stream32 | `ready_first` / 1 | Off | 5 / 209 | `finished`; checkpoint and clean exit verified | [Run 7](https://forge.coreweave.com/wandb/nvidia/rohit-unified-teacher-supervl3p5/runs/supervl3p5-visual-coding-v2-rf-s1-stream32-20261021) |

Additional V2 references:

| Reference | Settings | Completed steps | W&B |
| --- | --- | ---: | --- |
| Unified teachers, in-order | `in_order`, lookahead 1, R3 off | 187; W&B `finished` | [In-order](https://forge.coreweave.com/wandb/nvidia/rohit-unified-teacher-supervl3p5/runs/super-vl-35-mixed-teachers-v2-test-batch4h-20261002) |
| Run 7 before stream32 | `ready_first`, staleness 1, streaming minimum 128, R3 off | 5; W&B `finished` | [Run 7 reference](https://forge.coreweave.com/wandb/nvidia/rohit-unified-teacher-supervl3p5/runs/supervl3p5-visual-coding-v2-ready-first-s1-nor3-20261020-r2) |

### Recipes

Set each recipe placeholder to its YAML in `<RECIPE_ROOT>`.

| Run | YAML | Source availability in this documentation commit |
| --- | --- | --- |
| 1 | `<RUN_1_YAML>` | Local recipe |
| 2 | `<RUN_2_YAML>` | Tracked |
| 3 | `<RUN_3_YAML>` | Tracked; local logging-tag edits remain separate |
| 4 | `<RUN_4_YAML>` | Tracked |
| 5 | `<RUN_5_YAML>` | Tracked |
| 6 | `<RUN_6_YAML>` | Local recipe |
| 7 | `<RUN_7_STREAM32_YAML>` | Local recipe; inherits the local OpenCode base YAML |
| In-order | `<INORDER_YAML>` | Tracked |
| Run 7 reference | `<OPENCODE_BASE_YAML>` | Local recipe |

This commit publishes the runbook and its documentation-index entry. It does not publish the local-only configs, runtime helpers, or OpenCode environment port. The commands below use the existing tested checkout and provisioned runtime; a fresh GitHub clone alone does not contain all Run 7 dependencies. Historical runs also used runtime YAMLs and command-line overrides. The reusable YAMLs do not imply that every earlier job used their current contents.

## Rerun a recipe

### Code and container

| Item | Tested value |
| --- | --- |
| GitHub branch | `NVIDIA-NeMo/RL:rohit/unified-teacher-supervl3p5` |
| Training source commit | `b8e429c3803ad6b355b8d2dad5a7dd9b95f4745e`, plus recorded local recipe/runtime files |
| Checkout | `<CODE_DIR>` |
| Container | `<CONTAINER_IMAGE>` |
| Gym framework | `c004bce8068eaa35690be8705d3e263be10581e8`, plus the custom environments described below |
| Megatron-Bridge | `1f8873bb00a8ddf3af811649f0a7efdb2363570c` |
| Megatron-Core | `6a3660905a2736b5670baed1ca5954372937918b` |
| Run 7 vLLM worker package | `0.29.0` |

Prepare `<RUNTIME_SETUP>` for your environment, then start one driver per run.

The tested runtime keeps the container's pinned driver, Megatron, vLLM, and Gym environments, applies the existing RADIO final-LayerNorm backport, and makes custom Gym environments available. These are prerequisites for the commands below. `--no-sync` preserves those installed environments.

### Fresh run

Run inside the container. Choose the YAML from the table and the exact data file from [Data files](#data-files). Use a new W&B ID and a new results directory.

```bash
set -euo pipefail
cd "<CODE_DIR>"

export MM_TRAINER_MODEL_PATH="<MODEL_DIR>"
export MM_TRAINER_DATA_PATH="<TRAINING_JSONL>"
export MM_TRAINER_WANDB_ID="supervl3p5-caprl-v2-fresh-$(date -u +%Y%m%dT%H%M%SZ)"
export MM_TRAINER_WANDB_NAME="$MM_TRAINER_WANDB_ID"
export MM_TRAINER_WANDB_PROJECT=rohit-unified-teacher-supervl3p5
export MM_TRAINER_WANDB_ENTITY=nvidia
export MM_TRAINER_RESULTS_DIR="<OUTPUT_DIR>"
export GYM_EXTRA_DIR="<CUSTOM_GYM_DIR>"
export CONFIG="<RECIPE_YAML>"

test -r "$MM_TRAINER_DATA_PATH"
test -r "<MODEL_CONFIG_JSON>"
test -r "$CONFIG"
test ! -e "$MM_TRAINER_RESULTS_DIR"
mkdir -p "$MM_TRAINER_RESULTS_DIR"

# Existing helper in the tested checkout; not included in this docs commit.
source "<ENV_SETUP_SCRIPT>"
export WANDB_RUN_ID="$MM_TRAINER_WANDB_ID"
export WANDB_NAME="$MM_TRAINER_WANDB_NAME"
export WANDB_RESUME=never

uv run --no-sync "<V2_ENTRYPOINT>" \
  --config "$CONFIG" \
  logger.wandb_enabled=true logger.wandb.resume=never \
  logger.wandb.project="$MM_TRAINER_WANDB_PROJECT" \
  logger.wandb.name="$MM_TRAINER_WANDB_NAME" \
  logger.wandb.id="$MM_TRAINER_WANDB_ID" \
  checkpointing.checkpoint_dir="<CHECKPOINT_DIR>" \
  token_capture.control_auth_token=null
```

Credentials come from the runtime environment. Forward the existing W&B credentials to the driver. Run 7 also requires an `OPENAI_API_KEY` accepted by its judge endpoint. The native W&B x-axis is **`nemo_rl/step`**.

The short UV command assumes `uv run --no-sync` already selects the prepared driver environment. If it does not, the validated explicit spelling is `uv run --no-sync --python "<DRIVER_PYTHON>" python "<V2_ENTRYPOINT>" ...`. Do not run a dependency sync as a substitute for preparing this container.

### Run 7 additions

Use the OpenCode dataset and the stream32 YAML from the tables, with a fresh results directory. The local port installer must be present. After sourcing the environment helper and before starting the driver:

```bash
: "${OPENAI_API_KEY:?Set the existing judge credential}"
export TORCHINDUCTOR_CACHE_DIR="<TORCHINDUCTOR_CACHE_DIR>"
export TRITON_CACHE_DIR="<TRITON_CACHE_DIR>"
uv run --no-sync --python "<DRIVER_PYTHON>" python \
  "<OPENCODE_INSTALLER>"
```

The installer ports the published OpenCode environment into the existing Gym setup. Keep the shared Gym-venv path in the Run 7 YAML, `<SHARED_GYM_VENV_DIR>`; an environment that was not shared previously caused missing `cv2` imports in Gym actors.

The stream32 delta is:

```yaml
defaults: <OPENCODE_BASE_YAML>

async_rl:
  min_groups_for_streaming_train: 32
```

This is the minimum number of prompt groups needed to begin streamed training. It does not reduce the full optimizer update of 128 prompts × 16 generations = 2,048 sequences. It is not a vLLM stream count.

### Continue an existing run

Reuse the same results/checkpoint directory and W&B ID, set `WANDB_RESUME=allow` and `logger.wandb.resume=allow`, and omit the fresh-directory check. First verify that a complete checkpoint exists and that no driver for that run is active. Keep the total step target and restored sampler/optimizer state consistent across restarts.

Runs 1–6 cannot currently resume from their recorded output directories because their checkpoints were deleted. Run 7's step-5 checkpoints are retained at `<RUN_7_CHECKPOINT>` and `<RUN_7_REFERENCE_CHECKPOINT>`. Its YAML target of 209 steps is a ceiling; five updates completed.

## What each run tests

### Runs 1 and 2: unified teachers, R3 on/off

The 16,000-row blend contains 2,000 image-tools rows, 4,000 static-image rows, and 10,000 video rows. The static pool has two 2,000-row parts. The video pool contains 5,376 CapRL QA rows and 4,624 SA-V box/point/referring rows. Image-tools agents can make multiple model/tool calls; this is not a uniformly single-turn dataset.

R3 records the MoE experts chosen during generation and replays those choices in the policy forward pass. Run 1 enables `policy.router_replay.enabled` and `policy.generation.vllm_kwargs.enable_return_routed_experts`; Run 2 disables them. Both use ready-first with staleness 2.

Run 1 includes the cherry-picked media-position fixes from PR #4763, whereas the historical Run 2 predates them. They also have different completed step counts and launch histories. Their reward difference is therefore not a controlled estimate of the effect of R3 alone. The historical unified-teacher data path did not enable the later CapRL-specific native-video conversion; do not infer tubelet parity for those older runs from Run 3's checks.

### Run 3: CapRL native video

One row contains 64 cached frames from one video. The native-video path combines these into one lossless frame manifest. With temporal patch size 2, it forms 32 tubelets, rather than treating the frames as 64 independent images. The vision encoder and projection stay frozen.

The focused real-row comparison matched the reference preprocessing: 8,192 visual tokens, 9,264 total prompt tokens, exactly equal prompt IDs and pixel/image-size tensors, one outbound `input_video`, and no outbound `input_image`. A real-weight generation check also passed without OOM. The production run then completed 50 updates.

The fix registered the actual Super processor class names in the native-video processor list. Synthetic tests alone had missed those classes; the regression included the real processor path.

### Run 4: SA-V visibility

This dataset isolates tracking answers that mark a target `"visible": false` without providing a location. Such answers must receive the intended verifier score. Each batch has 48 referring, 40 box, and 40 point rows. Box/point rows and 652 referring rows include hidden targets.

The V2 setup reuses the SA-V verifier, configs, requirements, and tests from the reference Gym revision `22f94c44116ac263673ea3aafb5f80b558800e25`. The verifier SHA-256 is `ed0bba03fb13029178cf4634f7bd5bc9facf1e51f2488124e0f135b462d71740`; all 25 reference test functions passed in the actual Gym actor environment.

Rows contain 2–16 separate frame images. Preserve this representation: the SA-V recipe does not enable the CapRL video-only conversion. **Ten iterations completed.** The 50-step recipe target was not reached; W&B currently reports `crashed`. This document does not label that run a successful 50-step completion.

### Run 5: RAVEN grouping

RAVEN rows share the same prompt text but refer to different puzzle images. GRPO advantages must use the siblings of each dispatched puzzle row. Grouping by prompt token IDs can merge unrelated images whose token sequences have the same shape.

The V2 fix derives group IDs from canonical rollout sample IDs. It preserves each puzzle's group when ready-first results interleave. The regression covers normalized/unnormalized rewards, interleaved rows, and a masked sibling; the 90-test controller/advantage suite passed. The production run completed 50 updates. Mean logged reward rose from 0.5206 over its first five updates to 0.7961 over its last five, with repeated training data.

### Run 6: static-image control

Each row has one static image and a distinct question text. RAVEN is excluded. Agent counts are 2,154 math-with-judge, 549 string-match, and 113 MCQA. The 2,816 rows reference 2,645 distinct image paths.

This measures general training behavior without the CapRL video conversion, SA-V visibility rule, or RAVEN repeated-prompt grouping issue. The production run completed 50 updates. Vision encoder/projection, sound encoder/projection, and MoE router are frozen, as in the single-source recipes.

### Run 7: visual coding with OpenCode

The 26,801-row dataset contains one base64 image per question, with `question`, `expected_answer`, `dataset`, and `super_pass_rate`. Selection keeps rows with pass rate 0.25–0.75; mean source pass rate is 0.555. Large sources include line tracking, mazes, shadow/same/different matching, metro maps, and handshake puzzles. These are images, not video inputs.

OpenCode can write and run Python, inspect images, and make further model calls. The environment uses the published Gym fork at `DanialTaheri/Gym@9011e8aee8c8693de1da9d8c6d6a5e7f2edd322e`, ported into the current Gym framework. It is not a wholesale replacement of the Gym submodule, nor a claim to reproduce the dataset author's unpublished edits.

| Setting | Value |
| --- | --- |
| Agent | `opencode_vision_agent`; OpenCode 1.17.11 |
| Agent concurrency | 512 |
| Session placement | `<SESSION_PLACEMENT>` |
| Context / per-call output cap | 65,536 / 16,384 tokens |
| Session timeout | 1,800 seconds; timeout scores 0 |
| Sampling | Temperature 1.0, top-p 1.0 |
| Compaction | Auto/prune both off |
| Thinking history | `truncate_history_thinking: false`; request-level thinking enabled |
| Local tokenizer template kwargs | `null`, following the reference convention |
| Judge | `us/azure/openai/gpt-4o-mini`, temperature 0 |
| Judge input | Question, gold answer, extracted answer, and text tail; no image |
| Reward | Binary answer-equivalence score |

`grpo.max_rollout_turns: 1` counts one outer Gym interaction. OpenCode performs multiple internal model/tool turns inside it; the stream32 run averaged about 8.8–9.8 internal turns per sequence. Do not interpret the outer limit as single-turn generation.

The local port removes an unsupported assistant reasoning-display field before strict Gym validation, preserves captured tokens/logprobs, and returns the actual images used by model requests. It avoids injecting the seed image a second time into the direct training response. Strict feature/token alignment checks remain active. These changes and the shared Gym-venv path are prerequisites for the validated run, separate from the stream32 optimization.

Both the initial reference and the stream32 version completed five full updates. Stream32 retains R3 off, staleness 1, a full 2,048-sequence update, optimizer offloading, synchronous checkpoint saves, and multimodal deduplication. Its focused validation passed 95 CPU tests and two nonzero-gradient GPU sequence-loss parity cases. All five production updates had finite nonzero gradients and TMPE at most 1.05.

A separate real-weight vLLM test compared `async_scheduling: true` and `false`. Both passed generation checks. Enabling it showed no speed gain in the small focused sample, so it remains **false** in the recommended production recipe. The prepared async-scheduling variant was not submitted as a production run.

## Details

### Common training settings

| Setting | Value |
| --- | --- |
| Runtime resources | `<RUNTIME_RESOURCES>` |
| Policy parallelism | `<POLICY_PARALLELISM>` |
| Generation parallelism | `<GENERATION_PARALLELISM>` |
| Batch | 128 prompts × 16 generations = 2,048 sequences/update |
| Context / ordinary output cap | 65,536 / 32,768 tokens; OpenCode per-call cap 16,384 |
| GRPO | Leave-one-out baseline; reward normalization; sequence-level loss |
| Ratio clipping | 0.2 lower / 0.28 upper |
| Learning rate | 3e-6; 10-step warmup; cosine decay over 60 steps; minimum 5e-7 |
| Sequence-logprob error threshold | 2.0 |
| Vision encoder / projection | Frozen |
| Logprob-phase optimizer offload | On; optimizer CPU offload off |
| Multimodal deduplication | On |
| LM head | FP32 enabled on policy and generation |
| Checkpoint saves | Synchronous; optimizer and data-plane state included |
| Diagnostics | Optional TMPE dump/stop disabled |

| Recipe family | Buffer / in-flight / streaming minimum, prompt groups | Target / epochs | Checkpoint period |
| --- | --- | --- | --- |
| Unified teachers, ready-first | 384 / 128 / 128 | R3 off 300, R3 on 150 / three epochs | 50 |
| Runs 3–4 | 256 / 128 / 128 | 50 / three epochs | 10 |
| Runs 5–6 | 256 / 128 / 128 | 50 / three epochs | 20 |
| Run 7 reference | 256 / 128 / 128 | 209 / one epoch | 10 |
| Run 7 stream32 | 256 / 128 / 32 | 209 / one epoch | 10 |

The 50-step single-source recipes can revisit their finite datasets. They are not single-pass evaluation runs. Historical unified-teacher continuations changed checkpointing from 10 to 50 steps. Staleness is a sampler-admission setting, not a guarantee that every measured sample-version lag equals that value.

Set `checkpoint_must_save_by: <CHECKPOINT_SAVE_DEADLINE>` for your runtime. This deadline is relative to the trainer's timer; include setup and final-save time when choosing it.

### Data files

All runs start from:

`<MODEL_DIR>`

For Run 7, 64 weight shards and 16 model/tokenizer/processor files were checked byte-for-byte against the reference checkpoint. The 64 weight-shard hashes matched. This is identity evidence for that comparison, not a guarantee that every future runtime uses unchanged files.

Replace each training-file placeholder with its dataset. The hashes identify the files used for the recorded runs.

| Runs | Training file | Rows | SHA-256 |
| --- | --- | ---: | --- |
| 1–2 and in-order | `<UNIFIED_TEACHERS_JSONL>` | 16,000 | `b1f23c69978d496de25aae7d47fae081c18eb773afb28c1db2fb6979c6be2dce` |
| 3 | `<CAPRL_JSONL>` | 2,816 | `a6a6b04cfa012f1cd660c7097938a1cada986cd3dae001fc81dedadb60275ec1` |
| 4 | `<SAV_JSONL>` | 3,328 | `46565ca53ffbc3e5c83727981a3623a5f940d6fa411392d278d9957c28b19c47` |
| 5 | `<RAVEN_JSONL>` | 2,560 | `bf86c00e04632fdd2af296da495d05418d048b314d833e28b91115dacb111c47` |
| 6 | `<STATIC_CONTROL_JSONL>` | 2,816 | `50e5db5bb3572df2ebb1d4d541bfc324ff83a525a05fab55625504b9477bdaa3` |
| 7, both variants | `<OPENCODE_JSONL>` | 26,801 | `fd955f49c30022a35acc45f876c6964f4c4d710686a25568d2a1b073c900d0de` |

Set `MM_TRAINER_DATA_PATH` explicitly. Provide the media through `<MEDIA_ROOT>`. Run 7 embeds its initial image in the JSONL. These recipes have no held-out validation dataset.

### Latest training results

Values are the latest native W&B summary values, checked on 2026-10-21. The reward column is **`train/reward`**, which can reflect training masks. Raw rollout reward is shown separately when `train/total_reward/mean` was logged. A missing raw-reward metric is not filled from the masked metric. TMPE is also affected by the configured masking.

| V2 run | Steps | Logged training reward | Raw rollout reward | TMPE | Masked for sequence-logprob error, latest / 2,048 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1: unified R3 on | 23 | 0.6413 | — | 1.0088 | 8 |
| 2: unified R3 off | 274 | 0.7957 | — | 1.0196 | 8 |
| 3: CapRL | 50 | 0.7769 | — | 1.0158 | 0 |
| 4: SA-V | 10 | 0.6707 | — | 1.0095 | 1 |
| 5: RAVEN | 50 | 0.8163 | — | 1.0362 | 72 |
| 6: static control | 50 | 0.6899 | — | 1.0140 | 0 |
| 7: OpenCode stream32 | 5 | 0.6721 | 0.5850 | 1.0392 | 7 |
| In-order reference | 187 | 0.6804 | — | 1.0179 | 10 |
| Run 7 reference | 5 | 0.6552 | 0.5781 | 1.0390 | 6 |

These are training results on different datasets and different numbers of updates. They do not rank dataset difficulty or establish held-out accuracy, convergence parity, or an isolated R3 effect. In particular, low TMPE after masking does not show that all outliers are fixed.

Run 7 stream32's complete native updates:

| Step | Raw reward | TMPE | Logprob-error masks | Native total step time, minutes |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.5879 | 1.0363 | 11 | 55.36 |
| 2 | 0.5205 | 1.0366 | 7 | 38.70 |
| 3 | 0.5454 | 1.0369 | 10 | 42.45 |
| 4 | 0.5713 | 1.0370 | 10 | 42.49 |
| 5 | 0.5850 | 1.0392 | 7 | 42.89 |

Every update had a 2,048-entry reward histogram and finite nonzero gradients. Shared warm steps 2–4 averaged 2,472.72 seconds, with aggregate throughput 105.92 valid tokens/second/GPU. Throughput is `sum(valid_tokens) / (num_gpus × sum(total_step_time))`; do not average per-step rates or compare unrelated reward numerators. This short window is not a convergence test.

### Artifact index and checkpoint availability

Set `<ARTIFACT_ROOT>` for your environment. Replace each output and checkpoint placeholder with the corresponding location.

| Run | Output | Checkpoint availability |
| --- | --- | --- |
| 1 | `<RUN_1_OUTPUT_DIR>` | None; no complete policy checkpoint was found before cleanup |
| 2 | `<RUN_2_OUTPUT_DIR>` | Deleted |
| 3 | `<RUN_3_OUTPUT_DIR>` | Deleted; step 50 was verified before cleanup |
| 4 | `<RUN_4_OUTPUT_DIR>` | Deleted; step 10 was verified before cleanup |
| 5 | `<RUN_5_OUTPUT_DIR>` | Deleted; step 50 was verified before cleanup |
| 6 | `<RUN_6_OUTPUT_DIR>` | Deleted; step 50 was verified before cleanup |
| 7 | `<RUN_7_OUTPUT_DIR>` | Retained: step 5 at `<RUN_7_CHECKPOINT>` |
| In-order | `<INORDER_OUTPUT_DIR>` | Deleted |
| Run 7 reference | `<RUN_7_REFERENCE_OUTPUT_DIR>` | Retained: step 5 at `<RUN_7_REFERENCE_CHECKPOINT>` |

Run 7 stream32's driver exited with code 0. Its checkpoint verifier checked the step/version metadata, optimizer entries, controller/storage metadata, rollout-recovery payload hash, and all referenced byte ranges across 64 DCP shards. This checks checkpoint completeness; it is not a full reload/resume test.

Useful records for Run 7 stream32:

- `<SUBMITTED_PAYLOAD>` and `<UV_LAUNCH_SCRIPT>`: exact command and UV script used.
- `<DRIVER_LOG>`: production driver output.
- `<RUNTIME_CONFIG_CHECKS>`: 45 checked runtime settings.
- `<NATIVE_UPDATE_CHECKS>`: five complete native W&B updates and their checks.
- `<CHECKPOINT_VERIFICATION>`: retained checkpoint completeness checks.
- `<DRIVER_EXIT_RECORD>`: final driver return code.
- `<OPTIMIZATION_AUDIT>`: complete optimization/validation result.
- `<SCHEDULER_COMPARISON>`: focused async-scheduling A/B result; no production variant was launched.
- `<OPTIMIZATION_REPORT>`: chronological optimization report.

Exact launch receipts, runtime configs, validation records, and retained logs remain under the corresponding `<RUN_OUTPUT_DIR>`, even when checkpoints are absent.
