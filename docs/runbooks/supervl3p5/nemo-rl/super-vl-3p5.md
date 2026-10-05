# SuperVL3.5: NeMo-RL production V2

Use [supervl3p5-v2-production.yaml](configs/supervl3p5-v2-production.yaml) and [run_v2.sh](scripts/run_v2.sh) inside the provisioned container. This is the mixed-teacher recipe with R3 disabled, sequence-error masking at 2.0, and checkpointing every 10 steps.

## Run

Assume this bundle is at `/opt/nemo-rl/docs/runbooks/supervl3p5`. On HSG:

```bash
export BOOK=/opt/nemo-rl/docs/runbooks/supervl3p5
source "$BOOK/nemo-rl/scripts/hsg.env.sh"
export MM_TRAINER_WANDB_ID=supervl3p5-v2-in-order-inspection
export MM_TRAINER_RESULTS_DIR=$HSG_EXPERIMENTS/$MM_TRAINER_WANDB_ID
source "$BOOK/nemo-rl/scripts/env.sh"
bash "$BOOK/nemo-rl/scripts/preflight.sh"
bash "$BOOK/nemo-rl/scripts/run_v2.sh"
```

- Choose a new W&B ID and output directory for a new experiment. Provide credentials through the environment or existing login.
- The UV command is `uv run --no-sync --python /opt/nemo_rl_venv/bin/python python examples/run_grpo_single_controller.py --config CONFIG`. Additional Hydra overrides pass through the script.
- The launcher does not allocate nodes or start Ray. Use an existing 32-node allocation and its external Ray cluster. Keep the existing container mounts, including the recursively mounted NeMo-RL tree.
- On a fresh allocation, provision each node before Ray workers start. The existing HSG setup command is `bash "$HSG_RUNTIME/run_supervl_mixed_teachers_nv_main.sh" --setup`; it prepares Lens, MCore helpers, media dependencies, worker venvs, and the vLLM patch. The standalone UV launcher assumes that setup has finished.
- Source the runtime environment on all nodes before Ray starts. The driver must attach to the 32-node cluster rather than silently start a local one.
- The observed batch runs used 32 nodes × 4 GPUs, partition `batch`, account `nemotron_sw_post`, and a four-hour limit. The YAML training deadline is 3h15m; it leaves time for setup and the final save. Reuse the current allocation for retries while time remains.

Check the external cluster from its head container:

```bash
uv run --no-sync --python "$DRIVER_PYTHON" python -c \
  'import ray; ray.init(address="auto"); nodes=[n for n in ray.nodes() if n["Alive"]]; print(len(nodes), ray.cluster_resources()); assert len(nodes)==32; assert ray.cluster_resources().get("GPU",0)==128'
```

For another cluster, replace the path exports in `hsg.env.sh`. Keep the pinned checkout, included SA-V Gym verifier, image dependencies, and worker environments listed in [environment.md](../environment.md). A plain stock container plus upstream Git SHA has not been qualified as an equivalent setup.

## Production settings

| Setting | Value |
|---|---|
| Entry point | `examples/run_grpo_single_controller.py` (V2); `grpo.async_grpo: null` |
| Placement | 32 × 4 GPUs, segment size 8; separate generation fleet 16 × 4 |
| Prompt groups / samples | 128 × 16 = training global batch 2048; microbatch 1 |
| Duration | 125 steps, one epoch over 16,000 rows; no shuffle |
| Response / total context | 32,768 / 65,536 tokens |
| Policy | TP2 / PP1 / CP2 / EP16 / ETP1; sequence parallel; full recompute |
| Generation | vLLM TP4 / PP1 / EP4; GPU utilization 0.8; max sequences 128 |
| vLLM execution | Async engine enabled; internal `async_scheduling: false`; prefix cache disabled |
| Logprobs / numeric settings | Raw logprobs; FP32 LM head and Mamba SSM cache; patched RADIO final norm |
| R3 | `router_replay.enabled: false`; routed-expert return disabled |
| Filtering | `seq_logprob_error_threshold: 2.0`; optional TMPE diagnostic dump/stop disabled |
| Memory / save | Logprob optimizer offload enabled; optimizer CPU offload disabled; `async_save: false` |
| Packing | 65,536-token microbatches; round to 64; modified first-fit decreasing; fused loss |
| Frozen components | Vision, vision projection, sound modules, and MoE routers |
| Video | 64 frames, temporal patch size 2, target patches 1024 |
| In-order async | Lookahead 1; inflight 128; buffer 256; streaming minimum 128 groups |
| Data plane / capture | Transfer Queue `simple`, 64 storage units; token capture enabled |
| Checkpoints | Every 10 steps; keep latest two; save optimizer and data plane |

- Effective CP is **2**, even though an inherited source filename includes `cp1`.
- `deduplicate_multimodal_data: true` is configured. Its memory benefit for every Gym payload path has not been established.
- Validation is disabled in this recipe. Training reward alone does not establish held-out quality.

## Ready-first and reduced checks

Ready-first uses [supervl3p5-v2-ready-first.yaml](configs/supervl3p5-v2-ready-first.yaml), with an independent output directory and W&B ID:

```bash
export MM_TRAINER_WANDB_ID=supervl3p5-v2-ready-first-s2-inspection
export MM_TRAINER_WANDB_NAME=$MM_TRAINER_WANDB_ID
export MM_TRAINER_RESULTS_DIR=$HSG_EXPERIMENTS/$MM_TRAINER_WANDB_ID
export SUPER_CONFIG=$BOOK/nemo-rl/configs/supervl3p5-v2-ready-first.yaml
bash "$BOOK/nemo-rl/scripts/run_v2.sh"
```

- `max_staleness_versions: 2`, inflight 128, buffer 384, streaming minimum 128 groups. Importance sampling correction stays enabled; `force_on_policy_ratio` stays false.
- `_override_: true` replaces the async block and removes the inherited in-order sampler fields.
- The staleness setting limits admission/dispatch lead. It does not guarantee that a late completed rollout is rejected solely for its final age.
- Inherited replacement settings do not drive ready-first's untargeted groups; the related warning is expected.

For bring-up, choose [supervl3p5-v2-smoke.yaml](configs/supervl3p5-v2-smoke.yaml) with a new ID/output directory. It uses 128 × 2 samples, GBS 256, 512 output tokens, and two steps. It retains the 32-node placement and saves each smoke step. Passing this check does not validate full production memory or long responses. Unset `SUPER_CONFIG` to return to production.

## Data and rewards

- Dataset: `training.jsonl`, 16,000 rows. Preflight checks referenced local paths and decodes `file://` paths exactly once.
- Each row has `agent_ref` and `responses_create_params`; task-specific fields carry answers or labels for the selected Gym verifier. Keep these fields when copying rows.
- Routes include GUI coordinates, math, MCQA, string matching, SA-V tracking, and image tools. The current math route has `should_use_judge: false`; this YAML does not launch a separate set of teacher LMs.
- Image rows contain `input_image` entries. Some video tasks supply 64 cached image frames with `_is_video_frame: true` and `_video_source` pointing to the original MP4. Retain the complete frame list and its metadata. Do not convert these rows to a different media form without checking preprocessing parity.
- Bare paths and percent-encoded file URIs are supported by the local URI fix. Source data/media stay read-only. The image-tools `crop_dir` is under the writable experiment directory.

These excerpts show the actual row shape; they omit prompts, labels, and most frames and are not standalone training rows:

```yaml
# Row 0: SA-V box tracking
agent_ref: {type: responses_api_agents, name: sav_box_tracks_agent}
verifier: sav_tracks
responses_create_params:
  input:
    - role: user
      type: message
      content:
        - {type: input_text, text: "<tracking prompt>"}
        - {type: input_image, image_url: "/lustre/.../000055.jpg", detail: auto}
# Preserve task, objects, coordinate_space, and the remaining frames.

# Row 8: video QA carried as cached frames
agent_ref: {type: responses_api_agents, name: mcqa_simple_agent}
expected_answer: C
responses_create_params:
  input:
    - role: user
      type: message
      content:
        - type: input_image
          image_url: "/lustre/.../frame_0000.png"
          _is_video_frame: true
          _video_source: "/lustre/.../ytb_fhQKeAeIpdA.mp4"
# Preserve all 64 frames, question, options, and grading fields.
```

## Checkpoints, resume, and storage

- Relaunch with the same YAML, output directory, and W&B ID to resume an unfinished experiment from its latest saved checkpoint. Confirm the startup log restores weights, optimizer, step, and data-plane state before accepting new training metrics.
- A stop at the allocation deadline can save a non-multiple-of-10 step. Regular saves remain every 10 steps.
- `metric_name: null` retains recent checkpoints; it does not retain the best historical reward automatically.
- Full resume checkpoints were roughly 1.60–1.72 TiB each. Budget space for retained checkpoints, one new save, and logs/crops/caches before restart. A quota error can stop both training and setup.
- Cleanup now leaves only `step_110` in each of the two historical experiments. Ready-first's completed step 125 and its global reward peak at step 82 are not available as checkpoints.
- Keep the completed ready-first run as history. Starting from its retained step 110 under the old W&B ID would rewind the recorded history; use a separate experiment for a new branch of training. Export from retained step 110 follows [the Bridge runbook's RL export section](../megatron-bridge/README.md#export-a-retained-rl-checkpoint).

## Recorded outcomes and monitoring

| Run | Last training step | Retained checkpoint | Reward at retained step / masked TMPE |
|---|---:|---:|---|
| [In-order](https://forge.coreweave.com/wandb/nvidia/rohit-unified-teacher-supervl3p5/runs/super-vl-35-mixed-teachers-v2-test-batch4h-20261002) | 111; disk quota stopped progress | 110 | 0.74926 / 1.01951 |
| [Ready-first, staleness 2](https://forge.coreweave.com/wandb/nvidia/rohit-unified-teacher-supervl3p5/runs/super-vl-35-mixed-teachers-v2-ready-first-s2-20261002) | 125; completed | 110 | 0.68100 / approximately 1.02 |

- Ready-first's final step 125 had reward 0.66813, masked TMPE 1.01616, and 13 masked sequences out of 2048. Its highest observed training reward was 0.74447 at step 82; no step-82 checkpoint was retained.
- These are individual training-step rewards, not evaluation scores or smoothed comparisons. Both runs are stopped; there are no follow-up jobs left.
- Monitor reward over multiple steps, masked TMPE, raw sequence-error outliers, rejected count/fraction, response lengths, rollout/train/save time, and memory. Filtered TMPE around 1.015–1.02 is useful evidence about accepted sequences; it does not show that raw mismatches disappeared.
- Fixed-weight native vLLM decode/fresh-prefill checks covered about 238,000 tokens (TMPE 1.0128); synthetic assembly checks covered 256 rollouts. Those checks did not reproduce severe mismatches. Live packing, media expansion, and changing-weight rollout paths remain separate questions.
- Recent steps took minutes: roughly 4–9 for ready-first and 5–18 for in-order, with variable response/tool lengths. During startup or save, check worker and checkpoint logs before treating a quiet driver as stalled.

## Common failures

| Symptom | First check |
|---|---|
| Missing image with `%20`, `%28`, etc. | Confirm the URI patch is loaded and the once-decoded path exists on the worker; preserve bare paths |
| OOM during logprobs or save | Verify full effective YAML, synchronous save, logprob optimizer offload, packing and generation limits; reduced samples can help isolate it |
| High raw TMPE with low filtered TMPE | Count rejected sequences and inspect lengths/media/assembly; masking is containment, not a proved root-cause fix |
| Ray actors pending indefinitely | Confirm all 32 nodes, 128 GPUs, placement resources, and matching mounts/worker Python paths |
| Restore/setup failure or failed checkpoint write | Check filesystem quota and free space before submitting another retry |

Draft structure follows the [Nano Omni RL guide](https://docs.nvidia.com/nemo/rl/nightly/guides/models/nemotron/nemotron-3-nano-omni.html). Super settings and outcomes above come from the active recipe and its run evidence.
