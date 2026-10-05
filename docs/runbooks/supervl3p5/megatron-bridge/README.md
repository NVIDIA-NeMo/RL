# SuperVL3.5: Megatron-Bridge

This draft covers the existing SuperVL3.5 checkpoint, HF/Megatron conversion commands, inference checks, and SFT qualification work. Use the pinned recursive checkout and container in [environment.md](../environment.md).

| Workflow | Current status |
|---|---|
| Model loading through Bridge in the production RL policy | Exercised by the 32-node RL runs |
| Persistent HF → Bridge → HF conversion and weight round-trip | Commands checked against source; Super execution pending |
| Standalone deterministic text/image/video inference below | Commands checked against source; Super execution pending |
| Export of the retained NeMo-RL checkpoint | Converter interface checked; export and post-export inference pending |
| SuperVL3.5 SFT / LoRA | No qualified Super recipe in this checkout; requires a model-specific recipe |

## Prepare the environment

Work inside a provisioned GPU container with writable output space. The checkpoint's architecture is `NemotronH_Omni_Reasoning_V3`; its language model combines Mamba, attention, and MoE blocks, with a RADIO vision encoder. Its sound configuration is null.

```bash
export BOOK=/opt/nemo-rl/docs/runbooks/supervl3p5
source "$BOOK/nemo-rl/scripts/hsg.env.sh"
export BRIDGE_DIR=$RL_DIR/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
export MEGATRON_WORKER_PYTHON=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
export HF_MODEL=$MM_TRAINER_MODEL_PATH
export WORKSPACE=/path/to/writable/supervl3p5-conversion
export PYTHONPATH=$RL_DIR:$BRIDGE_DIR/src:$BRIDGE_DIR/3rdparty/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}
export VIRTUAL_ENV=${MEGATRON_WORKER_PYTHON%/bin/python}
export CUDA_DEVICE_MAX_CONNECTIONS=1
mkdir -p "$WORKSPACE"
cd "$BRIDGE_DIR"
uv run --no-project --no-sync --python "$MEGATRON_WORKER_PYTHON" python -c \
  'import importlib.metadata, megatron.bridge, megatron.core; print(megatron.bridge.__file__); print(megatron.core.__file__); print("nemo-run", importlib.metadata.version("nemo-run"))'
```

- Set `WORKSPACE` to a real output directory. Keep the source HF model unchanged.
- The conversion launcher requires NeMo Run; its documented launcher dependency is `nemo-run==0.10.0`. Check that the selected Bridge environment contains it before conversion. This draft does not change image dependencies.
- `VIRTUAL_ENV` selects the populated environment for `convert.sh`, avoiding an unrelated project environment. Keep the project's existing mounts and nested source paths.
- The four-GPU TP4/EP4 commands below are a proposed local qualification topology. Confirm memory capacity before use; full Super conversion on this topology has not been measured. A one-process CPU import can require hundreds of GiB of RAM.

## HF import, export, and weight round-trip

Use Bridge's actual `scripts/conversion/convert.sh` interface. It starts distributed workers itself.

```bash
./scripts/conversion/convert.sh import \
  --executor local --device gpu --gpus-per-node 4 \
  --hf-model "$HF_MODEL" --megatron-path "$WORKSPACE/bridge-import" \
  --tp 4 --pp 1 --ep 4 --etp 1 --trust-remote-code

# Use the actual iteration directory produced by import.
export BRIDGE_ITER=$WORKSPACE/bridge-import/iter_0000000
./scripts/conversion/convert.sh export \
  --executor local --device gpu --gpus-per-node 4 \
  --hf-model "$HF_MODEL" --megatron-path "$BRIDGE_ITER" \
  --hf-path "$WORKSPACE/hf-export" \
  --tp 4 --pp 1 --ep 4 --etp 1 --trust-remote-code

./scripts/conversion/convert.sh roundtrip \
  --executor local --device gpu --gpus-per-node 4 \
  --hf-model "$HF_MODEL" --tp 4 --pp 1 --ep 4 --etp 1 \
  --trust-remote-code
```

- Keep strict tensor checks first. The Nano example's expected missing-tensor list has not been verified for Super. If strict export fails, classify each missing tensor before deciding whether `--not-strict` is safe.
- Round-trip compares weights in memory; it does not test the checkpoint files just written. Persistent import/export also needs an exported-HF load and inference check.
- Record command, source revisions, topology, conversion logs, missing/unexpected keys, tensor comparisons, and output size. No Super tolerance or export-success claim is established yet.
- Bridge exports expect compatible provider metadata, normally `run_config.yaml`. A NeMo-RL `step_110/config.yaml` is a different format. Use the RL converter below for a training checkpoint.

## Deterministic inference checks

The available helper is `examples/models/nemotron/nemotron_3_omni/hf_to_megatron_generate_nemotron_omni.py`. It accepts a local HF model path and uses AutoBridge. Test each modality separately:

```bash
export INFER=examples/models/nemotron/nemotron_3_omni/hf_to_megatron_generate_nemotron_omni.py
uv run --no-project --no-sync --python "$MEGATRON_WORKER_PYTHON" python -m torch.distributed.run \
  --nproc_per_node=4 "$INFER" --hf_model_path "$HF_MODEL" \
  --tp 4 --pp 1 --ep 4 --etp 1 --system_prompt /no_think \
  --prompt "What is 2 + 2?" --max_new_tokens 32

export IMAGE=/path/to/readable/test-image.jpg
uv run --no-project --no-sync --python "$MEGATRON_WORKER_PYTHON" python -m torch.distributed.run \
  --nproc_per_node=4 "$INFER" --hf_model_path "$HF_MODEL" \
  --tp 4 --pp 1 --ep 4 --etp 1 --system_prompt /no_think \
  --image_path "$IMAGE" --prompt "Describe the image." --max_new_tokens 64

export VIDEO=/path/to/readable/test-video.mp4
uv run --no-project --no-sync --python "$MEGATRON_WORKER_PYTHON" python -m torch.distributed.run \
  --nproc_per_node=4 "$INFER" --hf_model_path "$HF_MODEL" \
  --tp 4 --pp 1 --ep 4 --etp 1 --system_prompt /no_think \
  --video_path "$VIDEO" --prompt "Describe the video." --max_new_tokens 64
```

- These use greedy generation; they are prepared checks, not recorded Super results. The helper recomputes prefixes and is unsuitable as a throughput benchmark.
- The helper currently hardcodes **8 video frames at 1 FPS**, temporal patch size 2. RL uses **64 frames**. Setting the RL `NUM_FRAMES` export does not change those helper constants. Qualify the same frame indices, prompt tokens, and vision preprocessing before using this helper for policy/rollout parity.
- The helper selects temporal vision settings by modality. Passing `--image_path` and passing `--video_path` exercise different paths.
- Add `--megatron_model_path "$BRIDGE_ITER"` to test a compatible imported Bridge checkpoint. Compare generated token IDs and teacher-forced logprobs against the original HF model, then against the exported HF model with identical inputs.
- Do not add an audio success claim: this Super checkpoint has no sound configuration.

## Export a retained RL checkpoint

The appropriate existing NeMo-RL entry point is `examples/converters/convert_megatron_to_hf.py`. Export still needs execution and reload checks for this Super checkpoint.

```bash
export RL_CHECKPOINT=$HSG_EXPERIMENTS/super-vl-35-mixed-teachers-v2-test-batch4h-20261002/checkpoints/step_110
export RL_WEIGHTS=$RL_CHECKPOINT/policy/weights/iter_0000000
test -f "$RL_CHECKPOINT/config.yaml"
test -d "$RL_WEIGHTS"
cd "$RL_DIR"
# The converter reads hf_overrides; the training recipe uses hf_config_overrides.
# Write only converter metadata, keeping the effective HF overrides.
uv run --no-sync --python "$MEGATRON_WORKER_PYTHON" python - <<'PY'
import os
from pathlib import Path
import yaml
source = yaml.safe_load((Path(os.environ["RL_CHECKPOINT"]) / "config.yaml").read_text())["policy"]
policy = {"model_name": source["model_name"], "tokenizer": source["tokenizer"], "hf_overrides": source.get("hf_config_overrides", source.get("hf_overrides", {}))}
(Path(os.environ["WORKSPACE"]) / "rl-export-config.yaml").write_text(yaml.safe_dump({"policy": policy}))
PY
uv run --no-sync --python "$MEGATRON_WORKER_PYTHON" python \
  examples/converters/convert_megatron_to_hf.py \
  --config "$WORKSPACE/rl-export-config.yaml" --hf-model-name "$HF_MODEL" \
  --megatron-ckpt-path "$RL_WEIGHTS" --hf-ckpt-path "$WORKSPACE/rl-step110-hf"
```

- Both retained step-110 checkpoints store weights in `iter_0000000`; the training step lives in separate state. Confirm the actual directory before export. This converter loads a large model; qualify host memory capacity before running it.
- The source HF model supplies configuration/tokenizer context. Keep its original processor, custom model code, and chat template available when validating the exported model.
- Start with strict export. The RL flag is `--no-strict` (Bridge's flag is `--not-strict`); neither should conceal unexplained missing tensors.
- Save weights to a separate directory. Training resume requires the original optimizer/data-plane checkpoint, not just an HF export.
- The ready-first keeper is also `step_110`, under `super-vl-35-mixed-teachers-v2-ready-first-s2-20261002`. Its global reward peak at step 82 has no surviving checkpoint.

## SFT and LoRA qualification

There is no qualified SuperVL3.5 SFT or LoRA recipe in the inspected checkout. The available `examples/models/nemotron/nemotron_vl/finetune_nemotron_nano_v2_vl.py` explicitly builds **Nano V2 VL 12B** configs. Changing only its HF path does not establish Super support.

Before documenting a runnable Super SFT command:

1. Add a Super-specific provider/recipe for the hybrid MoE language model and RADIO vision encoder. Set TP/CP/EP, frozen components, precision, sequence lengths, and optimizer settings explicitly.
2. Define image and video dataset processing with the checkpoint's processor/chat template. Check expanded media token counts and loss masks; match the 64-frame video training path where intended.
3. Run a short forward/backward check, checkpoint save/restore, and exported-HF reload on the selected topology. Add separate LoRA target and freeze checks if LoRA is offered.
4. Record memory, loss, token/logprob parity, and full conversion comparisons before marking the recipe supported.

Production GRPO evidence and the remaining raw-TMPE limitations are in [the RL runbook](../nemo-rl/super-vl-3p5.md). The layout follows the [Nano Omni Bridge README](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/nemotron/nemotron_3_omni/README.md); Nano validation results do not qualify these Super workflows.
