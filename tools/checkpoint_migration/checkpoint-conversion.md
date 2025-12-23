## Checkpoint Migration Guide (nemo.tron to megatron.bridge)

This guide explains how to convert existing NeMo Tron-style checkpoints to the new Megatron Bridge format when only the `run_config.yaml` schema has changed. Checkpoint file layout remains the same; you only need to update `run_config.yaml`.

### When to use this

Use this when your checkpoint directory looks like:

```
<CKPT_ROOT>/
├── iter_0000000/
│   ├── __0_0.distcp
│   ├── __0_1.distcp
│   ├── common.pt
│   ├── metadata.json
│   ├── run_config.yaml   # to be converted
│   └── train_state.pt
└── latest_train_state.pt
```

and your `run_config.yaml` has top-level `_target_: nemo.tron.config.ConfigContainer`.

### What changes

- `_target_` paths are migrated from `nemo.tron.*` to `megatron.bridge.*`.
- Top-level keys are renamed from `*_config` to shorter names (e.g., `checkpoint_config` → `checkpoint`).
- Some field defaults are updated for compatibility (e.g., `save_rng: false` under `checkpoint`).
- Some model targets are mapped to new provider classes (Qwen2 and DeepSeek V3 examples included).

### Converter script

We provide a helper script to update `run_config.yaml` in-place or to a new file:

```
python3 scripts/convert_run_config.py /path/to/iter_xxxxxx/run_config.yaml
```

Preview without writing:

```
python3 scripts/convert_run_config.py --dry-run /path/to/run_config.yaml
``;

Write to a new file:

```
python3 scripts/convert_run_config.py /path/to/run_config.yaml -o /path/to/new_run_config.yaml
```

### Supported mappings (key examples)

- Top-level: `nemo.tron.config.ConfigContainer` → `megatron.bridge.training.config.ConfigContainer`
- Sections: `checkpoint_config→checkpoint`, `logger_config→logger`, `model_config→model`, `optimizer_config→optimizer`, `dataset_config→dataset`, `scheduler_config→scheduler`, `rng_config→rng`, `train_config→train`, `dist_config→dist`, `ddp_config→ddp`, `ft_config→ft`, `profiling_config→profiling`, `straggler_config→straggler`, `tokenizer_config→tokenizer`, `rerun_state_machine_config→rerun_state_machine`
- Sub-config targets:
  - Checkpoint: `nemo.tron.config.CheckpointConfig` → `megatron.bridge.training.config.CheckpointConfig`
  - Logger: `nemo.tron.config.LoggerConfig` → `megatron.bridge.training.config.LoggerConfig`
  - RNG: `nemo.tron.config.RNGConfig` → `megatron.bridge.training.config.RNGConfig`
  - Rerun State Machine: `nemo.tron.config.RerunStateMachineConfig` → `megatron.bridge.training.config.RerunStateMachineConfig`
- Model targets:
  - Qwen2: `nemo.collections.llm.gpt.model.qwen2.Qwen2Config` → `megatron.bridge.models.qwen.qwen_provider.Qwen2ModelProvider`
  - DeepSeek V3: `nemo.collections.llm.gpt.model.deepseek.DeepSeekV3Config` → `megatron.bridge.models.deepseek.deepseek_provider.DeepSeekV3Provider`

### Defaults and cleanups applied

- `checkpoint.save_rng=false`, `checkpoint.load_main_params_from_ckpt=false`, `checkpoint.use_persistent_ckpt_worker=true`
- Remove legacy keys (if present): `auto_detect_ckpt_format`, `ckpt_convert_update_legacy_dist_opt_format`
- `logger.logging_level=20` (INFO) and `logger.log_energy=false` by default
- If present in model config: set `apply_rope_fusion=true`, `bias_dropout_fusion=true`, `masked_softmax_fusion=false`, `perform_initialization=false`
- Ensure `model.pipeline_dtype=bfloat16` if null
- Ensure `model.mtp_enabled=false` if missing
- If `model.hf_adapter` is present: set `transformers_version=4.53.3` and `trust_remote_code=true`

### End-to-end workflow

1. Convert `run_config.yaml` in each checkpoint directory you intend to resume from:
   - In-place: `python3 scripts/convert_run_config.py /path/to/iter_0000000/run_config.yaml`
2. Resume or load using Megatron Bridge training utilities with the updated checkpoint.

### Notes and limitations

- This tool focuses on configuration schema changes. It does not move or edit tensor files.
- For models beyond Qwen2/DeepSeekV3, add further `_target_` mappings as needed. See `MODEL_TARGET_MAP` in `scripts/convert_run_config.py`.
- If your original config contained custom callables/hooks under `data_step_fn`/`forward_step_fn`, those are deprecated in the new schema and intentionally not carried over.


