This directory is only intended for testing, temporary validation, and config alignment.

If you have experimental or test-only configs, place them here for local comparison and debugging.

This directory is not part of the final deliverable, and final PR submissions should not include `examples/configs_for_test/`.

Configs currently in this directory:

- `cross_tokenizer_distillation_llama1b_qwen8b_off-policy-distillation-gh.yaml`
  Mirrors the config layout used on the `xtoken/off-policy-distillation-gh` branch and serves as the baseline config. Use it to verify that subsequent development does not change the results produced with this setup.
  To run this config:
  ```bash
  git checkout xtoken/off-policy-distillation-gh
  
  HF_HUB_OFFLINE=1 NRL_FORCE_REBUILD_VENVS=true uv run python examples/run_off_policy_distillation.py --config examples/configs_for_test/cross_tokenizer_distillation_llama1b_qwen8b_off-policy-distillation-gh.yaml
  ```

- `cross_tokenizer_distillation_llama1b_qwen8b_refactored.yaml`
  Refactored test config for the current branch schema. Use it as the aligned reference when validating that the refactored setup remains runtime-equivalent to the gh-branch version.
  To run this config:
  ```bash
  git checkout ruit/xtoken_rafactor
  
  HF_HUB_OFFLINE=1 NRL_FORCE_REBUILD_VENVS=true uv run python examples/run_off_policy_distillation.py --config examples/configs_for_test/cross_tokenizer_distillation_llama1b_qwen8b_refactored.yaml
  ```