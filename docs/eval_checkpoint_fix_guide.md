# Evaluation Checkpoint Fix Guide

This guide explains how to fix HF checkpoints for VLMEvalKit evaluation when they fail due to configuration issues.

## Quick Reference

```bash
# Launch evals (normal case - should work for recently converted checkpoints)
ops/eval.sh <run_name>@<step> MathVista_MINI MMMU_Pro_V

# Example
ops/eval.sh super-clevr-v3-mmpr-det@135 MathVista_MINI
```

---

## Common Errors and Fixes

### Error 1: "model type not recognized"

**Error message:**
```
The checkpoint you are trying to load has model type `NemotronH_Nano_VL_V2` but Transformers does not recognize this architecture.
```

**Cause:** Old checkpoint uses `NemotronH_Nano_VL_V2` instead of `NemotronH_Nano_VL`

**Fix:**
```python
import json

config_path = "<checkpoint>/config.json"
with open(config_path, 'r') as f:
    config = json.load(f)

# Change model type
config['architectures'] = ['NemotronH_Nano_VL']
config['model_type'] = 'NemotronH_Nano_VL'

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
```

---

### Error 2: "NemotronH_Nano_VL_Config not found"

**Error message:**
```
AttributeError: module 'transformers_modules.mcore_to_hf.configuration_nemotron_h' has no attribute 'NemotronH_Nano_VL_Config'
```

**Cause:** Missing `auto_map` in config.json and/or old `configuration_nemotron_h.py` without the config class

**Fix - Step 1:** Add `auto_map` to config.json
**Fix - Step 2:** Copy `configuration_nemotron_h.py` and `custom_vlm.py` from a working checkpoint

---

### Error 3: "hybrid without layers_block_type"

**Error message:**
```
ValueError: The model is an hybrid without alayers_block_type or an attn_type_list in the hf_config, cannot determine the num of mamba layers
```

**Cause:** Old checkpoint conversion didn't include `layers_block_type` in `text_config`

**Fix:** Add `layers_block_type` to text_config in config.json

---

## Complete Fix Script

Run this to fix any checkpoint:

```bash
# Save as fix_checkpoint.sh and run: bash fix_checkpoint.sh <checkpoint_path>

CHECKPOINT=$1
REF=/lustre/fsw/portfolios/llmservice/users/ikarmanov/nemo-rl/results/super-clevr-v3-mmpr-det/tp_1_hf/step_135/mcore_to_hf

# Copy required Python files
cp $REF/configuration_nemotron_h.py $CHECKPOINT/
cp $REF/custom_vlm.py $CHECKPOINT/

# Fix config.json
python3 << EOF
import json

LAYERS_BLOCK_TYPE = [
    "mamba", "mlp", "mamba", "mlp", "mamba", "mlp", "mamba", "attention", "mlp",
    "mamba", "mlp", "mamba", "mlp", "mamba", "mlp", "mamba", "attention", "mlp",
    "mamba", "mlp", "mamba", "mlp", "mamba", "mlp", "mamba", "attention", "mlp",
    "mamba", "mlp", "mamba", "mlp", "mamba", "mlp", "mamba", "attention", "mlp",
    "mamba", "mlp", "mamba", "mlp", "mamba", "mlp", "mamba", "attention", "mlp",
    "mamba", "mlp", "mamba", "mlp", "mamba", "mlp", "mamba", "attention", "mlp",
    "mamba", "mlp", "mamba", "mlp", "mamba", "mlp", "mamba", "mlp"
]

config_path = "$CHECKPOINT/config.json"
with open(config_path, 'r') as f:
    config = json.load(f)

# Fix 1: Model type
if config.get('model_type') == 'NemotronH_Nano_VL_V2':
    config['architectures'] = ['NemotronH_Nano_VL']
    config['model_type'] = 'NemotronH_Nano_VL'
    print("Fixed: model_type")

# Fix 2: auto_map
if 'auto_map' not in config:
    config['auto_map'] = {
        'AutoConfig': 'configuration_nemotron_h.NemotronH_Nano_VL_Config',
        'AutoModel': 'custom_vlm.NemotronH_Nano_VL',
        'AutoModelForCausalLM': 'custom_vlm.NemotronH_Nano_VL'
    }
    print("Fixed: auto_map")

# Fix 3: layers_block_type
if 'text_config' in config and 'layers_block_type' not in config['text_config']:
    config['text_config']['layers_block_type'] = LAYERS_BLOCK_TYPE
    print("Fixed: layers_block_type")

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print("Done!")
EOF
```

---

## Quick Checklist

Before running eval, verify checkpoint has:

- [ ] `"model_type": "NemotronH_Nano_VL"` (not V2)
- [ ] `auto_map` section at top level of config.json
- [ ] `layers_block_type` in `text_config`
- [ ] `configuration_nemotron_h.py` with `NemotronH_Nano_VL_Config` class
- [ ] `custom_vlm.py` present

---

## Reference Working Checkpoint

Copy files from this known-working checkpoint:
```
/lustre/fsw/portfolios/llmservice/users/ikarmanov/nemo-rl/results/super-clevr-v3-mmpr-det/tp_1_hf/step_135/mcore_to_hf
```
