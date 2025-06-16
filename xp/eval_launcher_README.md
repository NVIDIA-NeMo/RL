# Evaluation Launcher for NeMo RL

The `eval_launcher.py` script provides a convenient way to launch evaluation experiments on SLURM clusters with automatic DCP to HuggingFace checkpoint conversion and parameter sweeps.

## Features

- **Smart Checkpoint Caching**: Automatically caches converted HF checkpoints to avoid redundant conversions
- **Race Condition Protection**: Uses file locking to prevent multiple jobs from converting the same checkpoint simultaneously
- **Force Re-conversion**: Option to delete and re-convert existing checkpoints
- **In-Job Conversion**: DCP to HF conversion happens inside the SLURM job, not on the login node
- **SLURM Resource Management**: Handles job submission with configurable resources
- **Parameter Sweeps**: Support for sweeping over multiple evaluation parameters
- **Dry Run Mode**: Preview commands before execution

## How It Works

### Smart Caching System

When converting DCP checkpoints:
1. The script generates a unique hash based on the DCP checkpoint path and config file
2. Converted checkpoints are stored in `$LOG/nemo-rl/hf_eval_checkpoints/` with descriptive names
3. Before conversion, the script checks if a cached version already exists
4. If found, it uses the cached checkpoint; otherwise, it performs the conversion
5. All conversion happens inside the SLURM job to utilize compute resources

### Race Condition Handling

When multiple jobs try to convert the same checkpoint:
1. The script uses file locking (`flock`) to ensure only one job performs the conversion
2. Other jobs wait for the conversion to complete and then use the converted checkpoint
3. This prevents duplicate work and potential file corruption from concurrent writes

### Force Conversion

When `--force-conversion` is used:
1. Any existing converted checkpoint is deleted first
2. A fresh conversion is performed regardless of cache status
3. This is useful when you suspect the cached version is corrupted or outdated

## Usage

### Basic Evaluation

Evaluate a HuggingFace model:
```bash
python xp/eval_launcher.py generation.model_name=meta-llama/Llama-3.2-1B-Instruct
```

### Evaluate with DCP Checkpoint Conversion

Convert a DCP checkpoint and evaluate (with automatic caching):
```bash
python xp/eval_launcher.py \
    --dcp-ckpt-path results/grpo/step_170/policy/weights/ \
    --dcp-config results/grpo/step_170/config.yaml
```

The converted checkpoint will be cached at:
`$LOG/nemo-rl/hf_eval_checkpoints/step_170_<hash>/`

### Force Re-conversion

To delete existing cache and force re-conversion:
```bash
python xp/eval_launcher.py \
    --dcp-ckpt-path results/grpo/step_170/policy/weights/ \
    --dcp-config results/grpo/step_170/config.yaml \
    --force-conversion
```

### Specify Custom HF Path

To save the converted checkpoint at a specific location:
```bash
python xp/eval_launcher.py \
    --dcp-ckpt-path results/grpo/step_170/policy/weights/ \
    --dcp-config results/grpo/step_170/config.yaml \
    --hf-ckpt-path /custom/path/to/hf_checkpoint
```

### Multi-node Evaluation

Run evaluation on multiple nodes:
```bash
python xp/eval_launcher.py \
    --nodes 2 \
    --gpus 8 \
    generation.model_name=Qwen/Qwen2.5-32B \
    generation.vllm_cfg.tensor_parallel_size=4
```

### Parameter Sweeps

Create a sweep configuration file (e.g., `eval_sweep.yaml`):
```yaml
# Optional: Specify the config file to use for all experiments
config_path: examples/configs/eval.yaml

# Parameters to sweep over
generation.temperature: [0.6, 0.8, 1.0]
generation.top_p: [0.9, 0.95]
eval.num_tests_per_prompt: [8, 16]
data.dataset_name: ["HuggingFaceH4/MATH-500", "openai/gsm8k"]
```

Run the sweep:
```bash
python xp/eval_launcher.py \
    --sweep eval_sweep.yaml \
    --dcp-ckpt-path results/grpo/step_170/policy/weights/ \
    --dcp-config results/grpo/step_170/config.yaml
```

**Config Path Handling:**
- If `config_path` is specified in the sweep file, it will be used for all experiments
- If both `--config` CLI argument and `config_path` in sweep file are provided, they must match
- If neither is provided, the script will use its default config handling

When using sweeps with the same checkpoint, the conversion only happens once due to the caching system. Multiple jobs will coordinate through file locking to avoid race conditions.

### Custom Evaluation Config

Use a custom evaluation configuration:
```bash
python xp/eval_launcher.py \
    --config examples/configs/eval_custom.yaml \
    generation.model_name=agentica-org/DeepScaleR-1.5B-Preview
```

## Command Line Arguments

### Evaluation Arguments
- `--config`: Path to evaluation config YAML file (default: examples/configs/eval.yaml)
- `--sweep`: Path to sweep config YAML file for parameter sweeps

### Checkpoint Conversion Arguments
- `--dcp-ckpt-path`: Path to DCP checkpoint to convert
- `--dcp-config`: Path to config file for DCP checkpoint (required with --dcp-ckpt-path)
- `--hf-ckpt-path`: Path to save converted HF checkpoint (optional, auto-generated with caching if not provided)
- `--skip-conversion`: Skip checkpoint conversion (use with pre-converted checkpoints)
- `--force-conversion`: Force re-conversion even if cached version exists (deletes existing cache first)

### SLURM Arguments
- `--nodes`: Number of nodes to use (default: from config or 1)
- `--gpus`: Number of GPUs per node (default: 8)
- `--time`: Time limit for the job (default: "2:0:0")
- `--account`: SLURM account to use
- `--partition`: SLURM partition to use (default: "batch")
- `--container`: Container to use
- `--mounts`: Mount points (default: "/lustre:/lustre")
- `--jobname`: Base name for the job (default: "eval")
- `--dry`: Print commands without executing (dry run mode)

### Additional Parameters
Any additional parameters can be passed directly and will be forwarded to the evaluation script:
```bash
python xp/eval_launcher.py \
    generation.model_name=model_path \
    data.dataset_name=HuggingFaceH4/MATH-500 \
    eval.num_tests_per_prompt=16
```

## Examples

### Example 1: Evaluate DeepScaleR on MATH-500
```bash
python xp/eval_launcher.py \
    --nodes 1 \
    --gpus 8 \
    --time "4:0:0" \
    generation.model_name=agentica-org/DeepScaleR-1.5B-Preview \
    generation.temperature=0.6 \
    generation.top_p=0.95 \
    generation.vllm_cfg.max_model_len=32768 \
    data.dataset_name=HuggingFaceH4/MATH-500 \
    data.dataset_key=test \
    eval.num_tests_per_prompt=16
```

### Example 2: Convert and Evaluate GRPO Checkpoint (with caching)
```bash
python xp/eval_launcher.py \
    --dcp-ckpt-path /path/to/grpo/checkpoint/weights/ \
    --dcp-config /path/to/grpo/checkpoint/config.yaml \
    --nodes 2 \
    --jobname grpo_eval \
    data.dataset_name=openai/gsm8k
```

### Example 3: Dry Run with Sweep
```bash
python xp/eval_launcher.py \
    --dry \
    --sweep xp/eval_sweep_example.yaml \
    --dcp-ckpt-path results/checkpoint/weights/ \
    --dcp-config results/checkpoint/config.yaml
```

### Example 4: View What Commands Will Be Run
Use dry run to see the full command that will be executed inside the SLURM job:
```bash
python xp/eval_launcher.py \
    --dry \
    --dcp-ckpt-path results/grpo/step_170/policy/weights/ \
    --dcp-config results/grpo/step_170/config.yaml
```

### Example 5: Force Re-conversion for Updated Checkpoint
```bash
python xp/eval_launcher.py \
    --dcp-ckpt-path results/grpo/final/policy/weights/ \
    --dcp-config results/grpo/final/config.yaml \
    --force-conversion \
    --jobname grpo_final_fresh
```

## Environment Variables

The script uses the following environment variables with defaults:
- `LOG`: Base log directory (default: "/tmp")
- `ACCOUNT`: SLURM account (default: "default")
- `CON`: Container directory (default: "/containers")

## Notes

1. **Caching System**: The script uses MD5 hashing of the checkpoint paths to create unique identifiers for cached checkpoints
2. **Race Condition Protection**: File locking ensures that only one job performs the conversion when multiple jobs are launched simultaneously
3. **Force Conversion**: The `--force-conversion` flag deletes any existing converted checkpoint before re-converting
4. **In-Job Conversion**: All DCP to HF conversion happens inside the SLURM job, utilizing compute resources efficiently
5. **Automatic Path Detection**: The script extracts meaningful identifiers from checkpoint paths for readable cache directory names
6. **Error Handling**: If conversion fails, the job will exit with an error message

## Cache Directory Structure

Converted checkpoints are stored as:
```
$LOG/nemo-rl/hf_eval_checkpoints/
├── step_170_a1b2c3d4e5f6g7h8/
│   ├── config.json
│   ├── model.safetensors
│   └── ...
├── step_170_a1b2c3d4e5f6g7h8.lock  (temporary during conversion)
├── checkpoint_final_9i8j7k6l5m4n3o2/
│   └── ...
└── ...
```

Where the directory name format is: `<checkpoint_identifier>_<hash>`

## Troubleshooting

### Multiple Jobs Converting Same Checkpoint
If you launch multiple evaluation jobs with the same DCP checkpoint:
- The first job to acquire the lock will perform the conversion
- Other jobs will wait and use the converted checkpoint once ready
- You'll see messages like "Another process is converting, waiting for completion..."

### Corrupted or Outdated Cache
If you suspect the cached checkpoint is corrupted or outdated:
- Use `--force-conversion` to delete and re-convert
- Check the conversion logs for any errors
- Verify the checkpoint files exist and are readable

### Config Path Conflicts
If you get an assertion error about config path mismatch:
- Either remove `--config` from command line and let sweep file specify it
- Or remove `config_path` from sweep file and use `--config` CLI argument
- Or ensure both specify the exact same path

Example error and solutions:
```
AssertionError: Command line config 'configs/eval_custom.yaml' does not match sweep config config_path 'configs/eval.yaml'
```

**Solution 1:** Use sweep file config
```bash
python xp/eval_launcher.py --sweep my_sweep.yaml  # Remove --config
```

**Solution 2:** Use CLI config
```yaml
# Remove config_path from sweep file
generation.temperature: [0.6, 0.8]
```

**Solution 3:** Make them match
```bash
python xp/eval_launcher.py --config configs/eval.yaml --sweep my_sweep.yaml
``` 