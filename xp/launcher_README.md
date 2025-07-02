# Experiment Launcher for NeMo RL

The `launcher.py` script provides a convenient way to launch training and experiment jobs on SLURM clusters with automatic resource management and parameter sweeps.

## Features

- **SLURM Job Management**: Handles job submission with configurable resources
- **Parameter Sweeps**: Support for sweeping over multiple training parameters
- **Flexible Script Execution**: Run any Python script with configurable arguments
- **Resource Auto-detection**: Automatically reads cluster configuration from config files
- **Dry Run Mode**: Preview commands before execution
- **Automatic Logging Setup**: Sets up logging directories and wandb integration

## How It Works

### Script Execution

The launcher executes Python scripts using `uv run` with:
1. Configurable base configuration files
2. Parameter overrides from sweep configs or command line
3. Automatic logging and checkpoint directory setup
4. SLURM resource allocation

### Parameter Sweeps

When using parameter sweeps:
1. Load sweep configuration from YAML file
2. Generate all combinations of specified parameters
3. Launch separate SLURM jobs for each parameter combination
4. Each job gets a unique name and logging directory

### Resource Management

The script automatically:
1. Reads cluster configuration from config files (`cluster.num_nodes`)
2. Allows CLI overrides for nodes and other SLURM parameters
3. Sets up appropriate logging and checkpoint directories
4. Configures wandb logging with unique job names

## Usage

### Basic Script Execution

Run a training script with default configuration:
```bash
python xp/launcher.py \
    --script examples/train_grpo.py \
    --config examples/configs/grpo.yaml
```

### Multi-node Training

Run training on multiple nodes:
```bash
python xp/launcher.py \
    --script examples/train_grpo.py \
    --config examples/configs/grpo.yaml \
    --nodes 4 \
    --time "8:0:0"
```

### Parameter Sweeps

Create a sweep configuration file (e.g., `grpo_sweep.yaml`):
```yaml
# Required: Specify the script to run
script_path: examples/train_grpo.py

# Optional: Specify the config file to use for all experiments
config_path: examples/configs/grpo.yaml

# Parameters to sweep over
grpo.kl_penalty: [0.01, 0.02, 0.05]
grpo.learning_rate: [1e-6, 5e-6, 1e-5]
optim.lr: [1e-6, 5e-6, 1e-5]
model.model_name: ["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B"]
```

Run the sweep:
```bash
python xp/launcher.py --sweep grpo_sweep.yaml
```

**Config Path Handling:**
- If `config_path` is specified in the sweep file, it will be used for all experiments
- If both `--config` CLI argument and `config_path` in sweep file are provided, they must match
- If neither is provided, scripts will run without a base config file

### Custom Parameters

Add custom parameters that will be passed to the script:
```bash
python xp/launcher.py \
    --script examples/train_grpo.py \
    --config examples/configs/grpo.yaml \
    model.model_name=meta-llama/Llama-3.2-7B \
    grpo.kl_penalty=0.02 \
    optim.lr=5e-6
```

## Command Line Arguments

### Core Arguments
- `--script`: Path to the Python script to run (required unless specified in sweep).
- `--config`: Path to base YAML config file (optional).
- `--sweep`: Path to sweep config YAML file for parameter sweeps.

### Launcher Control
- `--jobname`: Base name for the job and logging directories. If not provided, it's inferred from the sweep or script file name.
- `--dry`: Print commands without executing them (dry run mode).
- `--sleep`: Time in seconds to sleep between submitting jobs in a sweep (default: 0). A countdown timer will be displayed.

### SLURM Configuration
- `--nodes`: Number of nodes to use (default: from config or 1).
- `--time`: Time limit for the job (default: "4:0:0").
- `--account`: SLURM account to use (defaults to `$ACCOUNT` environment variable).
- `--partition`: SLURM partition to use (default: "batch").
- `--container`: Container to use (defaults to `$CON/nemo_rl_base.sqsh`).
- `--mounts`: Mount points (default: "/lustre:/lustre").

### Additional Parameters
Any additional parameters can be passed directly and will be forwarded to the script:
```bash
python xp/launcher.py \
    --script examples/train_grpo.py \
    model.model_name=custom_model \
    grpo.num_episodes=1000 \
    logger.wandb.project=my_project
```

## Examples

### Example 1: Basic GRPO Training
```bash
python xp/launcher.py \
    --script examples/train_grpo.py \
    --config examples/configs/grpo.yaml \
    --nodes 2 \
    --time "6:0:0" \
    --jobname grpo_llama_7b \
    model.model_name=meta-llama/Llama-3.2-7B
```

### Example 2: Parameter Sweep for Learning Rate
```bash
# Create sweep_lr.yaml
cat > sweep_lr.yaml << EOF
script_path: examples/train_grpo.py
config_path: examples/configs/grpo.yaml
optim.lr: [1e-6, 5e-6, 1e-5, 2e-5]
grpo.kl_penalty: [0.01, 0.02]
EOF

python xp/launcher.py --sweep sweep_lr.yaml --nodes 2
```

### Example 3: Multi-Model Comparison
```bash
# Create sweep_models.yaml
cat > sweep_models.yaml << EOF
script_path: examples/train_grpo.py
config_path: examples/configs/grpo.yaml
model.model_name: 
  - "meta-llama/Llama-3.2-1B"
  - "meta-llama/Llama-3.2-3B"
  - "microsoft/DialoGPT-medium"
grpo.learning_rate: [1e-6, 5e-6]
EOF

python xp/launcher.py \
    --sweep sweep_models.yaml \
    --time "12:0:0" \
    --jobname model_comparison
```

### Example 4: Dry Run to Preview Commands
```bash
python xp/launcher.py \
    --dry \
    --script examples/train_grpo.py \
    --config examples/configs/grpo.yaml \
    model.model_name=meta-llama/Llama-3.2-7B
```

### Example 5: PPO Training with Custom Settings
```bash
python xp/launcher.py \
    --script examples/train_ppo.py \
    --config examples/configs/ppo.yaml \
    --nodes 4 \
    --time "10:0:0" \
    --jobname ppo_large_model \
    model.model_name=meta-llama/Llama-3.2-8B \
    ppo.epochs=4 \
    ppo.learning_rate=3e-6 \
    logger.wandb.project=ppo_experiments
```

### Example 6: Training with Custom Logging
```bash
python xp/launcher.py \
    --script examples/train_grpo.py \
    --jobname custom_experiment \
    logger.wandb.project=my_research \
    logger.wandb.group=grpo_experiments \
    logger.wandb.tags='["experiment1", "baseline"]'
```

## Environment Variables

The script uses the following environment variables:
- `LOG`: Base log directory (required)
- `ACCOUNT`: SLURM account (required)
- `CON`: Container directory (required)

These must be set in your environment before running the launcher.

## Automatic Setup

The launcher automatically configures:

1. **Logging Directories**: `$LOG/nemo-rl/{job_name}/`
2. **Checkpoint Directories**: `$LOG/nemo-rl/{job_name}/checkpoints/`
3. **Wandb Integration**: Enabled by default with job name as run name
4. **Cluster Configuration**: Reads from config file or uses CLI overrides

## Sweep Configuration Format

Sweep YAML files support:
```yaml
# Required if not provided via CLI
script_path: path/to/script.py

# Optional base config
config_path: path/to/config.yaml

# Parameters to sweep - each will be combined with all others
parameter_name: [value1, value2, value3]
nested.parameter: [val1, val2]
boolean_param: [true, false]
```

The launcher generates all combinations of parameters using itertools.product().

## Job Naming

The base name for a job is determined with the following precedence:
1. The `--jobname` argument, if provided.
2. The basename of the sweep file, if `--sweep` is used (e.g., `grpo_sweep` from `grpo_sweep.yaml`).
3. The basename of the script file (e.g., `train_grpo` from `train_grpo.py`).

For sweeps that generate multiple jobs, a numeric index is appended to the base name for each job (e.g., `{base_name}_1`, `{base_name}_2`, etc.).

## Resource Configuration

### From Config File
```yaml
cluster:
  num_nodes: 2
  gpus_per_node: 8  # Currently fixed at 8 in sbatch command
```

### CLI Override
```bash
python xp/launcher.py --nodes 4  # Overrides config file
```

## Notes

1. **Fixed GPU Count**: Currently fixed at 8 GPUs per node in the sbatch command
2. **Wandb Integration**: Automatically enabled with job name as run name
3. **Parameter Precedence**: CLI extra args override sweep parameters
4. **Directory Structure**: Automatically creates organized logging directories
5. **Error Handling**: Validates configuration consistency between CLI and sweep files

## Troubleshooting

### Script Path Requirements
If you get an error about missing script path:
- Provide `--script` on command line, or
- Add `script_path` to your sweep configuration

### Config Path Conflicts
If you get an assertion error about config path mismatch:
```
AssertionError: Command line config 'configs/custom.yaml' does not match sweep config config_path 'configs/base.yaml'
```

**Solutions:**
1. Remove `--config` and let sweep file specify it
2. Remove `config_path` from sweep file
3. Ensure both specify the same path

### Environment Variables Not Set
```bash
# Set required environment variables
export LOG="/path/to/logs"
export ACCOUNT="your_slurm_account" 
export CON="/path/to/containers"
```

### Job Submission Failures
- Check SLURM account permissions
- Verify partition availability
- Ensure container path exists
- Check node availability for requested resources

### Parameter Override Issues
- Use quotes for string values: `param="string value"`
- Boolean values: `param=true` or `param=false`
- Nested parameters: `section.param=value`

## Directory Structure

Each job creates:
```
$LOG/nemo-rl/{job_name}/
├── checkpoints/          # Model checkpoints
├── logs/                 # Training logs
├── wandb/               # Wandb run data
└── ...                  # Other training artifacts
```

For sweeps, the structure will be based on the indexed job names:
```
$LOG/nemo-rl/
├── {base_name}_1/
├── {base_name}_2/
├── {base_name}_3/
└── ...
```
