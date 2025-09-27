#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import hashlib
import itertools
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import yaml


# Environment variables
LOG_DIR = os.environ.get("LOG", "/tmp") + "/nemo-rl"
ACCOUNT = os.environ.get("ACCOUNT", "default")
CONTAINER = os.environ.get("CON", "/containers") + "/nemo_rl_base.sqsh"
MOUNTS = "/lustre:/lustre"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Launch Slurm evaluation experiments with parameter sweeps")
    
    # Evaluation specific arguments
    parser.add_argument("--config", type=str, help="Path to the evaluation config YAML file")
    parser.add_argument("--sweep", type=str, help="Path to the sweep config YAML file")
    
    # Checkpoint conversion arguments
    parser.add_argument("--dcp-ckpt-path", type=str, help="Path to DCP checkpoint to convert")
    parser.add_argument("--dcp-config", type=str, help="Path to config file for DCP checkpoint")
    parser.add_argument("--hf-ckpt-path", type=str, help="Path to save converted HF checkpoint (optional)")
    parser.add_argument("--skip-conversion", action="store_true", help="Skip checkpoint conversion")
    parser.add_argument("--force-conversion", action="store_true", help="Force re-conversion even if cached version exists")
    
    # SLURM arguments
    parser.add_argument("--nodes", type=int, default=None, help="Number of nodes to use")
    parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs per node")
    parser.add_argument("--time", type=str, default="2:0:0", help="Time limit for the job")
    parser.add_argument("--account", type=str, default=ACCOUNT, help="Slurm account to use")
    parser.add_argument("--partition", type=str, default="batch", help="Slurm partition to use")
    parser.add_argument("--container", type=str, default=CONTAINER, help="Container to use")
    parser.add_argument("--mounts", type=str, default=MOUNTS, help="Mounts to use")
    parser.add_argument("--jobname", type=str, default=None, help="Base name for the job")
    parser.add_argument("--dry", action="store_true", help="Print commands without executing them")
    
    args, extra_args = parser.parse_known_args()
    return args, extra_args


def load_yaml(file_path: str) -> Dict[str, Any]:
    """Load a YAML file."""
    try:
        with open(file_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: Config file not found at {file_path}.")
        return {}
    except yaml.YAMLError as e:
        print(f"Warning: Error parsing YAML file {file_path}: {e}.")
        return {}


def get_num_nodes(config_path: Optional[str], cli_nodes: Optional[int]) -> int:
    """
    Determine the number of nodes to use.
    Reads from the config file (cluster.num_nodes) as default,
    overrides with CLI argument if provided.
    """
    if cli_nodes is not None:
        return cli_nodes
    
    config_nodes = None
    if config_path:
        config_data = load_yaml(config_path)
        config_nodes = config_data.get("cluster", {}).get("num_nodes")

    if isinstance(config_nodes, int) and config_nodes > 0:
        return config_nodes
        
    print("Warning: Number of nodes not specified in CLI or config, defaulting to 1.")
    return 1


def get_num_gpus(config_path: Optional[str], cli_gpus: Optional[int]) -> int:
    """
    Determine the number of GPUs per node to use.
    Reads from the config file (cluster.gpus_per_node) as default,
    overrides with CLI argument if provided.
    """
    if cli_gpus is not None:
        return cli_gpus
    
    config_gpus = None
    if config_path:
        config_data = load_yaml(config_path)
        config_gpus = config_data.get("cluster", {}).get("gpus_per_node")

    if isinstance(config_gpus, int) and config_gpus > 0:
        return config_gpus
        
    print("Warning: Number of GPUs per node not specified in CLI or config, defaulting to 8.")
    return 8


def get_checkpoint_hash(dcp_ckpt_path: str, dcp_config: str) -> str:
    """
    Generate a consistent hash based on the DCP checkpoint path and config.
    
    Returns:
        str: A hash string that uniquely identifies this checkpoint
    """
    # Normalize paths to ensure consistency
    normalized_ckpt = os.path.abspath(dcp_ckpt_path)
    normalized_config = os.path.abspath(dcp_config)
    
    # Create a unique identifier from both paths
    identifier = f"{normalized_ckpt}:{normalized_config}"
    
    # Generate a hash
    hash_obj = hashlib.md5(identifier.encode())
    return hash_obj.hexdigest()[:16]  # Use first 16 chars for readability


def get_hf_checkpoint_path(dcp_ckpt_path: str, dcp_config: str, log_dir: str) -> str:
    """
    Get the path where the HF checkpoint should be stored.
    
    Returns:
        str: Path to the HF checkpoint directory
    """
    checkpoint_hash = get_checkpoint_hash(dcp_ckpt_path, dcp_config)
    hf_checkpoints_dir = os.path.join(log_dir, "hf_eval_checkpoints")
    
    # Extract model name from path for readability
    path_parts = dcp_ckpt_path.strip('/').split('/')
    model_identifier = ""
    for part in reversed(path_parts):
        if part and part != "weights" and part != "policy" and part != "actor":
            model_identifier = part
            break
    
    # Create a descriptive directory name with hash
    checkpoint_name = f"{model_identifier}_{checkpoint_hash}" if model_identifier else checkpoint_hash
    return os.path.join(hf_checkpoints_dir, checkpoint_name)


def create_conversion_command(
    dcp_ckpt_path: str,
    dcp_config: str,
    hf_ckpt_path: str,
    force_conversion: bool = False
) -> str:
    """
    Create the command to convert DCP to HF format, with caching logic.
    
    Returns:
        str: Shell command to execute for conversion
    """
    # Create a lock file path based on the checkpoint hash
    lock_file = f"{hf_ckpt_path}.lock"
    
    if force_conversion:
        # Force conversion case - delete existing and always convert with locking
        conversion_cmd = (
            f"mkdir -p $(dirname '{hf_ckpt_path}') && "
            f"exec 200>'{lock_file}' && "
            f"if flock -n 200; then "
            f"echo 'Force conversion enabled, removing existing checkpoint if present...' && "
            f"rm -rf '{hf_ckpt_path}' && "
            f"echo 'Converting DCP to HF format (forced)...' && "
            f"uv run python examples/convert_dcp_to_hf.py "
            f"--config {dcp_config} "
            f"--dcp-ckpt-path {dcp_ckpt_path} "
            f"--hf-ckpt-path {hf_ckpt_path} && "
            f"echo 'Successfully converted checkpoint to: {hf_ckpt_path}'; "
            f"flock -u 200; "
            f"else "
            f"echo 'Another process is force converting, waiting for completion...' && "
            f"flock 200 && "
            f"echo 'Force conversion completed by another process'; "
            f"fi; "
            f"exec 200>&-"  # Close the file descriptor
        )
    else:
        # Normal case - check cache first with lock file for race condition handling
        conversion_cmd = (
            # First check if conversion is already complete
            f"if [ -d '{hf_ckpt_path}' ] && [ -f '{hf_ckpt_path}/config.json' ]; then "
            f"echo 'Found cached HF checkpoint at: {hf_ckpt_path}' && "
            f"echo 'Using cached checkpoint, skipping conversion'; "
            f"else "
            # Try to acquire lock for conversion
            f"mkdir -p $(dirname '{hf_ckpt_path}') && "
            f"exec 200>'{lock_file}' && "
            f"if flock -n 200; then "
            # We got the lock, check again if someone else completed it while we were waiting
            f"if [ -d '{hf_ckpt_path}' ] && [ -f '{hf_ckpt_path}/config.json' ]; then "
            f"echo 'Another process completed conversion, using cached checkpoint'; "
            f"else "
            # Proceed with conversion
            f"echo 'Acquired lock, converting DCP to HF format...' && "
            f"uv run python examples/convert_dcp_to_hf.py "
            f"--config {dcp_config} "
            f"--dcp-ckpt-path {dcp_ckpt_path} "
            f"--hf-ckpt-path {hf_ckpt_path} && "
            f"echo 'Successfully converted checkpoint to: {hf_ckpt_path}'; "
            f"fi; "
            f"flock -u 200; "
            f"else "
            # Could not get lock, wait for the other process
            f"echo 'Another process is converting, waiting for completion...' && "
            f"flock 200 && "
            f"echo 'Conversion completed by another process'; "
            f"fi; "
            f"exec 200>&-; "  # Close the file descriptor
            f"fi"
        )
    
    return conversion_cmd


def verify_with_sweep(sweep_config: Dict[str, Any], config_path: Optional[str]) -> Optional[str]:
    """
    Verify the sweep config and determine the final config path.

    Args:
        sweep_config: The loaded sweep configuration
        config_path: Config path from command line (if any)

    Returns:
        str: The determined config path (can be None)
    
    Raises:
        AssertionError: If inconsistencies are found between CLI and sweep config.
    """
    sweep_config_path = sweep_config.get("config_path")

    # Determine config path
    if config_path is None:
        # Use config path from sweep if available
        final_config_path = sweep_config_path 
    else:
        final_config_path = config_path
        if sweep_config_path and final_config_path != sweep_config_path:
            raise AssertionError(
                f"Command line config '{final_config_path}' does not match "
                f"sweep config config_path '{sweep_config_path}'. Please ensure consistency."
            )
    
    return final_config_path


def generate_parameter_combinations(sweep_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate all combinations of parameters from the sweep config."""
    # Remove config_path from the parameters to sweep over (similar to script_path in original launcher)
    sweep_params = {k: v for k, v in sweep_config.items() if k not in ["config_path"]}
    
    # Extract parameter names and their values
    param_names = list(sweep_params.keys())
    param_values = [sweep_params[name] for name in param_names]
    
    # Generate all combinations
    combinations = []
    for values in itertools.product(*param_values):
        param_dict = dict(zip(param_names, values))
        combinations.append(param_dict)
    
    return combinations


def parse_extra_args(extra_args: List[str]) -> Dict[str, Any]:
    """Parse extra arguments into a dictionary of parameter overrides."""
    if not extra_args:
        return {}
    
    # Filter out the '--' separator
    filtered_args = [arg for arg in extra_args if arg != '--']
    if not filtered_args:
        return {}
    
    # Parse arguments into a dictionary
    overrides = {}
    for arg in filtered_args:
        if '=' in arg:
            key, value = arg.split('=', 1)
            # Remove leading dashes if present
            key = key.lstrip('-')
            # Try to convert value to appropriate type
            try:
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '', 1).isdigit() and value.count('.') == 1:
                    value = float(value)
            except ValueError:
                pass
            overrides[key] = value
        else:
            # Handle boolean flags
            key = arg.lstrip('-')
            overrides[key] = True
    
    return overrides


def format_parameter_override(param_dict: Dict[str, Any]) -> str:
    """Format parameter dictionary into command line override string."""
    overrides = []
    for key, value in param_dict.items():
        if isinstance(value, str):
            overrides.append(f"{key}='{value}'")
        else:
            overrides.append(f"{key}={value}")
    return " ".join(overrides)


def launch_eval_experiment(
    config: Optional[str],
    model_path: str,
    param_overrides: str,
    nodes: int,
    gpus_per_node: int,
    time: str,
    account: str,
    partition: str,
    container: str,
    mounts: str,
    job_name: str,
    conversion_cmd: Optional[str] = None,
    hf_checkpoint_path: Optional[str] = None,
    dry_run: bool = False,
    extra_args: List[str] = None,
) -> Tuple[str, Optional[str]]:
    """Launch a single evaluation experiment using Slurm."""
    # Construct the evaluation command
    eval_command = f"uv run examples/run_eval.py"
    if config:
        eval_command += f" --config {config}"
    
    log_dir = os.path.join(LOG_DIR, job_name)
    
    # Parse extra arguments into overrides
    all_args = list(extra_args) if extra_args else []
    
    # If we have a conversion command, use the HF checkpoint path directly
    if conversion_cmd and hf_checkpoint_path:
        model_path_arg = f"generation.model_name={hf_checkpoint_path}"
    else:
        model_path_arg = f"generation.model_name={model_path}"
    
    default_args = [
        model_path_arg,
        f"cluster.num_nodes={nodes}",
        f"cluster.gpus_per_node={gpus_per_node}",
    ]
    all_args.extend(default_args)
    extra_overrides = parse_extra_args(all_args)
    
    # If we have parameter overrides, parse them and merge with extra overrides
    if param_overrides:
        # Split the param_overrides string into key-value pairs
        param_dict = {}
        for param in param_overrides.split():
            if '=' in param:
                key, value = param.split('=', 1)
                # Remove quotes if present
                value = value.strip("'")
                param_dict[key] = value
        
        # Update with extra overrides (they take precedence)
        param_dict.update(extra_overrides)
        eval_command += " " + format_parameter_override(param_dict)
    elif extra_overrides:
        # If no param_overrides but we have extra_overrides, use those
        eval_command += " " + format_parameter_override(extra_overrides)
    
    # Combine conversion and evaluation commands
    if conversion_cmd:
        # The conversion command sets HF_MODEL_PATH, so we need to ensure it's available for the eval command
        full_command = f"{conversion_cmd} && {eval_command}"
    else:
        full_command = eval_command
    
    # Construct the sbatch command
    sbatch_cmd = [
        f"BASE_LOG_DIR={log_dir}",
        f"NUM_ACTOR_NODES={nodes}",
        f"CONTAINER=\"{container}\"",
        f"MOUNTS=\"{mounts}\"",
        f"COMMAND=\"{full_command}\"",
        "sbatch",
        f"--nodes={nodes}",
        f"--account={account}",
        f"--job-name={job_name}",
        f"--partition={partition}",
        f"--time={time}",
        f"--gres=gpu:{gpus_per_node}",
        "ray.sub",
    ]
    
    # Join the command
    full_cmd = " \\\n".join(sbatch_cmd)
    
    if dry_run:
        # For dry run, show the command that will be executed
        if conversion_cmd:
            print("\n--- Command that will be executed ---")
            print(full_command)
            print("--- End of command ---\n")
        return full_cmd, None
    else:
        result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error submitting job: {result.stderr}")
            return full_cmd, None
        job_id = result.stdout.strip() if result.stdout else None
        return full_cmd, job_id


def print_experiment_info(
    job_name: str,
    params: Dict[str, Any],
    cmd: str,
    job_id: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """Print information about an experiment in a consistent format."""
    print(f"\n[{job_name}]")
    print("Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print("\nCommand:")
    print(cmd)
    if not dry_run and job_id:
        print(f"\nSubmitted batch job {job_id}")
    print("\n" + "="*80 + "\n")


def main():
    """Main entry point."""
    args, extra_args = parse_args()
    
    # Handle sweep config and verify consistency
    config_path = args.config
    param_combinations = [{}]  # Default to no parameters
    
    if args.sweep:
        sweep_config = load_yaml(args.sweep)
        config_path = verify_with_sweep(sweep_config, args.config)
        param_combinations = generate_parameter_combinations(sweep_config)
    
    # Handle checkpoint paths
    model_path = None
    conversion_cmd = None
    hf_checkpoint_path = None
    
    if args.dcp_ckpt_path and not args.skip_conversion:
        if not args.dcp_config:
            raise ValueError("--dcp-config must be provided when converting DCP checkpoint")
        
        # Get the HF checkpoint path (either specified or auto-generated)
        if args.hf_ckpt_path:
            hf_checkpoint_path = args.hf_ckpt_path
        else:
            # Use the smart caching system
            hf_checkpoint_path = get_hf_checkpoint_path(
                args.dcp_ckpt_path, 
                args.dcp_config,
                LOG_DIR
            )
        
        print(f"\n=== DCP to HF Conversion Setup ===")
        print(f"DCP checkpoint: {args.dcp_ckpt_path}")
        print(f"HF checkpoint will be at: {hf_checkpoint_path}")
        
        # Create the conversion command
        conversion_cmd = create_conversion_command(
            args.dcp_ckpt_path,
            args.dcp_config,
            hf_checkpoint_path,
            args.force_conversion
        )
        
        # Model path will be set to the HF checkpoint path
        model_path = hf_checkpoint_path
        
    elif args.skip_conversion and args.hf_ckpt_path:
        model_path = args.hf_ckpt_path
    else:
        # Check if model path is provided in extra args
        for arg in extra_args:
            if arg.startswith("generation.model_name="):
                model_path = arg.split("=", 1)[1].strip("'\"")
                break
    
    if not model_path:
        raise ValueError("Model path must be provided either through checkpoint conversion or generation.model_name parameter")
    
    # Determine the number of nodes and GPUs using the verified config path
    num_nodes = get_num_nodes(config_path, args.nodes)
    num_gpus = get_num_gpus(config_path, args.gpus)
    
    # Print header
    mode = "DRY RUN" if args.dry else "SUBMITTING"
    print(f"\n=== {mode} - Evaluation Experiments ===\n")
    if config_path:
        print(f"Config: {config_path}")
    if conversion_cmd:
        print(f"Model: Will be converted from DCP checkpoint")
        print(f"Target HF path: {hf_checkpoint_path}")
    else:
        print(f"Model: {model_path}")
    print(f"Nodes: {num_nodes}")
    print(f"GPUs per node: {num_gpus}")
    
    # Launch experiments
    for i, params in enumerate(param_combinations):
        # Format parameter overrides
        param_overrides = format_parameter_override(params)
        
        # Generate job name if not provided
        job_name = args.jobname or "eval"
        if len(param_combinations) > 1:
            job_name = f"{job_name}_sweep_{i+1}"
        
        # Launch experiment
        cmd, job_id = launch_eval_experiment(
            config=config_path,
            model_path=model_path,
            param_overrides=param_overrides,
            nodes=num_nodes,
            gpus_per_node=num_gpus,
            time=args.time,
            account=args.account,
            partition=args.partition,
            container=args.container,
            mounts=args.mounts,
            job_name=job_name,
            conversion_cmd=conversion_cmd,
            hf_checkpoint_path=hf_checkpoint_path,
            dry_run=args.dry,
            extra_args=extra_args,
        )
        
        # Print experiment info
        print_experiment_info(
            job_name=job_name,
            params=params,
            cmd=cmd,
            job_id=job_id,
            dry_run=args.dry,
        )


if __name__ == "__main__":
    main() 