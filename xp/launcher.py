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
import itertools
import os
import subprocess
from typing import Any, Dict, List, Tuple, Optional
import yaml


# Environment variables
ACCOUNT = os.environ["ACCOUNT"]
PARTITION = os.environ["PARTITION"]
LOG_DIR = os.environ["LOG"] + "/nemo-rl"
WORKSPACE = os.environ["WORKSPACE"]
CONTAINER = os.environ["CON"] + "/nemo_rl_base.sqsh"
MOUNTS = f"/lustre:/lustre,{WORKSPACE}:/workspace"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Launch Slurm experiments with parameter sweeps")
    parser.add_argument("--script", type=str, default=None, help="Path to the Python script to run")
    parser.add_argument("--config", type=str, default=None, help="Path to the base YAML config file")
    parser.add_argument("--sweep", type=str, default=None, help="Path to the sweep config YAML file")
    parser.add_argument("--nodes", type=int, default=None, help="Number of nodes to use")
    parser.add_argument("--time", type=str, default="4:0:0", help="Time limit for the job")
    parser.add_argument("--account", type=str, default=ACCOUNT, help="Slurm account to use")
    parser.add_argument("--partition", type=str, default=PARTITION, help="Slurm partition to use")
    parser.add_argument("--container", type=str, default=CONTAINER, help="Container to use")
    parser.add_argument("--mounts", type=str, default=MOUNTS, help="Mounts to use")
    parser.add_argument("--jobname", type=str, default=None, help="Base name for the job")
    parser.add_argument("--dry", action="store_true", help="Print commands without executing them")
    parser.add_argument("--chain", type=int, default=1, help="Number of jobs per sweep configuration to chain together")
    args, extra_args = parser.parse_known_args()
    return args, extra_args


def load_yaml(file_path: str) -> Dict[str, Any]:
    """Load a YAML file."""
    try:
        with open(file_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: Config file not found at {file_path}. Using default node count.")
        return {}
    except yaml.YAMLError as e:
        print(f"Warning: Error parsing YAML file {file_path}: {e}. Using default node count.")
        return {}


def get_num_nodes(config_path: Optional[str], cli_nodes: Optional[int]) -> int:
    """
    Determine the number of nodes to use.
    Reads from the config file (cluster.num_nodes) as default,
    overrides with CLI argument if provided.
    """
    # Prioritize CLI argument
    if cli_nodes is not None:
        return cli_nodes
    
    config_nodes = None
    if config_path:
        config_data = load_yaml(config_path)
        config_nodes = config_data.get("cluster", {}).get("num_nodes")

    # Use config value if valid
    if isinstance(config_nodes, int) and config_nodes > 0:
        return config_nodes
        
    # Default to 1 node if not specified or invalid
    print("Warning: Number of nodes not specified in CLI or config, defaulting to 1.")
    return 1


def verify_with_sweep(sweep_config: Dict[str, Any], script_path: Optional[str], config_path: Optional[str]) -> Tuple[str, str]:
    """
    Verify the sweep config and determine the final script and config paths.

    Returns:
        Tuple[str, str]: The determined script path and config path.
    Raises:
        AssertionError: If inconsistencies are found or required paths are missing.
    """
    sweep_script_path = sweep_config.get("script_path")
    sweep_config_path = sweep_config.get("config_path")

    # Determine script path
    if script_path is None:
        final_script_path = sweep_script_path
    else:
        final_script_path = script_path
        if sweep_script_path and final_script_path != sweep_script_path:
            raise AssertionError(
               f"Command line script '{os.path.basename(final_script_path)}' does not match "
               f"sweep config script_name '{sweep_script_path}'. Please ensure consistency."
            )

    # Determine config path
    if config_path is None:
        # config_path can be None if not provided by CLI or sweep
        final_config_path = sweep_config_path 
    else:
        final_config_path = config_path
        if sweep_config_path and final_config_path != sweep_config_path:
            raise AssertionError(
                f"Command line config '{final_config_path}' does not match "
                f"sweep config config_path '{sweep_config_path}'. Please ensure consistency."
            )
    
    if final_script_path is None:
        raise AssertionError(
            "Script path must be provided either via --script in command line or 'script_path' in the sweep config."
        )

    return final_script_path, final_config_path


def identify_varying_parameters(sweep_config: Dict[str, Any]) -> Dict[str, bool]:
    """
    Identify which parameters are varying (multiple values) vs constant (single values).
    
    Returns:
        Dict mapping parameter names to True if varying, False if constant.
    """
    sweep_params = {k: v for k, v in sweep_config.items() if k not in ["script_path", "config_path"]}
    varying_params = {}
    
    for param_name, param_values in sweep_params.items():
        # Consider a parameter varying if it has more than one value
        varying_params[param_name] = len(param_values) > 1
    
    return varying_params


def generate_job_name_from_params(base_name: str, params: Dict[str, Any], varying_params: Dict[str, bool]) -> str:
    """
    Generate a meaningful job name based on varying parameter values.
    
    Args:
        base_name: Base job name (e.g., script name or user-provided name)
        params: Dictionary of parameter values for this specific combination
        varying_params: Dictionary indicating which parameters are varying
    
    Returns:
        Generated job name including varying parameter values
    """
    varying_parts = []
    
    for param_name, param_value in params.items():
        if varying_params.get(param_name, False):  # Only include varying parameters
            # Simplify parameter names for readability
            # Extract the last part after dots (e.g., "grpo.kl_penalty" -> "kl_penalty")
            short_name = param_name.split('.')[-1]
            
            # Simplify common parameter values
            if isinstance(param_value, float):
                if param_value < 1e-3:
                    # Use scientific notation for very small values
                    value_str = f"{param_value:.0e}"
                else:
                    # Use regular decimal notation, removing trailing zeros
                    value_str = f"{param_value:g}"
            elif isinstance(param_value, str):
                # For model names, extract just the model part
                if "/" in param_value:
                    value_str = param_value.split("/")[-1]
                else:
                    value_str = param_value
                # Replace common characters that might cause issues
                value_str = value_str.replace("-", "").replace("_", "").replace(" ", "")
            else:
                value_str = str(param_value)
            
            varying_parts.append(f"{short_name}_{value_str}")
    
    # If no varying parameters, just return base name
    if not varying_parts:
        return base_name
    
    # Combine base name with varying parameter values
    return f"{base_name}_{'_'.join(varying_parts)}"


def generate_parameter_combinations(sweep_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate all combinations of parameters from the sweep config."""
    # Remove script_name from the parameters to sweep over
    sweep_params = {k: v for k, v in sweep_config.items() if k not in ["script_path", "config_path"]}
    
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


def launch_experiment(
    script: str,
    config: Optional[str],
    param_overrides: str,
    nodes: int,
    time: str,
    account: str,
    partition: str,
    container: str,
    mounts: str,
    job_name: str,
    dry_run: bool = False,
    extra_args: List[str] = None,
    dependency: Optional[str] = None,
    log_dir_name: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """Launch a single experiment using Slurm."""
    # Construct the command
    command = f"uv run {script}"
    if config:
        command += f" --config {config}"
    
    # Use log_dir_name if provided, otherwise fall back to job_name
    log_dir = os.path.join(LOG_DIR, log_dir_name or job_name)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    
    # Parse extra arguments into overrides
    # Add default args before parsing extra args
    all_args = list(extra_args) if extra_args else []
    default_args = [
        f"logger.log_dir={log_dir}",
        "logger.wandb_enabled=True",
        f"logger.wandb.name={job_name}",
        f"checkpointing.checkpoint_dir={checkpoint_dir}",
        f"cluster.num_nodes={nodes}"
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
        command += " " + format_parameter_override(param_dict)
    elif extra_overrides:
        # If no param_overrides but we have extra_overrides, use those
        command += " " + format_parameter_override(extra_overrides)
    
    # Construct the sbatch command
    sbatch_cmd = [
        f"BASE_LOG_DIR={log_dir}",
        f"NUM_ACTOR_NODES={nodes}",
        f"CONTAINER=\"{container}\"",
        f"MOUNTS=\"{mounts}\"",
        f"COMMAND=\"{command}\"",
        "sbatch",
        f"--nodes={nodes}",
        f"--account={account}",
        f"--job-name={job_name}",
        f"--partition={partition}",
        f"--time={time}",
        "--gres=gpu:8",
    ]
    
    # Add dependency if specified
    if dependency:
        if dependency == "singleton":
            sbatch_cmd.append("--dependency=singleton")
        else:
            sbatch_cmd.append(f"--dependency={dependency}")
    
    sbatch_cmd.append("ray.sub")
    
    # Join the command
    full_cmd = " \\\n".join(sbatch_cmd)
    
    if dry_run:
        return full_cmd, None
    else:
        result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
        job_id = None
        if result.stdout:
            # Extract job ID from sbatch output like "Submitted batch job 4101132"
            stdout_lines = result.stdout.strip().split('\n')
            for line in stdout_lines:
                if 'Submitted batch job' in line:
                    job_id = line.split()[-1]  # Get the last word (job ID)
                    break
        return full_cmd, job_id


def print_experiment_info(
    job_name: str,
    params: Dict[str, Any],
    cmd: str,
    job_id: Optional[str] = None,
    dry_run: bool = False,
    dependency: Optional[str] = None,
) -> None:
    """Print information about an experiment in a consistent format."""
    print(f"\n[{job_name}]")
    print("Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    if dependency:
        print(f"Dependency: {dependency}")
    print("\nCommand:")
    print(cmd)
    if not dry_run and job_id:
        print(f"\nSubmitted batch job {job_id}")
    print("\n" + "="*80 + "\n")


def main():
    """Main entry point."""
    args, extra_args = parse_args()

    script_path, config_path = args.script, args.config
    
    # Load sweep config if provided
    param_combinations = [{}]  # Default to no parameters
    varying_params = {}  # Track which parameters are varying
    if args.sweep:
        sweep_config = load_yaml(args.sweep)
        script_path, config_path = verify_with_sweep(sweep_config, script_path, config_path)
        param_combinations = generate_parameter_combinations(sweep_config)
        varying_params = identify_varying_parameters(sweep_config)
    elif not script_path:
         raise ValueError("Script path must be provided via --script or sweep config")

    # Determine the number of nodes after potentially getting config_path from sweep
    num_nodes = get_num_nodes(config_path, args.nodes)
    
    # Print header
    mode = "DRY RUN" if args.dry else "SUBMITTING"
    total_jobs = len(param_combinations) * args.chain
    print(f"\n=== {mode} - Experiments ===")
    print(f"Parameter combinations: {len(param_combinations)}")
    print(f"Jobs per chain: {args.chain}")
    print(f"Total jobs: {total_jobs}\n")
    
    if args.chain > 1:
        print("NOTE: Jobs in each chain will run sequentially using Slurm singleton dependency.")
        print("Chains for different parameter combinations will run in parallel.\n")
    
    # Launch experiments
    job_counter = 0
    for i, params in enumerate(param_combinations):
        # Format parameter overrides
        param_overrides = format_parameter_override(params)
        
        # Generate base job name
        base_job_name = args.jobname or (
            os.path.splitext(os.path.basename(args.sweep))[0] if args.sweep 
            else os.path.splitext(os.path.basename(script_path))[0]
        )
        
        # Create shared job name for all jobs in this chain
        if len(param_combinations) > 1:
            # Use varying parameters to generate meaningful job names
            shared_job_name = generate_job_name_from_params(base_job_name, params, varying_params)
        else:
            shared_job_name = base_job_name
        
        # Shared log directory for all jobs in this chain
        log_dir_name = shared_job_name
        
        # Submit all jobs in this chain
        for chain_idx in range(args.chain):
            # Use singleton dependency for jobs after the first in the chain
            dependency = "singleton" if chain_idx > 0 else None
            
            # Launch experiment
            cmd, job_id = launch_experiment(
                script=script_path,
                config=config_path,
                param_overrides=param_overrides,
                nodes=num_nodes,
                time=args.time,
                account=args.account,
                partition=args.partition,
                container=args.container,
                mounts=args.mounts,
                job_name=shared_job_name,  # Same name for all jobs in chain
                dry_run=args.dry,
                extra_args=extra_args,
                dependency=dependency,
                log_dir_name=log_dir_name,
            )
            
            # Print experiment info
            print_experiment_info(
                job_name=f"{shared_job_name}_chain{chain_idx+1}" if args.chain > 1 else shared_job_name,
                params=params,
                cmd=cmd,
                job_id=job_id,
                dry_run=args.dry,
                dependency=dependency,
            )
            
            job_counter += 1
    
    print(f"{total_jobs} jobs submitted: {len(param_combinations)} sweeps, with chain size {args.chain}.")


if __name__ == "__main__":
    main()
