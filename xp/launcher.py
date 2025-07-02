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
import time
import yaml
import sys # Added for countdown timer


# Environment variables
LOG_DIR = os.environ["LOG"] + "/nemo-rl"
ACCOUNT = os.environ["ACCOUNT"]
CONTAINER = os.environ["CON"] + "/nemo_rl_base.sqsh"
MOUNTS = "/lustre:/lustre"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Launch Slurm experiments with parameter sweeps")
    parser.add_argument("--script", type=str, default=None, help="Path to the Python script to run")
    parser.add_argument("--config", type=str, default=None, help="Path to the base YAML config file")
    parser.add_argument("--sweep", type=str, default=None, help="Path to the sweep config YAML file")
    parser.add_argument("--nodes", type=int, default=None, help="Number of nodes to use")
    parser.add_argument("--time", type=str, default="4:0:0", help="Time limit for the job")
    parser.add_argument("--account", type=str, default=ACCOUNT, help="Slurm account to use")
    parser.add_argument("--partition", type=str, default="batch", help="Slurm partition to use")
    parser.add_argument("--container", type=str, default=CONTAINER, help="Container to use")
    parser.add_argument("--mounts", type=str, default=MOUNTS, help="Mounts to use")
    parser.add_argument("--jobname", type=str, default=None, help="Base name for the job")
    parser.add_argument("--dry", action="store_true", help="Print commands without executing them")
    parser.add_argument("--sleep", type=int, default=0, help="Sleep time in seconds between jobs")
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
) -> Tuple[str, Optional[str]]:
    """Launch a single experiment using Slurm."""
    # Construct the command
    command = f"uv run {script}"
    if config:
        command += f" --config {config}"
    
    log_dir = os.path.join(LOG_DIR, job_name)
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
        "ray.sub",
    ]
    
    # Join the command
    full_cmd = " \\\n".join(sbatch_cmd)
    
    if dry_run:
        return full_cmd, None
    else:
        result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
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

    script_path, config_path = args.script, args.config
    
    # Load sweep config if provided
    param_combinations = [{}]  # Default to no parameters
    if args.sweep:
        sweep_config = load_yaml(args.sweep)
        script_path, config_path = verify_with_sweep(sweep_config, script_path, config_path)
        param_combinations = generate_parameter_combinations(sweep_config)
    elif not script_path:
         raise ValueError("Script path must be provided via --script or sweep config")

    # Determine the number of nodes after potentially getting config_path from sweep
    num_nodes = get_num_nodes(config_path, args.nodes)
    
    # Print header
    mode = "DRY RUN" if args.dry else "SUBMITTING"
    print(f"\n=== {mode} - Experiments ===\n")
    
    # Launch experiments
    for i, params in enumerate(param_combinations):
        # Format parameter overrides
        param_overrides = format_parameter_override(params)
        
        # Generate job name if not provided
        job_name = args.jobname or (
            os.path.splitext(os.path.basename(args.sweep))[0] if args.sweep 
            else os.path.splitext(os.path.basename(script_path))[0]
        )
        if len(param_combinations) > 1:
            job_name = f"{job_name}_{i+1}"
        
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
            job_name=job_name,
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

        if args.sleep and i < len(param_combinations) - 1: # Don't sleep after the last job
            print(f"Sumbitted {i+1} of {len(param_combinations)} jobs")
            print(f"Sleeping for {args.sleep} seconds before the next job...")
            for remaining in range(args.sleep, 0, -1):
                sys.stdout.write(f"\rTime remaining: {remaining:2d} seconds")
                sys.stdout.flush()
                time.sleep(1)
            sys.stdout.write("\rDone sleeping.                           \n") # Clear the countdown line
            sys.stdout.flush()
        
        print(f"All {len(param_combinations)} jobs submitted")


if __name__ == "__main__":
    main()
