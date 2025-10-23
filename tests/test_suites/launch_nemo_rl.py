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

"""Launch a NeMo-RL test as a Ray job on a Slurm cluster"""

import argparse
import os

from nemo_run.core.execution.slurm import SlurmExecutor, SSHTunnel
from nemo_run.run.ray.job import RayJob


def parse_args():
    """Parse command line arguments with environment variable defaults"""
    parser = argparse.ArgumentParser(
        description="Launch a NeMo-RL test as a Ray job on a Slurm cluster"
    )

    # Test specification - support multiple ways to specify tests
    test_group = parser.add_mutually_exclusive_group(required=True)
    test_group.add_argument(
        "--test-path",
        default=os.environ.get("TEST_PATH"),
        help="Path to test folder or file (e.g., tests/test_suites/llm/sft-llama3.1-8b-1n8g-fsdp2tp1-dynamicbatch/)",
    )
    test_group.add_argument(
        "--test-markers",
        default=os.environ.get("TEST_MARKERS"),
        help="Pytest markers to filter tests (e.g., 'algo_sft and suite_nightly and num_gpus_8')",
    )
    test_group.add_argument(
        "--test-name",
        default=os.environ.get("TEST_NAME"),
        help="Test name (e.g., sft-llama3.1-8b-1n8g-fsdp2tp1-dynamicbatch)",
    )

    # CI/Infrastructure
    parser.add_argument(
        "--ci-job-id", default=os.environ.get("CI_JOB_ID"), help="CI job ID"
    )
    parser.add_argument(
        "--hf-home",
        default=os.environ.get("HF_HOME"),
        help="Hugging Face home directory",
    )
    parser.add_argument(
        "--hf-datasets-cache",
        default=os.environ.get("HF_DATASETS_CACHE"),
        help="Hugging Face datasets cache directory",
    )
    parser.add_argument(
        "--hf-hub-offline",
        default=os.environ.get("HF_HUB_OFFLINE", "1"),
        help="Hugging Face Hub offline mode",
    )
    parser.add_argument(
        "--hf-token", default=os.environ.get("RL_HF_TOKEN"), help="Hugging Face token"
    )
    parser.add_argument(
        "--host",
        default=os.getenv("HOST", "cw-dfw-cs-001-login-01.nvidia.com"),
        help="SSH host",
    )
    parser.add_argument(
        "--identity", default=os.environ.get("IDENTITY"), help="SSH identity file"
    )
    parser.add_argument(
        "--image",
        default=os.environ.get("BUILD_IMAGE_NAME_SBATCH"),
        help="Container image",
    )
    parser.add_argument(
        "--job-name", default=os.environ.get("CI_JOB_NAME"), help="Job name"
    )
    parser.add_argument(
        "--job-time", default=os.environ.get("TIME"), help="Job time limit"
    )
    parser.add_argument(
        "--nemorun-home",
        default=os.environ.get("NEMORUN_HOME"),
        help="NeMo Run home directory",
    )
    parser.add_argument(
        "--nrl-deepscaler-16k-ckpt",
        default=os.environ.get("NRL_DEEPSCALER_16K_CKPT"),
        help="NRL DeepScaler 16K checkpoint path",
    )
    parser.add_argument(
        "--nrl-deepscaler-24k-ckpt",
        default=os.environ.get("NRL_DEEPSCALER_24K_CKPT"),
        help="NRL DeepScaler 24K checkpoint path",
    )
    parser.add_argument(
        "--nrl-deepscaler-8k-ckpt",
        default=os.environ.get("NRL_DEEPSCALER_8K_CKPT"),
        help="NRL DeepScaler 8K checkpoint path",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=int(os.getenv("NUM_NODES", "1")),
        help="Number of nodes",
    )
    parser.add_argument(
        "--partition", default=os.getenv("PARTITION", "batch"), help="Slurm partition"
    )
    parser.add_argument(
        "--slurm-account", default=os.environ.get("SLURM_ACCOUNT"), help="Slurm account"
    )
    parser.add_argument("--user", default=os.environ.get("RL_USER"), help="SSH user")
    parser.add_argument(
        "--wandb-api-key",
        default=os.environ.get("WANDB_API_KEY"),
        help="Weights & Biases API key",
    )

    # Pytest options
    parser.add_argument(
        "--pytest-args",
        default=os.environ.get("PYTEST_ARGS", ""),
        help="Additional pytest arguments (e.g., '-v -s')",
    )

    return parser.parse_args()


def build_pytest_command(args):
    """Build the pytest command based on arguments."""
    pytest_cmd = "uv run --no-sync pytest"

    # Determine test target
    if args.test_path:
        pytest_cmd += f" {args.test_path}"
    elif args.test_markers:
        pytest_cmd += f" -m '{args.test_markers}' tests/test_suites/"
    elif args.test_name:
        pytest_cmd += f" tests/test_suites/llm/{args.test_name}/"
    else:
        raise ValueError("Must specify --test-path, --test-markers, or --test-name")

    # Add additional pytest arguments
    if args.pytest_args:
        pytest_cmd += f" {args.pytest_args}"

    return pytest_cmd


def main():
    """Run a NeMo-RL test as a Ray job on a Slurm cluster"""
    args = parse_args()

    # Build pytest command
    pytest_command = build_pytest_command(args)

    executor = SlurmExecutor(
        account=args.slurm_account,
        partition=args.partition,
        nodes=args.num_nodes,
        gpus_per_node=8,
        gres="gpu:8",
        time=args.job_time,
        container_image=args.image,
        container_mounts=["/lustre:/lustre"],
        env_vars={
            "HF_HOME": args.hf_home,
            "HF_DATASETS_CACHE": args.hf_datasets_cache,
            "HF_HUB_OFFLINE": args.hf_hub_offline,
            "HF_TOKEN": args.hf_token,
            "NRL_DEEPSCALER_16K_CKPT": args.nrl_deepscaler_16k_ckpt,
            "NRL_DEEPSCALER_24K_CKPT": args.nrl_deepscaler_24k_ckpt,
            "NRL_DEEPSCALER_8K_CKPT": args.nrl_deepscaler_8k_ckpt,
            "WANDB_API_KEY": args.wandb_api_key,
        },
        tunnel=SSHTunnel(
            host=args.host,
            user=args.user,
            job_dir=f"{args.nemorun_home}/job_dir",
            identity=args.identity,
        ),
    )

    job = RayJob(name=f"{args.job_name}-{args.ci_job_id}", executor=executor)
    job.start(
        command=f"cd /opt/nemo-rl && git init && {pytest_command}",
        workdir=f"{args.nemorun_home}/work_dir",
    )
    job.logs(follow=True, timeout=60 * 60 * 24)
    if job.status()["state"] != "COMPLETED":
        exit(1)


if __name__ == "__main__":
    main()
