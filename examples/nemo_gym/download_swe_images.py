#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "datasets>=2.19.0",
# ]
# ///

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

"""
Downloads SWE-bench evaluation containers as Apptainer .sif files for use
with the OpenHands SWE agent in NeMo RL training.

Images are pulled from three HuggingFace datasets:
  - R2E-Gym/R2E-Gym-Subset
  - SWE-Gym/SWE-Gym
  - princeton-nlp/SWE-bench_Verified

Each image is saved as <prefix>_<image_tag>.sif under the output directory.
Pass SIF_DIR=/path/to/sif when launching super_launch.sh for Stage 2.2
and the container_formatter paths will be set automatically.

Usage:
    chmod +x examples/nemo_gym/download_swe_images.py
    ./examples/nemo_gym/download_swe_images.py --sif-dir /path/to/sif [--concurrency 16]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import tempfile

from datasets import load_dataset


IMAGE_SOURCES = [
    {
        "name": "R2E-Gym",
        "dataset": "R2E-Gym/R2E-Gym-Subset",
        "split": "train",
        "prefix": "r2egym",
        "images": lambda ds: [row["docker_image"] for row in ds],
    },
    {
        "name": "SWE-Gym",
        "dataset": "SWE-Gym/SWE-Gym",
        "split": "train",
        "prefix": "swegym",
        "images": lambda ds: [
            f"xingyaoww/sweb.eval.x86_64.{row['instance_id'].replace('__', '_s_')}" for row in ds
        ],
    },
    {
        "name": "SWE-Bench Verified",
        "dataset": "princeton-nlp/SWE-bench_Verified",
        "split": "test",
        "prefix": "swebench",
        "images": lambda ds: [
            f"swebench/sweb.eval.x86_64.{row['instance_id'].replace('__', '_1776_')}" for row in ds
        ],
    },
]


async def pull_image(
    semaphore: asyncio.Semaphore,
    image_name: str,
    sif_dir: str,
    prefix: str,
):
    async with semaphore:
        local_image_path = os.path.join(sif_dir, f"{prefix}_{image_name.split('/')[-1]}.sif")
        if os.path.exists(local_image_path):
            print(f"Already exists: {local_image_path}", flush=True)
            return

        print(f"Pulling image: {image_name}", flush=True)

        cache_dir = tempfile.mkdtemp(prefix="apptainer_cache_")
        temp_dir = tempfile.mkdtemp(prefix="apptainer_tmp_")
        tmp_sif = local_image_path + ".tmp"
        try:
            proc = await asyncio.create_subprocess_exec(
                "apptainer",
                "build",
                tmp_sif,
                f"docker://{image_name}",
                env={**os.environ, "APPTAINER_CACHEDIR": cache_dir, "APPTAINER_TMPDIR": temp_dir},
            )
            await proc.communicate()
            if proc.returncode != 0:
                print(f"FAILED (rc={proc.returncode}): {image_name}", flush=True)
            else:
                os.rename(tmp_sif, local_image_path)
                print(f"Pulled: {image_name}", flush=True)
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)
            if os.path.exists(tmp_sif):
                os.remove(tmp_sif)


async def main(sif_dir: str, concurrency: int) -> None:
    os.makedirs(sif_dir, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)

    for source in IMAGE_SOURCES:
        print(f"\n=== {source['name']} ({source['dataset']}) ===", flush=True)
        ds = load_dataset(source["dataset"], "default", split=source["split"])
        image_names = source["images"](ds)
        tasks = [pull_image(semaphore, name, sif_dir, source["prefix"]) for name in image_names]
        await asyncio.gather(*tasks)
        print(f"Done: {len(image_names)} images for {source['name']}", flush=True)


if __name__ == "__main__":
    if shutil.which("apptainer") is None:
        print(
            "Error: 'apptainer' not found on PATH.\n"
            "Install Apptainer (https://apptainer.org/docs/admin/main/installation.html) "
            "before running this script.",
            file=__import__("sys").stderr,
        )
        raise SystemExit(1)

    parser = argparse.ArgumentParser(
        description="Download SWE-bench evaluation containers as Apptainer .sif files."
    )
    parser.add_argument(
        "--sif-dir",
        type=str,
        required=True,
        help="Directory to save .sif files",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Number of parallel image pulls (default: 16)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.sif_dir, args.concurrency))
