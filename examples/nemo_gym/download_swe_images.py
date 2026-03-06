import argparse
import asyncio
import os
import shutil
import tempfile

from datasets import load_dataset


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
        try:
            proc = await asyncio.create_subprocess_exec(
                "singularity",
                "pull",
                local_image_path,
                f"docker://{image_name}",
                env={**os.environ, "APPTAINER_CACHEDIR": cache_dir, "APPTAINER_TMPDIR": temp_dir},
            )
            await proc.communicate()
            if proc.returncode != 0:
                print(f"FAILED (rc={proc.returncode}): {image_name}", flush=True)
                if os.path.exists(local_image_path):
                    os.remove(local_image_path)
            else:
                print(f"Pulled: {image_name}", flush=True)
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)


async def install_apptainer():
    """
    Install Apptainer on the system.
    """
    cmd = (
        "apt-get update"
        " && apt-get install -y git build-essential gcc wget"
        " && wget -O /tmp/apptainer_1.3.1_amd64.deb"
        " https://github.com/apptainer/apptainer/releases/download/v1.3.1/apptainer_1.3.1_amd64.deb"
        " && apt install -y /tmp/apptainer_1.3.1_amd64.deb"
    )
    proc = await asyncio.create_subprocess_exec("bash", "-c", cmd)
    await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"install_apptainer failed (rc={proc.returncode})")


async def main(sif_dir: str, concurrency: int):
    semaphore = asyncio.Semaphore(concurrency)

    # Install apptainer
    await install_apptainer()

    # Pull images
    image_sources = [
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

    for source in image_sources:
        ds = load_dataset(source["dataset"], "default", split=source["split"])
        image_names = source["images"](ds)
        tasks = [pull_image(semaphore, name, sif_dir, source["prefix"]) for name in image_names]
        await asyncio.gather(*tasks)
        print(f"Pulled {len(image_names)} images for {source['name']}")


if __name__ == "__main__":
    """
    uv run --with datasets pull_super_images.py --sif-dir /path/to/sif --concurrency 16

    # add paths to config in Nemo-RL
    container_formatter:
    - "/path/to/sif/r2egym_{instance_id}.sif"
    - "/path/to/sif/swegym_sweb.eval.x86_64.{instance_id}.sif"
    - "/path/to/sif/swebench_sweb.eval.x86_64.{instance_id}.sif"
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--sif-dir", type=str, required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()

    sif_dir = args.sif_dir
    concurrency = args.concurrency

    asyncio.run(main(sif_dir, concurrency))
