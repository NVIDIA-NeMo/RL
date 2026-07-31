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

"""Prefetch NeMo Gym internal venvs by doing a dry run of NemoGym initialization.

This complements nemo_rl/utils/prefetch_venvs.py (which prefetches Ray actor venvs)
by also triggering NeMo Gym's own internal venv creation for its servers (code_gen,
math, etc.). It reuses the real code path (NemoGym -> _spinup) with dry_run=True
so no actual policy model is needed.

Gym's per-server venvs are built as a side effect of ``NemoGym._spinup()`` into the
node-local container filesystem, so a single actor only covers whichever node it
happened to land on. Because the Gym runners are SPREAD-placed, any uncovered node
fails every rollout it receives with ``broken py_executable`` -> HTTP 500 -> a
zero-reward back-fill, which silently depresses reward instead of erroring out.
We therefore spin up one NemoGym actor *per node* under a STRICT_SPREAD placement
group, mirroring ``create_local_venv_on_each_node``, and assert coverage afterwards.
"""

import argparse
import glob
import os
import sys

import ray
from omegaconf import OmegaConf
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.nemo_gym import (
    NemoGym,
    NemoGymConfig,
    get_nemo_gym_uv_cache_dir,
    get_nemo_gym_venv_dir,
)
from nemo_rl.utils.config import load_config
from nemo_rl.utils.venvs import create_local_venv_on_each_node

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)


def _alive_nodes() -> list[dict]:
    """Ray nodes eligible to host a prefetch actor.

    Skips nodes with 0 CPUs (e.g. unschedulable head nodes) — including them
    makes the STRICT_SPREAD placement group infeasible. Same filter as
    ``nemo_rl.utils.venvs.create_local_venv_on_each_node``.
    """
    return [
        n
        for n in ray.nodes()
        if n.get("Alive", False) and n.get("Resources", {}).get("CPU", 0) > 0
    ]


def _gym_venv_root() -> str:
    """Directory that Gym builds its per-server venvs under.

    With NEMO_GYM_VENV_DIR set, Gym builds into that directory; otherwise it
    builds in-tree next to each server. Both layouts nest the venv as
    ``<root>/<category>/<server>/.venv`` (e.g. ``responses_api_agents/swe_agents``).
    """
    venv_dir = get_nemo_gym_venv_dir()
    if venv_dir:
        return venv_dir
    # <repo>/examples/nemo_gym/prefetch_venvs.py -> <repo>/3rdparty/Gym-workspace/Gym
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    return os.path.join(repo_root, "3rdparty", "Gym-workspace", "Gym")


@ray.remote(num_cpus=1)
def _list_gym_venvs(root: str) -> list[str]:
    """Return the Gym server venvs that exist on the node running this task.

    Matches both the nested ``<category>/<server>/.venv`` layout and a flat
    ``<server>/.venv`` one, so this does not depend on which root applies.
    """
    found = set()
    for pattern in ("*/.venv/bin/python", "*/*/.venv/bin/python"):
        for p in glob.glob(os.path.join(root, pattern)):
            found.add(os.path.relpath(p, root))
    return sorted(found)


def _assert_venvs_on_every_node(pg, num_nodes: int) -> None:
    """Fail loudly unless every node ended up with the same, non-empty venv set.

    Reported success with only partial coverage is the exact failure this guards:
    it does not crash the run, it just silently turns half the rollouts into
    zero-reward trajectories.
    """
    root = _gym_venv_root()
    per_node = ray.get(
        [
            _list_gym_venvs.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=i
                )
            ).remote(root)
            for i in range(num_nodes)
        ]
    )

    print(f"\nGym venv coverage under {root}:")
    for i, venvs in enumerate(per_node):
        print(f"  node[{i}]: {venvs if venvs else '<NONE>'}")

    empty = [i for i, v in enumerate(per_node) if not v]
    if empty:
        raise RuntimeError(
            f"NeMo Gym venv prefetch left {len(empty)}/{num_nodes} node(s) with no venv "
            f"under {root} (node indices {empty}). Rollouts scheduled onto those nodes "
            "would fail with 'broken py_executable' and be back-filled as zero reward."
        )

    distinct = {tuple(v) for v in per_node}
    if len(distinct) > 1:
        raise RuntimeError(
            f"NeMo Gym venv prefetch produced inconsistent venv sets across nodes: {distinct}"
        )


def prefetch_nemo_gym_venvs(config_paths: list[str]) -> None:
    """Prefetch NeMo Gym venvs for each config by doing a dry-run initialization.

    Args:
        config_paths: List of paths to NeMo RL config files that contain
            an env.nemo_gym section.
    """
    init_ray()

    nemo_gym_py_exec = get_actor_python_env("nemo_rl.environments.nemo_gym.NemoGym")
    if nemo_gym_py_exec.startswith("uv"):
        nemo_gym_py_exec = create_local_venv_on_each_node(
            nemo_gym_py_exec, "nemo_rl.environments.nemo_gym.NemoGym"
        )

    nodes = _alive_nodes()
    num_nodes = len(nodes)
    if num_nodes == 0:
        raise RuntimeError("No alive Ray nodes with CPUs available for venv prefetch.")
    print(f"Prefetching NeMo Gym venvs on {num_nodes} node(s).")

    # Reserve one CPU per node so each NemoGym actor is pinned to a distinct node.
    pg = placement_group(bundles=[{"CPU": 1}] * num_nodes, strategy="STRICT_SPREAD")
    ray.get(pg.ready())

    succeeded = []
    failed = []

    try:
        _prefetch_configs(
            config_paths, nemo_gym_py_exec, pg, num_nodes, succeeded, failed
        )
    finally:
        ray.util.remove_placement_group(pg)

    print(f"\n{'=' * 60}")
    print("NeMo Gym venv prefetch summary")
    print("=" * 60)
    print(f"  Succeeded: {len(succeeded)}")
    for path in succeeded:
        print(f"    - {path}")
    if failed:
        print(f"  Failed: {len(failed)}")
        for path, err in failed:
            print(f"    - {path}: {err}")

    if failed:
        sys.exit(1)


def _prefetch_configs(
    config_paths: list[str],
    nemo_gym_py_exec: str,
    pg,
    num_nodes: int,
    succeeded: list,
    failed: list,
) -> None:
    for config_path in config_paths:
        print(f"\n{'=' * 60}")
        print(f"Processing config: {config_path}")
        print("=" * 60)

        try:
            config = load_config(config_path)
            config = OmegaConf.to_container(config, resolve=True)

            # Recipes pick up env.nemo_gym through their `defaults:` chain, so
            # only the resolved config can answer this. Skipping here (rather
            # than erroring) lets callers run this unconditionally for any recipe.
            if "nemo_gym" not in (config.get("env") or {}):
                print(f"No env.nemo_gym section; skipping {config_path}")
                continue

            nemo_gym_dict = dict(config["env"]["nemo_gym"])
            nemo_gym_dict["dry_run"] = True
            uv_cache_dir = get_nemo_gym_uv_cache_dir()
            if uv_cache_dir is not None:
                nemo_gym_dict.setdefault("uv_cache_dir", uv_cache_dir)
            uv_venv_dir = get_nemo_gym_venv_dir()
            if uv_venv_dir is not None:
                nemo_gym_dict.setdefault("uv_venv_dir", uv_venv_dir)

            nemo_gym_cfg = NemoGymConfig(
                model_name="dummy-model",
                base_urls=["http://localhost:8000"],
                ray_gpu_nodes=[],
                ray_gpu_pgs=[],
                ray_num_gpus_per_node=0,
                ray_namespace=None,
                initial_global_config_dict=nemo_gym_dict,
                invalid_tool_call_patterns=None,
            )

            nemo_gym_opts = {
                # Don't restart to surface any issue from a failed build.
                "max_restarts": 0,
                "max_task_retries": 0,
                "runtime_env": {
                    "py_executable": nemo_gym_py_exec,
                    "env_vars": {
                        **os.environ,
                        "VIRTUAL_ENV": nemo_gym_py_exec,
                        "UV_PROJECT_ENVIRONMENT": nemo_gym_py_exec,
                    },
                },
            }

            print(
                f"Creating {num_nodes} NeMo Gym environment(s) (dry_run=True), one per node..."
            )
            actors = [
                NemoGym.options(
                    **nemo_gym_opts,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg, placement_group_bundle_index=i
                    ),
                ).remote(nemo_gym_cfg)
                for i in range(num_nodes)
            ]

            try:
                print("Waiting for NeMo Gym to finish initialization on every node...")
                # A build failure on any single node fails the whole prefetch.
                ray.get([a._spinup.remote() for a in actors])
                print("NeMo Gym initialized successfully on every node.")
            finally:
                print("Killing NeMo Gym actors...")
                for a in actors:
                    ray.kill(a)

            _assert_venvs_on_every_node(pg, num_nodes)

            succeeded.append(config_path)
            print(f"Done with config: {config_path}")

        except Exception as e:
            print(f"Error processing {config_path}: {e}")
            failed.append((config_path, str(e)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prefetch NeMo Gym internal venvs via dry-run initialization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Prefetch venvs for a single config
  uv run python examples/nemo_gym/prefetch_venvs.py \\
    examples/nemo_gym/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml

  # Prefetch venvs for multiple configs sequentially
  uv run python examples/nemo_gym/prefetch_venvs.py \\
    examples/nemo_gym/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml \\
    examples/nemo_gym/grpo_qwen3_30ba3b_instruct.yaml
""",
    )
    parser.add_argument(
        "configs",
        nargs="+",
        help="One or more NeMo RL config file paths containing an env.nemo_gym section.",
    )
    args = parser.parse_args()

    prefetch_nemo_gym_venvs(args.configs)
