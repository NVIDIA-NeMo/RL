# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "native_mxfp8_source_refit"


def test_native_smoke_overlays_enable_runtime_inventory_assertion() -> None:
    for model_scope in ("qwen30", "nano"):
        config = (EXPERIMENT_DIR / f"{model_scope}-fp8param-true.yaml").read_text()
        assert f"native_mxfp8_storage_assertion: {model_scope}" in config

    nano_config = (EXPERIMENT_DIR / "nano-fp8param-false.yaml").read_text()
    assert "te_precision_config_file: experiments/native_mxfp8_source_refit/te_nano_routed.yaml" in nano_config
    assert "first_last_layers_bf16: true" in nano_config
    assert "num_layers_at_end_in_bf16: 8" in nano_config


def test_ray_tmpdir_is_resolved_before_ray_head_and_workers_start() -> None:
    launcher = (EXPERIMENT_DIR / "submit_oci_hsg.sh").read_text()
    ray_sub = (REPO_ROOT / "ray.sub").read_text()

    assert "export RAY_TMPDIR_ROOT=" in launcher
    assert "export RAY_TMPDIR=" not in launcher
    assert ray_sub.index("export RAY_TMPDIR=") < ray_sub.index("ray start --head")
    assert ray_sub.index("export RAY_TMPDIR=") < ray_sub.index("ray start --address")


def test_launcher_exports_resolved_slurm_helper_path_to_batch_shell() -> None:
    launcher = (EXPERIMENT_DIR / "submit_oci_hsg.sh").read_text()

    assert "resolve_slurm_helper_path()" in launcher
    assert 'readlink -f "${helper_path}"' in launcher
    assert "realpath" in launcher
    assert "SLURM_HELPER_PATH=$(resolve_slurm_helper_path)" in launcher
    assert '--export="ALL,SLURM_HELPER_PATH=${SLURM_HELPER_PATH}"' in launcher
    assert "Required Slurm helper ${helper} not found" in launcher


def test_ray_sub_bootstraps_slurm_helper_path_before_queries() -> None:
    ray_sub = (REPO_ROOT / "ray.sub").read_text()

    path_bootstrap = 'export PATH="${SLURM_HELPER_PATH:-/usr/local/bin:/usr/bin:/bin:/cm/shared/apps/slurm/current/bin}:${PATH:-}"'
    assert path_bootstrap in ray_sub
    assert ray_sub.index(path_bootstrap) < ray_sub.index("maybe_gres_arg()")
