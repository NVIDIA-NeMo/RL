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

"""Regression guard for the MXFP8 flashinfer trtllm-gen MoE *fast path*.

The Nemotron-Super MXFP8 RL recipes only run on the optimized MoE kernel when a
small, fragile set of signals are all present. They live in two places:

  * the recipe's test-suite shell script
    (``tests/test_suites/llm/<recipe>.sh``) -- a few ``export`` lines plus a
    ``run_grpo.py`` CLI override, and
  * the recipe YAML (``examples/configs/recipes/llm/<recipe>.yaml``, resolved
    through its ``defaults:`` inheritance chain).

Historically, dropping any one of these silently fell back to a slow / bf16 MoE
path: generation stayed *correct* (every request still returned), so only a
throughput regression ever revealed it. These tests lock today's known-good
values so such a drop fails loudly in CI instead.

These are pure file-parsing / config-resolution checks -- no GPU, no model
download -- so they run in the default unit-test lane. The matching *runtime*
perf-floor guard lives in the nightly suite (see
``tests/test_suites/llm/grpo-nemotron-super-*-mxfp8.sh`` and
``tests/check_moe_fast_path.py``).
"""

import os
import re
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_rl.utils.config import (
    load_config_with_inheritance,
    register_omegaconf_resolvers,
)

register_omegaconf_resolvers()

# Repo root: this file is tests/unit/models/generation/test_moe_fast_path_guard.py
REPO_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).joinpath(
    "..", "..", "..", ".."
).resolve()
SUITE_DIR = REPO_ROOT / "tests" / "test_suites" / "llm"
RECIPE_DIR = REPO_ROOT / "examples" / "configs" / "recipes" / "llm"

# Recipes whose generation path is *supposed* to use the MXFP8 flashinfer
# trtllm-gen MoE kernel. Add new MXFP8 recipes here so they inherit the guard.
MXFP8_FAST_PATH_RECIPES = [
    "grpo-nemotron-super-8n8g-mxfp8",
    "grpo-nemotron-super-16n4g-mxfp8",
]

# --- known-good shell-script signals (env exports + CLI overrides) -----------
# Each engages part of the fast path; the message explains the failure mode.
REQUIRED_ENV_EXPORTS = {
    "NRL_VLLM_USE_V1": "1",
    "VLLM_USE_FLASHINFER_MOE_FP8": "1",
    # 'latency' selects the trtllm-gen low-latency MoE tactic; dropping it (or
    # switching to 'throughput') falls back to a slower MoE backend.
    "VLLM_FLASHINFER_MOE_BACKEND": "latency",
}
REQUIRED_CLI_OVERRIDE = "policy.generation.vllm_kwargs.attention_backend=FLASH_ATTN"


def _read_suite_script(recipe: str) -> str:
    path = SUITE_DIR / f"{recipe}.sh"
    if not path.is_file():
        # The Nemotron MXFP8 recipes are not present in every tree (e.g. public
        # upstream). Skip rather than fail so the guard is CI-safe everywhere;
        # it actively guards wherever the recipe exists.
        pytest.skip(f"MXFP8 recipe script not present: {path}")
    return path.read_text()


def _resolved_config(recipe: str) -> dict:
    path = RECIPE_DIR / f"{recipe}.yaml"
    if not path.is_file():
        pytest.skip(f"MXFP8 recipe config not present: {path}")
    cfg = load_config_with_inheritance(str(path))
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert resolved is not None, f"Recipe config resolved to empty: {path}"
    return resolved


def _parse_exports(script_text: str) -> dict:
    """Return {VAR: VALUE} for every ``export VAR=VALUE`` line in the script."""
    exports = {}
    for m in re.finditer(
        r"^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=([^\s#]+)", script_text, re.MULTILINE
    ):
        exports[m.group(1)] = m.group(2).strip().strip("\"'")
    return exports


@pytest.mark.parametrize("recipe", MXFP8_FAST_PATH_RECIPES)
def test_mxfp8_suite_script_engages_fast_path(recipe):
    """The recipe's .sh must export the flashinfer-MoE env vars + FLASH_ATTN.

    These env vars and the attention-backend override are how generation is
    routed onto the trtllm-gen MXFP8 MoE kernel. Dropping any of them silently
    re-disables the optimized path.
    """
    text = _read_suite_script(recipe)
    exports = _parse_exports(text)

    for var, expected in REQUIRED_ENV_EXPORTS.items():
        assert var in exports, (
            f"{recipe}.sh no longer exports {var}. This silently drops the "
            f"MXFP8 flashinfer trtllm-gen MoE fast path back to a slow MoE "
            f"backend. Re-add: export {var}={expected}"
        )
        assert exports[var] == expected, (
            f"{recipe}.sh sets {var}={exports[var]!r}, expected {expected!r}. "
            f"This changes the MoE path away from the known-good fast path."
        )

    assert REQUIRED_CLI_OVERRIDE in text, (
        f"{recipe}.sh no longer passes '{REQUIRED_CLI_OVERRIDE}' to "
        f"run_grpo.py. FLASH_ATTN is required for the optimized generation path."
    )


@pytest.mark.parametrize("recipe", MXFP8_FAST_PATH_RECIPES)
def test_mxfp8_recipe_config_engages_fast_path(recipe):
    """The resolved recipe YAML must keep the MXFP8 MoE fast-path invariants.

    Values are snapshotted from the known-good resolved config. A drift here
    means an edit to the recipe (or to a config it inherits from) changed a
    load-bearing fast-path setting.
    """
    cfg = _resolved_config(recipe)
    gen = cfg["policy"]["generation"]
    vllm_cfg = gen["vllm_cfg"]
    vllm_kwargs = gen.get("vllm_kwargs", {})

    # Generation must go through vLLM.
    assert gen.get("backend") == "vllm", (
        f"{recipe}: generation backend is {gen.get('backend')!r}, expected "
        f"'vllm' -- the MXFP8 flashinfer MoE path only exists under vLLM."
    )

    # MXFP8 weights, not bf16: precision=fp8 + is_mx engages the MX path.
    assert vllm_cfg.get("precision") == "fp8", (
        f"{recipe}: vllm_cfg.precision is {vllm_cfg.get('precision')!r}, "
        f"expected 'fp8'. Without it generation runs bf16, not MXFP8."
    )
    assert gen.get("fp8_cfg", {}).get("is_mx") is True, (
        f"{recipe}: fp8_cfg.is_mx is {gen.get('fp8_cfg', {}).get('is_mx')!r}, "
        f"expected True. is_mx=False selects plain FP8, not the MXFP8 path."
    )

    # No expert-parallel at generation: EP>1 measurably regresses small-M MoE.
    assert vllm_cfg.get("expert_parallel_size") == 1, (
        f"{recipe}: vllm_cfg.expert_parallel_size is "
        f"{vllm_cfg.get('expert_parallel_size')!r}, expected 1. Enabling EP "
        f"regresses the small-M MoE throughput this recipe depends on."
    )
    assert vllm_cfg.get("pipeline_parallel_size") == 1, (
        f"{recipe}: vllm_cfg.pipeline_parallel_size is "
        f"{vllm_cfg.get('pipeline_parallel_size')!r}, expected 1."
    )

    # Mamba SSM cache must stay fp32 (correctness + the tuned cache path).
    assert vllm_kwargs.get("mamba_ssm_cache_dtype") == "float32", (
        f"{recipe}: vllm_kwargs.mamba_ssm_cache_dtype is "
        f"{vllm_kwargs.get('mamba_ssm_cache_dtype')!r}, expected 'float32'."
    )

    # Known-good cudagraph mode for this recipe (locked to catch drift).
    compilation_config = vllm_kwargs.get("compilation_config", {}) or {}
    assert compilation_config.get("cudagraph_mode") == 0, (
        f"{recipe}: vllm_kwargs.compilation_config.cudagraph_mode is "
        f"{compilation_config.get('cudagraph_mode')!r}, expected 0 "
        f"(the known-good value for this MXFP8 recipe)."
    )
