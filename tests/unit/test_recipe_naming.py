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
"""A recipe's filename is how we describe what it covers, so it has to be true.

The name is the only place that says a test exists to exercise sequence packing,
or that two tests are a deliberate pair differing by one knob. Nothing else
records it. That only works if the name cannot drift from the config, which is
what this module checks.

Grammar::

    <algo>-<model>-<N>n<M>g-<backend[parallelism]>[-<feature>...][.vN]

The mandated sections -- algo, model, cluster tuple, backend -- are checked
**both ways**: the name must match the config and the config must match the
name. Feature tokens are checked **one way only**: if the name claims a feature
the config must have it, but a config may use a feature without advertising it.
That asymmetry is deliberate. Of the recipes checked here, 64 enable sequence
packing and 8 say ``seqpack``, because the token means "this test exists to
cover packing", not "packing happens to be on".
"""

import glob
import os
import re

import pytest
import yaml

dir_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(dir_path, "..", ".."))
recipes_dir = os.path.join(project_root, "examples", "configs", "recipes")
test_suites_dir = os.path.join(project_root, "tests", "test_suites")

# Directories exempt from the grammar.
#
# performance/ recipes are tuned for peak throughput and are named for the shape
# being benchmarked rather than the backend; their hyperparameters are frozen, so
# renaming them to satisfy a regex would break the perf series for no gain.
# reproduce/ recipes encode what the reproduction varies -- the DeepScaleR chain
# puts its context length where the cluster tuple would go -- and the
# parallelism is not the point.
EXEMPT_DIRS = ("performance", "reproduce")

ALGOS = {
    "sft",
    "dpo",
    "grpo",
    "vlm_grpo",
    "distillation",
    "rm",
    "dapo",
    "prorlv2",
    "ppo",
    "mopd",
    "gdpo",
}

# A backend token, optionally fused with its parallelism (fsdp2tp1,
# megatrontp2pp2). Both the fused and the separated form are accepted because
# both are already in wide use and neither is wrong.
BACKEND_RE = re.compile(
    r"^(megatron\w*|mcore\w*|fsdp\d*\w*|dtensor\d*\w*|automodel|hsdp\w*)$", re.I
)
MEGATRON_TOKENS = ("megatron", "mcore")
DTENSOR_TOKENS = ("fsdp", "dtensor", "automodel", "hsdp")

# Feature tokens whose presence in a name makes a claim we can verify. A token
# maps to every key that could satisfy it -- activation checkpointing is
# configured under whichever backend the recipe uses, so a DTensor recipe named
# actckpt sets it under dtensor_cfg, not megatron_cfg.
FEATURE_CLAIMS = {
    "seqpack": ("policy.sequence_packing.enabled",),
    "pack": ("policy.sequence_packing.enabled",),
    "dynamicbatch": ("policy.dynamic_batching.enabled",),
    "actckpt": (
        "policy.megatron_cfg.activation_checkpointing",
        "policy.dtensor_cfg.activation_checkpointing",
    ),
}


def _load_merged(path, depth=0):
    if depth > 8 or not os.path.exists(path):
        return {}
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("defaults", None)
    if not parent:
        return cfg
    resolved = os.path.normpath(os.path.join(os.path.dirname(path), parent))
    return _deep_merge(_load_merged(resolved, depth + 1), cfg)


def _deep_merge(a, b):
    out = dict(a)
    for key, value in b.items():
        if isinstance(out.get(key), dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _get(cfg, dotted):
    current = cfg
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _config_value(script_path, key):
    in_block = False
    with open(script_path) as f:
        for line in f:
            if re.match(r"^# =+ BEGIN CONFIG =+", line):
                in_block = True
                continue
            if re.match(r"^# =+ END CONFIG =+", line):
                break
            if in_block and line.startswith(f"{key}="):
                return line[len(key) + 1 :].split("#")[0].strip().strip("\"'")
    return None


def _parse(stem):
    """Split a recipe name into its grammar sections, or return the failure."""
    # The version marker belongs to the base recipe, so a wrapper that appends
    # its own token lands after it (…-fsdp2tp1.v3-tq_simple). Strip it wherever
    # it appears rather than assuming it is terminal.
    rest = re.sub(r"\.v\d+(?=-|$)", "", stem)
    algo = None
    for candidate in sorted(ALGOS, key=len, reverse=True):
        if rest.startswith(f"{candidate}-"):
            algo, rest = candidate, rest[len(candidate) + 1 :]
            break
    if algo is None:
        return None, "does not start with a known algorithm"

    match = re.search(r"-(\d+)n(\d+)g(?=-|$)", f"-{rest}")
    if not match:
        return None, "has no <N>n<M>g cluster tuple"
    tuple_token = match.group(0)[1:]
    index = rest.find(tuple_token)
    model = rest[: index - 1] if index > 0 else ""
    after = rest[index + len(tuple_token) :].lstrip("-")
    tokens = [t for t in after.split("-") if t]
    if not tokens:
        return None, "has no backend token after the cluster tuple"
    if not BACKEND_RE.match(tokens[0]):
        return None, f"has {tokens[0]!r} where a backend token belongs"
    return (
        {
            "algo": algo,
            "model": model,
            "nodes": int(match.group(1)),
            "gpus": int(match.group(2)),
            "backend": tokens[0],
            "features": tokens[1:],
        },
        None,
    )


def _recipes():
    for path in sorted(
        glob.glob(os.path.join(recipes_dir, "**", "*.yaml"), recursive=True)
    ):
        rel = os.path.relpath(path, recipes_dir)
        if rel.split(os.sep)[1:2] and rel.split(os.sep)[1] in EXEMPT_DIRS:
            continue
        if rel.split(os.sep)[0] in EXEMPT_DIRS:
            continue
        yield rel, path


RECIPES = list(_recipes())
assert RECIPES, "no recipes found"


@pytest.mark.parametrize("rel,path", RECIPES, ids=[r for r, _ in RECIPES])
def test_recipe_name_parses(rel, path):
    parsed, problem = _parse(os.path.basename(path)[: -len(".yaml")])
    assert parsed is not None, (
        f"{rel} {problem}.\n"
        f"Expected <algo>-<model>-<N>n<M>g-<backend[parallelism]>[-<feature>...][.vN], "
        f"e.g. grpo-llama3.1-8b-instruct-1n8g-megatron-fp8."
    )


@pytest.mark.parametrize("rel,path", RECIPES, ids=[r for r, _ in RECIPES])
def test_recipe_name_matches_its_config(rel, path):
    parsed, problem = _parse(os.path.basename(path)[: -len(".yaml")])
    if parsed is None:
        pytest.skip(f"name does not parse: {problem}")

    cfg = _load_merged(path)
    driver = os.path.join(test_suites_dir, rel[: -len(".yaml")] + ".sh")
    problems = []

    # --- mandated sections: both directions ---
    declared_nodes = _config_value(driver, "NUM_NODES")
    if declared_nodes is not None and int(declared_nodes) != parsed["nodes"]:
        problems.append(
            f"name says {parsed['nodes']} node(s) but the driver sets NUM_NODES={declared_nodes}"
        )

    declared_gpus = int(_config_value(driver, "GPUS_PER_NODE") or 8)
    if declared_gpus != parsed["gpus"]:
        problems.append(
            f"name says {parsed['gpus']} GPUs per node but the driver sets GPUS_PER_NODE={declared_gpus}"
        )

    config_gpus = _get(cfg, "cluster.gpus_per_node")
    if config_gpus is not None and config_gpus != parsed["gpus"]:
        problems.append(
            f"name says {parsed['gpus']} GPUs per node but cluster.gpus_per_node={config_gpus}"
        )

    megatron = bool(_get(cfg, "policy.megatron_cfg.enabled"))
    dtensor = bool(_get(cfg, "policy.dtensor_cfg.enabled"))
    backend = parsed["backend"].lower()
    if backend.startswith(MEGATRON_TOKENS) and not megatron:
        problems.append(
            f"name says {parsed['backend']!r} but megatron_cfg.enabled is not set"
        )
    if backend.startswith(DTENSOR_TOKENS) and not dtensor:
        problems.append(
            f"name says {parsed['backend']!r} but dtensor_cfg.enabled is not set"
        )

    # --- feature tokens: one direction only (see module docstring) ---
    for feature in parsed["features"]:
        keys = FEATURE_CLAIMS.get(feature.lower())
        if keys and not any(_get(cfg, key) for key in keys):
            problems.append(
                f"name claims {feature!r} but none of {', '.join(keys)} is enabled"
            )

    assert not problems, f"{rel}:\n  " + "\n  ".join(problems)
