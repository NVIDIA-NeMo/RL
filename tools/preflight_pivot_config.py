#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline preflight for a NeMo-Gym resources-server config + its data.

Catches the config mistakes that otherwise only surface ~4 minutes into a
multi-node job, as an "almost-server" that is silently skipped:

  * dataset entries missing the `license` field (required for train/validation;
    without it pydantic falls through to BenchmarkDatasetConfig and the server
    never starts)
  * jsonl paths that do not exist or are empty
  * rows whose `agent_ref.name` does not match an agent defined in the config
  * image paths inside the rows that do not resolve on disk

It AST-extracts the REAL DatasetConfig out of nemo_gym/config_types.py so the
license/type rules cannot drift from the framework. Only needs pydantic + PyYAML,
so it runs on a login node without the training container.

Usage:
    python tools/preflight_pivot_config.py \
        3rdparty/Gym-workspace/Gym/resources_servers/image_tools/configs/image_tools_pivot.yaml
"""

from __future__ import annotations

import ast
import json
import os
import sys
import types

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GYM = os.path.join(REPO, "3rdparty", "Gym-workspace", "Gym")
CONFIG_TYPES = os.path.join(GYM, "nemo_gym", "config_types.py")

NEEDED = {
    "DatasetType",
    "JsonlDatasetGitlabIdentifer",
    "JsonlDatasetHuggingFaceIdentifer",
    "GitlabDatasetSource",
    "HuggingFaceDatasetSource",
    "DatasetSource",
    "DatasetConfig",
}


def load_real_dataset_config():
    """Pull DatasetConfig (and its deps) out of config_types.py without importing it."""
    tree = ast.parse(open(CONFIG_TYPES).read())
    keep = []
    for node in tree.body:
        name = None
        if isinstance(node, ast.ClassDef):
            name = node.name
        elif (
            isinstance(node, ast.Assign)
            and node.targets
            and isinstance(node.targets[0], ast.Name)
        ):
            name = node.targets[0].id
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        if name in NEEDED:
            keep.append(node)

    mod = types.ModuleType("_cfg_types")
    preamble = (
        "import warnings\n"
        "from typing import Annotated, Literal, Optional, Union, List, Dict, Any\n"
        "from pydantic import BaseModel, Field, model_validator\n"
    )
    exec(preamble, mod.__dict__)
    exec(ast.unparse(ast.Module(body=keep, type_ignores=[])), mod.__dict__)
    if "DatasetConfig" not in mod.__dict__:
        raise SystemExit("FATAL: could not extract DatasetConfig from config_types.py")
    # config_types.py defers annotations, so pydantic needs an explicit rebuild
    # against this namespace before the model is usable.
    mod.DatasetConfig.model_rebuild(_types_namespace=mod.__dict__)
    return mod.DatasetConfig


# Input content parts accepted by ResponseInputMessageContentListParam.
# `output_text` is an OUTPUT part and is not valid anywhere in `input`.
INPUT_PART_TYPES = {"input_text", "input_image", "input_file"}


def validate_input_item(msg: dict) -> str | None:
    """Check one `responses_create_params.input` item against the Gym union.

    Rules read off nemo_gym/openai_utils.py:150-169 --

      NeMoGymResponseOutputMessage : requires `id` (no default; only the model
                                     server mints one), so a hand-authored
                                     assistant turn can never satisfy it.
      NeMoGymEasyInputMessage      : content is `str` OR a list of *input*
                                     content parts.
      NeMoGymMessage               : role user/system/developer only.

    So a prefix assistant turn must use STRING content. A
    [{"type": "output_text", ...}] list matches neither branch and fails the
    whole request with a confusing union error listing `call_id: Field required`.
    """
    if not isinstance(msg, dict):
        return f"input item is {type(msg).__name__}, expected dict"
    role = msg.get("role")
    if role not in ("user", "assistant", "system", "developer"):
        return f"bad role {role!r}"
    content = msg.get("content")

    if isinstance(content, str):
        return None
    if not isinstance(content, list):
        return f"role={role} content is {type(content).__name__}, expected str or list"

    if role == "assistant":
        return (
            "role=assistant with list content: only string content validates "
            "(NeMoGymResponseOutputMessage needs an `id`)"
        )
    for part in content:
        if not isinstance(part, dict):
            return f"role={role} content part is {type(part).__name__}, expected dict"
        ptype = part.get("type")
        if ptype not in INPUT_PART_TYPES:
            return f"role={role} content part type {ptype!r} not in {sorted(INPUT_PART_TYPES)}"
    return None


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cfg_path = sys.argv[1]
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(REPO, cfg_path)

    DatasetConfig = load_real_dataset_config()
    cfg = yaml.safe_load(open(cfg_path))

    errors: list[str] = []
    warnings: list[str] = []
    agent_names: set[str] = set()
    datasets: list[tuple[str, dict]] = []

    for instance_name, body in (cfg or {}).items():
        if not isinstance(body, dict):
            continue
        for server_type, servers in body.items():
            if server_type == "responses_api_agents":
                agent_names.add(instance_name)
            for _server, scfg in (servers or {}).items():
                if isinstance(scfg, dict) and scfg.get("datasets"):
                    for d in scfg["datasets"]:
                        datasets.append((instance_name, d))

    print(f"config: {os.path.relpath(cfg_path, REPO)}")
    print(f"  agents defined: {sorted(agent_names) or '(none)'}")
    print(f"  dataset entries: {len(datasets)}")

    # --- 1. validate each dataset entry against the real model --------------
    for owner, d in datasets:
        try:
            DatasetConfig(**d)
        except Exception as e:  # noqa: BLE001
            first = str(e).splitlines()
            detail = " | ".join(x.strip() for x in first[1:4])
            errors.append(
                f"[{owner}] dataset '{d.get('name')}' FAILED DatasetConfig: {detail}"
            )
            continue

        # --- 2. the file must exist and be non-empty ------------------------
        fpath = d.get("jsonl_fpath", "")
        abspath = fpath if os.path.isabs(fpath) else os.path.join(GYM, fpath)
        if not os.path.exists(abspath):
            if d.get("gitlab_identifier") or d.get("huggingface_identifier"):
                warnings.append(
                    f"[{owner}] '{d.get('name')}' missing locally but has a remote identifier: {fpath}"
                )
            else:
                errors.append(f"[{owner}] '{d.get('name')}' jsonl not found: {abspath}")
            continue

        n = img_total = img_missing = bad_ref = 0
        msg_errors: list[str] = []
        with open(abspath) as fh:
            for i, line in enumerate(fh):
                if i >= 500:  # sample
                    break
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    errors.append(
                        f"[{owner}] '{d.get('name')}' line {i + 1} is not valid JSON"
                    )
                    break
                n += 1
                ref = (row.get("agent_ref") or {}).get("name")
                if agent_names and ref not in agent_names:
                    bad_ref += 1
                for j, m in enumerate(
                    row.get("responses_create_params", {}).get("input", [])
                ):
                    problem = validate_input_item(m)
                    if problem and len(msg_errors) < 3:
                        msg_errors.append(f"line {i + 1} item {j}: {problem}")
                    c = m.get("content") if isinstance(m, dict) else None
                    if isinstance(c, list):
                        for part in c:
                            if (
                                isinstance(part, dict)
                                and part.get("type") == "input_image"
                            ):
                                img_total += 1
                                if not os.path.exists(part.get("image_url", "")):
                                    img_missing += 1

        if msg_errors:
            errors.append(
                f"[{owner}] '{d.get('name')}': malformed input items -> "
                + "; ".join(msg_errors)
            )

        if n == 0:
            errors.append(f"[{owner}] '{d.get('name')}' is empty: {abspath}")
        if bad_ref:
            errors.append(
                f"[{owner}] '{d.get('name')}': {bad_ref}/{n} rows have agent_ref not in {sorted(agent_names)}"
            )
        if img_missing:
            errors.append(
                f"[{owner}] '{d.get('name')}': {img_missing}/{img_total} image paths do not resolve"
            )
        print(
            f"  ok  {d.get('name'):12s} rows_sampled={n:4d} images={img_total:4d} missing={img_missing}"
        )

    for w in warnings:
        print(f"  WARN {w}")
    if errors:
        print("\nFAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\nPREFLIGHT PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
