# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from omegaconf import OmegaConf

from nemo_rl.utils.config import load_config


def test_concurrency_fragment_changes_only_existing_ccc_batch_size(tmp_path):
    fragment = (
        Path(__file__).resolve().parents[3]
        / "training_configs/super_rl/ccc_verifier_concurrency.yaml"
    )
    gym = {
        "config_paths": [
            "policy.yaml",
            "ccc.yaml",
            "cot.yaml",
            "ns_tools.yaml",
            "scicode.yaml",
            "equivalence.yaml",
        ],
        "competitive_coding_challenges_resources_server": {
            "resources_servers": {
                "competitive_coding_challenges": {
                    "test_batch_size": 4,
                    "test_file": "/data/ccc.jsonl",
                    "num_parallel_requests": 64,
                    "time_scale": 2.0,
                    "reward_mode": "binary",
                    "shared_dir": "/shared",
                }
            }
        },
        "math_with_judge": {"keep": "cot"},
        "ns_tools": {"keep": "tir"},
        "scicode": {"keep": "scicode"},
        "equivalence_llm_judge": {"keep": "equivalence"},
        "policy_model": {"enable_thinking": True},
    }
    base = {
        "env": {"nemo_gym": gym},
        "policy": {"generation": {"max_new_tokens": 102400}},
        "data": {"train": {"data_path": "/data/train.jsonl"}},
    }
    OmegaConf.save(OmegaConf.create(base), tmp_path / "base.yaml")
    OmegaConf.save(
        OmegaConf.create({"defaults": ["base.yaml", str(fragment)]}),
        tmp_path / "child.yaml",
    )
    actual = OmegaConf.to_container(load_config(tmp_path / "child.yaml"), resolve=True)
    expected = OmegaConf.to_container(OmegaConf.create(base), resolve=True)
    expected["env"]["nemo_gym"]["competitive_coding_challenges_resources_server"][
        "resources_servers"
    ]["competitive_coding_challenges"]["test_batch_size"] = 32
    assert actual == expected
