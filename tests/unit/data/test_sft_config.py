# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from pathlib import Path

import yaml


def test_sft_config_supplies_default_chat_key() -> None:
    config_path = Path(__file__).parents[3] / "examples/configs/sft.yaml"

    with config_path.open() as config_file:
        config = yaml.safe_load(config_file)

    assert config["data"]["default"]["chat_key"] == "messages"
