# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from examples import run_vlm_grpo


def test_vlm_entrypoint_materializes_video_contract(monkeypatch):
    """The VLM runner must override vLLM VideoMediaIO's 32-frame default."""

    class Materialized(RuntimeError):
        pass

    config = SimpleNamespace(policy={"generation": {}}, data={})
    materialize = MagicMock(side_effect=Materialized)

    monkeypatch.setattr(
        run_vlm_grpo,
        "parse_args",
        lambda: (Namespace(config="config.yaml"), []),
    )
    monkeypatch.setattr(run_vlm_grpo, "load_config", lambda _: {})
    monkeypatch.setattr(
        run_vlm_grpo.OmegaConf,
        "to_container",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(run_vlm_grpo, "MasterConfig", lambda **_: config)
    monkeypatch.setattr(run_vlm_grpo, "materialize_vllm_video_config", materialize)

    with pytest.raises(Materialized):
        run_vlm_grpo.main()

    materialize.assert_called_once_with(config.policy, config.data)
