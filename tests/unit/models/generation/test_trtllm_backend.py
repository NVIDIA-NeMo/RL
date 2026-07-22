# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from nemo_rl.models.generation.trtllm.trtllm_backend import NcclExtension


def test_collective_refit_always_resets_prefix_cache():
    extension = NcclExtension.__new__(NcclExtension)
    model = MagicMock()
    model.modules.return_value = []
    model_loader = MagicMock()
    engine = MagicMock()
    engine.model_engine = SimpleNamespace(model=model, model_loader=model_loader)
    engine.control_action.return_value = nullcontext()

    extension.engine = engine
    extension.device_id = 0
    extension.model_update_group = MagicMock()
    extension.state_dict_info = {}

    with (
        patch(
            "nemo_rl.models.generation.trtllm.trtllm_backend.packed_broadcast_consumer"
        ),
        patch("torch.cuda.synchronize"),
        patch("torch.cuda.current_stream") as current_stream,
    ):
        result = extension.update_weights_from_collective(
            drain=False,
            recompute_kv=False,
        )

    assert result is True
    model_loader.begin_update_weights.assert_called_once_with()
    model_loader.finalize_update_weights.assert_called_once_with()
    model_loader.abort_update_weights.assert_not_called()
    current_stream.return_value.synchronize.assert_called_once_with()
    engine.reset_prefix_cache.assert_called_once_with()
