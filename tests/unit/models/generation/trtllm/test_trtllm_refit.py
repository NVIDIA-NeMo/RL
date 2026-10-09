# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.trtllm


def test_collective_refit_always_resets_prefix_cache():
    from nemo_rl.models.generation.trtllm import trtllm_backend as backend

    extension = backend.NcclExtension.__new__(backend.NcclExtension)
    model = MagicMock()
    model.modules.return_value = []
    model_loader = MagicMock()
    engine = MagicMock()
    # begin_weight_update/finish_weight_update call these on the model engine.
    engine.model_engine = SimpleNamespace(
        model=model,
        model_loader=model_loader,
        unwrap_compiled_model_for_refit=MagicMock(),
        restore_compiled_model_after_refit=MagicMock(),
    )
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
    ):
        result = extension.update_weights_from_collective(
            drain=False,
            recompute_kv=False,
        )

    assert result is True
    model_loader.begin_update_weights.assert_called_once_with()
    model_loader.finalize_update_weights.assert_called_once_with()
    model_loader.abort_update_weights.assert_not_called()
    engine.model_engine.unwrap_compiled_model_for_refit.assert_called_once_with()
    engine.model_engine.restore_compiled_model_after_refit.assert_called_once_with(
        engine.resource_manager
    )
    engine.reset_prefix_cache.assert_called_once_with()
