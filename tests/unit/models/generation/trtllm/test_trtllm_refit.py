# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.trtllm


@pytest.mark.parametrize("recompute_kv", [False, True])
def test_collective_refit_recomputes_active_requests_only_when_requested(
    recompute_kv,
):
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
            recompute_kv=recompute_kv,
        )

    assert result is True
    model_loader.begin_update_weights.assert_called_once_with()
    model_loader.finalize_update_weights.assert_called_once_with()
    model_loader.abort_update_weights.assert_not_called()
    engine.model_engine.unwrap_compiled_model_for_refit.assert_called_once_with()
    engine.model_engine.restore_compiled_model_after_refit.assert_called_once_with(
        engine.resource_manager
    )
    # finish_weight_update always resets the prefix cache; recompute_kv adds a
    # recompute of in-flight requests before it.
    calls = [c[0] for c in engine.method_calls]
    assert [
        c for c in calls if c in ("recompute_active_requests", "reset_prefix_cache")
    ] == [
        *(["recompute_active_requests"] if recompute_kv else []),
        "reset_prefix_cache",
    ]
