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
"""CPU tests for the Megatron worker under shared-prefix execution.

The driver plans the forwards of each rank. A plan that shares prompts carries
one tree id per row, where a tree is one prompt with its completions. The
worker's ``_attach_or_repack_with_order`` turns the tree ids and each row's
prompt-length tag into batch columns. The worker's constructor runs the model
check when the policy config shares prompts.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch

# Probe the concrete module: test_modelopt_worker_utils.py leaves stub
# ``megatron.bridge`` packages in sys.modules that a package probe accepts.
pytest.importorskip("megatron.bridge.training.checkpointing")

import nemo_rl.models.policy.workers.megatron_policy_worker as worker_module  # noqa: E402
from nemo_rl.data.packing.shared_prefix.row_metadata import (  # noqa: E402
    PROMPT_LENGTH_TAG,
    SHARED_PREFIX_PROMPT_LENGTHS,
    SHARED_PREFIX_TREE_IDS,
    UNSHARED_ROW,
)
from nemo_rl.data_plane import KVBatchMeta  # noqa: E402
from nemo_rl.data_plane.schema import (  # noqa: E402
    ELEM_COUNTS_PER_GB,
    MICRO_BATCH_INDICES,
    MICRO_BATCH_LENGTHS,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict  # noqa: E402

pytestmark = pytest.mark.mcore

# Two forwards over three rows: a pack of one tree, then a dense bin of one row.
_PACKING_PLAN = {
    MICRO_BATCH_INDICES: [[[0, 2], [2, 3]]],
    MICRO_BATCH_LENGTHS: [[6, 4]],
    ELEM_COUNTS_PER_GB: [3],
}
_TREE_IDS = [0, 0, UNSHARED_ROW]
_PROMPT_LENGTHS = [2, 2, 0]


def _bare_worker() -> Any:
    return object.__new__(worker_module.MegatronPolicyWorkerImpl)


def _batch(rows: int = 3) -> BatchedDataDict[Any]:
    return BatchedDataDict(
        {
            "input_ids": torch.zeros(rows, 4, dtype=torch.long),
            "input_lengths": torch.full((rows,), 4, dtype=torch.long),
        }
    )


def _tq_meta(
    *,
    extra_info: dict[str, Any],
    prompt_lengths: Optional[list[int]] = _PROMPT_LENGTHS,
) -> KVBatchMeta:
    tags = None
    if prompt_lengths is not None:
        tags = [{PROMPT_LENGTH_TAG: length} for length in prompt_lengths]
    return KVBatchMeta(
        partition_id="train",
        task_name="prev_lp",
        sample_ids=[f"sample{index}" for index in range(3)],
        extra_info=extra_info,
        tags=tags,
    )


def test_shared_plan_adds_tree_ids_and_prompt_lengths_to_the_batch() -> None:
    meta = _tq_meta(extra_info={**_PACKING_PLAN, SHARED_PREFIX_TREE_IDS: _TREE_IDS})

    data, row_order = _bare_worker()._attach_or_repack_with_order(_batch(), meta)

    assert row_order is None
    # The base class attached the driver's packing plan.
    assert data.micro_batch_indices == _PACKING_PLAN[MICRO_BATCH_INDICES]
    assert data.micro_batch_lengths == _PACKING_PLAN[MICRO_BATCH_LENGTHS]
    assert data.elem_counts_per_gb == _PACKING_PLAN[ELEM_COUNTS_PER_GB]
    for column, values in (
        (SHARED_PREFIX_TREE_IDS, _TREE_IDS),
        (SHARED_PREFIX_PROMPT_LENGTHS, _PROMPT_LENGTHS),
    ):
        assert data[column].dtype == torch.long
        assert data[column].tolist() == values
        # Each forward's slice keeps the entries of its own rows.
        assert data.slice(0, 2)[column].tolist() == values[:2]
        assert data.slice(2, 3)[column].tolist() == values[2:]


def test_plan_without_tree_ids_adds_no_column() -> None:
    # Rows carry their prompt-length tag even when the driver plans dense forwards.
    meta = _tq_meta(extra_info=_PACKING_PLAN)

    data, row_order = _bare_worker()._attach_or_repack_with_order(_batch(), meta)

    assert row_order is None
    assert data.micro_batch_indices == _PACKING_PLAN[MICRO_BATCH_INDICES]
    assert set(data) == {"input_ids", "input_lengths"}


def test_batch_without_tree_ids_returns_the_base_class_result() -> None:
    meta = _tq_meta(extra_info={})
    batch = _batch()
    repacked = _batch()

    # Without a driver plan the base class repacks locally and returns the row order.
    with patch.object(
        worker_module.TQWorkerMixin,
        "_attach_or_repack_with_order",
        return_value=(repacked, [2, 0, 1]),
    ) as base:
        data, row_order = _bare_worker()._attach_or_repack_with_order(batch, meta)

    base.assert_called_once_with(batch, meta)
    assert data is repacked
    assert row_order == [2, 0, 1]


@pytest.mark.parametrize(
    ("rows", "tree_ids", "prompt_lengths"),
    [
        (3, [0, 0], _PROMPT_LENGTHS),
        (3, [0, 0, UNSHARED_ROW, UNSHARED_ROW], _PROMPT_LENGTHS),
        (3, _TREE_IDS, None),
        (2, _TREE_IDS, _PROMPT_LENGTHS),
    ],
    ids=[
        "too_few_tree_ids",
        "too_many_tree_ids",
        "no_prompt_length_tags",
        "fewer_fetched_rows",
    ],
)
def test_shared_plan_needs_one_entry_per_fetched_row(
    rows: int, tree_ids: list[int], prompt_lengths: Optional[list[int]]
) -> None:
    meta = _tq_meta(
        extra_info={**_PACKING_PLAN, SHARED_PREFIX_TREE_IDS: tree_ids},
        prompt_lengths=prompt_lengths,
    )

    with pytest.raises(ValueError, match="one tree id and one prompt length"):
        _bare_worker()._attach_or_repack_with_order(_batch(rows), meta)


def _run_setup_dispatch(worker: SimpleNamespace) -> MagicMock:
    """Run the statement of the worker's constructor that checks the model.

    The constructor needs GPUs and a model, so the statement runs on its own with
    ``worker`` as ``self``. The check it calls is replaced and returned.
    """
    tree = ast.parse(Path(worker_module.__file__).read_text())
    constructor = next(
        node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "MegatronPolicyWorkerImpl"
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    (dispatch,) = [
        node
        for node in ast.walk(constructor)
        if isinstance(node, ast.If) and "share_prefixes" in ast.unparse(node.test)
    ]
    code = compile(
        ast.Module(body=[dispatch], type_ignores=[]), worker_module.__file__, "exec"
    )
    with patch(
        "nemo_rl.models.megatron.shared_prefix.setup.validate_shared_prefix_model"
    ) as check:
        exec(code, {"self": worker})
    return check


@pytest.mark.parametrize(
    ("mtp_enabled", "model_packs", "model_slices", "model_prepares_inputs"),
    [
        (False, False, False, False),
        (True, True, False, True),
        (False, False, True, True),
    ],
    ids=["worker_prepares_inputs", "model_packs", "model_slices_for_cp"],
)
def test_setup_checks_the_model_when_the_config_shares_prompts(
    mtp_enabled: bool,
    model_packs: bool,
    model_slices: bool,
    model_prepares_inputs: bool,
) -> None:
    worker = SimpleNamespace(
        cfg={"sequence_packing": {"enabled": True, "share_prefixes": True}},
        megatron_cfg=SimpleNamespace(model=object()),
        mtp_enabled=mtp_enabled,
        delegate_pack_to_model=model_packs,
        model_slices_context_parallel_inputs=model_slices,
    )

    check = _run_setup_dispatch(worker)

    check.assert_called_once_with(
        worker.megatron_cfg.model,
        policy_cfg=worker.cfg,
        mtp_enabled=mtp_enabled,
        model_prepares_inputs=model_prepares_inputs,
    )


@pytest.mark.parametrize(
    "sequence_packing",
    [{"enabled": True}, {"enabled": True, "share_prefixes": False}],
    ids=["key_absent", "key_false"],
)
def test_setup_skips_the_model_check_without_the_option(
    sequence_packing: dict[str, Any],
) -> None:
    worker = SimpleNamespace(cfg={"sequence_packing": sequence_packing})

    _run_setup_dispatch(worker).assert_not_called()
