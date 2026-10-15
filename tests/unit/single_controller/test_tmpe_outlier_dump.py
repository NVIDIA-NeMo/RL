from __future__ import annotations

import asyncio
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.algorithms.single_controller_utils.tmpe_outlier_dump import (
    DEFAULT_TMPE_OUTLIER_THRESHOLD,
    TMPE_METRIC_FIELDS,
    TMPE_OUTLIER_DUMP_PATH_ENV,
    TMPE_OUTLIER_THRESHOLD_ENV,
    TMPEOutlierDumpConfig,
    dump_tmpe_outlier_rows,
    find_tmpe_outlier_rows,
    tmpe_outlier_dump_config_from_env,
)
from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def test_tmpe_outlier_dump_config_is_default_off_and_path_gated(
    tmp_path: Path,
) -> None:
    assert tmpe_outlier_dump_config_from_env({}) is None
    assert tmpe_outlier_dump_config_from_env({TMPE_OUTLIER_DUMP_PATH_ENV: " "}) is None

    config = tmpe_outlier_dump_config_from_env(
        {TMPE_OUTLIER_DUMP_PATH_ENV: str(tmp_path)}
    )

    assert config == TMPEOutlierDumpConfig(
        output_path=tmp_path,
        threshold=DEFAULT_TMPE_OUTLIER_THRESHOLD,
    )


@pytest.mark.parametrize("threshold", ["not-a-number", "0", "nan", "inf"])
def test_tmpe_outlier_dump_rejects_invalid_diagnostic_threshold(
    tmp_path: Path,
    threshold: str,
) -> None:
    with pytest.raises(ValueError, match=TMPE_OUTLIER_THRESHOLD_ENV):
        tmpe_outlier_dump_config_from_env(
            {
                TMPE_OUTLIER_DUMP_PATH_ENV: str(tmp_path),
                TMPE_OUTLIER_THRESHOLD_ENV: threshold,
            }
        )


def test_find_tmpe_outlier_rows_uses_shifted_raw_sequence_tmpe() -> None:
    # Row 0 is below threshold, row 1 has TMPE=2, and row 2 has no valid
    # predicted tokens after the first-token shift.
    generation_logprobs = torch.zeros(3, 4)
    generation_logprobs[0, 1:] = math.log(1.2)
    generation_logprobs[1, 1:] = math.log(2.0)
    data = BatchedDataDict(
        {
            "token_mask": torch.tensor(
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0],
                ]
            ),
            "generation_logprobs": generation_logprobs,
            "prev_logprobs": torch.zeros(3, 4),
        }
    )

    indices, errors = find_tmpe_outlier_rows(data, threshold=1.5)

    assert indices == [1]
    assert errors.tolist() == pytest.approx([1.2, 2.0, 0.0])


def _dump_batch() -> BatchedDataDict:
    return BatchedDataDict(
        {
            "input_ids": torch.tensor([[10, 11, 0], [20, 21, 22]]),
            "input_lengths": torch.tensor([2, 3]),
            "token_mask": torch.tensor(
                [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
            ),
            "generation_logprobs": torch.tensor(
                [[0.0, 0.1, 0.0], [0.0, 0.7, 0.8]]
            ),
            "prev_logprobs": torch.zeros(2, 3),
            "pixel_values": PackedTensor(
                [
                    torch.full((1, 3, 2, 2), 1.0),
                    torch.full((2, 3, 2, 2), 2.0),
                ],
                dim_to_pack=0,
            ),
            "imgs_sizes": PackedTensor(
                [torch.tensor([[2, 2]]), torch.tensor([[2, 2], [2, 2]])],
                dim_to_pack=0,
            ),
            "num_frames": PackedTensor(
                [torch.tensor([1]), torch.tensor([2])],
                dim_to_pack=0,
            ),
        }
    )


def test_dump_tmpe_outlier_rows_writes_only_outliers_atomically(
    tmp_path: Path,
) -> None:
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["normal/sample", "outlier/sample"],
        sequence_lengths=[2, 3],
        tags=[{"kind": "normal"}, {"kind": "video", "weight_version": 4}],
    )
    config = TMPEOutlierDumpConfig(output_path=tmp_path, threshold=1.5)

    written = dump_tmpe_outlier_rows(
        data=_dump_batch(),
        meta=meta,
        sequence_errors=torch.tensor([1.1, 2.0]),
        config=config,
        step=7,
        trainer_version=4,
    )

    assert len(written) == 1
    assert written[0].parent == tmp_path
    assert list(tmp_path.glob("*.tmp")) == []
    assert list(tmp_path.glob("*.pt")) == written

    payload = torch.load(written[0], map_location="cpu", weights_only=False)
    assert payload["sample_id"] == "outlier/sample"
    assert payload["tags"] == {"kind": "video", "weight_version": 4}
    assert payload["sequence_length"] == 3
    assert payload["sequence_tmpe"] == pytest.approx(2.0)
    assert payload["threshold"] == pytest.approx(1.5)
    torch.testing.assert_close(payload["input_ids"], torch.tensor([20, 21, 22]))
    torch.testing.assert_close(payload["input_lengths"], torch.tensor(3))
    torch.testing.assert_close(
        payload["generation_logprobs"], torch.tensor([0.0, 0.7, 0.8])
    )
    torch.testing.assert_close(payload["prev_logprobs"], torch.zeros(3))
    assert isinstance(payload["pixel_values"], PackedTensor)
    assert isinstance(payload["imgs_sizes"], PackedTensor)
    assert isinstance(payload["num_frames"], PackedTensor)
    torch.testing.assert_close(
        payload["pixel_values"].tensors[0],
        torch.full((2, 3, 2, 2), 2.0),
    )
    torch.testing.assert_close(
        payload["imgs_sizes"].tensors[0], torch.tensor([[2, 2], [2, 2]])
    )
    torch.testing.assert_close(payload["num_frames"].tensors[0], torch.tensor([2]))


def test_controller_dump_fetches_full_payload_for_outlier_rows_only(
    tmp_path: Path,
) -> None:
    all_data = _dump_batch()
    trainer = MagicMock()

    def read_from_dataplane(meta: KVBatchMeta, *, select_fields: list[str]):
        if select_fields == list(TMPE_METRIC_FIELDS):
            assert meta.sample_ids == ["normal", "outlier"]
            return BatchedDataDict({field: all_data[field] for field in select_fields})
        assert meta.sample_ids == ["outlier"]
        selected: dict[str, object] = {}
        for field in select_fields:
            value = all_data[field]
            selected[field] = (
                value.slice([1])
                if isinstance(value, PackedTensor)
                else value[1:2]
            )
        return BatchedDataDict(selected)

    trainer.read_from_dataplane.side_effect = read_from_dataplane
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._trainer = trainer
    ctrl._tmpe_outlier_dump_config = TMPEOutlierDumpConfig(
        output_path=tmp_path,
        threshold=1.5,
    )
    ctrl._train_steps = 12
    ctrl._trainer_version = 5
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["normal", "outlier"],
        fields=[
            "input_ids",
            "input_lengths",
            "token_mask",
            "generation_logprobs",
            "pixel_values",
            "imgs_sizes",
            "num_frames",
        ],
        sequence_lengths=[2, 3],
        tags=[{"kind": "text"}, {"kind": "video"}],
        extra_info={"sentinel": "unchanged"},
    )

    asyncio.run(ctrl._maybe_dump_tmpe_outliers(meta))

    assert trainer.read_from_dataplane.call_count == 2
    assert meta.extra_info == {"sentinel": "unchanged"}
    files = list(tmp_path.glob("*.pt"))
    assert len(files) == 1
    payload = torch.load(files[0], map_location="cpu", weights_only=False)
    assert payload["sample_id"] == "outlier"
    assert payload["tags"] == {"kind": "video"}
    assert payload["step"] == 12
    assert payload["trainer_version"] == 5


def test_controller_dump_disabled_performs_no_data_plane_read() -> None:
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._trainer = SimpleNamespace(read_from_dataplane=MagicMock())
    ctrl._tmpe_outlier_dump_config = None
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["sample"],
    )

    asyncio.run(ctrl._maybe_dump_tmpe_outliers(meta))

    ctrl._trainer.read_from_dataplane.assert_not_called()
