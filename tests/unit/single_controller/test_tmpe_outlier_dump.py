from __future__ import annotations

import asyncio
import hashlib
import json
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
    TMPEOutlierDumpRef,
    dump_tmpe_outlier_admissions,
    dump_tmpe_outlier_rows,
    find_tmpe_outlier_rows,
    tmpe_outlier_dump_config_from_env,
)
from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import ROUTED_EXPERTS_FIELD
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from tests.unit.single_controller.test_logical_advantage import _batch, _controller


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
            "token_mask": torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]),
            "generation_logprobs": torch.tensor([[0.0, 0.1, 0.0], [0.0, 0.7, 0.8]]),
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


def test_dump_preserves_generation_routes_and_source_tensors(tmp_path: Path) -> None:
    data = _dump_batch()
    routes = torch.arange(2 * 3 * 4 * 2, dtype=torch.int16).reshape(2, 3, 4, 2)
    data[ROUTED_EXPERTS_FIELD] = routes
    original_routes = routes.clone()
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["normal", "outlier"],
        sequence_lengths=[2, 3],
    )

    [written] = dump_tmpe_outlier_rows(
        data=data,
        meta=meta,
        sequence_errors=torch.tensor([1.1, 2.0]),
        config=TMPEOutlierDumpConfig(output_path=tmp_path, threshold=1.5),
        step=0,
        trainer_version=0,
    )

    payload = torch.load(written, map_location="cpu", weights_only=False)
    torch.testing.assert_close(payload[ROUTED_EXPERTS_FIELD], original_routes[1])
    torch.testing.assert_close(routes, original_routes)
    # A retained dump must remain independent of later replay/padding writes.
    routes.fill_(-1)
    torch.testing.assert_close(payload[ROUTED_EXPERTS_FIELD], original_routes[1])


@pytest.mark.parametrize("has_routes", [False, True])
def test_controller_dump_fetches_full_payload_for_outlier_rows_only(
    tmp_path: Path,
    has_routes: bool,
) -> None:
    all_data = _dump_batch()
    if has_routes:
        all_data[ROUTED_EXPERTS_FIELD] = torch.arange(
            2 * 3 * 4 * 2, dtype=torch.int16
        ).reshape(2, 3, 4, 2)
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
                value.slice([1]) if isinstance(value, PackedTensor) else value[1:2]
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
            *([ROUTED_EXPERTS_FIELD] if has_routes else []),
        ],
        sequence_lengths=[2, 3],
        tags=[{"kind": "text"}, {"kind": "video"}],
        extra_info={"sentinel": "unchanged"},
    )

    dumps = asyncio.run(ctrl._maybe_dump_tmpe_outliers(meta))

    assert trainer.read_from_dataplane.call_count == 2
    assert meta.extra_info == {"sentinel": "unchanged"}
    files = list(tmp_path.glob("*.pt"))
    assert len(files) == 1
    assert dumps == [TMPEOutlierDumpRef(sample_id="outlier", path=files[0])]
    payload = torch.load(files[0], map_location="cpu", weights_only=False)
    assert payload["sample_id"] == "outlier"
    assert payload["tags"] == {"kind": "video"}
    assert payload["step"] == 12
    assert payload["trainer_version"] == 5
    selected_fields = trainer.read_from_dataplane.call_args.kwargs["select_fields"]
    assert (ROUTED_EXPERTS_FIELD in selected_fields) is has_routes
    if has_routes:
        torch.testing.assert_close(
            payload[ROUTED_EXPERTS_FIELD], all_data[ROUTED_EXPERTS_FIELD][1]
        )
    else:
        assert payload.get(ROUTED_EXPERTS_FIELD) is None


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

    assert asyncio.run(ctrl._maybe_dump_tmpe_outliers(meta)) == []

    ctrl._trainer.read_from_dataplane.assert_not_called()


def test_admission_sidecars_keep_raw_filtered_rows_and_match_by_identity(
    tmp_path: Path,
) -> None:
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["normal", "outlier"],
        sequence_lengths=[2, 3],
    )
    paths = dump_tmpe_outlier_rows(
        data=_dump_batch(),
        meta=meta,
        sequence_errors=torch.tensor([1.1, 2.0]),
        config=TMPEOutlierDumpConfig(output_path=tmp_path, threshold=1.01),
        step=7,
        trainer_version=4,
    )
    raw_digests = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    initial = torch.ones(2)
    final = torch.tensor([1.0, 0.0])
    masks = [initial.clone(), final.clone()]
    written = dump_tmpe_outlier_admissions(
        # Reference order differs from metadata order on purpose.
        dumps=[
            TMPEOutlierDumpRef(sample_id="outlier", path=paths[1]),
            TMPEOutlierDumpRef(sample_id="normal", path=paths[0]),
        ],
        meta=meta,
        input_sample_mask=initial,
        mask_sample=torch.zeros(2, dtype=torch.bool),
        truncated=torch.tensor([False, True]),
        pre_owner_sample_mask=final,
        final_sample_mask=final,
        token_mask=_dump_batch()["token_mask"],
        overlong_filtering=True,
        seq_logprob_error_threshold=None,
        step=7,
        trainer_version=4,
    )
    records = {
        json.loads(path.read_text())["sample_id"]: json.loads(path.read_text())
        for path in written
    }
    assert records["normal"]["admitted"] is True
    assert records["normal"]["raw_dump"] == paths[0].name
    assert records["normal"]["shifted_target_tokens"] == 1
    assert records["outlier"]["admitted"] is False
    assert records["outlier"]["truncated"] is True
    assert records["outlier"]["input_sample_mask"] == 1.0
    assert records["outlier"]["final_sample_mask"] == 0.0
    assert (
        records["outlier"]["scope"] == "post_advantage_writeback; not_optimizer_commit"
    )
    assert [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    ] == raw_digests
    torch.testing.assert_close(initial, masks[0])
    torch.testing.assert_close(final, masks[1])
    final.fill_(1)
    assert json.loads(written[0].read_text())["admitted"] is False
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("failure", [None, "sidecar", "writeback"])
def test_actual_owner_admission_is_recorded_only_after_successful_writeback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    meta, data = _batch([("a", [(0.0, 2), (1.0, 1)])], padding=True)
    # A truncated child invalidates its otherwise valid parent; the sibling
    # owner remains admitted. Execution padding must never become an outlier.
    data["truncated"][1] = True
    data["input_ids"] = torch.tensor([[10, 11, 12]] * meta.size)
    data["input_lengths"] = torch.full((meta.size,), 3)
    data["prev_logprobs"] = torch.zeros(meta.size, 3)
    data["generation_logprobs"] = torch.full((meta.size, 3), math.log(2))
    data["generation_logprobs"][2] = math.log(1.6)
    meta.fields = list(data.keys())
    control_data = data.clone()
    control = _controller(meta, control_data, grpo={"overlong_filtering": True})
    control._policy_logprobs_required = True
    asyncio.run(control._advantage_stage(meta))
    ctrl = _controller(meta, data, grpo={"overlong_filtering": True})
    ctrl._policy_logprobs_required = True
    ctrl._train_steps, ctrl._trainer_version = 12, 5
    ctrl._tmpe_outlier_dump_config = TMPEOutlierDumpConfig(tmp_path, 1.5)

    def read_from_dataplane(selected_meta, *, select_fields):
        indices = [meta.sample_ids.index(sid) for sid in selected_meta.sample_ids]
        return BatchedDataDict({name: data[name][indices] for name in select_fields})

    ctrl._trainer = SimpleNamespace(
        read_from_dataplane=MagicMock(side_effect=read_from_dataplane)
    )
    dumps = asyncio.run(ctrl._maybe_dump_tmpe_outliers(meta))
    assert len(dumps) == 3
    raw_digests = {
        dump.path: hashlib.sha256(dump.path.read_bytes()).hexdigest() for dump in dumps
    }

    def fail(*args, **kwargs):
        raise OSError("injected storage error")

    if failure == "sidecar":
        monkeypatch.setattr(
            "nemo_rl.algorithms.single_controller.dump_tmpe_outlier_admissions", fail
        )
    elif failure == "writeback":
        monkeypatch.setattr(ctrl._dp_client, "put_samples", fail)

    if failure == "writeback":
        with pytest.raises(OSError, match="injected storage error"):
            asyncio.run(ctrl._advantage_stage(meta, tmpe_outlier_dumps=dumps))
    else:
        _, valid = asyncio.run(ctrl._advantage_stage(meta, tmpe_outlier_dumps=dumps))
        assert valid
        torch.testing.assert_close(data["sample_mask"], control_data["sample_mask"])
        torch.testing.assert_close(data["advantages"], control_data["advantages"])
        assert data["sample_mask"].tolist() == [0.0, 0.0, 1.0, 0.0]
        # Existing after_mask telemetry precedes owner fanout: its maximum
        # still includes this rejected parent. The new sidecar records the
        # actual final admission without overwriting or hiding that raw peak.
        metrics = ctrl._step_log_dict["seq_logprob_error_metrics"][0]
        assert metrics["max_seq_mult_prob_error_after_mask"] == pytest.approx(2.0)
        assert dumps[2].sample_id == meta.sample_ids[2]
        payload = torch.load(dumps[2].path, map_location="cpu", weights_only=False)
        assert payload["sequence_tmpe"] == pytest.approx(1.6)
    if failure is None:
        records = {
            json.loads(path.read_text())["sample_id"]: json.loads(path.read_text())
            for path in tmp_path.glob("*.admission.json")
        }
        parent, child, sibling = [records[sid] for sid in meta.sample_ids[:3]]
        assert not parent["truncated"] and parent["pre_owner_sample_mask"] == 1
        assert parent["final_sample_mask"] == 0 and not parent["admitted"]
        assert child["truncated"] and child["pre_owner_sample_mask"] == 0
        assert child["final_sample_mask"] == 0 and not child["admitted"]
        assert sibling["final_sample_mask"] == 1 and sibling["admitted"]
        assert all(
            record["step"] == 12 and record["trainer_version"] == 5
            for record in records.values()
        )
    else:
        assert not list(tmp_path.glob("*.admission.json"))
    assert ctrl._trainer.read_from_dataplane.call_count == 2
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in raw_digests
    } == raw_digests


def test_admission_atomic_rename_failure_leaves_raw_dump_and_no_partial_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw.pt"
    raw.write_bytes(b"immutable raw dump")

    def fail(*args, **kwargs):
        raise OSError("injected rename failure")

    monkeypatch.setattr(
        "nemo_rl.algorithms.single_controller_utils.tmpe_outlier_dump.os.replace", fail
    )
    with pytest.raises(OSError, match="rename failure"):
        dump_tmpe_outlier_admissions(
            dumps=[TMPEOutlierDumpRef("sample", raw)],
            meta=KVBatchMeta("train", "train", ["sample"]),
            input_sample_mask=torch.ones(1),
            mask_sample=torch.zeros(1, dtype=torch.bool),
            truncated=torch.zeros(1, dtype=torch.bool),
            pre_owner_sample_mask=torch.ones(1),
            final_sample_mask=torch.ones(1),
            token_mask=torch.ones(1, 3),
            overlong_filtering=True,
            seq_logprob_error_threshold=None,
            step=0,
            trainer_version=0,
        )
    assert raw.read_bytes() == b"immutable raw dump"
    assert list(tmp_path.iterdir()) == [raw]


def test_raw_dump_read_failure_returns_no_admission_references(tmp_path: Path) -> None:
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._tmpe_outlier_dump_config = TMPEOutlierDumpConfig(tmp_path, 1.5)
    ctrl._trainer = SimpleNamespace(
        read_from_dataplane=MagicMock(side_effect=OSError("diagnostic read failed"))
    )
    meta = KVBatchMeta("train", "train", ["sample"])
    assert asyncio.run(ctrl._maybe_dump_tmpe_outliers(meta)) == []
    assert not list(tmp_path.iterdir())
