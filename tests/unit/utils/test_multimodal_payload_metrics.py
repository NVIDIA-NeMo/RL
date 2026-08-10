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

import numpy as np
import torch
from PIL import Image

from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.utils import multimodal_payload_metrics
from nemo_rl.utils.multimodal_payload_metrics import (
    collect_multimodal_payload_metrics,
    collect_sharded_multimodal_payload_metrics,
    drain_multimodal_payload_metrics,
    merge_multimodal_payload_metrics,
    print_multimodal_payload_metrics,
)


def test_payload_metrics_do_no_work_when_disabled(monkeypatch):
    def fail_if_called(_value):
        raise AssertionError("disabled metrics must not serialize or scan")

    monkeypatch.setattr(
        multimodal_payload_metrics,
        "protocol5_serialized_nbytes",
        fail_if_called,
    )
    assert (
        collect_multimodal_payload_metrics(
            {"pixel_values": PackedTensor(torch.ones(1), dim_to_pack=0)},
            "disabled",
            enabled=False,
        )
        == {}
    )


def test_payload_metrics_accumulate_for_per_step_logger(capsys):
    drain_multimodal_payload_metrics()
    first = {
        "payload_bytes/replay_push/serialized": 110,
        "payload_bytes/replay_push/physical_media": 20,
        "payload_bytes/replay_push/logical_media": 100,
        "payload_bytes/replay_push/estimated_saved": 80,
        "payload_counts/replay_push/physical_segments": 1,
        "payload_counts/replay_push/logical_segments": 5,
        "payload_counts/replay_push/calls": 1,
        "payload_ratio/replay_push/physical_to_logical": 0.2,
    }
    second = {
        "payload_bytes/replay_push/serialized": 220,
        "payload_bytes/replay_push/physical_media": 40,
        "payload_bytes/replay_push/logical_media": 200,
        "payload_bytes/replay_push/estimated_saved": 160,
        "payload_counts/replay_push/physical_segments": 2,
        "payload_counts/replay_push/logical_segments": 10,
        "payload_counts/replay_push/calls": 1,
        "payload_ratio/replay_push/physical_to_logical": 0.2,
    }

    print_multimodal_payload_metrics(first)
    print_multimodal_payload_metrics(second)
    logged = drain_multimodal_payload_metrics()

    assert logged["payload_bytes/replay_push/serialized"] == 330
    assert logged["payload_bytes/replay_push/physical_media"] == 60
    assert logged["payload_bytes/replay_push/logical_media"] == 300
    assert logged["payload_ratio/replay_push/physical_to_logical"] == 0.2
    assert logged["payload_counts/replay_push/calls"] == 2
    assert logged["payload_bytes/replay_push/serialized_mean_per_call"] == 165.0
    assert drain_multimodal_payload_metrics() == {}
    assert capsys.readouterr().out.count("▶ [PAYLOAD]") == 2


def test_payload_metric_accumulator_is_bounded(capsys):
    drain_multimodal_payload_metrics()
    metrics = {
        "payload_bytes/replay_push/serialized": 10,
        "payload_bytes/replay_push/physical_media": 2,
        "payload_bytes/replay_push/logical_media": 4,
        "payload_counts/replay_push/calls": 1,
        "payload_ratio/replay_push/physical_to_logical": 0.5,
    }

    for _ in range(100):
        print_multimodal_payload_metrics(metrics)

    assert isinstance(multimodal_payload_metrics._PENDING_PAYLOAD_METRICS, dict)
    assert len(multimodal_payload_metrics._PENDING_PAYLOAD_METRICS) <= 7
    logged = drain_multimodal_payload_metrics()
    assert logged["payload_counts/replay_push/calls"] == 100
    assert logged["payload_bytes/replay_push/serialized"] == 1000
    assert logged["payload_bytes/replay_push/serialized_mean_per_call"] == 10.0
    capsys.readouterr()


def test_payload_metric_merge_preserves_per_call_maxima():
    merged = merge_multimodal_payload_metrics(
        [
            {
                "payload_bytes/policy_train/serialized_max_shard": 10,
                "payload_counts/policy_train/shards": 4,
                "payload_counts/policy_train/calls": 4,
            },
            {
                "payload_bytes/policy_train/serialized_max_shard": 15,
                "payload_counts/policy_train/shards": 4,
                "payload_counts/policy_train/calls": 4,
            },
        ]
    )

    assert merged["payload_bytes/policy_train/serialized_max_shard"] == 15
    assert merged["payload_counts/policy_train/shards"] == 4
    assert merged["payload_counts/policy_train/calls"] == 8


def test_payload_metrics_measure_physical_and_logical_dedup_savings():
    tensor = torch.ones(256, 256)
    flag_off = {
        "input_ids": torch.ones(4, 2, dtype=torch.long),
        "pixel_values": PackedTensor([tensor.clone() for _ in range(4)], dim_to_pack=0),
    }
    flag_on_media = PackedTensor(tensor, dim_to_pack=0).enable_deduplication()
    flag_on = {
        "input_ids": flag_off["input_ids"],
        "pixel_values": PackedTensor.concat([flag_on_media] * 4),
    }

    off = collect_multimodal_payload_metrics(flag_off, "off", enabled=True)
    on = collect_multimodal_payload_metrics(flag_on, "on", enabled=True)

    one_tensor_bytes = tensor.numel() * tensor.element_size()
    assert off["payload_bytes/off/physical_media"] == 4 * one_tensor_bytes
    assert on["payload_bytes/on/physical_media"] == one_tensor_bytes
    assert on["payload_bytes/on/logical_media"] == 4 * one_tensor_bytes
    assert on["payload_bytes/on/estimated_saved"] == 3 * one_tensor_bytes
    assert on["payload_bytes/on/serialized"] < off["payload_bytes/off/serialized"]


def test_payload_metrics_measure_bf16_policy_transport_savings():
    pixels = PackedTensor(torch.ones(4, 3, 16, 16, dtype=torch.float32), dim_to_pack=0)
    batch = BatchedDataDict({"pixel_values": pixels})
    bf16_payload = batch.get_multimodal_dict(
        as_tensors=False, pixel_dtype=torch.bfloat16
    )

    fp32_metrics = collect_multimodal_payload_metrics(batch, "fp32", enabled=True)
    bf16_metrics = collect_multimodal_payload_metrics(
        bf16_payload, "bf16", enabled=True
    )

    assert (
        bf16_metrics["payload_bytes/bf16/physical_media"] * 2
        == fp32_metrics["payload_bytes/fp32/physical_media"]
    )
    assert (
        bf16_metrics["payload_bytes/bf16/serialized"]
        < fp32_metrics["payload_bytes/fp32/serialized"]
    )


def test_payload_metrics_serialize_exact_positional_argument_tuple(monkeypatch):
    media = PackedTensor(torch.ones(2, 2), dim_to_pack=0)
    payload = ({"pixel_values": media}, "timer", False)
    serialized_values = []

    def record_serialized_value(value):
        serialized_values.append(value)
        return 123

    monkeypatch.setattr(
        multimodal_payload_metrics,
        "protocol5_serialized_nbytes",
        record_serialized_value,
    )

    metrics = collect_multimodal_payload_metrics(
        payload,
        "ray_args",
        enabled=True,
    )

    assert serialized_values == [payload]
    assert metrics["payload_bytes/ray_args/serialized"] == 123
    assert metrics["payload_counts/ray_args/physical_segments"] == 1


def test_sharded_payload_metrics_measure_exact_worker_arguments():
    media = PackedTensor(torch.ones(4, 4), dim_to_pack=0)
    media.enable_deduplication()
    payload = {
        "input_ids": torch.ones(4, 2, dtype=torch.long),
        "pixel_values": PackedTensor.concat([media] * 4),
    }
    shards = [
        {
            "input_ids": payload["input_ids"][:2],
            "pixel_values": payload["pixel_values"].slice([0, 1]),
        },
        {
            "input_ids": payload["input_ids"][2:],
            "pixel_values": payload["pixel_values"].slice([2, 3]),
        },
    ]

    metrics = collect_sharded_multimodal_payload_metrics(
        shards,
        "train",
        enabled=True,
    )

    assert metrics["payload_counts/train/shards"] == 2
    assert metrics["payload_bytes/train/physical_media_total"] == 2 * 4 * 4 * 4
    assert metrics["payload_bytes/train/logical_media_total"] == 4 * 4 * 4 * 4
    assert metrics["payload_bytes/train/estimated_saved_total"] == 2 * 4 * 4 * 4
    assert metrics["payload_counts/train/physical_segments_total"] == 2
    assert metrics["payload_counts/train/logical_segments_total"] == 4
    assert metrics["payload_ratio/train/physical_to_logical"] == 0.5


def test_nested_payload_metrics_aggregate_by_media_key():
    first_media = PackedTensor(torch.ones(2, 2), dim_to_pack=0).enable_deduplication()
    second_media = PackedTensor(torch.ones(2, 2), dim_to_pack=0).enable_deduplication()
    nested = {
        "trajectories": [
            {"batch": {"pixel_values": PackedTensor.concat([first_media] * 2)}},
            {"batch": {"pixel_values": PackedTensor.concat([second_media] * 3)}},
        ]
    }

    metrics = collect_multimodal_payload_metrics(
        nested,
        "replay",
        enabled=True,
    )

    assert metrics["payload_counts/replay/pixel_values/physical_segments"] == 2
    assert metrics["payload_counts/replay/pixel_values/logical_segments"] == 5
    assert not any("[0]" in key or "[1]" in key for key in metrics)


def test_native_audio_video_and_typed_image_metrics_count_shared_media():
    image = Image.new("RGB", (4, 3))
    video = np.ones((2, 3, 4, 3), dtype=np.uint8)
    audio = np.ones(16, dtype=np.float32)
    payload = {
        "vllm_content": [
            [
                {"type": "text", "text": "must not count as media"},
                {"type": "image", "image": image},
            ],
            [
                {"type": "text", "text": "different prompt text"},
                {"type": "image", "image": image},
            ],
        ],
        "vllm_videos": [[video], [video]],
        "vllm_audios": [[(audio, 16_000)], [(audio, 16_000)]],
    }

    metrics = collect_multimodal_payload_metrics(
        payload,
        "native",
        enabled=True,
    )

    one_logical_row = len(image.tobytes()) + video.nbytes + audio.nbytes
    assert metrics["payload_bytes/native/physical_media"] == one_logical_row
    assert metrics["payload_bytes/native/logical_media"] == 2 * one_logical_row
    assert metrics["payload_bytes/native/estimated_saved"] == one_logical_row
    assert metrics["payload_counts/native/physical_segments"] == 3
    assert metrics["payload_counts/native/logical_segments"] == 6


def test_nested_responses_input_image_metrics_count_shared_data_url():
    data_url = "data:image/png;base64,abc123"
    payload = {
        "nemo_gym_examples": [
            {
                "responses_create_params": {
                    "input": [
                        {
                            "role": "user",
                            "content": [{"type": "input_image", "image_url": data_url}],
                        }
                    ]
                }
            },
            {
                "responses_create_params": {
                    "input": [
                        {
                            "role": "user",
                            "content": [{"type": "input_image", "image_url": data_url}],
                        }
                    ]
                }
            },
        ]
    }

    metrics = collect_multimodal_payload_metrics(
        payload,
        "nemo_gym_request",
        enabled=True,
    )

    logical_row_bytes = len(data_url.encode())
    assert metrics["payload_bytes/nemo_gym_request/physical_media"] == logical_row_bytes
    assert (
        metrics["payload_bytes/nemo_gym_request/logical_media"] == 2 * logical_row_bytes
    )
    assert metrics["payload_counts/nemo_gym_request/physical_segments"] == 1
    assert metrics["payload_counts/nemo_gym_request/logical_segments"] == 2
