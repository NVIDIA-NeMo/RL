# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import json
from pathlib import Path

from PIL import Image, ImageDraw

from resources_servers.image_tools import (
    IMAGE_TOOL_NAMES,
    ImageToolsGymToolLogic,
    execute_image_tool,
    has_malformed_image_tool_raw_generation,
    parse_image_tool_calls,
)


def _xml_tool_call(name: str, arguments: dict) -> str:
    parameters = "".join(
        f"<parameter={key}>{json.dumps(value)}</parameter>"
        for key, value in arguments.items()
    )
    return f"<tool_call><function={name}>{parameters}</function></tool_call>"


def test_parse_image_tools_xml_tool_call() -> None:
    text = """<think>zoom the sign</think>
<tool_call>
<function=image_zoom_in_tool>
<parameter=bbox_2d>
[100, 200, 400, 500]
</parameter>
<parameter=label>
small sign
</parameter>
<parameter=img_idx>
0
</parameter>
</function>
</tool_call>"""

    calls = parse_image_tool_calls(text)

    assert len(calls) == 1
    assert calls[0]["name"] == "image_zoom_in_tool"
    assert calls[0]["arguments"] == {
        "bbox_2d": [100, 200, 400, 500],
        "label": "small sign",
        "img_idx": 0,
    }


def test_raw_generation_markup_allows_expected_think_close() -> None:
    text = """I need a closer view.
</think>
<tool_call>
<function=image_zoom_in_tool>
<parameter=bbox_2d>[100, 200, 400, 500]</parameter>
<parameter=label>small sign</parameter>
<parameter=img_idx>0</parameter>
</function>
</tool_call>"""

    assert not has_malformed_image_tool_raw_generation(text)


def test_raw_generation_markup_rejects_unbalanced_tool_call() -> None:
    text = """I need a closer view.
</think>
<tool_call>
<function=image_zoom_in_tool>
<parameter=bbox_2d>[100, 200, 400, 500]</parameter>
<parameter=label>small sign</parameter>
<tool_call>
<function=image_zoom_in_tool>"""

    assert has_malformed_image_tool_raw_generation(text)


def test_raw_generation_markup_rejects_partial_function_tag() -> None:
    text = """I need one more crop.
</think>
<tool_call>
<function=image_zoom_in_tool<|im_end|>"""

    assert has_malformed_image_tool_raw_generation(text)


def test_execute_image_tools_writes_resized_crop(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    Image.new("RGB", (100, 80), color=(10, 20, 30)).save(image_path)

    result = execute_image_tool(
        "image_zoom_in_tool",
        {"bbox_2d": [0, 0, 500, 500], "label": "upper left", "img_idx": 0},
        [str(image_path)],
        tmp_path / "crops",
    )

    crop_path = Path(result["path"])
    assert crop_path.exists()
    # Zoom now flows through the shared dispatch, so it reports the same
    # result keys as every other tool.
    assert result["result"]["box"] == [0, 0, 500, 500]
    assert result["result"]["factor"] == 3.0
    assert result["result"]["img_idx"] == 0
    with Image.open(crop_path) as crop:
        assert crop.width % 32 == 0
        assert crop.height % 32 == 0
        assert result["result"]["size"] == [crop.width, crop.height]


def test_image_tools_logic_returns_crop_then_final_answer(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    Image.new("RGB", (96, 96), color=(255, 255, 255)).save(image_path)
    logic = ImageToolsGymToolLogic(
        {
            "crop_dir": str(tmp_path / "work"),
            "max_tool_calls": 8,
            "max_tool_calls_per_turn": 1,
            "stop_strings": ["</tool_call>"],
        }
    )
    metadata = {
        "ground_truth": "cat",
        "image_paths": [str(image_path)],
        "dataset": "unit",
    }
    tool_log = [
        {
            "role": "assistant",
            "content": (
                "<tool_call><function=image_zoom_in_tool>"
                "<parameter=bbox_2d>[0, 0, 500, 500]</parameter>"
                "<parameter=label>object</parameter>"
                "<parameter=img_idx>0</parameter>"
                "</function></tool_call>"
            ),
        }
    ]

    observation, reward, done, stops, next_metadata, answer, terminal_gt = (
        logic.process_nonterminal_turn(tool_log, metadata)
    )

    assert reward == 0.0
    assert not done
    assert stops == ["</tool_call>"]
    assert answer is None
    assert terminal_gt is None
    assert observation["role"] == "user"
    assert any(item.get("type") == "image" for item in observation["content"])
    assert next_metadata is not None
    assert next_metadata["tool_call_count"] == 1
    assert Path(next_metadata["crop_paths"][0]).exists()

    final_log = [{"role": "assistant", "content": "<think>done</think>\ncat"}]
    _, _, done, _, _, answer, terminal_gt = logic.process_nonterminal_turn(
        final_log, next_metadata
    )

    assert done
    assert answer == "<think>done</think>\ncat"
    assert terminal_gt == "cat"


def test_image_tools_logic_executes_all_dataset_tools_and_chains_indices(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    first = Image.new("RGB", (64, 64), color="white")
    second = Image.new("RGB", (64, 64), color="white")
    first_draw = ImageDraw.Draw(first)
    first_draw.rectangle((5, 5, 15, 15), fill=(255, 0, 0))
    first_draw.rectangle((30, 30, 42, 42), fill=(255, 0, 0))
    ImageDraw.Draw(second).rectangle((10, 5, 20, 15), fill=(0, 0, 255))
    first.save(first_path)
    second.save(second_path)

    logic = ImageToolsGymToolLogic(
        {
            "crop_dir": str(tmp_path / "outputs"),
            "crop_format": "png",
            "crop_min_pixels": 32 * 32,
            "crop_max_pixels": 1024 * 1024,
            "max_tool_calls": len(IMAGE_TOOL_NAMES),
            "max_tool_calls_per_turn": 1,
        }
    )
    metadata = {
        "ground_truth": "done",
        "image_paths": [str(first_path), str(second_path)],
        "dataset": "all-tools-smoke",
    }
    calls = [
        (
            "image_crop_tool",
            {"bbox_2d": [0, 0, 500, 500], "label": "crop", "img_idx": 0},
        ),
        (
            "image_zoom_in_tool",
            {"bbox_2d": [0, 0, 1000, 1000], "factor": 2, "label": "zoom", "img_idx": 2},
        ),
        ("image_rotate_tool", {"degrees": 90, "label": "rotate", "img_idx": 3}),
        ("image_flip_tool", {"axis": "horizontal", "label": "flip", "img_idx": 4}),
        ("image_diff_tool", {"img_idx_a": 0, "img_idx_b": 1, "label": "diff"}),
        (
            "image_side_by_side_tool",
            {
                "img_indices": [0, 1, 6],
                "labels": ["a", "b", "diff"],
                "label": "compare",
            },
        ),
        (
            "image_overlay_tool",
            {"img_idx_a": 0, "img_idx_b": 1, "alpha": 0.25, "label": "overlay"},
        ),
        (
            "count_objects_tool",
            {
                "color": [255, 0, 0],
                "tolerance": 0,
                "min_size": 10,
                "label": "count",
                "img_idx": 0,
            },
        ),
        (
            "find_color_tool",
            {"color": [255, 0, 0], "tolerance": 0, "label": "red", "img_idx": 0},
        ),
        ("color_at_tool", {"point_2d": [100, 100], "label": "sample", "img_idx": 0}),
    ]

    seen_names = set()
    for expected_new_idx, (name, arguments) in enumerate(calls, start=2):
        observation, reward, done, _, next_metadata, answer, terminal_gt = (
            logic.process_nonterminal_turn(
                [{"role": "assistant", "content": _xml_tool_call(name, arguments)}],
                metadata,
            )
        )
        assert reward == 0.0
        assert not done
        assert answer is None
        assert terminal_gt is None
        assert next_metadata is not None
        payload = json.loads(observation["content"][1]["text"])
        assert payload["ok"] is True
        assert payload["new_img_indices"] == [expected_new_idx]
        if name == "count_objects_tool":
            assert payload["count"] == 2
            assert all(len(blob["bbox"]) == 4 for blob in payload["blobs"])
        elif name == "find_color_tool":
            assert payload["count"] == 2
            assert payload["match_fraction"] > 0
        elif name == "color_at_tool":
            assert payload["rgb"] == [255, 0, 0]
        assert Path(observation["content"][2]["image"]).exists()
        metadata = next_metadata
        seen_names.add(name)

    assert seen_names == IMAGE_TOOL_NAMES
    assert metadata["tool_call_count"] == len(IMAGE_TOOL_NAMES)
    assert len(metadata["image_paths"]) == 2 + len(IMAGE_TOOL_NAMES)
    assert len(metadata["crop_paths"]) == len(IMAGE_TOOL_NAMES)

    forced_observation, _, done, _, forced_metadata, _, _ = (
        logic.process_nonterminal_turn(
            [
                {
                    "role": "assistant",
                    "content": _xml_tool_call(
                        "color_at_tool",
                        {"point_2d": [0, 0], "label": "extra", "img_idx": 0},
                    ),
                }
            ],
            metadata,
        )
    )
    assert not done
    assert "FINAL turn" in forced_observation["content"]
    assert forced_metadata is not None
    assert forced_metadata["force_final_next"] is True

    _, _, done, _, _, answer, terminal_gt = logic.process_nonterminal_turn(
        [{"role": "assistant", "content": "Answer: done"}], forced_metadata
    )
    assert done
    assert answer == "Answer: done"
    assert terminal_gt == "done"
