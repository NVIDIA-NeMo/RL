from pathlib import Path

from tools.prepare_image_tools_sft_gym_data import (
    IMAGE_TOOL_NAMES,
    convert_sft_row,
    select_covering_rows,
)


def _row(tmp_path: Path, row_id: str, tools: set[str]) -> dict:
    image_path = tmp_path / f"{row_id}.png"
    image_path.write_bytes(b"not-decoded-during-conversion")
    tool_turn = "".join(
        f"<tool_call><function={name}></function></tool_call>" for name in tools
    )
    return {
        "id": row_id,
        "image": [image_path.name],
        "conversations": [
            {"from": "system", "value": "tool system prompt"},
            {"from": "human", "value": "Image-1: <image>\nQuestion?"},
            {"from": "gpt", "value": tool_turn},
        ],
        "__metadata__": {
            "media_root": str(tmp_path),
            "answer": "A",
            "dataset_version": "unit",
            "n_tool_calls": len(tools),
        },
    }


def test_convert_sft_row_interleaves_image_and_configures_string_match(
    tmp_path: Path,
) -> None:
    source = _row(tmp_path, "sample", {"image_crop_tool"})

    converted = convert_sft_row(source)

    assert converted["agent_ref"]["name"] == "image_tools_simple_agent"
    assert (
        converted["image_tools_base_agent_ref"]["name"] == "string_match_simple_agent"
    )
    assert converted["expected_answer"] == "A"
    assert converted["source_tool_names"] == ["image_crop_tool"]
    messages = converted["responses_create_params"]["input"]
    assert messages[0]["role"] == "system"
    assert [part["type"] for part in messages[1]["content"]] == [
        "input_text",
        "input_image",
        "input_text",
    ]
    assert messages[1]["content"][1]["image_url"] == str(
        (tmp_path / "sample.png").resolve()
    )


def test_select_covering_rows_returns_deterministic_small_cover(tmp_path: Path) -> None:
    tool_names = sorted(IMAGE_TOOL_NAMES)
    rows = [
        _row(tmp_path, "first", set(tool_names[:6])),
        _row(tmp_path, "second", set(tool_names[6:])),
        _row(tmp_path, "redundant", {tool_names[0]}),
    ]

    selected = select_covering_rows(rows, scan_limit=len(rows))

    assert [row["id"] for row in selected] == ["first", "second"]
