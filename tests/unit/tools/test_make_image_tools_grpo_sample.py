import json
from pathlib import Path

from PIL import Image

from tools.make_image_tools_grpo_sample import build_sample


def test_build_sample_creates_runnable_rows(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Use the image tools.")

    result = build_sample(
        tmp_path / "sample",
        system_prompt_path=prompt,
        train_repeats=3,
        validation_repeats=1,
    )

    with Path(result["train"]).open() as source:
        rows = [json.loads(line) for line in source]
    assert len(rows) == 3
    assert rows[0]["agent_ref"]["name"] == "image_tools_simple_agent"
    assert rows[0]["image_tools_base_agent_ref"]["name"] == "string_match_simple_agent"
    image_path = Path(
        rows[0]["responses_create_params"]["input"][1]["content"][1]["image_url"]
    )
    assert image_path.is_absolute()
    with Image.open(image_path) as image:
        assert image.size == (768, 512)

    with Path(result["validation"]).open() as source:
        assert sum(1 for _ in source) == 1


def test_default_sample_prompt_advertises_requested_tool(tmp_path: Path) -> None:
    result = build_sample(tmp_path / "sample")

    with Path(result["train"]).open() as source:
        row = json.loads(next(source))
    system_prompt = row["responses_create_params"]["input"][0]["content"]
    user_prompt = row["responses_create_params"]["input"][1]["content"][0]["text"]

    assert "<name>count_objects_tool</name>" in system_prompt
    assert "count_objects_tool" in user_prompt
