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
import json

import pytest

from nemo_rl.data.datasets.text_to_image_prompt import (
    TextToImagePromptDataset,
    text_to_image_collate_fn,
)


def test_txt_parsing_skips_blank_lines(tmp_path):
    p = tmp_path / "prompts.txt"
    p.write_text("a cat\n\n  \nan astronaut riding a horse\n")
    ds = TextToImagePromptDataset(p)
    assert len(ds) == 2
    assert ds[0]["prompt"] == "a cat"
    assert ds[1]["prompt"] == "an astronaut riding a horse"


def test_jsonl_parsing_optional_fields(tmp_path):
    p = tmp_path / "prompts.jsonl"
    p.write_text(
        json.dumps({"prompt": "a", "negative_prompt": "ugly"})
        + "\n"
        + json.dumps({"prompt": "b", "metadata": {"category": "x"}})
        + "\n"
    )
    ds = TextToImagePromptDataset(p)
    assert len(ds) == 2
    assert ds[0]["prompt"] == "a"
    assert ds[0]["negative_prompt"] == "ugly"
    assert ds[1]["prompt"] == "b"
    assert ds[1]["negative_prompt"] == " "  # default
    assert ds[1]["metadata"] == {"category": "x"}


def test_default_negative_prompt_overridable(tmp_path):
    p = tmp_path / "prompts.txt"
    p.write_text("a\n")
    ds = TextToImagePromptDataset(p, negative_prompt_default="bad")
    assert ds[0]["negative_prompt"] == "bad"


def test_unsupported_extension_raises(tmp_path):
    p = tmp_path / "prompts.csv"
    p.write_text("prompt\na\n")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        TextToImagePromptDataset(p)


def test_jsonl_missing_prompt_key_raises(tmp_path):
    p = tmp_path / "prompts.jsonl"
    p.write_text(json.dumps({"foo": "bar"}) + "\n")
    with pytest.raises(ValueError, match="without 'prompt' key"):
        TextToImagePromptDataset(p)


def test_collate_fn_packs_into_batched_data_dict(tmp_path):
    p = tmp_path / "prompts.txt"
    p.write_text("a\nb\nc\n")
    ds = TextToImagePromptDataset(p)
    batch = text_to_image_collate_fn([ds[0], ds[1], ds[2]])
    assert batch["prompts"] == ["a", "b", "c"]
    assert batch["negative_prompts"] == [" ", " ", " "]
    assert batch["idx"] == [0, 1, 2]
    assert batch["task_names"] == ["text_to_image"] * 3


def test_ocr_line_to_record_extracts_ground_truth():
    from tools.export_ocr_prompts import ocr_line_to_record

    line = 'The image displays "Hello World".'
    rec = ocr_line_to_record(line)
    assert rec == {
        "prompt": line,
        "metadata": {"ground_truth": "Hello World"},
    }


def test_ocr_line_to_record_rejects_line_without_quotes():
    import pytest

    from tools.export_ocr_prompts import ocr_line_to_record

    with pytest.raises(ValueError):
        ocr_line_to_record("no quoted text here")
