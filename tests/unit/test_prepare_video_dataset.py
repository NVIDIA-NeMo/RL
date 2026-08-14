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
import sys

import pytest

from examples.nemo_gym import prepare_video_dataset


def test_converter_skips_missing_local_videos_when_requested(
    monkeypatch, tmp_path, capsys
):
    input_path = tmp_path / "raw.jsonl"
    output_path = tmp_path / "gym.jsonl"
    existing_video = tmp_path / "existing.mp4"
    existing_video.write_bytes(b"video")
    missing_video = tmp_path / "missing.mp4"
    rows = [
        {
            "video": str(existing_video),
            "question": "What happens?\nA. Run\nB. Stop",
            "answer": "A",
            "verifier": "multiple-choice",
        },
        {
            "video": str(missing_video),
            "question": "What is missing?\nA. Video\nB. Audio",
            "answer": "A",
            "verifier": "multiple-choice",
        },
    ]
    input_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_video_dataset.py",
            "convert",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--skip-missing-local-videos",
        ],
    )

    args = prepare_video_dataset.parse_args()
    args.handler(args)

    converted_rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(converted_rows) == 1
    content = converted_rows[0]["responses_create_params"]["input"][0]["content"]
    assert content[0]["video_url"] == str(existing_video.resolve())
    metadata = converted_rows[0]["responses_create_params"]["metadata"]
    assert metadata["chat_template_kwargs"] == {"enable_thinking": True}
    assert "extra_body" not in metadata
    assert "extraction_mode" not in converted_rows[0]
    assert "Skipped 1 non-video or duplicate rows" in capsys.readouterr().out


def test_converter_can_skip_rows_without_video(monkeypatch, tmp_path, capsys):
    input_path = tmp_path / "raw.jsonl"
    output_path = tmp_path / "gym.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "question": "What happens?\nA. Run\nB. Stop",
                "answer": "A",
                "verifier": "multiple-choice",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_video_dataset.py",
            "convert",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--skip-rows-without-video",
        ],
    )

    args = prepare_video_dataset.parse_args()
    args.handler(args)

    assert output_path.read_text(encoding="utf-8") == ""
    assert "Skipped 1 non-video or duplicate rows" in capsys.readouterr().out


def test_clean_question_replaces_legacy_final_answer_instruction():
    question = 'Reason carefully, then answer using the format "Final answer: ..".'

    cleaned = prepare_video_dataset._clean_question(question)

    assert '"\\boxed{...}"' in cleaned
    assert "Final answer: .." not in cleaned


@pytest.mark.parametrize(
    ("question", "answer", "error"),
    [
        ("What happens?\nA. Run\nB. Stop", None, "expected_answer"),
        ("What happens?\nA. Run\nB. Stop", "AB", "one uppercase letter"),
        ("What happens without choices?", "A", "could not parse MCQA options"),
        ("What happens?\nA. Run\nB. Stop", "C", "not present in options"),
    ],
)
def test_converter_rejects_invalid_mcqa_rows(
    monkeypatch, tmp_path, question, answer, error
):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    input_path = tmp_path / "raw.jsonl"
    output_path = tmp_path / "gym.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "video": str(video_path),
                "question": question,
                "answer": answer,
                "verifier": "multiple-choice",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_video_dataset.py",
            "convert",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )

    with pytest.raises(ValueError, match=error):
        args = prepare_video_dataset.parse_args()
        args.handler(args)


def test_validate_row_requires_agent_name(tmp_path):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    row = {
        "agent_ref": {},
        "expected_answer": "A",
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_video", "video_url": str(video_path)}],
                }
            ]
        },
    }

    with pytest.raises(ValueError, match="agent_ref.name"):
        prepare_video_dataset.validate_row(row, line_number=1)
