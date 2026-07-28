import json
import sys

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
            "--require-video",
            "--skip-missing-local-videos",
        ],
    )

    args = prepare_video_dataset.parse_args()
    args.handler(args)

    converted_rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(converted_rows) == 1
    content = converted_rows[0]["responses_create_params"]["input"][0]["content"]
    assert content[0]["video_url"] == str(existing_video.resolve())
    assert "Skipped 1 non-video or duplicate rows" in capsys.readouterr().out


def test_clean_question_replaces_legacy_final_answer_instruction():
    question = 'Reason carefully, then answer using the format "Final answer: ..".'

    cleaned = prepare_video_dataset._clean_question(question)

    assert '"\\boxed{...}"' in cleaned
    assert "Final answer: .." not in cleaned
