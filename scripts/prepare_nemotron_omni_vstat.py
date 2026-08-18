#!/usr/bin/env python3
"""Download and prepare a small VSTAT split for Nemotron Omni video GRPO."""

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for extracted media and source JSONL files.",
    )
    parser.add_argument(
        "--repo-id",
        default="ShushengYang/VSTAT",
        help="Hugging Face dataset repository.",
    )
    parser.add_argument(
        "--num-rows",
        type=int,
        default=8,
        help="Maximum number of MCQ examples to prepare.",
    )
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def main() -> None:
    args = parse_args()
    if args.num_rows < 3:
        raise ValueError("--num-rows must be at least 3.")

    root = args.output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    parquet_path = hf_hub_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        filename="test.parquet",
    )
    archive_path = hf_hub_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        filename="videos.zip",
    )

    media_root = root / "media"
    marker = media_root / ".extracted"
    if not marker.exists():
        media_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(media_root)
        marker.touch()

    video_files = {
        path.name: path.resolve() for path in media_root.rglob("*.mp4")
    }
    converted = []
    for row in pq.read_table(parquet_path).to_pylist():
        if str(row.get("answer_type", "")).lower() != "mcq":
            continue

        relative_video = str(row["video"])
        video_path = media_root / relative_video
        if not video_path.exists():
            video_path = video_files.get(Path(relative_video).name)
        if video_path is None or not video_path.exists():
            continue

        choices = [str(choice) for choice in row.get("choices") or []]
        if not choices:
            continue
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        options = "\n".join(
            f"{letters[index]}. {choice}"
            for index, choice in enumerate(choices)
        )
        converted.append(
            {
                "prompt": (
                    "Answer the multiple-choice question using the video. "
                    "Return the final answer as a boxed letter.\n"
                    f"Question: {row['question']}\n{options}"
                ),
                "video": str(video_path),
                "answer": str(row["answer"]).strip().upper(),
                "verifier": "multiple-choice",
            }
        )
        if len(converted) >= args.num_rows:
            break

    if len(converted) < 3:
        raise RuntimeError(
            f"Only resolved {len(converted)} VSTAT MCQ rows with local videos."
        )

    split = max(2, len(converted) - 2)
    train_source = root / "train-source.jsonl"
    val_source = root / "val-source.jsonl"
    write_jsonl(train_source, converted[:split])
    write_jsonl(val_source, converted[split:])

    repo_root = Path(__file__).resolve().parents[1]
    converter = repo_root / "examples/nemo_gym/prepare_video_dataset.py"
    for source, output in (
        (train_source, root / "train-gym.jsonl"),
        (val_source, root / "val-gym.jsonl"),
    ):
        subprocess.run(
            [
                sys.executable,
                str(converter),
                "convert",
                "--input",
                str(source),
                "--output",
                str(output),
                "--require-video",
            ],
            cwd=repo_root,
            check=True,
        )

    print(f"Prepared {len(converted)} VSTAT examples under {root}")


if __name__ == "__main__":
    main()
