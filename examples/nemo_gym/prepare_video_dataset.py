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
"""Convert and validate one-video-per-row NeMo-Gym JSONL datasets."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

VERIFIER_TO_AGENT = {
    "mcqa": "mcqa_simple_agent",
    "multiple-choice": "mcqa_simple_agent",
    "multiple_choice": "mcqa_simple_agent",
    "mathruler": "math_with_judge_simple_agent",
    "math-with-judge": "math_with_judge_simple_agent",
    "math_with_judge": "math_with_judge_simple_agent",
}
OPTION_RE = re.compile(
    r"(?:^|\n|\s)(?:\(([A-Ja-j])\)|([A-Ja-j])[.)：:])\s+(.+?)"
    r"(?=(?:\s|\n)(?:\([A-Ja-j]\)|[A-Ja-j][.)：:])\s+|\Z)",
    re.S,
)


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def _video_parts(row: dict[str, Any]) -> list[dict[str, Any]]:
    parts = []
    for message in row.get("responses_create_params", {}).get("input", []):
        content = message.get("content", []) if isinstance(message, dict) else []
        if not isinstance(content, list):
            continue
        parts.extend(
            part
            for part in content
            if isinstance(part, dict)
            and part.get("type") in ("input_video", "video", "video_url")
        )
    return parts


def _part_source(part: dict[str, Any]) -> str:
    value = part.get("video_url") or part.get("video") or part.get("url")
    if isinstance(value, dict):
        value = value.get("url") or value.get("path")
    return value if isinstance(value, str) else ""


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if item]
    return [value] if value else []


def _clean_question(question: str) -> str:
    question = question.strip()
    for token in ("<image>", "<video>", "<so_embedding>"):
        question = question.replace(token, "").strip()
    if "\\boxed{" not in question:
        if '"Final answer: .."' in question:
            question = question.replace('"Final answer: .."', '"\\boxed{...}"')
        else:
            question += "\nPlease put the final answer within \\boxed{...}."
    return question


def _extract_mcqa_options(question: str) -> list[dict[str, str]]:
    options = []
    seen = set()
    for match in OPTION_RE.finditer(question):
        letter = (match.group(1) or match.group(2) or "").upper()
        text = " ".join(match.group(3).strip().split())
        if len(letter) == 1 and text and letter not in seen:
            options.append({letter: text})
            seen.add(letter)
    return options


def _source_media(row: dict[str, Any], key: str) -> list[Any]:
    value = row.get(key)
    if value is None and key == "video":
        value = row.get("videos")
    return _as_list(value)


def _agent_name(row: dict[str, Any], *, line_number: int) -> str:
    agent_ref = row.get("agent_ref")
    if not isinstance(agent_ref, dict):
        raise ValueError(f"line {line_number}: agent_ref must be an object")
    name = agent_ref.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            f"line {line_number}: agent_ref.name must be a non-empty string"
        )
    return name.strip()


def _validate_expected_answer(
    row: dict[str, Any], *, agent_name: str, line_number: int
) -> None:
    answer = row.get("expected_answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(
            f"line {line_number}: expected_answer must be a non-empty string"
        )
    if agent_name != "mcqa_simple_agent":
        return

    answer = answer.strip()
    if re.fullmatch(r"[A-J]", answer) is None:
        raise ValueError(
            f"line {line_number}: MCQA expected_answer must be one uppercase "
            f"letter A-J, got {answer!r}"
        )
    options = row.get("options")
    if not isinstance(options, list) or len(options) < 2:
        raise ValueError(
            f"line {line_number}: MCQA rows require at least two parsed options"
        )
    option_keys: set[str] = set()
    for option in options:
        if not isinstance(option, dict) or len(option) != 1:
            raise ValueError(
                f"line {line_number}: every MCQA option must contain one letter"
            )
        letter, text = next(iter(option.items()))
        if (
            not isinstance(letter, str)
            or re.fullmatch(r"[A-J]", letter) is None
            or not isinstance(text, str)
            or not text.strip()
            or letter in option_keys
        ):
            raise ValueError(f"line {line_number}: invalid MCQA option {option!r}")
        option_keys.add(letter)
    if answer not in option_keys:
        raise ValueError(
            f"line {line_number}: MCQA answer {answer!r} is not present in options"
        )


def validate_row(row: dict[str, Any], *, line_number: int) -> None:
    """Validate the static video contract used by NeMo-RL Gym preprocessing."""
    agent_name = _agent_name(row, line_number=line_number)
    _validate_expected_answer(row, agent_name=agent_name, line_number=line_number)
    parts = _video_parts(row)
    if len(parts) != 1:
        raise ValueError(
            f"line {line_number}: expected exactly one video, found {len(parts)}"
        )
    source = _part_source(parts[0])
    if source.startswith("file://"):
        source = unquote(urlparse(source).path)
    path = Path(source).expanduser()
    if not path.is_absolute():
        raise ValueError(
            f"line {line_number}: video path must be absolute, got {source!r}"
        )
    if not path.is_file():
        raise FileNotFoundError(
            f"line {line_number}: video file does not exist: {path}"
        )


def convert(args: argparse.Namespace) -> None:
    rows = []
    counts: Counter[str] = Counter()
    seen_prompts = set()
    skipped = 0
    for line_number, source_row in _read_jsonl(args.input):
        videos = _source_media(source_row, args.video_key)
        if not videos:
            if args.skip_rows_without_video:
                skipped += 1
                continue
            raise ValueError(f"line {line_number}: no video media found")
        if len(videos) != 1:
            raise ValueError(
                f"line {line_number}: expected exactly one video, found {len(videos)}"
            )
        raw_video = str(videos[0])
        parsed_video = urlparse(raw_video)
        try:
            if parsed_video.scheme == "file":
                video_path = Path(unquote(parsed_video.path)).resolve(strict=True)
            elif parsed_video.scheme:
                raise ValueError(
                    f"line {line_number}: only local video paths are supported"
                )
            else:
                video_path = Path(raw_video).expanduser().resolve(strict=True)
        except FileNotFoundError:
            if args.skip_missing_local_videos:
                skipped += 1
                continue
            raise

        prompt = source_row.get(args.prompt_key)
        if prompt is None and args.prompt_key == "prompt":
            prompt = source_row.get("question")
        if not isinstance(prompt, str):
            raise ValueError(
                f"line {line_number}: {args.prompt_key!r} must be a string"
            )

        row = dict(source_row)
        verifier = str(row.get("verifier", "multiple-choice"))
        if "agent_ref" not in row:
            agent_name = args.agent_name or VERIFIER_TO_AGENT.get(
                verifier.strip().lower()
            )
            if not agent_name:
                raise ValueError(
                    f"line {line_number}: unsupported verifier {verifier!r}; "
                    "pass --agent-name explicitly"
                )
            row["agent_ref"] = {
                "type": "responses_api_agents",
                "name": agent_name,
            }
        agent_name = _agent_name(row, line_number=line_number)

        content: list[dict[str, Any]] = []
        if args.system_prompt:
            input_messages = [{"role": "system", "content": args.system_prompt}]
        else:
            input_messages = []
        content.extend(
            [
                {"type": "input_video", "video_url": str(video_path)},
                {"type": "input_text", "text": _clean_question(prompt)},
            ]
        )
        input_messages.append({"role": "user", "content": content})
        row["responses_create_params"] = {
            "input": input_messages,
            "metadata": {
                "chat_template_kwargs": json.dumps({"enable_thinking": True}),
            },
        }
        raw_answer = source_row.get("answer")
        if raw_answer is None:
            raw_answer = source_row.get("expected_answer")
        answer = "" if raw_answer is None else str(raw_answer).strip()
        if len(answer) == 1 and answer.isalpha():
            answer = answer.upper()
        row["expected_answer"] = answer
        if agent_name == "mcqa_simple_agent":
            options = _extract_mcqa_options(prompt)
            if not options:
                raise ValueError(
                    f"line {line_number}: could not parse MCQA options from prompt"
                )
            row["options"] = options
            row["grading_mode"] = "strict_single_letter_boxed"

        validate_row(row, line_number=line_number)
        signature = json.dumps(
            row["responses_create_params"]["input"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if args.deduplicate_prompts and signature in seen_prompts:
            skipped += 1
            continue
        seen_prompts.add(signature)
        rows.append(row)
        counts[agent_name] += 1
        if args.max_rows and len(rows) >= args.max_rows:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} validated rows to {args.output}")
    if skipped:
        print(f"Skipped {skipped} non-video or duplicate rows")
    for agent_name, count in sorted(counts.items()):
        print(f"  {agent_name}: {count}")


def validate(args: argparse.Namespace) -> None:
    count = 0
    for line_number, row in _read_jsonl(args.input):
        validate_row(row, line_number=line_number)
        count += 1
    print(f"Validated {count} rows in {args.input}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("--input", type=Path, required=True)
    convert_parser.add_argument("--output", type=Path, required=True)
    convert_parser.add_argument("--prompt-key", default="prompt")
    convert_parser.add_argument("--video-key", default="video")
    convert_parser.add_argument("--agent-name")
    convert_parser.add_argument("--system-prompt")
    convert_parser.add_argument(
        "--skip-rows-without-video",
        action="store_true",
        help="Skip rows with no video field instead of aborting.",
    )
    convert_parser.add_argument(
        "--skip-missing-local-videos",
        action="store_true",
        help="Skip rows whose local video file is absent instead of aborting.",
    )
    convert_parser.add_argument("--max-rows", type=int, default=0)
    convert_parser.add_argument(
        "--deduplicate-prompts",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    convert_parser.set_defaults(handler=convert)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--input", type=Path, required=True)
    validate_parser.set_defaults(handler=validate)
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    parsed_args.handler(parsed_args)
