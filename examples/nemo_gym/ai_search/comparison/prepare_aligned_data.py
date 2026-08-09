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

"""Build the deterministic data fixture shared by all AI-search trainers."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


NUM_TRAIN_EXAMPLES = 32
NUM_VALIDATION_EXAMPLES = 8
NUM_EXAMPLES = NUM_TRAIN_EXAMPLES + NUM_VALIDATION_EXAMPLES

VERL_PROMPT = """Answer the given question using the private search engine. \
You must reason inside <think> and </think>. When you need evidence, emit exactly \
<search>query</search>; the engine returns passages inside <information> and \
</information>. Search again when a first passage names an intermediate entity. \
When you have the answer, emit only <answer>answer</answer>. Example final format: \
<answer>Example</answer>. Question: {question}\n"""

NEMO_SYSTEM_PROMPT = """You answer questions using a private document collection. \
You must call the search tool before answering. These questions require two linked \
facts, so search again when the first result names an intermediate person. Use \
concise queries, do not invent facts, and finish with exactly one line: \
Final Answer: <answer>."""

SEARCH_TOOL = {
    "type": "function",
    "name": "search",
    "description": "Search the private document collection and return ranked passages.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A concise natural-language search query.",
            },
            "top_k": {
                "type": "integer",
                "minimum": 1,
                "maximum": 3,
                "description": "Number of passages to return.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "strict": False,
}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_rows() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    corpus_rows: list[dict[str, Any]] = []
    nemo_rows: list[dict[str, Any]] = []
    verl_rows: list[dict[str, Any]] = []

    for index in range(NUM_EXAMPLES):
        project = f"Meridian-{index:03d}"
        custodian = f"Talven-{index:03d}"
        city = f"Orinth-{index:03d}"
        project_doc_id = f"project-{index:03d}"
        person_doc_id = f"person-{index:03d}"
        question = (
            f"In which city was the official custodian of Project {project} born?"
        )

        corpus_rows.extend(
            [
                {
                    "id": project_doc_id,
                    "title": f"Project {project} registry",
                    "text": (
                        f"The official custodian of Project {project} is {custodian}. "
                        f"The registry was audited in {2040 + index}."
                    ),
                },
                {
                    "id": person_doc_id,
                    "title": f"Biography of {custodian}",
                    "text": (
                        f"Archive custodian {custodian} was born in the city of {city}. "
                        "The biography was deposited after the registry audit."
                    ),
                },
                {
                    "id": f"project-distractor-{index:03d}",
                    "title": f"Project {project} logistics",
                    "text": (
                        f"Project {project} ships instruments through Port "
                        f"Caldra-{index:03d}; this logistics record does not identify "
                        "its official custodian."
                    ),
                },
                {
                    "id": f"person-distractor-{index:03d}",
                    "title": f"Awards received by {custodian}",
                    "text": (
                        f"{custodian} received the Vesper-{index:03d} medal in "
                        f"Lunara-{index:03d}. The ceremony location is not a birthplace."
                    ),
                },
            ]
        )

        nemo_rows.append(
            {
                "question": question,
                "answers": [city],
                "supporting_doc_ids": [project_doc_id, person_doc_id],
                "agent_ref": {
                    "type": "responses_api_agents",
                    "name": "ai_search_simple_agent",
                },
                "responses_create_params": {
                    "input": [
                        {"role": "system", "content": NEMO_SYSTEM_PROMPT},
                        {"role": "user", "content": question},
                    ],
                    "tools": [SEARCH_TOOL],
                    "parallel_tool_calls": False,
                    "tool_choice": "auto",
                },
            }
        )

        split = "train" if index < NUM_TRAIN_EXAMPLES else "validation"
        verl_rows.append(
            {
                "data_source": "nq",
                "prompt": [
                    {
                        "role": "user",
                        "content": VERL_PROMPT.format(question=question),
                    }
                ],
                "ability": "fact-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {"target": [city]},
                },
                "extra_info": {"split": split, "index": index},
                "question": question,
                "golden_answers": [city],
            }
        )

    return corpus_rows, nemo_rows, verl_rows


def prepare(output_root: Path, server_data_root: Path) -> dict[str, Any]:
    """Write canonical, NeMo Gym, and veRL views of the same fixture."""
    corpus_rows, nemo_rows, verl_rows = _build_rows()
    output_root.mkdir(parents=True, exist_ok=True)
    server_data_root.mkdir(parents=True, exist_ok=True)

    nemo_corpus_path = server_data_root / "corpus.jsonl"
    retrieval_corpus_path = output_root / "retrieval_corpus.jsonl"
    train_jsonl_path = server_data_root / "train.jsonl"
    validation_jsonl_path = server_data_root / "validation.jsonl"
    train_parquet_path = output_root / "train.parquet"
    validation_parquet_path = output_root / "validation.parquet"

    _write_jsonl(nemo_corpus_path, corpus_rows)
    _write_jsonl(
        retrieval_corpus_path,
        [
            {
                **row,
                "contents": f"{row['title']}\n{row['text']}",
            }
            for row in corpus_rows
        ],
    )
    _write_jsonl(train_jsonl_path, nemo_rows[:NUM_TRAIN_EXAMPLES])
    _write_jsonl(validation_jsonl_path, nemo_rows[NUM_TRAIN_EXAMPLES:])
    pd.DataFrame(verl_rows[:NUM_TRAIN_EXAMPLES]).to_parquet(
        train_parquet_path, index=False
    )
    pd.DataFrame(verl_rows[NUM_TRAIN_EXAMPLES:]).to_parquet(
        validation_parquet_path, index=False
    )

    paths = [
        nemo_corpus_path,
        retrieval_corpus_path,
        train_jsonl_path,
        validation_jsonl_path,
        train_parquet_path,
        validation_parquet_path,
    ]
    manifest = {
        "schema_version": 1,
        "train_examples": NUM_TRAIN_EXAMPLES,
        "validation_examples": NUM_VALIDATION_EXAMPLES,
        "documents": len(corpus_rows),
        "files": {str(path): _sha256(path) for path in paths},
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    plugin_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=plugin_root / "comparison" / "generated",
    )
    parser.add_argument(
        "--server-data-root",
        type=Path,
        default=plugin_root / "resources_servers" / "ai_search" / "data" / "aligned",
    )
    args = parser.parse_args()
    manifest = prepare(
        output_root=args.output_root.resolve(),
        server_data_root=args.server_data_root.resolve(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
