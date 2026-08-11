# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for conversion of Search-R1's official QA parquet files."""

import json

import pandas as pd

from prepare_search_r1_data import prepare


def _source_row(identifier: str, source: str) -> dict:
    prompt = "Question with <search>query</search> and <answer>answer</answer>."
    return {
        "id": identifier,
        "question": "Where?",
        "golden_answers": ["There"],
        "data_source": source,
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "fact-reasoning",
        "reward_model": {"style": "rule"},
        "extra_info": {"split": "train"},
    }


def test_prepare_preserves_official_prompt_and_answers(tmp_path) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    pd.DataFrame(
        [_source_row("train-0", "nq"), _source_row("train-1", "hotpotqa")]
    ).to_parquet(source_dir / "train.parquet", index=False)
    pd.DataFrame([_source_row("test-0", "nq")]).to_parquet(
        source_dir / "test.parquet", index=False
    )

    manifest = prepare(
        source_dir, output_dir, max_train_rows=1, max_validation_rows=None
    )

    train_row = json.loads((output_dir / "train.jsonl").read_text().splitlines()[0])
    assert train_row["responses_create_params"]["input"] == [
        {
            "role": "user",
            "content": "Question with <search>query</search> and <answer>answer</answer>.",
        }
    ]
    assert train_row["answers"] == ["There"]
    assert train_row["agent_ref"]["name"] == "ai_search_search_r1_agent"
    assert train_row["responses_create_params"]["tools"] == []
    assert manifest["train"]["rows"] == 1
    assert manifest["validation"]["rows"] == 1
