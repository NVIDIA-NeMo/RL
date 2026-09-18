# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import json

import pytest

from tools.prepare_image_tools_grpo import AGENT, prepare, validate_row, write_bundle


def row(path, identifier, source_id=None):
    return {
        "agent_ref": dict(AGENT),
        "image_tools_base_agent_ref": {
            "name": "string_match_simple_agent",
            "type": "responses_api_agents",
        },
        "expected_answer": "blue",
        "metadata": {"source_id": source_id},
        "dataset": "synthetic",
        "responses_create_params": {
            "tools": [],
            "parallel_tool_calls": False,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": identifier},
                        {"type": "input_image", "image_url": str(path)},
                    ],
                }
            ],
        },
    }


def dataset(tmp_path):
    paths = [tmp_path / f"image{i}.png" for i in range(5)]
    for i, path in enumerate(paths):
        path.write_bytes(f"synthetic-image-{i}".encode())
    rows = [
        row(paths[0], "a", "shared-source"),
        row(paths[1], "b", "shared-source"),
        row(paths[1], "c"),
        row(paths[2], "d"),
        row(paths[3], "e"),
        row(paths[4], "f"),
    ]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return source, rows, paths


def run(source, **kwargs):
    return prepare(
        source, validation_fraction=0.4, seed=13, max_output_tokens=512, **kwargs
    )


def test_transitive_groups_preserve_prompts_answers_and_cover_all_rows(tmp_path):
    source, originals, _ = dataset(tmp_path)
    before = source.read_bytes()
    splits, assets, report = run(source)
    assert source.read_bytes() == before
    assert report["rows"] == 6
    assert report["groups"] == 4
    assert len(assets) == 5
    memberships = {}
    for split, rows in splits.items():
        assert rows
        for current in rows:
            question = current["responses_create_params"]["input"][0]["content"][0][
                "text"
            ]
            memberships[question] = split
            original = next(
                r
                for r in originals
                if r["responses_create_params"]["input"][0]["content"][0]["text"]
                == question
            )
            assert (
                current["responses_create_params"]["input"]
                == original["responses_create_params"]["input"]
            )
            assert current["metadata"] == original["metadata"]
            assert current["expected_answer"] == original["expected_answer"]
            assert (
                current["image_tools_base_agent_ref"]
                == original["image_tools_base_agent_ref"]
            )
    assert memberships["a"] == memberships["b"] == memberships["c"]
    assert len(memberships) == 6


def test_split_and_ids_stable_under_source_shuffle(tmp_path):
    source, rows, _ = dataset(tmp_path)
    first, _, _ = run(source)
    source.write_text("".join(json.dumps(r) + "\n" for r in reversed(rows)))
    second, _, _ = run(source)
    assert {s: [r["task_id"] for r in v] for s, v in first.items()} == {
        s: [r["task_id"] for r in v] for s, v in second.items()
    }


def test_hashing_groups_identical_bytes_at_different_paths(tmp_path):
    source, _, paths = dataset(tmp_path)
    paths[4].write_bytes(paths[3].read_bytes())
    _, _, report = run(source, hash_images=True)
    assert report["groups"] == 3
    assert report["image_content_hashes_checked"] is True


def test_regroup_existing_splits_by_content_without_copying_images(tmp_path):
    source, rows, paths = dataset(tmp_path)
    paths[4].write_bytes(paths[3].read_bytes())
    left, right = tmp_path / "old_train.jsonl", tmp_path / "old_validation.jsonl"
    left.write_text("".join(json.dumps(r) + "\n" for r in rows[:-1]))
    right.write_text(json.dumps(rows[-1]) + "\n")
    splits, assets, report = run([left, right], hash_images=True)
    expected_splits, _, _ = run(source, hash_images=True)
    assert splits == expected_splits
    assert report["rows"] == len(rows)
    content_hashes = {a["source_path"]: a["sha256"] for a in assets}
    sets = {}
    for split, current_rows in splits.items():
        sets[split] = {
            content_hashes[
                r["responses_create_params"]["input"][0]["content"][1]["image_url"]
            ]
            for r in current_rows
        }
    assert not (sets["train"] & sets["validation"])


def test_image_remap_is_explicit_and_does_not_copy_assets(tmp_path):
    source, _, _ = dataset(tmp_path)
    target = tmp_path / "not-created"
    splits, assets, _ = run(source, asset_root=target)
    assert not target.exists()
    expected = {str(target / a["relative_path"]) for a in assets}
    actual = {
        r["responses_create_params"]["input"][0]["content"][1]["image_url"]
        for rs in splits.values()
        for r in rs
    }
    assert actual == expected
    assert all(len(a["relative_path"].split("/")) == 2 for a in assets)


@pytest.mark.parametrize("change", ["pivot", "grader", "tools", "cap", "missing_image"])
def test_bad_rows_are_not_silently_normalized(tmp_path, change):
    source, rows, paths = dataset(tmp_path)
    bad = copy.deepcopy(rows)
    if change == "pivot":
        bad[0]["expected_action"] = {}
    elif change == "grader":
        bad[0]["image_tools_base_agent_ref"]["name"] = "unknown"
    elif change == "tools":
        bad[0]["responses_create_params"]["tools"] = [{"type": "function"}]
    elif change == "cap":
        bad[0]["responses_create_params"]["max_output_tokens"] = True
    else:
        paths[0].unlink()
    source.write_text("".join(json.dumps(r) + "\n" for r in bad))
    with pytest.raises((ValueError, FileNotFoundError)):
        run(source)


def test_existing_lower_caps_and_new_bundle_only(tmp_path):
    source, rows, _ = dataset(tmp_path)
    rows[0]["responses_create_params"]["max_output_tokens"] = 256
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    splits, assets, report = run(source)
    caps = sorted(
        r["responses_create_params"]["max_output_tokens"]
        for rs in splits.values()
        for r in rs
    )
    assert caps == [256, 512, 512, 512, 512, 512]
    output = tmp_path / "bundle"
    write_bundle(output, splits, assets, report)
    before = (output / "train.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        write_bundle(output, splits, assets, report)
    assert (output / "train.jsonl").read_bytes() == before


@pytest.mark.parametrize("options", [None, [], [{"A": "red"}], [{"B": None}], ["B"]])
def test_mcqa_requires_top_level_options_containing_gold(tmp_path, options):
    current = row(tmp_path / "image.png", "mcqa")
    current["image_tools_base_agent_ref"]["name"] = "mcqa_simple_agent"
    current["expected_answer"] = "B"
    current["options"] = options
    current["metadata"]["options"] = [{"B": "blue"}]
    with pytest.raises(ValueError, match="MCQA"):
        validate_row(current, 1)
    current["options"] = [{"A": "red"}, {"B": "blue"}]
    validate_row(current, 1)
