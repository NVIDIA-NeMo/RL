import copy
import json

import pytest

from tools.check_visual_image_tools_mix import (
    GYM_V_TRAIN,
    GYM_V_VALIDATION,
    VISGYM_TRAIN,
    audit_mix,
)


def game_row(env, agent):
    return {
        "env_id": env,
        "task_id": env,
        "agent_ref": {"name": agent, "type": "responses_api_agents"},
        "task_source": agent,
        "responses_create_params": {"input": [], "max_output_tokens": 512},
    }


@pytest.fixture
def manifests():
    train = [game_row(env, "gym_v_agent") for env in sorted(GYM_V_TRAIN)]
    train += [game_row(env, "visgym_agent") for env in sorted(VISGYM_TRAIN)]
    image = game_row("image_tools/synthetic", "image_tools_simple_agent")
    image.update(
        expected_answer="blue",
        image_tools_base_agent_ref={
            "name": "string_match_simple_agent",
            "type": "responses_api_agents",
        },
        task_metadata={"split": "train"},
    )
    image["responses_create_params"].update(
        tools=[],
        parallel_tool_calls=False,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": "/synthetic/not-read.png"}
                ],
            }
        ],
    )
    train.append(image)
    validation = [game_row(env, "gym_v_agent") for env in sorted(GYM_V_VALIDATION)]
    return {"train": train, "validation": validation, "max_output_tokens": 512}


def test_full_coverage_routes_counts_and_no_mutation(manifests):
    original = copy.deepcopy(manifests)
    result = audit_mix(**manifests)
    assert result["train_agent_rows"] == {
        "gym_v_agent": 22,
        "visgym_agent": 12,
        "image_tools_simple_agent": 1,
    }
    assert result["rows"] == {"train": 35, "validation": 8}
    assert sum(result["unweighted_row_fractions"].values()) == pytest.approx(1)
    assert result["runtime_verified"] is False
    assert manifests == original


@pytest.mark.parametrize("index", [0, 22])
def test_missing_game_is_not_silently_dropped(manifests, index):
    manifests["train"].pop(index)
    with pytest.raises(ValueError, match="Missing games"):
        audit_mix(**manifests)


def test_image_family_required(manifests):
    manifests["train"].pop()
    with pytest.raises(ValueError, match="no image-tool"):
        audit_mix(**manifests)


@pytest.mark.parametrize("field", ["agent_ref", "task_source"])
def test_ambiguous_routing_rejected(manifests, field):
    manifests["train"][0].pop(field)
    with pytest.raises(ValueError, match=field):
        audit_mix(**manifests)


@pytest.mark.parametrize("cap", [True, 0, 1024, None])
def test_invalid_or_excessive_cap_rejected(manifests, cap):
    manifests["train"][0]["responses_create_params"]["max_output_tokens"] = cap
    with pytest.raises(ValueError, match="max_output_tokens"):
        audit_mix(**manifests)


def test_held_out_cannot_enter_train(manifests):
    manifests["train"].append(manifests["validation"].pop())
    with pytest.raises(ValueError, match="held-out training"):
        audit_mix(**manifests)


def test_validation_cannot_expand_to_images(manifests):
    manifests["validation"].append(manifests["train"].pop())
    with pytest.raises(ValueError, match="only held-out Gym-V"):
        audit_mix(**manifests)


def test_duplicate_task_rejected(manifests):
    manifests["train"].append(copy.deepcopy(manifests["train"][0]))
    with pytest.raises(ValueError, match="duplicate task_id"):
        audit_mix(**manifests)


@pytest.mark.parametrize("mutation", ["pivot", "validation"])
def test_image_data_contract_preserved(manifests, mutation):
    image = manifests["train"][-1]
    if mutation == "pivot":
        image["expected_action"] = "zoom"
    else:
        image["task_metadata"]["split"] = "validation"
    with pytest.raises(ValueError, match="not pivot|prepared training split"):
        audit_mix(**manifests)


def test_preparation_preserves_every_row_and_refuses_overwrite(manifests, tmp_path):
    from tools.prepare_visual_image_tools_mix import prepare

    games, images, validation, output = [
        tmp_path / name for name in ("games", "images", "validation", "output")
    ]
    for path, rows in (
        (games, manifests["train"][:-1]),
        (images, manifests["train"][-1:]),
        (validation, manifests["validation"]),
    ):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = prepare(games=games, images=images, validation=validation, output=output)
    assert [json.loads(line) for line in output.read_text().splitlines()] == manifests[
        "train"
    ]
    assert report["rows"] == {"train": 35, "validation": 8}
    with pytest.raises(FileExistsError):
        prepare(games=games, images=images, validation=validation, output=output)
