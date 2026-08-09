from tools.prepare_image_tools_gym_data import convert_row


def test_convert_row_wraps_agent_and_preserves_base_agent_ref() -> None:
    row = {
        "agent_ref": {
            "type": "responses_api_agents",
            "name": "mcqa_simple_agent",
        },
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Question?"}],
                }
            ],
        },
        "answer": "A",
    }

    converted = convert_row(row, "system prompt")

    assert converted["agent_ref"] == {
        "type": "responses_api_agents",
        "name": "image_tools_simple_agent",
    }
    assert converted["image_tools_base_agent_ref"] == row["agent_ref"]
    assert converted["responses_create_params"]["input"][0] == {
        "role": "system",
        "type": "message",
        "content": "system prompt",
    }
    assert (
        converted["responses_create_params"]["input"][1]
        == row["responses_create_params"]["input"][0]
    )
    assert row["agent_ref"]["name"] == "mcqa_simple_agent"


def test_convert_row_replaces_existing_leading_system_prompt() -> None:
    row = {
        "agent_ref": {
            "type": "resources_servers",
            "name": "string_match_simple_agent",
        },
        "responses_create_params": {
            "input": [
                {
                    "role": "system",
                    "type": "message",
                    "content": "old system prompt",
                },
                {
                    "role": "system",
                    "type": "message",
                    "content": "another old system prompt",
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Question?"}],
                },
            ],
        },
    }

    converted = convert_row(row, "image zoom system prompt")
    messages = converted["responses_create_params"]["input"]

    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0] == {
        "role": "system",
        "type": "message",
        "content": "image zoom system prompt",
    }
    assert messages[1] == row["responses_create_params"]["input"][2]
