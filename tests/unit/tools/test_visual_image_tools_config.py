import importlib.metadata
from pathlib import Path

from omegaconf import OmegaConf

from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from resources_servers.gym_v.schemas import GymVResourcesServerConfig
from resources_servers.visgym.schemas import VisGymResourcesServerConfig
from responses_api_agents.gymv_agent.app import GymVAgentConfig
from responses_api_agents.visgym_agent.app import TextActionAgentConfig


def test_three_family_bundle_resolves_and_preserves_games_settings(monkeypatch):
    gym = Path(__file__).resolve().parents[3] / "3rdparty/Gym-workspace/Gym"
    monkeypatch.setenv("NEMO_GYM_EXTRA_ROOTS", str(gym))
    monkeypatch.setenv("IMAGE_TOOLS_OUTPUT_DIR", "/fixture/image-crops")
    monkeypatch.setattr(importlib.metadata, "requires", lambda name: ["openai<=2.7.2"])
    monkeypatch.setattr("nemo_gym.global_config.openai_version", "2.6.1")
    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [
                        "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml",
                        "environments/visual_games_image_tools/config.yaml",
                    ],
                    "policy_model_name": "fixture",
                    "policy_api_key": "dummy",
                    "policy_base_url": ["http://127.0.0.1:8000/v1"],
                }
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    resolved = OmegaConf.to_container(resolved, resolve=True)
    expected_routes = {
        "gym_v_agent": ("gymv_agent", GymVAgentConfig, "gym_v_resources_server", 8),
        "visgym_agent": (
            "visgym_agent",
            TextActionAgentConfig,
            "visgym_resources_server",
            35,
        ),
    }
    for name, (component, schema, resource, steps) in expected_routes.items():
        config = schema.model_validate(
            {
                **resolved[name]["responses_api_agents"][component],
                "name": name,
                "host": "127.0.0.1",
                "port": 1,
            }
        )
        assert config.resources_server.name == resource
        assert config.max_steps == steps
        assert config.max_total_sequence_length == 32768
        assert config.done_if_no_boxed_answer is True
        assert config.return_transitions is False
    for name, component, schema in (
        ("gym_v_resources_server", "gym_v", GymVResourcesServerConfig),
        ("visgym_resources_server", "visgym", VisGymResourcesServerConfig),
    ):
        config = schema.model_validate(
            {
                **resolved[name]["resources_servers"][component],
                "name": name,
                "host": "127.0.0.1",
                "port": 1,
            }
        )
        assert config.num_workers == 1
        assert config.enforce_horizon_cap is True
        assert config.return_transitions is False
    assert (
        resolved["gym_v_resources_server"]["resources_servers"]["gym_v"][
            "valid_action_bonus"
        ]
        == 0.05
    )
    image = resolved["image_tools_simple_agent"]["responses_api_agents"][
        "image_tools_agent"
    ]
    assert image["max_output_tokens"] == 512
    assert image["max_tool_calls"] == 20
    assert set(image["resource_servers_by_agent"]) == {
        "string_match_simple_agent",
        "math_with_judge_simple_agent",
        "mcqa_simple_agent",
    }
