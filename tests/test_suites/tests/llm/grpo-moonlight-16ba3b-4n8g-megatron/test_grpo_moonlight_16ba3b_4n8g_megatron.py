from tests.test_suites.base_config import BaseNeMoRLTest, NeMoRLTestConfig


class TestGrpoMoonlight16ba3b4n8gMegatron(BaseNeMoRLTest):
    """Test grpo-moonlight-16ba3b-4n8g-megatron."""

    config = NeMoRLTestConfig(
        test_name="grpo-moonlight-16ba3b-4n8g-megatron",
        algorithm="grpo",
        model_class="llm",
        test_suites=["nightly"],
        time_limit_minutes=150,
        overrides={
            "grpo.max_num_steps": 30,
        },
    )
