from tests.test_suites.base_config import BaseNeMoRLTest, NeMoRLTestConfig


class TestGrpoQwen257bInstruct4n8gMegatron(BaseNeMoRLTest):
    """Test grpo-qwen2.5-7b-instruct-4n8g-megatron."""

    config = NeMoRLTestConfig(
        test_name="grpo-qwen2.5-7b-instruct-4n8g-megatron",
        algorithm="grpo",
        model_class="llm",
        test_suites=["release"],
        time_limit_minutes=180,
        overrides={
            "grpo.max_num_steps": 30,
        },
    )
