from tests.test_suites.base_config import BaseNeMoRLTest, NeMoRLTestConfig


class TestGrpoLlama321bInstruct1n8gMegatron(BaseNeMoRLTest):
    """Test grpo-llama3.2-1b-instruct-1n8g-megatron."""

    config = NeMoRLTestConfig(
        test_name="grpo-llama3.2-1b-instruct-1n8g-megatron",
        algorithm="grpo",
        model_class="llm",
        test_suites=["nightly", "long"],
        time_limit_minutes=180,
        overrides={
            "grpo.max_num_steps": 500,
        },
    )
