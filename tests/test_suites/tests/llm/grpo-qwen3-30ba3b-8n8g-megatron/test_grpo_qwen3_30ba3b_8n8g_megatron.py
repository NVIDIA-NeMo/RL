from tests.test_suites.base_config import BaseNeMoRLTest, NeMoRLTestConfig


class TestGrpoQwen330ba3b8n8gMegatron(BaseNeMoRLTest):
    config = NeMoRLTestConfig(
        test_name="grpo-qwen3-30ba3b-8n8g-megatron",
        algorithm="grpo",
        model_class="llm",
        test_suites=["release"],
        time_limit_minutes=240,
        overrides={
            "grpo.max_num_steps": 30,
        },
    )
