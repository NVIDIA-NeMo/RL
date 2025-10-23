from tests.test_suites.base_config import BaseNeMoRLTest, NeMoRLTestConfig


class TestDapoQwen257b(BaseNeMoRLTest):
    """Test dapo-qwen2.5-7b."""

    config = NeMoRLTestConfig(
        test_name="dapo-qwen2.5-7b",
        algorithm="dapo",
        model_class="llm",
        test_suites=["release"],
        time_limit_minutes=240,
        overrides={
            "grpo.max_num_steps": 20,
        },
    )
