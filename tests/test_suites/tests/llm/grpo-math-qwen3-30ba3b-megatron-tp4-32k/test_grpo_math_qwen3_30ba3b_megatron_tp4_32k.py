from tests.test_suites.base_config import BaseNeMoRLTest, NeMoRLTestConfig


class TestGrpoMathQwen330ba3bMegatronTp432k(BaseNeMoRLTest):
    """Test grpo-math-qwen3-30ba3b-megatron-tp4-32k."""

    config = NeMoRLTestConfig(
        test_name="grpo-math-qwen3-30ba3b-megatron-tp4-32k",
        algorithm="grpo",
        model_class="llm",
        test_suites=["nightly"],
        time_limit_minutes=240,
        overrides={
            "grpo.max_num_steps": 3,
        },
    )
