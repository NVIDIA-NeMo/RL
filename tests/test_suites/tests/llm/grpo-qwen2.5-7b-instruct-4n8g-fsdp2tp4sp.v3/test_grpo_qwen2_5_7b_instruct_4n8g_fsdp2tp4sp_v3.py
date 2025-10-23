from tests.test_suites.base_config import BaseNeMoRLTest, NeMoRLTestConfig


class TestGrpoQwen257bInstruct4n8gFsdp2tp4spV3(BaseNeMoRLTest):
    """Test grpo-qwen2.5-7b-instruct-4n8g-fsdp2tp4sp.v3."""

    config = NeMoRLTestConfig(
        test_name="grpo-qwen2.5-7b-instruct-4n8g-fsdp2tp4sp.v3",
        algorithm="grpo",
        model_class="llm",
        test_suites=["nightly"],
        time_limit_minutes=180,
        overrides={
            "grpo.max_num_steps": 30,
        },
    )
