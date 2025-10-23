from tests.test_suites.base_config import BaseNeMoRLTest, NeMoRLTestConfig


class TestGrpoGemma31bIt1n8gFsdp2tp1(BaseNeMoRLTest):
    config = NeMoRLTestConfig(
        test_name="grpo-gemma3-1b-it-1n8g-fsdp2tp1",
        algorithm="grpo",
        model_class="llm",
        test_suites=["nightly", "long"],
        time_limit_minutes=120,
        overrides={
            "grpo.max_num_steps": 400,
        },
    )
