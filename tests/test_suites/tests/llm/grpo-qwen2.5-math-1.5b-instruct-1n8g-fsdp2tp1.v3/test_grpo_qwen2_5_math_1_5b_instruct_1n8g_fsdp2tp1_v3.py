from tests.test_suites.base_config import BaseNeMoRLTest, NeMoRLTestConfig


class TestGrpoQwen25Math15bInstruct1n8gFsdp2tp1V3(BaseNeMoRLTest):
    config = NeMoRLTestConfig(
        test_name="grpo-qwen2.5-math-1.5b-instruct-1n8g-fsdp2tp1.v3",
        algorithm="grpo",
        model_class="llm",
        test_suites=["nightly", "long"],
        time_limit_minutes=120,
        overrides={
            "grpo.max_num_steps": 450,
        },
    )
