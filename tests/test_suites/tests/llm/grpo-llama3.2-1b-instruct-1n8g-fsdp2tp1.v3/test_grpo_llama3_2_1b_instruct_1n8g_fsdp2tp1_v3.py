from tests.test_suites.base_config import BaseNeMoRLTest, NeMoRLTestConfig


class TestGrpoLlama321bInstruct1n8gFsdp2tp1V3(BaseNeMoRLTest):
    config = NeMoRLTestConfig(
        test_name="grpo-llama3.2-1b-instruct-1n8g-fsdp2tp1.v3",
        algorithm="grpo",
        model_class="llm",
        test_suites=["nightly", "long"],
        time_limit_minutes=120,
        overrides={
            "grpo.max_num_steps": 500,
        },
    )
