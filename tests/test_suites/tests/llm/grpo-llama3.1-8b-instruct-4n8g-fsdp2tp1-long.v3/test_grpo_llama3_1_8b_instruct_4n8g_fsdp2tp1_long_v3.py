from tests.test_suites.base_config import BaseNeMoRLTest, NeMoRLTestConfig


class TestGrpoLlama318bInstruct4n8gFsdp2tp1LongV3(BaseNeMoRLTest):
    config = NeMoRLTestConfig(
        test_name="grpo-llama3.1-8b-instruct-4n8g-fsdp2tp1-long.v3",
        algorithm="grpo",
        model_class="llm",
        test_suites=["release", "long"],
        time_limit_minutes=240,
        overrides={
            "grpo.max_num_steps": 500,
        },
    )
