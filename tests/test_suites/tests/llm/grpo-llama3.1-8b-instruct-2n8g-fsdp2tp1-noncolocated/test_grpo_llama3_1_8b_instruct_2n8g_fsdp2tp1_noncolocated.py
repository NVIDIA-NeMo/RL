from tests.test_suites.base_config import BaseNeMoRLTest, NeMoRLTestConfig


class TestGrpoLlama318bInstruct2n8gFsdp2tp1Noncolocated(BaseNeMoRLTest):
    config = NeMoRLTestConfig(
        test_name="grpo-llama3.1-8b-instruct-2n8g-fsdp2tp1-noncolocated",
        algorithm="grpo",
        model_class="llm",
        test_suites=["nightly"],
        time_limit_minutes=120,
        overrides={
            "grpo.max_num_steps": 30,
        },
    )
