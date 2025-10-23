from tests.test_suites.base_config import BaseNeMoRLTest, NeMoRLTestConfig


class TestGrpoLlama318bInstruct1n8gMegatronFp8E2e(BaseNeMoRLTest):
    config = NeMoRLTestConfig(
        test_name="grpo-llama3.1-8b-instruct-1n8g-megatron-fp8-e2e",
        algorithm="grpo",
        model_class="llm",
        test_suites=["nightly"],
        time_limit_minutes=240,
        overrides={
            "grpo.max_num_steps": 100,
        },
    )
