from tests.test_suites.base_config import BaseNeMoRLTest, NeMoRLTestConfig


class TestDpoLlama318bInstruct4n8gMegatronV2(BaseNeMoRLTest):
    config = NeMoRLTestConfig(
        test_name="dpo-llama3.1-8b-instruct-4n8g-megatron.v2",
        algorithm="dpo",
        model_class="llm",
        test_suites=["release"],
        time_limit_minutes=140,
        overrides={
            "dpo.max_num_steps": 150,
        },
    )
