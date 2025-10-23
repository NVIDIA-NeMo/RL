from tests.test_suites.base_config import BaseNeMoRLTest, NeMoRLTestConfig


class TestDpoLlama318bInstruct4n8gMegatrontp2pp2Quick(BaseNeMoRLTest):
    config = NeMoRLTestConfig(
        test_name="dpo-llama3.1-8b-instruct-4n8g-megatrontp2pp2-quick",
        algorithm="dpo",
        model_class="llm",
        test_suites=["quick", "nightly"],
        time_limit_minutes=30,
        overrides={
            "dpo.max_num_steps": 20,
        },
    )
