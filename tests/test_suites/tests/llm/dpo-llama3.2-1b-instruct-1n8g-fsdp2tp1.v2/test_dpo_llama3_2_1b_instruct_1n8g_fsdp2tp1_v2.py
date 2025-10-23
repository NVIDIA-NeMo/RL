from tests.test_suites.base_config import BaseNeMoRLTest, NeMoRLTestConfig


class TestDpoLlama321bInstruct1n8gFsdp2tp1V2(BaseNeMoRLTest):
    config = NeMoRLTestConfig(
        test_name="dpo-llama3.2-1b-instruct-1n8g-fsdp2tp1.v2",
        algorithm="dpo",
        model_class="llm",
        test_suites=["nightly"],
        time_limit_minutes=45,
        overrides={
            "dpo.max_num_steps": 150,
        },
    )
