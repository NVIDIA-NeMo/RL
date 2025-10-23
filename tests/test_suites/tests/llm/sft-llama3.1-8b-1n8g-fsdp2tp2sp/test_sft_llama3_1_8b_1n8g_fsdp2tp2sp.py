from tests.test_suites.base_config import BaseNeMoRLTest, NeMoRLTestConfig


class TestSftLlama318b1n8gFsdp2tp2sp(BaseNeMoRLTest):
    config = NeMoRLTestConfig(
        test_name="sft-llama3.1-8b-1n8g-fsdp2tp2sp",
        algorithm="sft",
        model_class="llm",
        test_suites=["nightly"],
        time_limit_minutes=45,
        overrides={
            "sft.max_num_steps": 50,
        },
    )
