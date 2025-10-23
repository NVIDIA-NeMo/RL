from tests.test_suites.base_config import BaseNeMoRLTest, NeMoRLTestConfig


class TestSftLlama318b1n8gFsdp2tp1Long(BaseNeMoRLTest):
    config = NeMoRLTestConfig(
        test_name="sft-llama3.1-8b-1n8g-fsdp2tp1-long",
        algorithm="sft",
        model_class="llm",
        test_suites=["release", "long"],
        time_limit_minutes=120,
        overrides={
            "sft.max_num_steps": 250,
        },
    )
