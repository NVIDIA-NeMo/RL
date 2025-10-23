from tests.test_suites.base_config import BaseNeMoRLTest, NeMoRLTestConfig


class TestSftLlama321b1n8gFsdp2tp1V3(BaseNeMoRLTest):
    config = NeMoRLTestConfig(
        test_name="sft-llama3.2-1b-1n8g-fsdp2tp1.v3",
        algorithm="sft",
        model_class="llm",
        test_suites=["nightly"],
        time_limit_minutes=15,
        overrides={
            "sft.max_num_steps": 250,
        },
    )
