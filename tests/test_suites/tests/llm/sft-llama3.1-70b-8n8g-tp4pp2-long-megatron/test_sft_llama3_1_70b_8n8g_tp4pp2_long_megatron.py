from tests.test_suites.base_config import BaseNeMoRLTest, NeMoRLTestConfig


class TestSftLlama3170b8n8gTp4pp2LongMegatron(BaseNeMoRLTest):
    """Test sft-llama3.1-70b-8n8g-tp4pp2-long-megatron."""

    config = NeMoRLTestConfig(
        test_name="sft-llama3.1-70b-8n8g-tp4pp2-long-megatron",
        algorithm="sft",
        model_class="llm",
        test_suites=["release", "long"],
        time_limit_minutes=240,
        overrides={
            "sft.max_num_steps": 300,
        },
    )
