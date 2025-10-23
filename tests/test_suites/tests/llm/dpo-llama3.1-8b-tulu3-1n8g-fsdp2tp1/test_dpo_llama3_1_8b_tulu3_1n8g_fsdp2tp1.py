from tests.test_suites.base_config import BaseNeMoRLTest, NeMoRLTestConfig


class TestDpoLlama318bTulu31n8gFsdp2tp1(BaseNeMoRLTest):
    """Test dpo-llama3.1-8b-tulu3-1n8g-fsdp2tp1."""

    config = NeMoRLTestConfig(
        test_name="dpo-llama3.1-8b-tulu3-1n8g-fsdp2tp1",
        algorithm="dpo",
        model_class="llm",
        test_suites=["release"],
        time_limit_minutes=45,
        overrides={
            "dpo.max_num_steps": 150,
        },
    )
