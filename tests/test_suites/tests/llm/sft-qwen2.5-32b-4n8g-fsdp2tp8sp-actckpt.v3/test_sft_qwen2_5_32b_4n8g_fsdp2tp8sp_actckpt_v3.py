from tests.test_suites.base_config import BaseNeMoRLTest, NeMoRLTestConfig


class TestSftQwen2532b4n8gFsdp2tp8spActckptV3(BaseNeMoRLTest):
    """Test sft-qwen2.5-32b-4n8g-fsdp2tp8sp-actckpt.v3."""

    config = NeMoRLTestConfig(
        test_name="sft-qwen2.5-32b-4n8g-fsdp2tp8sp-actckpt.v3",
        algorithm="sft",
        model_class="llm",
        test_suites=["nightly"],
        time_limit_minutes=30,
        overrides={
            "sft.max_num_steps": 20,
        },
    )
