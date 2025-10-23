from tests.test_suites.base_config import BaseNeMoRLTest, NeMoRLTestConfig


class TestGrpoQwen2532b32n8gFsdp2tp8spActckptLongV3(BaseNeMoRLTest):
    """Test grpo-qwen2.5-32b-32n8g-fsdp2tp8sp-actckpt-long.v3."""

    config = NeMoRLTestConfig(
        test_name="grpo-qwen2.5-32b-32n8g-fsdp2tp8sp-actckpt-long.v3",
        algorithm="grpo",
        model_class="llm",
        test_suites=["release", "long"],
        time_limit_minutes=240,
        overrides={
            "grpo.max_num_steps": 20,
        },
    )
