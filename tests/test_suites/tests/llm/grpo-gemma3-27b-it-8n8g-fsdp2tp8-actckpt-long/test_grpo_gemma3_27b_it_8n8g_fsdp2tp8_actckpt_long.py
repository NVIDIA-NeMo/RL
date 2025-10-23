from tests.test_suites.base_config import BaseNeMoRLTest, NeMoRLTestConfig


class TestGrpoGemma327bIt8n8gFsdp2tp8ActckptLong(BaseNeMoRLTest):
    """Test grpo-gemma3-27b-it-8n8g-fsdp2tp8-actckpt-long."""

    config = NeMoRLTestConfig(
        test_name="grpo-gemma3-27b-it-8n8g-fsdp2tp8-actckpt-long",
        algorithm="grpo",
        model_class="llm",
        test_suites=["release", "long"],
        time_limit_minutes=240,
        overrides={
            "grpo.max_num_steps": 20,
        },
    )
