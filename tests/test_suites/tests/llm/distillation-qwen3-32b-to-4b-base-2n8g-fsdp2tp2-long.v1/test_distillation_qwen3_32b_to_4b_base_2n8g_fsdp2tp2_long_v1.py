from tests.test_suites.base_config import BaseNeMoRLTest, NeMoRLTestConfig


class TestDistillationQwen332bTo4bBase2n8gFsdp2tp2LongV1(BaseNeMoRLTest):
    """Test distillation-qwen3-32b-to-4b-base-2n8g-fsdp2tp2-long.v1."""

    config = NeMoRLTestConfig(
        test_name="distillation-qwen3-32b-to-4b-base-2n8g-fsdp2tp2-long.v1",
        algorithm="distillation",
        model_class="llm",
        test_suites=["release", "long"],
        time_limit_minutes=240,
        overrides={
            "distillation.max_num_steps": 100,
        },
    )
