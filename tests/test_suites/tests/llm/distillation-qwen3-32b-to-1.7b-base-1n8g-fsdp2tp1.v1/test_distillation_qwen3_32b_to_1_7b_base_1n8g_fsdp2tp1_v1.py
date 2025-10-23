from tests.test_suites.base_config import BaseNeMoRLTest, NeMoRLTestConfig


class TestDistillationQwen332bTo17bBase1n8gFsdp2tp1V1(BaseNeMoRLTest):
    """Test distillation-qwen3-32b-to-1.7b-base-1n8g-fsdp2tp1.v1."""

    config = NeMoRLTestConfig(
        test_name="distillation-qwen3-32b-to-1.7b-base-1n8g-fsdp2tp1.v1",
        algorithm="distillation",
        model_class="llm",
        test_suites=["nightly"],
        time_limit_minutes=60,
        overrides={
            "distillation.max_num_steps": 10,
            "distillation.val_period": 20,
        },
    )
