from tests.test_suites.base_config import BaseNeMoRLTest, NeMoRLTestConfig


class TestDistillationQwen332bTo4bBase1n8gFsdp2tp2DynamicbatchV1(BaseNeMoRLTest):
    config = NeMoRLTestConfig(
        test_name="distillation-qwen3-32b-to-4b-base-1n8g-fsdp2tp2-dynamicbatch.v1",
        algorithm="distillation",
        model_class="llm",
        test_suites=["release"],
        time_limit_minutes=120,
        overrides={
            "distillation.max_num_steps": 20,
        },
    )
