from tests.test_suites.base_config import BaseNeMoRLTest, NeMoRLTestConfig


class TestVlmGrpoQwen25Vl3bInstructClevr1n2gDtensor2tp1V1(BaseNeMoRLTest):
    config = NeMoRLTestConfig(
        test_name="vlm_grpo-qwen2.5-vl-3b-instruct-clevr-1n2g-dtensor2tp1.v1",
        algorithm="grpo",
        model_class="vlm",
        test_suites=["nightly"],
        time_limit_minutes=180,
        overrides={
            "grpo.max_num_steps": 200,
        },
    )
