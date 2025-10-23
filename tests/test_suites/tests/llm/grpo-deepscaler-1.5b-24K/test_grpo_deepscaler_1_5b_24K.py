from tests.test_suites.base_config import BaseNeMoRLTest, NeMoRLTestConfig


class TestGrpoDeepscaler15b24k(BaseNeMoRLTest):
    config = NeMoRLTestConfig(
        test_name="grpo-deepscaler-1.5b-24K",
        algorithm="grpo",
        model_class="llm",
        test_suites=["nightly"],
        time_limit_minutes=240,
        overrides={
            "grpo.max_num_steps": 10,
        },
    )
