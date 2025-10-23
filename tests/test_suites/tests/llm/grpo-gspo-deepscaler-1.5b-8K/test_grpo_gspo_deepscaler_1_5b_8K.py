from tests.test_suites.base_config import BaseNeMoRLTest, NeMoRLTestConfig


class TestGrpoGspoDeepscaler15b8k(BaseNeMoRLTest):
    config = NeMoRLTestConfig(
        test_name="grpo-gspo-deepscaler-1.5b-8K",
        algorithm="gspo",
        model_class="llm",
        test_suites=["nightly"],
        time_limit_minutes=240,
        overrides={
            "grpo.max_num_steps": 40,
        },
    )
