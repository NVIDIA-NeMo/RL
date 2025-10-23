from tests.test_suites.base_config import BaseNeMoRLTest, NeMoRLTestConfig


class TestGrpoMathLlamaNemotronSuper49bV54n8gFsdp2tp8(BaseNeMoRLTest):
    """Test grpo-math-llama-nemotron-super-49b-v.5-4n8g-fsdp2tp8."""

    config = NeMoRLTestConfig(
        test_name="grpo-math-llama-nemotron-super-49b-v.5-4n8g-fsdp2tp8",
        algorithm="grpo",
        model_class="llm",
        test_suites=["nightly"],
        time_limit_minutes=30,
        overrides={
            "grpo.max_num_steps": 2,
        },
    )
