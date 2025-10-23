from tests.test_suites.base_config import BaseNeMoRLTest, NeMoRLTestConfig


class TestDPOLlama31_8B_Instruct_4N8G_FSDP2TP2_Quick(BaseNeMoRLTest):
    config = NeMoRLTestConfig(
        test_name="dpo-llama3.1-8b-instruct-4n8g-fsdp2tp2-quick.v2",
        algorithm="dpo",
        model_class="llm",
        test_suites=["quick"],
        time_limit_minutes=30,
        overrides={
            "dpo.max_num_steps": 20,
        },
    )
