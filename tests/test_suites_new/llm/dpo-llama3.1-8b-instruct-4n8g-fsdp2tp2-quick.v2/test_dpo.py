from tests.test_suites_new.base_config import BaseNeMoRLTest, NeMoRLTestConfig


class TestDPOLlama31_8B_Instruct_123(BaseNeMoRLTest):
    config = NeMoRLTestConfig(
        test_name="dpo-llama3.1-8b-instruct-4n8g-fsdp2tp2-quick.v2",
        algorithm="dpo",
        model_class="vlm",
        test_suites=["quick"],
        time_limit_minutes=30,
        overrides={
            "dpo.max_num_steps": 20,
            "cluster.num_nodes": 1111,
            "cluster.gpus_per_node": 123,
        },
    )
