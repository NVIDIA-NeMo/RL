from tests.test_suites.base_config import BaseNeMoRLTest, TestConfig


class TestDPOLlama31_8B_Instruct_4N8G_FSDP2TP2_Quick(BaseNeMoRLTest):
    """Test DPO with Llama 3.1 8B Instruct on 4 nodes, 32 GPUs, FSDP2, TP=2, quick run.

    This test corresponds to: dpo-llama3.1-8b-instruct-4n8g-fsdp2tp2-quick.v2.sh
    """

    # Test configuration
    config = TestConfig(
        # Test metadata
        test_name="dpo-llama3.1-8b-instruct-4n8g-fsdp2tp2-quick.v2",
        algorithm="dpo",
        test_suites=["quick"],
        time_limit_minutes=30,
        # Configuration overrides
        overrides={
            "dpo.max_num_steps": 20,  # Override for quick test
        },
    )
    # The following are auto-extracted from YAML:
    # - model_name: "llama3.1-8b-instruct" (from policy.model_name)
    # - num_nodes: 4 (from cluster.num_nodes)
    # - num_gpus_per_node: 8 (from cluster.gpus_per_node)
    # - backend: "dtensor" (detected from policy.dtensor_cfg.enabled)
    # - tensor_parallel: 2 (from policy.dtensor_cfg.tensor_parallel_size)
