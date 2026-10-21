"""Check real SGLang config resolution/pickling without launching model workers.

Use the supplied SGLang interpreter inside the allocated container. ServerArgs
may inspect GPU capabilities, but this check loads no weights and starts no
serving processes, sockets, Ray cluster, or CUDA model workers.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "/flint-work/RL/examples/configs/recipes/llm/grpo-flint-49layer-1n4g-automodel-sglang-actckpt-quick.yaml"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from sglang.srt.plugins import load_plugins

    load_plugins()

    from omegaconf import OmegaConf
    from sglang.srt.server_args import ServerArgs

    from nemo_rl.models.generation.sglang.sglang_worker import SGLangGenerationWorker
    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

    register_omegaconf_resolvers()
    config = OmegaConf.to_container(load_config(args.config), resolve=True)
    generation = config["policy"]["generation"]
    actor_class = SGLangGenerationWorker.__ray_metadata__.modified_class
    worker = object.__new__(actor_class)
    worker.sglang_cfg = generation
    worker.gpus_per_node = config["cluster"]["gpus_per_node"]
    worker.num_gpus_per_engine = generation["sglang_cfg"]["sglang_server_config"][
        "num_gpus_per_engine"
    ]
    worker.rank = 0
    worker.base_gpu_id = 0
    server_kwargs = worker._compute_server_args(
        dist_init_addr="127.0.0.1:12340",
        nccl_port=12341,
        host="127.0.0.1",
        port=12342,
    )
    assert server_kwargs["trust_remote_code"] is False
    assert server_kwargs["max_total_tokens"] == 4096
    assert server_kwargs["max_total_tokens"] >= (
        server_kwargs["max_running_requests"] * server_kwargs["context_length"]
    )
    server_args = ServerArgs(**server_kwargs)
    config_class = type(server_args.get_model_config().hf_config)
    assert config_class.__module__.startswith("flint_sglang."), config_class.__module__
    assert server_args.get_model_config().hf_config.num_hidden_layers == 49
    assert server_args.max_total_tokens == 4096
    serialized = pickle.dumps(server_args)
    restored = pickle.loads(serialized)
    assert type(restored.get_model_config().hf_config) is config_class
    assert restored.get_model_config().hf_config.num_hidden_layers == 49
    assert restored.max_total_tokens == 4096
    report = {
        "status": "passed",
        "interpreter": sys.executable,
        "config_class": f"{config_class.__module__}.{config_class.__name__}",
        "num_hidden_layers": 49,
        "pickle_bytes": len(serialized),
        "server_kwargs": server_kwargs,
        "model_workers_started": 0,
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
