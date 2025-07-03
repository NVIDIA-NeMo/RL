import argparse
import itertools
import os
import pprint
import random
from datetime import datetime, timedelta
from typing import Iterator

from omegaconf import OmegaConf
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.simulated_user.unique_numbers import (
    UniqueNumbersEnv,
    UniqueNumbersMetadata,
)
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)

PROMPT = (
    "I will play a game with you. I have a list of integers in mind and can NOT tell you. "
    "Your goal is to guess the count of UNIQUE numbers in my list. The only 2 things you can do is the following: "
    "You can either ask me 'what is number k?' to get the number at position k in my list, "
    "or answer 'there are m unique numbers' whenever you feel you want to make a guess."
    "Please do not say anything else. You cannot ask me to provide the list of integers."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run GRPO with unique numbers simulator")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    args, overrides = parser.parse_known_args()
    return args, overrides


def generate_datum(tokenizer: AutoTokenizer, env_cfg: dict, task_name: str, idx: int, add_system_prompt: bool) -> DatumSpec:
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_system_prompt=add_system_prompt,
        add_generation_prompt=True,
        add_special_tokens=False,
    ).strip()
    token_ids = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

    def _generate_numbers(min_length, max_length, max_integer, default_max_turns) -> UniqueNumbersMetadata:
        length = random.randint(min_length, max_length)
        numbers = [random.randint(0, max_integer) for _ in range(length)]
        return UniqueNumbersMetadata(
            numbers=numbers,
            unique_count=len(set(numbers)),
            turn=0,
            max_turns=default_max_turns,
        )

    metadata = _generate_numbers(
        min_length=env_cfg["cfg"]["min_length"],
        max_length=env_cfg["cfg"]["max_length"],
        max_integer=env_cfg["cfg"]["max_integer"],
        default_max_turns=env_cfg["cfg"]["max_turns"],
    )

    message_log: LLMMessageLogType = [
        {"role": "user", "content": formatted_prompt, "token_ids": token_ids}
    ]
    return {
        "message_log": message_log,
        "length": len(token_ids),
        "extra_env_info": metadata,
        "loss_multiplier": 1.0,
        "idx": idx,
        "task_name": task_name,
    }


class IterableNumbersDataset(IterableDataset):
    def __init__(self, tokenizer, env_cfg, task_name, add_system_prompt, length):
        super().__init__()
        self.tokenizer = tokenizer
        self.env_cfg = env_cfg
        self.task_name = task_name
        self.add_system_prompt = add_system_prompt
        self.length = length

    def __iter__(self) -> Iterator[DatumSpec]:
        for i in itertools.count():
            yield generate_datum(
                tokenizer=self.tokenizer,
                env_cfg=self.env_cfg,
                task_name=self.task_name,
                idx=i,
                add_system_prompt=self.add_system_prompt,
            )

    def __len__(self):
        return self.length


def setup_data(tokenizer, env_cfg, task_name, length, val_length, add_system_prompt):
    env_config = env_cfg[task_name]
    env = UniqueNumbersEnv.options(num_gpus=0).remote(cfg=dict(env_config["cfg"]))
    task_to_env = {task_name: env}

    train_ds = IterableNumbersDataset(
        tokenizer=tokenizer,
        env_cfg=env_config,
        task_name=task_name,
        add_system_prompt=add_system_prompt,
        length=length,
    )
    val_ds = IterableNumbersDataset(
        tokenizer=tokenizer,
        env_cfg=env_config,
        task_name=task_name,
        add_system_prompt=add_system_prompt,
        length=val_length,
    )
    val_task_to_env = task_to_env
    return train_ds, val_ds, task_to_env, val_task_to_env


def main():
    args, overrides = parse_args()
    if not args.config:
        args.config = os.path.join(os.path.dirname(__file__), "configs", "grpo_unique_numbers_gemma1b.yaml")
    config = load_config(args.config)
    if overrides:
        config = parse_hydra_overrides(config, overrides)
    config: MasterConfig = OmegaConf.to_container(config, resolve=True)

    now_pst = datetime.utcnow() + timedelta(hours=-7)
    config["logger"]["wandb"]["name"] = config["logger"]["wandb"]["name"].replace("__NOW__", now_pst.strftime("%m/%d-%H:%M"))

    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    if config["checkpointing"]["enabled"]:
        print(f"\U0001F4CA Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}")

    pprint.pprint(config)

    init_ray()

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    config["policy"]["generation"] = configure_generation_config(config["policy"]["generation"], tokenizer)

    ds_length = config["grpo"]["num_prompts_per_step"] * config["grpo"]["num_generations_per_prompt"] * config["grpo"]["max_num_steps"]
    dataset, val_dataset, task_to_env, val_task_to_env = setup_data(
        tokenizer=tokenizer,
        env_cfg=config["env"],
        task_name="unique_numbers",
        length=ds_length,
        val_length=config["grpo"]["max_val_samples"],
        add_system_prompt=config["data"]["add_system_prompt"],
    )

    (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)

    grpo_train(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        task_to_env,
        val_task_to_env,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )


if __name__ == "__main__":
    main()
