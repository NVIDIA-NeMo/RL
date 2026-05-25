# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import itertools
import math
import os
import pprint
import random
from typing import Iterator

from omegaconf import OmegaConf
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer, set_seed
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.games.calculator_tool import (
    CALCULATOR_TOOL,
    CalculatorMetadata,
    CalculatorToolEnv,
)
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir

SYSTEM_PROMPT = (
    "You are a math problem solver with access to a calculator tool. "
    "Break down complex problems step by step and use the calculator "
    "for each computation. After computing your final answer, provide "
    "it using <answer>NUMBER</answer> tags."
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run GRPO training for Calculator Tool-Call"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


# ---------------------------------------------------------------------------
# Problem generators — each returns (problem_text, expected_answer)
# All problems require 2-4 calculator steps with non-trivial decimal math.
# ---------------------------------------------------------------------------


def _shopping_discount_tax(rng: random.Random) -> tuple[str, float]:
    """3 steps: subtotal -> discount -> tax."""
    n1 = rng.randint(2, 5)
    p1 = rng.choice([14.99, 19.49, 24.99, 29.95, 34.99])
    n2 = rng.randint(1, 3)
    p2 = rng.choice([39.50, 44.99, 54.95, 64.99, 79.50])
    disc = rng.choice([12, 15, 18, 22, 25])
    tax = rng.choice([6.5, 7.25, 8.5, 9.25, 10.5])
    subtotal = n1 * p1 + n2 * p2
    after_disc = subtotal * (1 - disc / 100)
    ans = round(after_disc * (1 + tax / 100), 2)
    return (
        f"You buy {n1} shirts at ${p1} each and {n2} pants at ${p2} each. "
        f"The store applies a {disc}% discount on the subtotal, "
        f"then adds {tax}% sales tax. What is the final cost?",
        ans,
    )


def _compound_interest_withdraw(rng: random.Random) -> tuple[str, float]:
    """3 steps: compound -> balance -> withdraw."""
    principal = rng.choice([1500, 2500, 3500, 4500, 7500])
    rate = rng.choice([3.5, 4.25, 5.5, 6.75, 8.0])
    years = rng.choice([2, 3, 4])
    withdraw_pct = rng.choice([25, 30, 35, 40])
    balance = principal * (1 + rate / 100) ** years
    ans = round(balance * (1 - withdraw_pct / 100), 2)
    return (
        f"You invest ${principal} at {rate}% annual interest compounded yearly. "
        f"After {years} years, you withdraw {withdraw_pct}% of the balance. "
        f"How much remains in the account?",
        ans,
    )


def _travel_two_legs_km(rng: random.Random) -> tuple[str, float]:
    """3 steps: leg1 distance + leg2 distance -> convert to km."""
    s1 = rng.choice([45, 55, 60, 65, 72])
    h1 = rng.choice([1.5, 2, 2.5, 3])
    s2 = rng.choice([30, 35, 40, 50])
    mins2 = rng.choice([40, 50, 75, 90, 100])
    d1 = s1 * h1
    d2 = s2 * (mins2 / 60)
    ans = round((d1 + d2) * 1.609, 2)
    return (
        f"A car travels at {s1} mph for {h1} hours, then at {s2} mph "
        f"for {mins2} minutes. What is the total distance in kilometers? "
        f"(1 mile = 1.609 km)",
        ans,
    )


def _revenue_costs_tax_bonus(rng: random.Random) -> tuple[str, float]:
    """4 steps: revenue - costs = profit -> tax -> after-tax -> bonus."""
    rev = rng.choice([125000, 175000, 250000, 340000, 450000])
    cost_pct = rng.choice([57, 62, 68, 73])
    tax_pct = rng.choice([18, 21, 24, 28])
    bonus_pct = rng.choice([5, 8, 10, 12])
    profit = rev * (1 - cost_pct / 100)
    after_tax = profit * (1 - tax_pct / 100)
    ans = round(after_tax * (1 - bonus_pct / 100), 2)
    return (
        f"A company has revenue of ${rev:,}. Operating costs are {cost_pct}% of "
        f"revenue. They pay {tax_pct}% tax on profit, then distribute {bonus_pct}% "
        f"of the after-tax profit as employee bonuses. "
        f"How much profit remains after bonuses?",
        ans,
    )


def _recipe_scaling_cost(rng: random.Random) -> tuple[str, float]:
    """3 steps: scale ingredient -> convert units -> compute cost."""
    orig_servings = rng.choice([4, 6, 8])
    target_servings = rng.choice([10, 14, 18, 22])
    cups = rng.choice([1.5, 2, 2.5, 3])
    grams_per_cup = rng.choice([120, 128, 140, 150])
    price_per_kg = rng.choice([3.49, 4.29, 5.99, 7.49])
    scaled_cups = cups * (target_servings / orig_servings)
    grams = scaled_cups * grams_per_cup
    ans = round(grams / 1000 * price_per_kg, 2)
    return (
        f"A recipe for {orig_servings} servings needs {cups} cups of flour. "
        f"You want to make {target_servings} servings. "
        f"If 1 cup of flour = {grams_per_cup}g and flour costs "
        f"${price_per_kg}/kg, what is the flour cost?",
        ans,
    )


def _multi_currency_purchase(rng: random.Random) -> tuple[str, float]:
    """3 steps: convert currencies -> sum -> apply fee."""
    eur_amount = rng.choice([45.50, 62.75, 89.99, 120.00, 155.50])
    eur_rate = rng.choice([1.08, 1.09, 1.10, 1.12])
    gbp_amount = rng.choice([35.99, 48.50, 72.25, 95.00])
    gbp_rate = rng.choice([1.25, 1.27, 1.29, 1.31])
    fee_pct = rng.choice([2.5, 3.0, 3.5])
    usd_total = eur_amount * eur_rate + gbp_amount * gbp_rate
    ans = round(usd_total * (1 + fee_pct / 100), 2)
    return (
        f"You buy an item for {eur_amount} EUR (rate: 1 EUR = {eur_rate} USD) "
        f"and another for {gbp_amount} GBP (rate: 1 GBP = {gbp_rate} USD). "
        f"Your bank charges a {fee_pct}% foreign transaction fee on the total. "
        f"What is your total cost in USD?",
        ans,
    )


def _paint_room_cost(rng: random.Random) -> tuple[str, float]:
    """4 steps: wall area -> subtract window/door -> gallons needed -> cost."""
    length = rng.choice([12.5, 14, 15.5, 18, 20])
    width = rng.choice([10, 11.5, 13, 14.5])
    height = rng.choice([8, 9, 10])
    window_area = rng.choice([12, 15, 18, 24])
    sqft_per_gallon = rng.choice([350, 375, 400])
    price_per_gallon = rng.choice([28.99, 34.49, 42.99, 54.99])
    wall_area = 2 * (length + width) * height
    paintable = wall_area - window_area
    gallons = paintable / sqft_per_gallon
    gallons_needed = math.ceil(gallons)
    ans = round(gallons_needed * price_per_gallon, 2)
    return (
        f"A room is {length} ft x {width} ft with {height} ft ceilings. "
        f"Windows and doors total {window_area} sq ft. "
        f"Paint covers {sqft_per_gallon} sq ft/gallon and costs "
        f"${price_per_gallon}/gallon. You must buy whole gallons. "
        f"What is the paint cost?",
        ans,
    )


def _loan_monthly_simple(rng: random.Random) -> tuple[str, float]:
    """3 steps: total interest -> total owed -> monthly payment."""
    principal = rng.choice([5000, 8000, 12000, 15000, 20000])
    annual_rate = rng.choice([4.5, 5.25, 6.0, 7.5, 8.25])
    years = rng.choice([2, 3, 4, 5])
    total_interest = principal * (annual_rate / 100) * years
    total_owed = principal + total_interest
    ans = round(total_owed / (years * 12), 2)
    return (
        f"You take a ${principal:,} simple-interest loan at {annual_rate}% "
        f"annual rate for {years} years. "
        f"What is the monthly payment (total owed / total months)?",
        ans,
    )


def _three_item_weighted_avg(rng: random.Random) -> tuple[str, float]:
    """3 steps: weighted sum -> total weight -> divide."""
    w1 = rng.choice([2.5, 3.0, 3.5, 4.0])
    v1 = rng.choice([78.5, 82.0, 85.5, 91.0, 95.5])
    w2 = rng.choice([1.5, 2.0, 2.5])
    v2 = rng.choice([65.0, 70.5, 74.0, 88.0])
    w3 = rng.choice([3.0, 4.0, 5.0])
    v3 = rng.choice([72.5, 77.0, 83.5, 90.0])
    weighted_sum = w1 * v1 + w2 * v2 + w3 * v3
    total_weight = w1 + w2 + w3
    ans = round(weighted_sum / total_weight, 2)
    return (
        f"A student's grades are weighted: subject A has weight {w1} and "
        f"score {v1}, subject B has weight {w2} and score {v2}, "
        f"subject C has weight {w3} and score {v3}. "
        f"What is the weighted average grade?",
        ans,
    )


def _shipping_cost_intl(rng: random.Random) -> tuple[str, float]:
    """3 steps: item total + handling -> shipping rate -> customs duty."""
    n_items = rng.randint(2, 5)
    price_each = rng.choice([18.75, 24.99, 32.50, 45.99, 67.50])
    handling = rng.choice([4.99, 6.50, 8.99, 12.50])
    ship_per_kg = rng.choice([3.25, 4.50, 5.75, 7.99])
    weight_kg = rng.choice([1.2, 1.8, 2.5, 3.4])
    duty_pct = rng.choice([8, 12, 15, 18])
    items_cost = n_items * price_each
    shipping = handling + ship_per_kg * weight_kg
    subtotal = items_cost + shipping
    ans = round(subtotal * (1 + duty_pct / 100), 2)
    return (
        f"You order {n_items} items at ${price_each} each from overseas. "
        f"Handling fee is ${handling}, shipping is ${ship_per_kg}/kg for "
        f"a {weight_kg} kg package. Customs duty is {duty_pct}% on the "
        f"total (items + shipping). What is the total cost?",
        ans,
    )


# Weighted template pools — all multi-step problems
_TEMPLATES = [
    (_shopping_discount_tax, 15),
    (_compound_interest_withdraw, 12),
    (_travel_two_legs_km, 12),
    (_revenue_costs_tax_bonus, 10),
    (_recipe_scaling_cost, 10),
    (_multi_currency_purchase, 10),
    (_paint_room_cost, 8),
    (_loan_monthly_simple, 8),
    (_three_item_weighted_avg, 8),
    (_shipping_cost_intl, 7),
]
_TEMPLATE_FNS = [t[0] for t in _TEMPLATES]
_TEMPLATE_WEIGHTS = [t[1] for t in _TEMPLATES]


def generate_problem(rng: random.Random) -> tuple[str, float]:
    """Generate a random multi-step math word problem and its expected answer."""
    fn = rng.choices(_TEMPLATE_FNS, weights=_TEMPLATE_WEIGHTS, k=1)[0]
    return fn(rng)


def generate_calculator_datum(
    tokenizer: AutoTokenizer,
    idx: int,
    max_tool_calls: int,
    tolerance: float,
    relative_tolerance: float,
) -> DatumSpec:
    """Generate a single calculator tool-call datum."""
    rng = random.Random()
    problem, expected_answer = generate_problem(rng)

    message_list = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]

    # Use apply_chat_template with tools to embed tool schema in the prompt
    initial_prompt_content = tokenizer.apply_chat_template(
        message_list,
        tools=[CALCULATOR_TOOL],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    tokenized_prompt = tokenizer(
        initial_prompt_content, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]

    message_log: LLMMessageLogType = [
        {
            "role": "user",
            "content": initial_prompt_content,
            "token_ids": tokenized_prompt,
        }
    ]

    metadata = CalculatorMetadata(
        expected_answer=expected_answer,
        problem=problem,
        tool_calls_remaining=max_tool_calls,
        max_tool_calls=max_tool_calls,
        tolerance=tolerance,
        relative_tolerance=relative_tolerance,
    )

    datum: DatumSpec = {
        "message_log": message_log,
        "length": len(tokenized_prompt),
        "extra_env_info": metadata,
        "loss_multiplier": 1.0,
        "idx": idx,
        "task_name": "calculator",
        "stop_strings": ["</tool_call>", "</answer>"],
    }
    return datum


class IterableCalculatorDataset(IterableDataset):
    """An IterableDataset that generates calculator problems indefinitely."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_tool_calls: int,
        tolerance: float,
        relative_tolerance: float,
        length: int,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_tool_calls = max_tool_calls
        self.tolerance = tolerance
        self.relative_tolerance = relative_tolerance
        self.length = length

    def __iter__(self) -> Iterator[DatumSpec]:
        for i in itertools.count():
            yield generate_calculator_datum(
                tokenizer=self.tokenizer,
                idx=i,
                max_tool_calls=self.max_tool_calls,
                tolerance=self.tolerance,
                relative_tolerance=self.relative_tolerance,
            )

    def __len__(self):
        return self.length


def setup_calculator_data(
    tokenizer: AutoTokenizer,
    env_cfg: dict,
    task_name: str,
    length: int,
    val_length: int,
) -> tuple[IterableDataset, IterableDataset, dict, dict]:
    """Set up the iterable data generator and env map for the calculator task."""
    env_config = env_cfg[task_name]
    cfg = dict(env_config.get("cfg", {}))

    env = CalculatorToolEnv.options(num_gpus=0).remote(cfg=cfg)
    task_to_env = {task_name: env}

    max_tool_calls = cfg.get("max_tool_calls", 3)
    tolerance = cfg.get("tolerance", 0.01)
    relative_tolerance = cfg.get("relative_tolerance", 0.10)

    training_dataset = IterableCalculatorDataset(
        tokenizer=tokenizer,
        max_tool_calls=max_tool_calls,
        tolerance=tolerance,
        relative_tolerance=relative_tolerance,
        length=length,
    )
    validation_dataset = IterableCalculatorDataset(
        tokenizer=tokenizer,
        max_tool_calls=max_tool_calls,
        tolerance=tolerance,
        relative_tolerance=relative_tolerance,
        length=val_length,
    )

    return training_dataset, validation_dataset, task_to_env, task_to_env


def main():
    """Main entry point."""
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "grpo_calculator.yaml"
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config = OmegaConf.to_container(config, resolve=True)
    config = MasterConfig(**config)
    print("Applied CLI overrides")

    print("Final config:")
    pprint.pprint(config)

    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(
            f"Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}"
        )

    init_ray()
    set_seed(config["grpo"]["seed"])

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    ds_length = (
        config["grpo"]["num_prompts_per_step"]
        * config["grpo"]["num_generations_per_prompt"]
        * config["grpo"]["max_num_steps"]
    )
    dataset, val_dataset, task_to_env, val_task_to_env = setup_calculator_data(
        tokenizer=tokenizer,
        env_cfg=config["env"],
        task_name="calculator",
        length=ds_length,
        val_length=config["grpo"]["max_val_samples"],
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
