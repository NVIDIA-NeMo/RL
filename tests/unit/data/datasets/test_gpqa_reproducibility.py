# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import random

import pytest
from datasets import Dataset

from nemo_rl.data.datasets.eval_datasets.gpqa import GPQADataset


@pytest.mark.parametrize("variant", ["main", "diamond"])
def test_gpqa_choices_follow_application_seed(monkeypatch, variant):
    rows = [
        {
            "Question": f"Question {i}",
            "Correct Answer": f"Correct {i}",
            "Incorrect Answer 1": f"First distractor {i}",
            "Incorrect Answer 2": f"Second distractor {i}",
            "Incorrect Answer 3": f"Third distractor {i}",
        }
        for i in range(32)
    ]
    dataset = Dataset.from_list(rows)
    monkeypatch.setattr(
        "nemo_rl.data.datasets.eval_datasets.gpqa.load_dataset",
        lambda *args, **kwargs: dataset,
    )
    state = random.getstate()
    try:

        def load(seed):
            random.seed(seed)
            return GPQADataset(variant=variant).rekeyed_ds.to_list()

        first = load(42)
        assert load(42) == first
        assert load(43) != first
        for row, original in zip(first, rows, strict=True):
            assert row["options"][row["answer"]] == original["Correct Answer"]
    finally:
        random.setstate(state)


def test_eval_seeds_before_loading_data(monkeypatch):
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    from examples import run_eval

    config = OmegaConf.create(
        {
            "eval": {"seed": 42},
            "data": {"dataset_name": "gpqa"},
            "generation": {},
            "tokenizer": {},
            "env": {},
        }
    )
    monkeypatch.setattr(
        run_eval, "parse_args", lambda: (SimpleNamespace(config="unused"), {})
    )
    monkeypatch.setattr(run_eval, "load_config", lambda _: config)
    monkeypatch.setattr(run_eval, "MasterConfig", SimpleNamespace)
    monkeypatch.setattr(run_eval, "init_ray", lambda: None)
    monkeypatch.setattr(run_eval, "get_tokenizer", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        run_eval, "configure_generation_config", lambda *args, **kwargs: {}
    )
    calls = []
    monkeypatch.setattr(run_eval, "set_seed", lambda seed: calls.append(seed))

    class DataReached(Exception):
        pass

    def setup_data(*args, **kwargs):
        assert calls == [42]
        raise DataReached

    monkeypatch.setattr(run_eval, "setup_data", setup_data)
    with pytest.raises(DataReached):
        run_eval.main()
