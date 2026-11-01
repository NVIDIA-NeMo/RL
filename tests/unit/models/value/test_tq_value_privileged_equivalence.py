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
"""GPU equivalence test for the SWE privileged critic's value path.

On a real Megatron critic behind the TQ data plane, the feature path (policy-layout
rows plus ``privilege_prefixes``) must match the plain path run on explicitly
concatenated ``prefix + tokens`` rows: values at every policy position, and the
value loss. Both paths share the critic lengths, so they are planned into the same
micro-batches and the comparison is exact in float32.
"""

from __future__ import annotations

import asyncio
import gc
from typing import Any

import pytest
import torch

from nemo_rl.algorithms import swe_privileged_critic as spc
from nemo_rl.algorithms.loss.loss_functions import MseValueLossConfig, MseValueLossFn
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data_plane import KVBatchMeta, build_data_plane_client
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.experience.payload import pack_payload
from nemo_rl.models.value.config import SwePrivilegedCriticConfig
from nemo_rl.models.value.tq_value import TQValue
from tests.unit.models.value.test_megatron_value_worker import (
    _apply_config_updates,
    _create_value_test_config,
)

_PARTITION = "privileged_critic_equivalence"
_FIELDS = [
    "input_ids",
    "input_lengths",
    "token_mask",
    "sample_mask",
    "values",
    "returns",
]
# Two instances with different reference blocks, so prefix lengths differ per row.
_ENV_INFOS = (
    {"instance_id": "repo__1", "patch": "diff --git a/x.py b/x.py\n+fix = 1"},
    {
        "instance_id": "repo__2",
        "patch": "diff --git a/y.py b/y.py\n-old()\n+new(arg)\n+more()",
        "FAIL_TO_PASS": ["tests/test_y.py::test_new"],
    },
)
_POLICY_LENGTHS = ((23, 31, 17, 40), (12, 36, 28, 9))


def _dp_cfg() -> dict[str, Any]:
    return {
        "enabled": True,
        "impl": "transfer_queue",
        "backend": "simple",
        "claim_meta_poll_interval_s": 0.5,
        "simple": {"storage_capacity": 1024, "num_storage_units": 1},
        "mooncake_cpu": {
            "global_segment_size": 8589934592,
            "local_buffer_size": 1073741824,
        },
    }


def _policy_rows(lengths: tuple[int, ...], seed: int) -> dict[str, torch.Tensor]:
    """Random rows in the policy layout: prompt then response, returns on the response."""
    gen = torch.Generator().manual_seed(seed)
    width = max(lengths)
    input_ids = torch.randint(10, 150000, (len(lengths), width), generator=gen)
    token_mask = torch.zeros(len(lengths), width)
    returns = torch.zeros(len(lengths), width)
    for row, length in enumerate(lengths):
        token_mask[row, length // 3 : length] = 1.0
        returns[row, :length] = torch.rand(length, generator=gen)
    return {
        "input_ids": input_ids,
        "input_lengths": torch.tensor(lengths, dtype=torch.int32),
        "token_mask": token_mask,
        "sample_mask": torch.ones(len(lengths)),
        "values": torch.zeros(len(lengths), width),
        "returns": returns,
    }


def _explicit_rows(
    rows: dict[str, torch.Tensor], prefixes: list[torch.Tensor]
) -> dict[str, torch.Tensor]:
    """The same rows with each prefix concatenated in front (critic layout)."""
    lengths = [int(n) + int(p.numel()) for n, p in zip(rows["input_lengths"], prefixes)]
    width = max(lengths)
    out = {
        "input_ids": torch.zeros(len(lengths), width, dtype=rows["input_ids"].dtype),
        "input_lengths": torch.tensor(lengths, dtype=torch.int32),
        "token_mask": torch.zeros(len(lengths), width),
        "sample_mask": rows["sample_mask"].clone(),
        "values": torch.zeros(len(lengths), width),
        "returns": torch.zeros(len(lengths), width),
    }
    for row, prefix in enumerate(prefixes):
        p, n = int(prefix.numel()), int(rows["input_lengths"][row])
        out["input_ids"][row, :p] = prefix.to(out["input_ids"].dtype)
        out["input_ids"][row, p : p + n] = rows["input_ids"][row, :n]
        out["token_mask"][row, p : p + n] = rows["token_mask"][row, :n]
        out["returns"][row, p : p + n] = rows["returns"][row, :n]
    return out


def _put(
    dp_client: Any, rows: dict[str, torch.Tensor], group_id: str, idx: int
) -> KVBatchMeta:
    sample_ids, fields, tags = pack_payload(
        rows, weight_version=0, group_id=group_id, prompt_idx=idx
    )
    dp_client.put_samples(
        sample_ids=sample_ids, partition_id=_PARTITION, fields=fields, tags=tags
    )
    return KVBatchMeta(
        partition_id=_PARTITION,
        task_name="train",
        sample_ids=sample_ids,
        fields=list(fields.keys()),
        sequence_lengths=[int(n) for n in rows["input_lengths"]],
        tags=[dict(tag) for tag in tags],
    )


@pytest.mark.mcore
@pytest.mark.hf_gated
@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "tp,batching",
    [(1, "dynamic_batching"), (2, "sequence_packing")],
    ids=["dp2_dynamic_batching", "tp2_sequence_packing"],
)
def test_privileged_values_and_loss_match_explicit_prefix_concatenation(
    tiny_qwen2_model_path: str, tp: int, batching: str
) -> None:
    config = _create_value_test_config(model_name=tiny_qwen2_model_path, tp=tp)
    config["max_total_sequence_length"] = 512
    config["train_global_batch_size"] = 8
    _apply_config_updates(config, {batching: True})
    tokenizer = get_tokenizer(config["tokenizer"])
    cluster = RayVirtualCluster(
        name=f"test-privileged-critic-equivalence-tp{tp}-{batching}",
        bundle_ct_per_node_list=[2],
        use_gpus=True,
        num_gpus_per_node=2,
        max_colocated_worker_groups=1,
    )
    # TQValue attaches to an existing controller; in a run the TQPolicy built
    # alongside the critic bootstraps it.
    controller = build_data_plane_client(_dp_cfg(), bootstrap=True)
    value = None
    try:
        value = TQValue(
            cluster=cluster, config=config, tokenizer=tokenizer, dp_cfg=_dp_cfg()
        )
        dp_client = value.dp_client
        dp_client.register_partition(
            partition_id=_PARTITION,
            fields=_FIELDS,
            num_samples=32,
            consumer_tasks=["value_fwd", "value_train"],
        )
        store = spc.SwePrivilegePrefixStore(
            tokenizer, SwePrivilegedCriticConfig(enabled=True, max_total_tokens=64)
        )

        # Two groups (two instances), each written in the policy layout and stamped
        # by the real store; then the same rows with the prefixes made explicit.
        policy_metas, explicit_metas, policy_rows, row_prefixes = [], [], [], []
        for idx, (env_info, lengths) in enumerate(zip(_ENV_INFOS, _POLICY_LENGTHS)):
            rows = _policy_rows(lengths, seed=idx)
            meta = _put(dp_client, rows, f"policy{idx}", idx)
            meta = asyncio.run(store.enrich(meta, env_info))
            prefixes = list(store.prefixes_for(meta).values())
            assert len(prefixes) == 1, "siblings must share one prefix"
            prefixes = prefixes * len(lengths)
            policy_metas.append(meta)
            policy_rows.append(rows)
            row_prefixes.extend(prefixes)
            explicit_metas.append(
                _put(dp_client, _explicit_rows(rows, prefixes), f"explicit{idx}", idx)
            )
        policy_meta = policy_metas[0].concat(*policy_metas[1:])
        explicit_meta = explicit_metas[0].concat(*explicit_metas[1:])
        prefix_by_key = store.prefixes_for(policy_meta)
        assert len({int(p.numel()) for p in row_prefixes}) > 1, (
            "want distinct prefix lengths"
        )

        # Forward: feature path vs plain path on the concatenated rows.
        value.prepare_for_inference()
        value.get_values_from_meta(policy_meta, privilege_prefixes=prefix_by_key)
        value.get_values_from_meta(explicit_meta)
        value.finish_inference()
        got = value.read_from_dataplane(policy_meta, select_fields=["values"])["values"]
        ref = value.read_from_dataplane(explicit_meta, select_fields=["values"])[
            "values"
        ]

        lengths = [n for rows in policy_rows for n in rows["input_lengths"].tolist()]
        shifted_err = 0.0
        for row, (n, prefix) in enumerate(zip(lengths, row_prefixes)):
            p = int(prefix.numel())
            torch.testing.assert_close(
                got[row, :n].float(), ref[row, p : p + n].float(), rtol=1e-4, atol=1e-4
            )
            # Negative control: a one-token misalignment must be visible.
            shifted_err = max(
                shifted_err,
                (got[row, : n - 1].float() - ref[row, p + 1 : p + n].float())
                .abs()
                .max()
                .item(),
            )
        assert shifted_err > 1e-2, (
            "values do not vary by position; the check is vacuous"
        )

        # Training: the value loss over the shifted targets equals the explicit rows'.
        loss_fn = MseValueLossFn(MseValueLossConfig(scale=1.0, cliprange=0.5))
        value.prepare_for_training()
        out_got = value.train_from_meta(
            policy_meta,
            loss_fn,
            eval_mode=True,
            gbs=8,
            privilege_prefixes=prefix_by_key,
        )
        out_ref = value.train_from_meta(explicit_meta, loss_fn, eval_mode=True, gbs=8)
        value.finish_training()
        torch.testing.assert_close(
            torch.as_tensor(out_got["loss"]).float(),
            torch.as_tensor(out_ref["loss"]).float(),
            rtol=1e-4,
            atol=1e-5,
        )
    finally:
        if value is not None:
            value.shutdown()
        controller.close()
        cluster.shutdown()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
