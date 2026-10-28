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

"""Test-only Automodel policy worker that runs one update as several chunks.

Kept out of pytest modules so Ray workers can import it without pytest.
"""

from typing import Any

import ray
import torch

from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.automodel.data import process_global_batch
from nemo_rl.models.automodel.train import LossPostProcessor
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.automodel_policy_worker import (
    AutomodelPolicyWorkerImpl,
    GlobalCounts,
)


class ChunkedUpdateAutomodelPolicyWorkerImpl(AutomodelPolicyWorkerImpl):
    def train_update_in_chunks(
        self,
        *,
        data: BatchedDataDict[Any],
        loss_fn: LossFunction,
        chunk_sizes_by_dp_rank: list[list[int]],
        logprobs_after_first_chunk: bool,
    ) -> dict[str, Any]:
        """Run one optimizer update over ``data`` as several ``_run_update_chunk`` calls.

        Args:
            data: This DP rank's rows of the update.
            loss_fn: Loss function.
            chunk_sizes_by_dp_rank: The chunk sizes of every DP rank; this rank
                splits ``data`` by its own entry.
            logprobs_after_first_chunk: Compute actor logprobs of the first
                chunk while the update is open, as a streaming trainer would.

        Returns:
            The metrics ``train()`` returns.
        """
        chunk_sizes = chunk_sizes_by_dp_rank[
            torch.distributed.get_rank(self.dp_mesh.get_group())
        ]
        data = data.to("cuda")
        counts = process_global_batch(
            data, loss_fn, self.dp_mesh.get_group(), batch_idx=0, batch_size=data.size
        )
        for part in self.model_runtime.parts:
            part.train()
        self._open_update(
            LossPostProcessor(
                loss_fn=loss_fn,
                cfg=self.cfg,
                cp_mesh=self.cp_mesh,
                cp_size=self.cp_size,
                dp_size=self.dp_size,
                sampling_params=self.sampling_params,
            ),
            self.cfg["train_micro_batch_size"],
            global_counts=GlobalCounts(
                global_valid_seqs=counts["global_valid_seqs"],
                global_valid_toks=counts["global_valid_toks"],
            ),
            eval_mode=False,
            num_global_batches=1,
            sequence_dim=1,
            on_microbatch_start=lambda mb_idx: None,
        )
        start = 0
        for chunk_idx, size in enumerate(chunk_sizes):
            self._run_update_chunk(
                data.slice(start, start + size), sync_expert_grads=True
            )
            if logprobs_after_first_chunk and chunk_idx == 0:
                self.get_logprobs(data=data.slice(start, start + size))
                for part in self.model_runtime.parts:
                    part.train()
            start += size
        closed = self._close_update()
        return self._finish_train_call(
            losses=[closed.loss],
            all_mb_metrics=closed.mb_metrics,
            grad_norm=closed.grad_norm,
            draft_grad_norm=closed.draft_grad_norm,
            eval_mode=False,
        )


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("automodel_policy_worker")
)  # pragma: no cover
class ChunkedUpdateAutomodelPolicyWorker(ChunkedUpdateAutomodelPolicyWorkerImpl):
    pass
