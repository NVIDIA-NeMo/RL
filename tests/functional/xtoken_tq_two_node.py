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
"""Real two-node S1 and frozen-payload P-KL S2; never creates a local cluster."""

from __future__ import annotations

import argparse
import re
import tempfile
import uuid
from datetime import timedelta
from pathlib import Path


def check_log(path: Path, steps: int) -> None:
    """Require transfer, bounded buffers and explicit empty-row evidence."""
    text = path.read_text(encoding="utf-8")
    transfers = re.findall(
        r"XTOKEN_TQ_TRANSFER producer=(\S+) consumer=(\S+) .*?"
        r"put_payload_bytes=(\d+) get_payload_bytes=(\d+) .*?student_buffer_bytes=(\d+)",
        text,
    )
    assert len(transfers) == steps, (len(transfers), steps)
    for producer, consumer, put_bytes, get_bytes, buffer_bytes in transfers:
        assert producer != consumer
        assert int(put_bytes) == int(get_bytes) > 0
        assert int(buffer_bytes) <= 64 * 1024 * 1024
    assert len({row[-1] for row in transfers}) == 1, "Receive buffer grew"
    cleared = re.findall(r"XTOKEN_TQ_CLEARED .*?remaining_rows=(\d+)", text)
    assert len(cleared) == steps and set(cleared) == {"0"}, cleared
    losses = re.findall(r"Loss:\s+([-+\d.eE]+)", text)
    assert len(losses) == steps
    assert all(float("-inf") < float(value) < float("inf") for value in losses)
    print(
        f"PASS: {steps} cross-node training steps, finite loss, bounded buffer, no rows"
    )


def run_two_node() -> None:
    """Run production PUT/GET, local reconstruction and P-KL on two GPU nodes."""
    import ray
    import torch
    from omegaconf import OmegaConf

    from nemo_rl.algorithms.loss.loss_functions import CrossTokenizerDistillationLossFn
    from nemo_rl.algorithms.x_token.loss_utils import (
        LocalizedAlignment,
        rebuild_teacher_full_logits_from_ipc,
    )
    from nemo_rl.data_plane.factory import build_data_plane_client
    from nemo_rl.data_plane.worker_mixin import TQWorkerMixin
    from nemo_rl.data_plane.xtoken import (
        XTOKEN_LOGITS_FIELD,
        publish_logits,
        select_tq_nodes,
    )
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
    from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
    from nemo_rl.models.policy.workers.dtensor_policy_worker_v2 import (
        DTensorPolicyWorkerV2Impl,
    )
    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

    register_omegaconf_resolvers()
    # address="auto" fails without an existing cluster; no single-host fallback.
    ray.init(address="auto")
    teacher_resource, student_resource = select_tq_nodes(ray.nodes())
    cfg = {
        "enabled": True,
        "impl": "transfer_queue",
        "backend": "simple",
        "claim_meta_poll_interval_s": 0.5,
        "simple": {"storage_capacity": 16, "num_storage_units": 4},
    }
    client = build_data_plane_client(cfg, bootstrap=True)
    partition = f"xtoken-acceptance-{uuid.uuid4().hex}"
    client.register_partition(partition, [XTOKEN_LOGITS_FIELD], 1, [])
    loss_cfg = OmegaConf.to_container(
        load_config("examples/configs/xtoken_off_policy_distillation.yaml").loss_fn,
        resolve=True,
    )

    @ray.remote(
        num_gpus=1,
        runtime_env={
            **get_runtime_env_for_policy_worker("dtensor_policy_worker_v2"),
            "py_executable": PY_EXECUTABLES.AUTOMODEL,
            "env_vars": {"PYTHONPATH": str(Path.cwd())},
        },
    )
    class Probe(TQWorkerMixin):
        def __init__(self, config, loss_config):
            self.setup_data_plane(config)
            torch.cuda.set_device(0)
            self._teacher_ipc_storage = None
            self._teacher_ipc_handle = None
            self.tmp = tempfile.TemporaryDirectory(prefix="xtoken-tq-")
            torch.distributed.init_process_group(
                "nccl",
                rank=0,
                world_size=1,
                init_method=f"file://{self.tmp.name}/rendezvous",
                timeout=timedelta(seconds=120),
            )
            projection = str(Path(self.tmp.name) / "projection.pt")
            torch.save(
                {
                    "indices": torch.stack(
                        (torch.arange(16), torch.arange(16) + 16), dim=1
                    ),
                    "likelihoods": torch.tensor([[0.7, 0.3]]).repeat(16, 1),
                },
                projection,
            )
            loss_config.update(
                student_vocab_size=16,
                teacher_vocab_sizes=[32],
                projection_matrix_paths=[projection],
                teacher_weights=[1.0],
                teacher_gold_loss=[False],
                teacher_xtoken_loss=[False],
                vocab_topk=32,
            )
            self.loss_fn = CrossTokenizerDistillationLossFn(loss_config)

        def payload(self):
            return (
                (torch.arange(64 * 32, dtype=torch.float32).reshape(1, 64, 32) * 37)
                % 509
            ) / 32

        def publish(self, sample_id):
            return publish_logits(
                self._require_dp_client(),
                self.payload(),
                partition_id=partition,
                sample_id=sample_id,
                producer_node_id=ray.get_runtime_context().get_node_id(),
                max_payload_bytes=64 * 1024 * 1024,
            )

        def receive_and_compare(self, reference):
            receipt = DTensorPolicyWorkerV2Impl.materialize_full_logits_tq(
                self, reference, max_payload_bytes=64 * 1024 * 1024
            )
            received = rebuild_teacher_full_logits_from_ipc(
                receipt.handles, cp_group=None, device=0
            )
            expected = self.payload().cuda()
            torch.testing.assert_close(received, expected, rtol=0, atol=0)
            torch.manual_seed(42)
            initial = torch.randn(1, 64, 16, device="cuda")
            ids = (torch.arange(64, device="cuda") % 16).unsqueeze(0)
            mask = torch.ones(1, 64, device="cuda")
            sample_mask = torch.ones(1, device="cuda")
            data = BatchedDataDict(
                input_ids=ids, token_mask=mask, sample_mask=sample_mask
            )
            align = LocalizedAlignment(
                sample_mask=sample_mask,
                student_chunk_id=torch.arange(64, device="cuda").unsqueeze(0),
                teacher_chunk_id=torch.arange(64, device="cuda").unsqueeze(0),
                pair_valid=mask.bool(),
                pair_is_correct=mask.bool(),
                student_input_ids=ids,
                student_token_mask=mask,
            )

            def update(teacher_logits):
                student = torch.nn.Parameter(initial.clone())
                optimizer = torch.optim.AdamW([student], lr=0.01)
                loss, _ = self.loss_fn(
                    data,
                    torch.ones((), device="cuda"),
                    mask.sum(),
                    student,
                    student,
                    {0: teacher_logits},
                    {0: align},
                )
                loss.backward()
                gradient = student.grad.detach().clone()
                optimizer.step()
                return loss.detach(), gradient, student.detach().clone()

            # Calibrate each tolerance using two repeats on the reference path.
            baseline = update(expected)
            repeat = update(expected)
            transported = update(received)
            errors = []
            tolerances = []
            for direct, again, tq in zip(baseline, repeat, transported):
                tolerance = max(
                    float((direct - again).abs().max()) * 4,
                    torch.finfo(torch.float32).eps,
                )
                torch.testing.assert_close(tq, direct, rtol=0, atol=tolerance)
                errors.append(float((tq - direct).abs().max()))
                tolerances.append(tolerance)
            assert baseline[1].abs().sum() > 0
            assert not torch.equal(baseline[2], initial)
            return {
                "consumer": receipt.consumer_node_id,
                "bytes": receipt.nbytes,
                "buffer_bytes": receipt.buffer_bytes,
                "buffer_ptr": self._teacher_ipc_storage.data_ptr(),
                "errors": errors,
                "tolerances": tolerances,
            }

    actors = []
    ids = []
    try:
        producer = Probe.options(resources=teacher_resource).remote(cfg, loss_cfg)
        consumer = Probe.options(resources=student_resource).remote(cfg, loss_cfg)
        actors.extend([producer, consumer])
        reports = []
        for _ in range(10):
            sample_id = f"{partition}/{uuid.uuid4().hex}"
            ids.append(sample_id)
            ref = ray.get(producer.publish.remote(sample_id), timeout=120)
            report = ray.get(consumer.receive_and_compare.remote(ref), timeout=120)
            assert report["consumer"] != ref.producer_node_id
            assert report["bytes"] == ref.nbytes > 0
            reports.append(report)
            client.clear_samples([sample_id], partition)
            assert client.list_sample_ids(partition) == []
        assert len({r["buffer_ptr"] for r in reports}) == 1
        assert len({r["buffer_bytes"] for r in reports}) == 1
        print(
            f"PASS S1/S2: teacher={ref.producer_node_id}, student={report['consumer']}, payload_bytes={ref.nbytes}, numerical={reports[-1]}"
        )
    finally:
        for actor in actors:
            ray.kill(actor, no_restart=True)
        for actor in actors:
            try:
                ray.get(actor.payload.remote(), timeout=120)
            except ray.exceptions.ActorDiedError:
                pass
        client.clear_samples(ids, partition)
        client.close()
        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-log", type=Path)
    parser.add_argument("--expected-steps", type=int, default=3)
    args = parser.parse_args()
    if args.check_log is not None:
        check_log(args.check_log, args.expected_steps)
    else:
        run_two_node()
