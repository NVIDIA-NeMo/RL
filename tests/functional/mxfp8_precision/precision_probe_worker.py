"""Real Megatron worker extension. Probes never call optimizer.step()."""

import copy
import hashlib
from collections import Counter
from pathlib import Path

import ray
import torch
import torch.distributed as dist
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from megatron.core.fp8_utils import (
    get_grouped_quantized_members,
    is_grouped_mxfp8tensor,
    is_mxfp8tensor,
)

from nemo_rl.algorithms.loss import NLLLossFn
from nemo_rl.models.megatron import train as train_module
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    MegatronPolicyWorkerImpl,
)
from nemo_rl.models.policy.workers.mxfp8_refit_source import (
    extract_native_mxfp8_components,
)
from precision_probe_common import (
    name_key,
    save_json,
    tensor_hash,
    order_logit_captures,
)
from probe_batches import prepare_probe_batch


def normalized(name):
    while name.startswith("module."):
        name = name[7:]
    return name


class PrecisionProbeImpl(MegatronPolicyWorkerImpl):
    def _records(self):
        names = {p: normalized(n) for n, p in self.model.named_parameters()}
        children = getattr(self.optimizer, "chained_optimizers", [self.optimizer])
        for opt in children:
            owners = {}
            for model_attr, master_attr in [
                ("model_float16_groups", "shard_fp32_from_float16_groups"),
                ("model_fp32_groups", "shard_fp32_groups"),
            ]:
                for models, masters in zip(
                    getattr(opt, model_attr, []), getattr(opt, master_attr, [])
                ):
                    for param, master in zip(models, masters):
                        assert master is not None and master.dtype == torch.float32
                        owners[param] = master
            params = {p for buffer in opt.buffers for p in buffer.params}
            for param in sorted(params, key=lambda p: names[p]):
                master = owners.get(param)
                span = (
                    opt._get_model_param_range_map(param)["param"]
                    if master is not None
                    else None
                )
                yield names[param], param, opt, master, span

    @torch.no_grad()
    def canonical_masters(self, folder, export):
        """Reconstruct FP32 parameters across optimizer shards, independent of bucket layout."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        count = 0
        for name, param, opt, master, span in self._records():
            ranks = dist.get_process_group_ranks(opt.data_parallel_group)
            path = folder / (name_key(name + ":" + ",".join(map(str, ranks))) + ".pt")
            if export:
                full = torch.zeros(param.numel(), dtype=torch.float32, device="cuda")
                if master is not None:
                    full[span.start : span.end].copy_(master)
                dist.all_reduce(full, group=opt.data_parallel_group)
                if ".experts." in name:
                    # Include real FP32 low bits, rather than testing only
                    # FP32 tensors whose values are already BF16-representable.
                    _, exponent = torch.frexp(full.abs())
                    ulp = torch.ldexp(torch.ones_like(full), exponent - 8)
                    ulp.masked_fill_(full == 0, 0)
                    fractions = torch.tensor(
                        [0.375, 0.625, -0.375, -0.625], device="cuda"
                    )
                    indices = (
                        torch.arange(full.numel(), dtype=torch.int32, device="cuda") % 4
                    )
                    full.add_(ulp * fractions[indices])
                    if master is not None:
                        master.copy_(full[span.start : span.end])
                if opt.data_parallel_group.rank() == 0:
                    torch.save(full.cpu(), path)
            else:
                full = torch.load(path, map_location="cuda", weights_only=True)
                assert full.numel() == param.numel() and full.dtype == torch.float32
                if master is not None:
                    master.copy_(full[span.start : span.end])
                restored = torch.zeros_like(full)
                if master is not None:
                    restored[span.start : span.end].copy_(master)
                dist.all_reduce(restored, group=opt.data_parallel_group)
                assert torch.equal(restored, full), (name, "FP32 master mismatch")
            digest.update(name.encode())
            digest.update(tensor_hash(full).encode())
            count += 1
        dist.barrier()
        self._publish_masters()
        state_file = folder / f"initial-nonparameter-state-rank-{dist.get_rank()}.pt"
        if export:
            tracker = get_cuda_rng_tracker().get_states()
            assert all(isinstance(v, torch.Tensor) for v in tracker.values()), (
                "Probe needs tensor RNG tracker states; graph-safe generators need a separate adapter"
            )
            torch.save(
                {
                    "buffers": {
                        normalized(n): b.detach().cpu().clone()
                        for n, b in self.model.named_buffers()
                    },
                    "rng_tracker": {n: v.cpu().clone() for n, v in tracker.items()},
                },
                state_file,
            )
        dist.barrier()
        self._probe_initial_state = torch.load(
            state_file, weights_only=True, map_location="cpu"
        )
        result = {"sha256": digest.hexdigest(), "parameters": count}
        assert count > 0
        return result

    def _publish_masters(self):
        children = getattr(self.optimizer, "chained_optimizers", [self.optimizer])
        for opt in children:
            opt._stage_model_params_from_main_params()
        for name, param, opt, master, _ in self._records():
            if master is None:
                continue
            span = opt._get_model_param_range_map(param)["gbuf_world_in_bucket"]
            index, _, bucket = opt.model_param_gbuf_map[param]
            staged = (
                opt.buffers[index]
                .buckets[bucket]
                .param_data.view(-1)[span.start : span.end]
            )
            assert torch.equal(staged, master.to(staged.dtype)), (
                name,
                "FP32 staging mismatch",
            )
        self.model.start_param_sync(force_sync=True)
        torch.cuda.synchronize()

    @torch.no_grad()
    def _masters_signature(self):
        return {
            name: tensor_hash(master)
            for name, _, _, master, _ in self._records()
            if master is not None
        }

    def refit_probe_state(self):
        return {
            "rank": dist.get_rank(),
            "masters": self._masters_signature(),
            "expert_storage": self._expert_storage(),
            "native_export": self._is_native_mxfp8_export(),
        }

    @torch.no_grad()
    def _gradient_hashes(self):
        return {
            normalized(n): tensor_hash(p.main_grad)
            for n, p in self.model.named_parameters()
            if getattr(p, "main_grad", None) is not None
        }

    @torch.no_grad()
    def _save_gradients(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        manifest = {}
        nonzero = False
        for name, p in self.model.named_parameters():
            grad = getattr(p, "main_grad", None)
            if grad is None:
                continue
            name = normalized(name)
            assert torch.isfinite(grad).all(), (name, "nonfinite gradient")
            nonzero = nonzero or bool(torch.count_nonzero(grad))
            filename = name_key(name) + ".pt"
            torch.save(grad.detach().cpu(), folder / filename)
            manifest[name] = filename
        assert manifest, "No gradients captured"
        assert nonzero, "The fixed teacher-forced objective produced no gradients"
        save_json(folder / "manifest.json", manifest)

    def teacher_forced_probe(
        self,
        data,
        folder,
        streaming,
        atol,
        rtol,
        logprob_detour=True,
        canonical_order=True,
    ):
        """Use the SC begin/microbatch API, capture logits and unreduced gradients, then abort."""
        folder = Path(folder) / f"rank-{dist.get_rank()}"
        folder.mkdir(parents=True, exist_ok=True)
        self.prepare_for_training()
        with torch.no_grad():
            buffers = {normalized(n): b for n, b in self.model.named_buffers()}
            for name, saved in self._probe_initial_state["buffers"].items():
                assert (
                    name in buffers
                    and buffers[name].shape == saved.shape
                    and buffers[name].dtype == saved.dtype
                ), (name, "Initial buffer layout differs")
                buffers[name].copy_(saved)
        data = {k: v.to("cuda") for k, v in data.items()}
        before = self._masters_signature()
        rng = torch.random.get_rng_state(), torch.cuda.get_rng_state()
        tracker = get_cuda_rng_tracker()
        saved_tracker = {k: v.clone() for k, v in tracker.get_states().items()}
        tracker.set_states(
            {k: v.clone() for k, v in self._probe_initial_state["rng_tracker"].items()}
        )
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        original_forward = train_module.model_forward
        original_step = self.optimizer.step

        def forbidden_step(*args, **kwargs):
            raise AssertionError("optimizer.step() is forbidden in numerical probes")

        self.optimizer.step = forbidden_step
        captured = []
        captured_row_ids = []
        gradient_prefixes = []

        def capture_gradient_prefix(next_row):
            targets = {
                "decoder.final_norm.weight",
                "decoder.layers.0.mixer.norm.weight",
                "decoder.layers.4.mixer.norm.weight",
                "decoder.layers.44.mixer.conv1d_weight",
            }
            values = {
                normalized(n): p.main_grad.detach().cpu().clone()
                for n, p in self.model.named_parameters()
                if normalized(n) in targets
                and getattr(p, "main_grad", None) is not None
            }
            assert values, "No diagnostic gradient targets found"
            gradient_prefixes.append({"next_row": next_row, "gradients": values})

        save_json(
            folder / "initial-state.json",
            {
                "expert_storage": self._expert_storage(),
                "buffers": {
                    normalized(n): tensor_hash(b) for n, b in self.model.named_buffers()
                },
                "rng_tracker": {
                    k: tensor_hash(v) for k, v in tracker.get_states().items()
                },
                "max_sequences_per_bin": self.cfg["sequence_packing"][
                    "max_sequences_per_bin"
                ],
            },
        )

        def forward(*args, **kwargs):
            data_dict = args[1] if len(args) > 1 else kwargs["data_dict"]
            row_ids = data_dict["probe_row_id"].reshape(-1)
            assert row_ids.numel() == 1, (
                "Controlled probe requires one sequence per microbatch"
            )
            if self.model.training:
                capture_gradient_prefix(int(row_ids.item()))
            output = original_forward(*args, **kwargs)
            assert torch.isfinite(output).all(), "Nonfinite teacher-forced logits"
            captured.append(output.detach().cpu())
            captured_row_ids.append(int(row_ids.item()))
            return output

        train_module.model_forward = forward
        preservation = None
        try:
            self.begin_train_step(
                NLLLossFn(),
                gbs=self.cfg["train_global_batch_size"],
                mbs=self.cfg["train_micro_batch_size"],
            )
            if self.cfg["megatron_cfg"]["distributed_data_parallel_config"].get(
                "grad_reduce_in_fp32"
            ):
                assert all(
                    p.main_grad.dtype == torch.float32
                    for p in self.model.parameters()
                    if getattr(p, "main_grad", None) is not None
                ), "FP32 gradient-buffer control did not allocate FP32 gradients"
            if streaming:
                half = len(data["input_ids"]) // 2
                left = prepare_probe_batch(
                    {k: v[:half] for k, v in data.items()},
                    self.cfg,
                    canonical_order=canonical_order,
                )
                right_data = {k: v[half:] for k, v in data.items()}
                right = prepare_probe_batch(
                    right_data, self.cfg, canonical_order=canonical_order
                )
                self.train_microbatch(left)
                left_calls = len(captured)
                gradients_before = self._gradient_hashes()
                assert gradients_before
                if logprob_detour:
                    self.prepare_for_lp_inference(keep_train_buffers=True)
                    # Mirrors SC's between-chunk logprob detour; no sampling or update.
                    self.get_logprobs(
                        data=prepare_probe_batch(
                            right_data,
                            self.cfg,
                            logprob=True,
                            canonical_order=canonical_order,
                        ),
                        micro_batch_size=1,
                        require_router_replay=False,
                    )
                    self.prepare_for_training()
                gradients_after = self._gradient_hashes()
                changed = sorted(
                    k
                    for k in set(gradients_before) | set(gradients_after)
                    if gradients_before.get(k) != gradients_after.get(k)
                )
                preservation = {"pass": not changed, "changed": changed}
                save_json(folder / "gradient-preservation.json", preservation)
                # LP forwards are diagnostics, not the training-logit comparison.
                del captured[left_calls:]
                del captured_row_ids[left_calls:]
                self.train_microbatch(right)
            else:
                self.train_microbatch(
                    prepare_probe_batch(data, self.cfg, canonical_order=canonical_order)
                )
            capture_gradient_prefix(None)
            torch.save(gradient_prefixes, folder / "gradient-prefixes.pt")
            if canonical_order:
                assert captured_row_ids == sorted(data["probe_row_id"].tolist()), (
                    "Controlled backward order did not match prompt order"
                )
            self._save_gradients(folder / "gradients")
            torch.save(
                order_logit_captures(captured, captured_row_ids), folder / "logits.pt"
            )
            save_json(folder / "logit-row-order.json", captured_row_ids)
            assert self._masters_signature() == before, (
                "A probe changed optimizer masters"
            )
            result = {
                "rank": dist.get_rank(),
                "master_hashes": before,
                "streaming": streaming,
                "logprob_detour": logprob_detour,
                "logit_calls": len(captured),
                "gradient_preservation": preservation,
                "optimizer_step_calls": 0,
                "raw_pre_finish_gradients": True,
                "canonical_accumulation_order": canonical_order,
                "gradient_dtype_counts": dict(
                    Counter(
                        str(p.main_grad.dtype)
                        for p in self.model.parameters()
                        if getattr(p, "main_grad", None) is not None
                    )
                ),
                "normalization": {
                    key: float(self._train_step_state[key].item())
                    if isinstance(self._train_step_state[key], torch.Tensor)
                    else self._train_step_state[key]
                    for key in [
                        "num_chunks",
                        "total_num_microbatches",
                        "local_valid_seqs",
                        "local_valid_toks",
                        "mtp_enabled",
                    ]
                },
                "mtp_loss_scaling_factor": getattr(
                    self._get_model_config(), "mtp_loss_scaling_factor", None
                ),
            }
            save_json(folder / "probe.json", result)
            if preservation is not None:
                assert preservation["pass"], (
                    "Accumulated gradients changed across the logprob detour"
                )
            return result
        finally:
            train_module.model_forward = original_forward
            self.optimizer.step = original_step
            self.abort_train_step()
            torch.random.set_rng_state(rng[0])
            torch.cuda.set_rng_state(rng[1])
            tracker.set_states(saved_tracker)

    @torch.no_grad()
    def _expert_storage(self):
        result = {}
        for name, param in self.model.named_parameters():
            if ".experts." not in name:
                continue
            members = (
                list(get_grouped_quantized_members(param, create_if_missing=True))
                if is_grouped_mxfp8tensor(param)
                else ([param] if is_mxfp8tensor(param) else [])
            )
            if members:
                for i, member in enumerate(members):
                    parts = extract_native_mxfp8_components(member)
                    result[f"{name}:{i}"] = [
                        tensor_hash(parts.weight),
                        tensor_hash(parts.weight_scale),
                    ]
            else:
                result[name] = tensor_hash(param)
        assert result
        return result

    @torch.no_grad()
    def test_stale_storage(self, folder, data):
        """Perturb FP32 expert masters, publish, and compare physical FP8 against a fresh cast."""
        data = prepare_probe_batch(
            {k: v.to("cuda") for k, v in data.items()}, self.cfg, logprob=True
        )
        self.prepare_for_lp_inference()
        old_logprobs = (
            self.get_logprobs(
                data=data, micro_batch_size=1, require_router_replay=False
            )["logprobs"]
            .detach()
            .cpu()
        )
        old_storage = self._expert_storage()
        old_column_scales = {}
        for name, param in self.model.named_parameters():
            if ".experts." not in name:
                continue
            members = (
                list(get_grouped_quantized_members(param, create_if_missing=True))
                if is_grouped_mxfp8tensor(param)
                else ([param] if is_mxfp8tensor(param) else [])
            )
            for index, member in enumerate(members):
                metadata = member.get_metadata()
                if metadata.get("columnwise_data") is not None:
                    cols = member.shape[-1]
                    rows = member.numel() // cols
                    old_column_scales[(normalized(name), index)] = tensor_hash(
                        metadata["columnwise_scale_inv"][: (rows + 31) // 32, :cols]
                    )
        report = []
        captured_column_failure = False
        restore = []
        for name, param, opt, master, span in self._records():
            if ".experts." not in name:
                continue
            if master is not None:
                restore.append((master, master.detach().clone()))
                master.mul_(2)
        assert restore, "No expert master shards found"
        # Negative control: master changes alone leave physical weights stale.
        assert self._expert_storage() == old_storage
        try:
            self._publish_masters()
            new_storage = self._expert_storage()
            assert new_storage != old_storage, (
                "Publication left expert weights/scales stale"
            )
            new_logprobs = (
                self.get_logprobs(
                    data=data, micro_batch_size=1, require_router_replay=False
                )["logprobs"]
                .detach()
                .cpu()
            )
            assert torch.isfinite(new_logprobs).all()
            changed_logits = not torch.equal(old_logprobs, new_logprobs)
            save_json(
                Path(folder) / f"refresh-rank-{dist.get_rank()}.json",
                {
                    "negative_control_stale_before_publish": True,
                    "storage_changed": True,
                    "logprobs_changed": changed_logits,
                },
            )
            assert changed_logits, (
                "Forward still observes stale expert workspace after refresh"
            )
            for name, param, opt, master, span in self._records():
                if ".experts." not in name:
                    continue
                full = torch.zeros(param.numel(), dtype=torch.float32, device="cuda")
                if master is not None:
                    full[span.start : span.end].copy_(master)
                dist.all_reduce(full, group=opt.data_parallel_group)
                members = (
                    list(get_grouped_quantized_members(param, create_if_missing=True))
                    if is_grouped_mxfp8tensor(param)
                    else ([param] if is_mxfp8tensor(param) else [])
                )
                if not members:
                    assert torch.equal(
                        param.detach().float().view(-1), full.to(param.dtype).float()
                    ), (name, "stale BF16 weights")
                    report.append({"name": name, "kind": "bf16", "pass": True})
                    continue
                offset = 0
                for index, member in enumerate(members):
                    bf16 = (
                        full[offset : offset + member.numel()]
                        .view(member.shape)
                        .to(torch.bfloat16)
                    )
                    offset += member.numel()
                    quantizer = copy.copy(
                        getattr(member, "_quantizer", None)
                        or getattr(param, "quantizer", None)
                    )
                    assert quantizer is not None, (name, "No real TE quantizer")
                    metadata = member.get_metadata()
                    has_columns = metadata.get("columnwise_data") is not None
                    if has_columns:
                        quantizer.columnwise_usage = True
                    fresh = quantizer.quantize(bf16)
                    live_parts = extract_native_mxfp8_components(member)
                    fresh_parts = extract_native_mxfp8_components(fresh)
                    weight_ok = tensor_hash(live_parts.weight) == tensor_hash(
                        fresh_parts.weight
                    )
                    scale_ok = tensor_hash(live_parts.weight_scale) == tensor_hash(
                        fresh_parts.weight_scale
                    )
                    column_ok = None
                    column_data_ok = None
                    column_scale_ok = None
                    column_scale_changed = None
                    if has_columns:
                        rows, cols = (
                            member.numel() // member.shape[-1],
                            member.shape[-1],
                        )
                        fm = fresh.get_metadata()
                        assert fm.get("columnwise_data") is not None, (
                            name,
                            "Fresh column representation missing",
                        )
                        column_data_ok = tensor_hash(
                            metadata["columnwise_data"]
                        ) == tensor_hash(fm["columnwise_data"])
                        # TE uses [ceil(M/32), K] column scale grids;
                        # exclude allocation padding, which is not semantic state.
                        column_scale_ok = tensor_hash(
                            metadata["columnwise_scale_inv"][: (rows + 31) // 32, :cols]
                        ) == tensor_hash(
                            fm["columnwise_scale_inv"][: (rows + 31) // 32, :cols]
                        )
                        column_ok = column_data_ok and column_scale_ok
                        column_scale_changed = old_column_scales.get(
                            (name, index)
                        ) != tensor_hash(
                            metadata["columnwise_scale_inv"][: (rows + 31) // 32, :cols]
                        )
                        if not column_ok and not captured_column_failure:
                            torch.save(
                                {
                                    "name": name,
                                    "member": index,
                                    "logical_shape": list(member.shape),
                                    "live_column_codes": metadata[
                                        "columnwise_data"
                                    ].cpu(),
                                    "fresh_column_codes": fm["columnwise_data"].cpu(),
                                    "live_column_scales": metadata[
                                        "columnwise_scale_inv"
                                    ][: (rows + 31) // 32, :cols].cpu(),
                                    "fresh_column_scales": fm["columnwise_scale_inv"][
                                        : (rows + 31) // 32, :cols
                                    ].cpu(),
                                    "quantizer_rowwise_usage": quantizer.rowwise_usage,
                                    "quantizer_columnwise_usage": quantizer.columnwise_usage,
                                },
                                Path(folder)
                                / f"column-failure-rank-{dist.get_rank()}.pt",
                            )
                            captured_column_failure = True
                    report.append(
                        {
                            "name": name,
                            "member": index,
                            "weight_pass": weight_ok,
                            "scale_pass": scale_ok,
                            "columnwise_pass": column_ok,
                            "columnwise_data_pass": column_data_ok,
                            "columnwise_scale_pass": column_scale_ok,
                            "columnwise_scale_changed_after_master_perturbation": column_scale_changed,
                            "logical_shape": list(member.shape),
                            "columnwise_scale_allocation_shape": list(
                                metadata["columnwise_scale_inv"].shape
                            )
                            if has_columns
                            else None,
                            "pass": weight_ok and scale_ok and column_ok is not False,
                        }
                    )
                assert offset == full.numel()
            save_json(Path(folder) / f"stale-rank-{dist.get_rank()}.json", report)
            assert report and all(row["pass"] for row in report), (
                "Stale quantized weights or scales"
            )
            return {"rank": dist.get_rank(), "checked": len(report)}
        finally:
            for master, original in restore:
                master.copy_(original)
            self._publish_masters()


@ray.remote  # pragma: no cover
class PrecisionProbeWorker(PrecisionProbeImpl):
    pass
