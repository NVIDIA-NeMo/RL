"""Observers of real Nano staging, accumulation and gradient finalization."""

import copy
import hashlib
import os
import re
from pathlib import Path

import ray
import torch
import torch.distributed as dist

from megatron.core.fp8_utils import (
    get_grouped_quantized_members,
    is_grouped_mxfp8tensor,
    is_mxfp8tensor,
)
from precision_probe_common import (
    mxfp8_storage_snapshot,
    semantic_model_state,
    save_json,
    tensor_error,
    tensor_hash,
    name_key,
)
from precision_probe_worker import PrecisionProbeImpl, normalized
from probe_batches import prepare_probe_batch
from nemo_rl.algorithms.loss import NLLLossFn
from boundary_probe_common import validate_named_spans, scaled_fp32_reference_input


class StopBeforeOptimizerUpdate(Exception):
    """The production finish path reached the optimizer boundary."""


class BoundaryProbeImpl(PrecisionProbeImpl):
    def __init__(self, config, *args, **kwargs):
        fp8 = config["megatron_cfg"]["fp8_cfg"]["fp8_param"]
        resumed = kwargs.get("weights_path") is not None
        os.environ["NANO_BOUNDARY_MODE"] = (
            f"fp8_{str(fp8).lower()}_" + ("resume" if resumed else "fresh")
        )
        super().__init__(config, *args, **kwargs)
        self._boundary_expected = {}
        self._boundary_case = None
        self._boundary_events = []
        self._boundary_grad_calls = {}
        self._boundary_track_master_changes = False

    def _nccl_reshard_refit(self, *args, **kwargs):
        from boundary_refit_observer import emit
        from rollout_diagnostics import model_state

        emit("trainer_before_refit", {"model": model_state(self.model)})
        result = super()._nccl_reshard_refit(*args, **kwargs)
        emit("trainer_after_refit", {"model": model_state(self.model)})
        return result

    def _broadcast_misc_params_packed(self, *args, **kwargs):
        self._boundary_sending_misc = True
        try:
            return super()._broadcast_misc_params_packed(*args, **kwargs)
        finally:
            self._boundary_sending_misc = False

    def _iter_params_with_optional_kv_scales(self, *args, **kwargs):
        from boundary_refit_observer import emit, fingerprint

        for name, value in super()._iter_params_with_optional_kv_scales(*args, **kwargs):
            if getattr(self, "_boundary_sending_misc", False):
                emit("misc_send", {"name": name, **fingerprint(value)})
            yield name, value

    @staticmethod
    def _members(param):
        if is_grouped_mxfp8tensor(param):
            return list(get_grouped_quantized_members(param, create_if_missing=True))
        return [param] if is_mxfp8tensor(param) else []

    @torch.no_grad()
    def loaded_snapshot(self, folder):
        """Observe loaded masters before any canonical-bank replacement or staging."""
        result = {"masters": {}, "model": {}, "matches_own_masters": {}}
        result["mamba_forward_references"] = {}
        for module_name, module in self.model.named_modules():
            cp = getattr(module, "cp", None)
            if cp is None or not hasattr(cp, "A_log_cp1"):
                continue
            aliases = {}
            for key in ["A_log", "D", "dt_bias", "conv1d_weight", "conv1d_bias"]:
                registered = getattr(module, key, None)
                consumed = getattr(cp, key + "_cp1", None)
                if registered is None and consumed is None:
                    continue
                assert isinstance(registered, torch.Tensor) and isinstance(consumed, torch.Tensor), (module_name, key)
                aliases[key] = {"same_object": registered is consumed,
                    "same_storage": registered.data_ptr() == consumed.data_ptr(),
                    "registered_dtype": str(registered.dtype), "consumed_dtype": str(consumed.dtype),
                    "registered_sha256": tensor_hash(registered), "consumed_sha256": tensor_hash(consumed)}
            cache_stale = getattr(module, "_A_neg_exp_cache_stale", None)
            cache = getattr(module, "_A_neg_exp_cache", None)
            result["mamba_forward_references"][normalized(module_name)] = {
                "aliases": aliases, "decode_cache_stale": cache_stale,
                # Stale caches may intentionally contain uninitialized storage.
                "valid_decode_cache_sha256": tensor_hash(cache) if cache_stale is False and isinstance(cache, torch.Tensor) else None}
        expected = {}
        for name, param, opt, master, span in self._records():
            full = torch.zeros(param.numel(), device="cuda", dtype=torch.float32)
            if master is not None:
                full[span.start : span.end].copy_(master)
            dist.all_reduce(full, group=opt.data_parallel_group)
            result["masters"][name] = tensor_hash(full)
            staged = full.to(param.dtype).view(param.shape)
            expected[name] = {"staged": tensor_hash(staged), "quantized": []}
            members = self._members(param)
            if members:
                offset = 0
                actual, fresh = [], []
                for member in members:
                    source = staged.view(-1)[offset : offset + member.numel()]
                    offset += member.numel()
                    quantizer = copy.copy(member._quantizer)
                    quantizer.rowwise_usage = True
                    quantizer.columnwise_usage = True
                    quantizer.optimize_for_gemm = False
                    quantized = quantizer.quantize(source.view(member.shape))
                    fresh.append(semantic_model_state(mxfp8_storage_snapshot(
                        member.shape, quantized.get_metadata()
                    )))
                    actual.append(semantic_model_state(mxfp8_storage_snapshot(
                        member.shape, member.get_metadata()
                    )))
                assert offset == param.numel()
                expected[name]["quantized"] = fresh
                result["model"][name] = actual
                result["matches_own_masters"][name] = actual == fresh
            else:
                result["model"][name] = tensor_hash(param)
                result["matches_own_masters"][name] = torch.equal(param, staged)
        self._boundary_expected = expected
        if folder is not None:
            save_json(Path(folder) / f"rank-{dist.get_rank()}.json", result)
        return {"rank": dist.get_rank(), "masters": result["masters"],
                "mismatching_model_parameters": [n for n, v in result["matches_own_masters"].items() if not v]}

    def _publish_masters(self):
        if self._boundary_track_master_changes:
            self.loaded_snapshot(None)
        return super()._publish_masters()

    def stale_boundary_probe(self, folder, data):
        self._boundary_case = "stale-storage"
        self._boundary_grad_calls = {}
        start = len(self._boundary_events)
        self._boundary_track_master_changes = True
        try:
            return self.test_stale_storage(folder=folder, data=data)
        finally:
            self._boundary_track_master_changes = False
            save_json(self._boundary_root / "stale-storage-events.json", self._boundary_events[start:])
            self._boundary_case = None

    def arm_boundary_observers(self, folder):
        """Attach observers after initialization, without replacing production operations."""
        from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup

        self._boundary_root = Path(folder) / f"rank-{dist.get_rank()}"
        self._boundary_root.mkdir(parents=True, exist_ok=True)
        names = {p: normalized(n) for n, p in self.model.named_parameters()}
        owner = self
        original_post = _ParamAndGradBucketGroup._post_param_sync
        original_reduce = _ParamAndGradBucketGroup.start_grad_sync

        def post(group):
            before = []
            for bucket in group.buckets:
                if bucket.param_data is None:
                    continue
                validate_named_spans([
                    {"name": names[p], "start": bucket.param_to_index[p][0],
                     "end": bucket.param_to_index[p][1], "numel": p.numel()}
                    for p in bucket.params], bucket.param_data.numel())
                for param in bucket.params:
                    start, end = bucket.param_to_index[param]
                    value = bucket.param_data.view(-1)[start:end].view(param.shape)
                    expected = owner._boundary_expected.get(names[param])
                    assert expected is not None, (names[param], "Missing master-derived staging coverage")
                    matches = tensor_hash(value) == expected["staged"]
                    if not matches:
                        failure = owner._boundary_root / "first-staging-failure.pt"
                        if not failure.exists():
                            torch.save({"name": names[param], "case": owner._boundary_case,
                                "staged": value.detach().cpu(), "expected": expected}, failure)
                    before.append({"name": names[param], "start": start, "end": end,
                                   "dtype": str(value.dtype), "staging_matches": matches,
                                   "gradient_contributions_seen": owner._boundary_grad_calls.get(names[param], 0)})
            original_post(group)
            after = []
            for bucket in group.buckets:
                if bucket.param_data is None:
                    continue
                all_quantized = all(owner._members(p) for p in bucket.params)
                zero_required = group.ddp_config.reuse_grad_buf_for_mxfp8_param_ag and all_quantized
                zero = not bool(torch.count_nonzero(bucket.param_data)) if zero_required else None
                after.append({"zero_required": bool(zero_required), "zero": zero})
                if zero_required:
                    if not zero:
                        owner._boundary_events.append({"kind": "buffer_clear_failure", "case": owner._boundary_case,
                            "parameters": sorted(names[p] for p in bucket.params)})
                    assert zero, "Shared BF16 staging buffer was not cleared"
                for param in bucket.params:
                    members = owner._members(param)
                    expected = owner._boundary_expected.get(names[param])
                    if members and expected is not None:
                        actual = [semantic_model_state(mxfp8_storage_snapshot(m.shape, m.get_metadata())) for m in members]
                        if actual != expected["quantized"]:
                            owner._boundary_events.append({"kind": "quantized_refresh_failure", "case": owner._boundary_case,
                                "name": names[param], "actual": actual, "expected": expected["quantized"]})
                            failure = owner._boundary_root / "first-quantized-refresh-failure.pt"
                            if not failure.exists():
                                torch.save({"name": names[param], "case": owner._boundary_case,
                                    "actual": [{k: v.detach().cpu() for k, v in m.get_metadata().items()
                                                if isinstance(v, torch.Tensor)} for m in members],
                                    "expected": expected["quantized"]}, failure)
                        assert actual == expected["quantized"], (names[param], "Post-sync FP8 codes/scales differ from masters")
            owner._boundary_events.append({"case": owner._boundary_case, "kind": "post_param_sync", "before": before, "after": after})
            assert all(v["staging_matches"] is not False for v in before), "Gathered staging values differ from FP32 masters"

        def reduce(group, force_all_reduce=False):
            # Reference collectives run before the production coalescing scope.
            # Keep the same original-dtype scaling before an FP32 reduction.
            assert group.ddp_config.num_distributed_optimizer_instances == 1
            assert not group.ddp_config.overlap_grad_reduce
            comm = group.intra_distributed_optimizer_instance_group if group.ddp_config.use_distributed_optimizer else group.data_parallel_group
            refs = []
            for bucket in group.buckets:
                assert not bucket.params_with_extra_main_grads, "Extra-grad copy path needs its own reference adapter"
                reference = scaled_fp32_reference_input(bucket.grad_data, bucket.gradient_scaling_factor)
                dist.all_reduce(reference, group=comm)
                if group.ddp_config.average_in_collective:
                    reference.div_(dist.get_world_size(comm))
                refs.append(reference)
            result = original_reduce(group, force_all_reduce=force_all_reduce)
            torch.cuda.synchronize()
            for bucket, reference in zip(group.buckets, refs):
                value = bucket.grad_data
                if group.ddp_config.use_distributed_optimizer and not force_all_reduce:
                    size = value.numel() // dist.get_world_size(comm)
                    start = dist.get_rank(comm) * size
                    value, reference = value[start:start + size], reference[start:start + size]
                owner._boundary_events.append({"case": owner._boundary_case, "kind": "gradient_reduction",
                    "ranks": dist.get_process_group_ranks(comm), "dtype": str(value.dtype),
                    "parameters": sorted(names[p] for p in bucket.params),
                    "fp32_reference_error": tensor_error(value, reference, atol=0, rtol=0)})
            return result

        _ParamAndGradBucketGroup._post_param_sync = post
        _ParamAndGradBucketGroup.start_grad_sync = reduce
        layouts = []
        for name, param, opt, master, span in self._records():
            gbuf, _, bucket_id = opt.model_param_gbuf_map[param]
            bucket = opt.buffers[gbuf].buckets[bucket_id]
            layouts.append({"name": name, "bucket": bucket_id, "buffer": gbuf,
                "param_dtype": str(bucket.param_data.dtype), "grad_dtype": str(param.main_grad.dtype),
                "param_to_index": list(bucket.param_to_index[param]),
                "owned_param_range": [span.start, span.end] if master is not None else None,
                "param_grad_share_storage": bucket.param_data.untyped_storage().data_ptr() == bucket.grad_data.untyped_storage().data_ptr()})
            def first_grad(grad, name=name, param=param):
                count = owner._boundary_grad_calls.get(name, 0)
                if count == 0:
                    zero = not bool(torch.count_nonzero(param.main_grad))
                    owner._boundary_events.append({"case": owner._boundary_case, "kind": "first_gradient", "name": name, "buffer_zero": zero})
                    assert zero, (name, "Gradient accumulation started on a nonzero buffer")
                owner._boundary_grad_calls[name] = count + 1
                return grad
            param.register_hook(first_grad)
        save_json(self._boundary_root / "buffer-layout.json", layouts)
        self._boundary_layer_seen = set()
        for module_name, module in self.model.named_modules():
            module_name = normalized(module_name)
            if not re.fullmatch(r"decoder\.layers\.\d+|decoder\.final_norm|embedding", module_name):
                continue
            def capture(module, args, output, name=module_name):
                if self._boundary_case is None or not self.model.training or name in self._boundary_layer_seen:
                    return
                self._boundary_layer_seen.add(name)
                value = output[0] if isinstance(output, tuple) else output
                if not isinstance(value, torch.Tensor):
                    raise AssertionError((name, "Unrecognized layer output"))
                folder = self._boundary_root / self._boundary_case / "layers"
                folder.mkdir(parents=True, exist_ok=True)
                torch.save(value.detach().cpu(), folder / (name_key(name) + ".pt"))
                self._boundary_events.append({"case": self._boundary_case,
                    "kind": "layer_forward", "name": name, "sha256": tensor_hash(value),
                    "shape": list(value.shape), "dtype": str(value.dtype)})
            module.register_forward_hook(capture)
        return {"rank": dist.get_rank(), "parameters": len(layouts)}

    def initialize_private_optimizer_state(self, data):
        """One real update only in the isolated seed, before installing a common bank."""
        self.prepare_for_training()
        self.begin_train_step(NLLLossFn(), gbs=self.cfg["train_global_batch_size"],
                              mbs=self.cfg["train_micro_batch_size"])
        try:
            self.train_microbatch(prepare_probe_batch(
                {k: v.to("cuda") for k, v in data.items()}, self.cfg
            ))
            self.finish_train_step()
        finally:
            if self._train_step_state is not None:
                self.abort_train_step()
        return {"rank": dist.get_rank(), "private_seed_updates": 1}

    def verify_and_publish_current_masters(self):
        self._publish_masters()
        return {"rank": dist.get_rank(), "published": True}

    def load_probe_controls(self, folder):
        self._probe_initial_state = torch.load(
            Path(folder) / f"initial-nonparameter-state-rank-{dist.get_rank()}.pt",
            weights_only=True, map_location="cpu",
        )
        return {"rank": dist.get_rank(), "loaded_controls": True}

    def teacher_forced_probe(self, *args, **kwargs):
        self._boundary_case = Path(kwargs["folder"]).name
        self._boundary_grad_calls = {}
        self._boundary_layer_seen = set()
        start = len(self._boundary_events)
        try:
            return super().teacher_forced_probe(*args, **kwargs)
        finally:
            save_json(self._boundary_root / f"{self._boundary_case}-events.json", self._boundary_events[start:])
            self._boundary_case = None
            self._boundary_grad_calls = {}
            # Finalization marked parameter gathers undispatched, but this probe
            # deliberately stopped before the optimizer would restage parameters.
            # Publish unchanged masters before the next independent case/refit.
            self._publish_masters()

    def _save_gradients(self, folder):
        super()._save_gradients(folder)
        original_step = self.optimizer.step
        reached = False

        def stop():
            nonlocal reached
            reached = True
            self._save_reduced_gradients(Path(folder).parent / "reduced-gradients")
            raise StopBeforeOptimizerUpdate()

        self.optimizer.step = stop
        try:
            self.finish_train_step()
        except StopBeforeOptimizerUpdate:
            pass
        finally:
            self.optimizer.step = original_step
        assert reached, "Production finish path never reached the optimizer boundary"

    @torch.no_grad()
    def _save_reduced_gradients(self, output):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        manifest = {}
        for name, param, opt, master, span in self._records():
            full = torch.zeros(param.numel(), device="cuda", dtype=torch.float32)
            if master is not None:
                full[span.start:span.end].copy_(param.main_grad.view(-1)[span.start:span.end])
            dist.all_reduce(full, group=opt.data_parallel_group)
            filename = name_key(name) + ".pt"
            torch.save(full.cpu(), output / filename)
            manifest[name] = filename
        save_json(output / "manifest.json", manifest)

    @torch.no_grad()
    def synthetic_reduction_probe(self, folder):
        """Same semantic local gradients in both layouts, without a model backward."""
        self._boundary_case = "synthetic-reduction"
        self._boundary_grad_calls = {}
        start = len(self._boundary_events)
        masters = self._masters_signature()
        self.prepare_for_training()
        self._publish_masters()
        self.model.zero_grad_buffer()
        source = {}
        for name, param, opt, _, _ in self._records():
            rank = dist.get_rank(opt.data_parallel_group)
            # Cancellation-sensitive values plus a name/element-dependent signal.
            values = torch.tensor([256., 1., -256., 1.], device="cuda")
            offset = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 4
            indices = (torch.arange(param.numel(), device="cuda", dtype=torch.int32) + offset) % 4
            grad = values[(indices + rank) % 4].to(param.main_grad.dtype)
            param.main_grad.view(-1).copy_(grad)
            source[name] = tensor_hash(param.main_grad)
        self.model.finish_grad_sync()
        torch.cuda.synchronize()
        self._save_reduced_gradients(Path(folder) / f"rank-{dist.get_rank()}" / "gradients")
        assert self._masters_signature() == masters
        save_json(self._boundary_root / "synthetic-source.json", source)
        save_json(self._boundary_root / "synthetic-reduction-events.json", self._boundary_events[start:])
        self.model.zero_grad_buffer()
        self._publish_masters()
        return {"rank": dist.get_rank(), "source": source, "optimizer_updates": 0}


@ray.remote  # pragma: no cover
class BoundaryProbeWorker(BoundaryProbeImpl):
    pass
