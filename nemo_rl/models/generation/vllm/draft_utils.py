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
"""vLLM-side draft co-training helpers, kept vllm-import-free at module scope.

Collects the draft-specific refit/module-sharing glue that ``vllm_backend.py``
and ``vllm_worker.py`` previously spread across each other (``vllm_worker.py``
re-declaring ``vllm_backend.py``'s method list because ``vllm_backend.py``
imports vllm eagerly). Every vllm import here is function-local so the Ray
driver process -- which never installs vllm -- can still import this module.
"""

import os
from typing import Any, Optional

# Speculative methods whose drafter is co-trained by the trainer and refit
# through the ``draft.*`` weight stream (dspark/dflash block drafters and the
# eagle3 TTT drafter). MTP is co-trained too but streams without the prefix.
COTRAINED_SPECULATIVE_METHODS = ("dspark", "dflash", "eagle3")

# Env flag that requests disable_draft_module_sharing() at import time in
# every process that resolves vllm_backend.py (the driver actor and any
# spawned vLLM executor workers, which import it for worker_extension_cls
# before loading the model).
DRAFT_DISABLE_MODULE_SHARING_ENV = "NRL_DRAFT_DISABLE_MODULE_SHARING"


def draft_module_sharing_disable_required(config: dict[str, Any]) -> bool:
    """Whether this worker's engine needs the draft module-sharing disable.

    Full-stream draft co-training refits the drafter's trained
    embed_tokens/lm_head. Under load_format="dummy" the pinned vLLM would
    alias those drafter modules to the target model's (no checkpoint load
    ever marks them as owned), and the draft refit would then overwrite the
    policy's serving weights through the alias. The method name alone
    doesn't imply a full stream -- Megatron block-drafter paths can use the
    same dspark/dflash method names with a headless exporter that relies on
    module sharing, just like the megatron eagle3 path does -- so gate on
    _draft_full_refit (true only for DTensor-v2 co-training, which always
    streams the drafter's entire state_dict) for all three methods.
    """
    load_format = config["vllm_cfg"]["load_format"]
    spec_cfg = config.get("vllm_kwargs", {}).get("speculative_config")
    if load_format != "dummy" or spec_cfg is None:
        return False
    method = spec_cfg.get("method")
    return method in COTRAINED_SPECULATIVE_METHODS and bool(
        config.get("_draft_full_refit")
    )


def disable_draft_module_sharing() -> None:
    """Keep the drafter's embed_tokens/lm_head separate from the target's.

    The pinned vLLM's ``load_dspark_model`` / ``load_dflash_model`` /
    ``load_eagle_model`` alias the drafter's embed_tokens and lm_head modules
    to the target model's whenever the drafter's ``load_weights`` has not
    marked them as owned -- which is always the case under
    ``load_format="dummy"``, where no checkpoint weights are read at startup.
    With draft co-training the refit stream carries trained
    ``draft.embed_tokens`` / ``draft.lm_head`` weights, and loading them through
    the aliased modules silently replaces the POLICY model's serving embed and
    lm_head with the draft's. Forcing ``_should_share`` to False makes the
    drafter keep the modules it built in ``__init__``; refit fills them before
    the first generation, and CUDA graphs capture drafter-owned storage.

    ``_should_share`` is defined in the eagle utils module and imported by
    value into the dspark/dflash utils modules, so each module's global must
    be rebound individually.

    A second, independent embed_tokens-sharing decision lives in
    ``SpecDecodeBaseProposer._maybe_share_embeddings`` (used by the eagle3
    proposer): it shares whenever ``self.model.has_own_embed_tokens`` is
    falsy, a flag only ever set (to True) by ``process_eagle_weight`` when a
    weight literally named "embed_tokens" is loaded -- which never happens
    under ``load_format="dummy"``. This is not reachable through
    ``_should_share`` at all, so it needs its own no-op patch; skipping it
    entirely leaves each drafter holding the embed_tokens it built in
    ``__init__``, matching the ``_should_share`` override above.

    Must run before engine creation (drafter load and CUDA-graph capture).
    """
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
    from vllm.v1.worker.gpu.spec_decode.dflash import utils as dflash_utils
    from vllm.v1.worker.gpu.spec_decode.dspark import utils as dspark_utils
    from vllm.v1.worker.gpu.spec_decode.eagle import utils as eagle_utils

    def _never_share(*args: Any, **kwargs: Any) -> bool:
        return False

    def _never_share_embeddings(self: Any, target_language_model: Any) -> None:
        return None

    eagle_utils._should_share = _never_share
    dspark_utils._should_share = _never_share
    dflash_utils._should_share = _never_share
    SpecDecodeBaseProposer._maybe_share_embeddings = _never_share_embeddings


# Incoming draft stream keys each drafter's loader intentionally skips. They
# are tolerated as extras in the refit manifest and excluded from the required
# key set. dspark/dflash: mask_embedding is a placeholder param, the
# confidence head is not wired into inference (dflash drafts never have one),
# and t2d is training-only. eagle3: t2d is training-only.
_DRAFT_SKIPPED_KEY_SUBSTRINGS = {
    "dspark": ("mask_embedding", "confidence_head", "t2d"),
    "dflash": ("mask_embedding", "confidence_head", "t2d"),
    "eagle3": ("t2d",),
}


def _is_full_draft_stream(draft_keys: "set[str] | list[str]") -> bool:
    """Whether a ``draft.*`` stream is the DTensor-v2 FULL drafter.

    The DTensor-v2 co-training path streams the vendored model's entire
    state_dict (always including per-layer keys like ``layers.0.*``, and
    normally ``embed_tokens`` too), regardless of method (dspark/dflash/
    eagle3). A megatron co-training exporter may instead stream a PARTIAL
    drafter -- the megatron eagle3 exporter intentionally omits
    ``embed_tokens`` and aliases its single collapsed layer as
    ``midlayer.*`` instead of ``layers.0.*``, relying on drafter module
    sharing at serve time -- and future megatron block drafters
    (dspark/dflash) may do the same.

    Neither signal alone is reliable: a FULL stream can be missing
    ``embed_tokens`` (e.g. a misconfigured trainer export, or
    ``train_embed_and_head=False``), and some callers only ever see a
    one-key slice of a FULL stream (e.g. a single IPC transport batch) with
    no ``layers.*`` key in it either. Treat the stream as FULL unless it
    carries neither signal, so a batch that happens to contain only
    ``lm_head.weight`` -- as the genuine megatron partial path's minimal
    payload does -- is still read as partial, while a FULL stream missing
    just ``embed_tokens`` is still caught by its remaining ``layers.*``
    keys.
    """
    return any("embed_tokens" in key or "layers." in key for key in draft_keys)


def _speculative_method_of(model_runner: Any) -> Optional[str]:
    spec_config = getattr(model_runner.vllm_config, "speculative_config", None)
    return getattr(spec_config, "method", None) if spec_config else None


def _draft_owns_speculator(model_runner: Any, method: Optional[str]) -> bool:
    """Whether this rank should own the co-trained speculator.

    vLLM keeps the drafter on the last pipeline stage, so earlier stages
    legitimately have no speculator and must skip draft payloads; every
    rank owns it in single-stage layouts.
    """
    from vllm.distributed.parallel_state import get_pp_group

    if method not in COTRAINED_SPECULATIVE_METHODS:
        return False
    try:
        return bool(get_pp_group().is_last_rank)
    except AssertionError:
        # No initialized PP group (single-process layouts, unit tests).
        return True


def _expected_draft_keys(draft_model: Any, method: str) -> set[str]:
    """Expected incoming ``draft.*`` keys derived from the drafter's layout.

    Inverts the drafter ``load_weights`` name handling shared by
    ``Qwen3DSparkForCausalLM`` / ``DFlashQwen3ForCausalLM`` /
    ``Eagle3Qwen3ForCausalLM``: trainer names are prefixed with ``model.``
    (except ``lm_head.*``, and ``d2t`` which maps to
    ``draft_id_to_target_id``), and fused parameters load from their
    stacked components (qkv_proj <- q/k/v_proj, gate_up_proj <-
    gate/up_proj). Parameters the loader never feeds from the stream (see
    _DRAFT_SKIPPED_KEY_SUBSTRINGS) are excluded.
    """
    fused_expansions = {
        "qkv_proj": ("q_proj", "k_proj", "v_proj"),
        "gate_up_proj": ("gate_proj", "up_proj"),
    }
    skipped = _DRAFT_SKIPPED_KEY_SUBSTRINGS[method]
    expected: set[str] = set()
    for name, _ in draft_model.named_parameters():
        if any(s in name for s in skipped):
            continue
        if name == "draft_id_to_target_id":
            expected.add("draft.d2t")
            continue
        trainer_name = name.removeprefix("model.")
        segments = trainer_name.split(".")
        fused = next((s for s in segments if s in fused_expansions), None)
        if fused is None:
            expected.add(f"draft.{trainer_name}")
        else:
            for part in fused_expansions[fused]:
                expected.add(
                    "draft." + ".".join(part if s == fused else s for s in segments)
                )
    return expected


def _validate_draft_refit_info(
    model_runner: Any,
    draft_model: Any,
    method: Optional[str],
    state_dict_info: dict[str, Any],
) -> None:
    """Hard-error when the trainer's ``draft.*`` manifest mismatches the drafter.

    Owning ranks require exactly the drafter's loadable key set (plus keys
    the loader intentionally skips); non-owning ranks ignore draft payloads
    entirely and are never required to have a drafter.
    """
    if method is None or method not in COTRAINED_SPECULATIVE_METHODS:
        return
    if not _draft_owns_speculator(model_runner, method):
        return
    provided = {k for k in state_dict_info if k.startswith("draft.")}
    if not provided:
        if os.environ.get(DRAFT_DISABLE_MODULE_SHARING_ENV) == "1":
            # The generation worker sets this env exactly when full-draft
            # co-training refit is enabled (dummy-loaded drafter waiting
            # for streamed weights). An empty draft manifest here means the
            # trainer failed to export the draft -- serving would silently
            # run a stale (dummy-initialized) drafter forever.
            raise RuntimeError(
                f"[draft] {method} co-training refit is enabled "
                f"({DRAFT_DISABLE_MODULE_SHARING_ENV}=1) but the refit "
                "manifest carries no draft.* keys; the trainer-side draft "
                "export is missing or misconfigured."
            )
        # Static-drafter mode: with policy.draft.enabled=false the drafter
        # is loaded from its checkpoint at startup and refits legitimately
        # carry only policy weights. Exact-key validation applies only when
        # the trainer co-trains (and therefore streams) the draft.
        return
    if draft_model is None:
        raise RuntimeError(
            "[draft] Draft refit validation requires the drafter model, but "
            "none was found at model_runner.drafter.model or "
            "model_runner.speculator.model on a speculator-owning rank."
        )
    if not _is_full_draft_stream(provided):
        # Megatron co-training (eagle3 today; block drafters may follow)
        # can stream a PARTIAL drafter (no embed_tokens, e.g. eagle3's
        # midlayer.* alias for the single layer) and relies on the
        # drafter sharing the target's embedding; exact-key validation
        # against the vLLM parameter layout only applies to the
        # DTensor-v2 full stream.
        return
    expected = _expected_draft_keys(draft_model, method)
    skipped = _DRAFT_SKIPPED_KEY_SUBSTRINGS[method]
    missing = expected - provided
    unexpected = {
        key for key in provided - expected if not any(s in key for s in skipped)
    }
    errors = []
    if missing:
        from nemo_rl.models.generation.vllm.utils import _format_refit_key_error

        errors.append(_format_refit_key_error("missing draft keys", missing))
    if unexpected:
        from nemo_rl.models.generation.vllm.utils import _format_refit_key_error

        errors.append(_format_refit_key_error("unexpected draft keys", unexpected))
    if errors:
        raise RuntimeError(
            f"[draft] {method} refit manifest does not match the vLLM "
            "drafter layout: " + "; ".join(errors)
        )


def _assert_drafter_owns_modules(
    model_runner: Any,
    draft_model: Any,
    draft_weights: list[tuple[str, Any]],
) -> None:
    """Refuse to load draft embed/lm_head through target-shared modules.

    vLLM's drafter loaders may alias the drafter's embed_tokens and
    lm_head to the target model's modules as a weight-sharing optimization;
    loading refit draft weights through such an alias would overwrite the
    policy's serving weights with the draft's (see
    ``disable_draft_module_sharing``). Guard every refit so any path
    that reintroduces the sharing fails loudly instead of silently
    corrupting generation.
    """
    target_model = model_runner.model
    target_lm = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    refits_embed = any("embed_tokens" in name for name, _ in draft_weights)
    refits_lm_head = any("lm_head" in name for name, _ in draft_weights)

    def _shares_weight(draft_module: Any, target_module: Any) -> bool:
        return (
            draft_module is not None
            and target_module is not None
            and draft_module.weight.data_ptr() == target_module.weight.data_ptr()
        )

    shared = []
    if refits_embed and _shares_weight(
        getattr(getattr(draft_model, "model", None), "embed_tokens", None),
        getattr(getattr(target_lm, "model", None), "embed_tokens", None),
    ):
        shared.append("embed_tokens")
    if refits_lm_head and _shares_weight(
        getattr(draft_model, "lm_head", None),
        getattr(target_lm, "lm_head", None),
    ):
        shared.append("lm_head")
    if shared:
        raise RuntimeError(
            "[draft] The drafter's "
            + "/".join(shared)
            + " share storage with the target model; loading refit draft "
            "weights through the alias would overwrite the policy's serving "
            "weights. Ensure disable_draft_module_sharing() ran "
            "before engine creation "
            f"({DRAFT_DISABLE_MODULE_SHARING_ENV}=1)."
        )
