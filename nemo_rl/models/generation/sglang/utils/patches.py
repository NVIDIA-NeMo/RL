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

import logging
import os
from importlib.util import find_spec

logger = logging.getLogger(__name__)


def _get_sglang_file(relative_path: str) -> str:
    spec = find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            f"sglang package not found while attempting to patch '{relative_path}'. "
        )

    base_dir = next(iter(spec.submodule_search_locations))
    file_path = os.path.join(base_dir, *relative_path.split("/"))
    if not os.path.exists(file_path):
        raise RuntimeError(
            f"Expected sglang file '{relative_path}' not found at '{file_path}'. "
            "The sglang version may have moved this file; compat patch cannot be applied."
        )
    return file_path


def _write_and_verify(
    file_path: str, content: str, sentinel: str | tuple[str, ...]
) -> None:
    sentinels = (sentinel,) if isinstance(sentinel, str) else sentinel
    tmp_path = f"{file_path}.nemo_rl_compat.{os.getpid()}.tmp"
    with open(tmp_path, "w") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, file_path)

    with open(file_path, "r") as f:
        verify = f.read()
    missing = [item for item in sentinels if item not in verify]
    if missing:
        raise RuntimeError(
            f"Compat patch verification failed for {file_path}: "
            f"sentinel(s) {missing} not present after write. "
            "The write may have been silently dropped by the filesystem."
        )


def _patch_sglang_safe_unpickler() -> None:
    file_to_patch = _get_sglang_file("srt/utils/common.py")

    with open(file_to_patch, "r") as f:
        content = f.read()

    module = "nemo_rl.models.generation.sglang.utils.train_utils"
    prefix_sentinel = f'"{module}."'
    if prefix_sentinel in content:
        return

    # Older sglang allowlists module prefixes; newer sglang (e.g. the Inkling2
    # build, 0.5.17+) allowlists exact (module, name) pairs and rejects prefixes.
    prefix_anchor = '        "torch.nn.parameter.",\n'
    pair_anchor = '        ("torch.nn.parameter", "Parameter"),\n'
    # Globals NeMo RL refit payloads reference: the flattened bucket containers
    # and the CUDA-IPC rebuild function installed by monkey_patch_torch_reductions.
    pair_names = (
        "FlattenedTensorMetadata",
        "FlattenedTensorBucket",
        "_rebuild_cuda_tensor_modified",
    )
    pair_lines = "".join(f'        ("{module}", "{name}"),\n' for name in pair_names)
    pair_sentinel = f'("{module}", "{pair_names[0]}")'

    if prefix_anchor in content:
        content = content.replace(
            prefix_anchor, prefix_anchor + f"        {prefix_sentinel},\n", 1
        )
        sentinel = prefix_sentinel
    elif pair_anchor in content:
        if pair_sentinel in content:
            return
        content = content.replace(pair_anchor, pair_anchor + pair_lines, 1)
        sentinel = pair_sentinel
    else:
        raise RuntimeError(
            "SafeUnpickler allowlist anchor not found in "
            f"{file_to_patch}: expected '{prefix_anchor.strip()}' or "
            f"'{pair_anchor.strip()}'."
        )

    _write_and_verify(file_to_patch, content, sentinel)
    logger.info("Patched SafeUnpickler allowlist in %s.", file_to_patch)


def _patch_sglang_transformers_compat_bootstrap() -> None:
    """Load NeMo RL's temporary Transformers patch in every SGLang process."""
    file_to_patch = _get_sglang_file("srt/utils/hf_transformers_patches.py")

    with open(file_to_patch, "r") as f:
        content = f.read()

    sentinel = (
        "    import nemo_rl  # noqa: F401  # Install temporary Transformers patch.\n"
    )
    if sentinel in content:
        return

    anchor = "\n    _applied = True\n"
    if anchor not in content:
        raise RuntimeError(
            f"Transformers patch bootstrap anchor '{anchor.strip()}' not found in "
            f"{file_to_patch}."
        )

    patched_content = content.replace(anchor, f"{anchor}\n{sentinel}", 1)
    compile(patched_content, file_to_patch, "exec")
    _write_and_verify(file_to_patch, patched_content, sentinel)
    logger.info("Patched Transformers compatibility bootstrap in %s.", file_to_patch)


def _patch_sglang_inkling2_bucketed_refit() -> None:
    """Let Inkling2 accept refit weights in buckets.

    The Inkling2 model's ``load_weights`` rejects any call that does not carry
    every parameter (and every fused-projection shard). That is right for the
    initial checkpoint load, but NeMo RL refits stream weights in buckets and
    SGLang calls ``load_weights`` once per bucket. Each call has already copied
    its tensors when the completeness check runs, so only the check is limited
    to the first (initial) load; per-tensor name, dtype and shape checks still
    apply to every bucket.

    Loaders that bypass ``load_weights`` (``load_format: dummy``, used when SGLang
    serves BF16 experts that only the first refit fills) call the model's
    ``post_load_weights`` hook instead, so the patch adds that hook to mark the
    initial load done.

    It also lets ``load_weights`` take routed-expert stacks in chunks named
    ``<name>@nemo_rl_experts_<start>_<stop>`` (``expert_refit_chunk_size``):
    each EP rank copies the chunks inside its expert range and skips the rest,
    so the trainer never has to gather a layer's full expert stack. No-op on
    SGLang builds without Inkling2.
    """
    spec = find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        return
    root = next(iter(spec.submodule_search_locations))
    # sgl-project/sglang#50186 adds the model at srt/models/inkling2.py; earlier
    # Inkling2 builds shipped it as tml/model.py.
    candidates = (
        os.path.join(root, "srt", "models", "inkling2.py"),
        os.path.join(root, "tml", "model.py"),
    )
    file_to_patch = next((c for c in candidates if os.path.exists(c)), None)
    if file_to_patch is None:
        return

    with open(file_to_patch, "r") as f:
        content = f.read()

    flag = "_nemo_rl_initial_load_done"
    edits = (
        (
            "# NeMo RL: refit buckets after the initial load carry a subset.",
            "        if missing:\n"
            '            raise ValueError(f"Missing checkpoint tensors: {sorted(missing)}")\n',
            "        # NeMo RL: refit buckets after the initial load carry a subset.\n"
            f'        if missing and not getattr(self, "{flag}", False):\n'
            '            raise ValueError(f"Missing checkpoint tensors: {sorted(missing)}")\n'
            f"        self.{flag} = True\n",
        ),
        (
            "# NeMo RL: loaders that bypass load_weights",
            "    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n",
            "    def post_load_weights(self):\n"
            "        # NeMo RL: loaders that bypass load_weights (load_format=dummy)\n"
            "        # initialize every parameter; later refits may arrive in buckets.\n"
            f"        self.{flag} = True\n"
            "\n"
            "    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n",
        ),
        (
            "# NeMo RL: chunked routed-expert refit",
            "    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n",
            "    def _nemo_rl_expert_chunks(self, params, weights):\n"
            "        # NeMo RL: chunked routed-expert refit. A tensor named\n"
            '        # "<name>@nemo_rl_experts_<start>_<stop>" carries experts [start, stop)\n'
            "        # of one stacked routed-expert tensor; copy it into this EP rank's\n"
            "        # slice (or skip it on other ranks) instead of requiring all experts.\n"
            "        import re\n"
            "\n"
            '        pattern = re.compile(r"(.+)@nemo_rl_experts_(\\d+)_(\\d+)")\n'
            "        ep_rank = get_parallel().moe_ep_rank\n"
            "        for checkpoint_name, weight in weights:\n"
            "            match = pattern.fullmatch(checkpoint_name)\n"
            "            if match is None:\n"
            "                yield checkpoint_name, weight\n"
            "                continue\n"
            "            source, start, stop = match[1], int(match[2]), int(match[3])\n"
            '            name = source.removeprefix("model.")\n'
            '            for leaf in ("w13_weight", "w2_weight"):\n'
            '                name = name.replace(f".mlp.{leaf}", f".mlp.experts.{leaf}")\n'
            "            param = params.get(name)\n"
            '            if param is None or ".experts." not in name or ".shared_experts." in name:\n'
            '                raise ValueError(f"Unrecognized chunked expert tensor: {checkpoint_name}")\n'
            "            if self.quant_config is not None:\n"
            "                expected_dtype = self.quant_config.tensor_dtypes[source]\n"
            "                if weight.dtype != expected_dtype:\n"
            "                    raise ValueError(\n"
            '                        f"{checkpoint_name}: expected {expected_dtype}, got {weight.dtype}"\n'
            "                    )\n"
            "            if param.dtype != weight.dtype:\n"
            "                raise ValueError(\n"
            '                    f"{checkpoint_name}: parameter dtype {param.dtype} would change checkpoint precision {weight.dtype}"\n'
            "                )\n"
            "            local = param.shape[0]\n"
            "            low, high = ep_rank * local, (ep_rank + 1) * local\n"
            "            if stop <= low or start >= high:\n"
            "                continue\n"
            "            if start < low or stop > high:\n"
            '                raise ValueError(f"{checkpoint_name}: chunk crosses expert-parallel ranks")\n'
            "            if tuple(weight.shape) != (stop - start, *param.shape[1:]):\n"
            "                raise ValueError(\n"
            '                    f"{checkpoint_name}: chunk shape {tuple(weight.shape)} does not match "\n'
            '                    f"{(stop - start, *param.shape[1:])} (expert TP must be 1)"\n'
            "                )\n"
            "            param.data[start - low : stop - low].copy_(weight)\n"
            "\n"
            "    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n",
        ),
        (
            "weights = self._nemo_rl_expert_chunks(params, weights)",
            "        params = dict(self.named_parameters())\n"
            "        loaded = set()\n"
            "        projections = (\n",
            "        params = dict(self.named_parameters())\n"
            "        weights = self._nemo_rl_expert_chunks(params, weights)\n"
            "        loaded = set()\n"
            "        projections = (\n",
        ),
    )
    patched_content = content
    applied = False
    for sentinel, anchor, replacement in edits:
        if sentinel in patched_content:
            continue
        if patched_content.count(anchor) != 1:
            raise RuntimeError(
                f"Inkling2 refit patch anchor not found exactly once in {file_to_patch}: "
                f"{anchor.strip().splitlines()[0]}"
            )
        patched_content = patched_content.replace(anchor, replacement, 1)
        applied = True
    if not applied:
        return
    compile(patched_content, file_to_patch, "exec")
    _write_and_verify(file_to_patch, patched_content, tuple(s for s, _, _ in edits))
    logger.info("Patched Inkling2 bucketed refit loading in %s.", file_to_patch)


def _override_sglang_imbalance_check_env() -> None:
    """Force-disable sglang's per-GPU memory imbalance check.

    Pop the legacy names so the shim has nothing to copy, then set
    ``ENABLE=false`` directly. Inherited env reaches the subprocesses
    cleaned, so the shim no longer overwrites our ENABLE on re-import.
    """
    for legacy in (
        "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK",
        "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK",
    ):
        os.environ.pop(legacy, None)
    os.environ["SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK"] = "false"


def _get_megatron_file(subpackage: str, relative_path: str) -> str | None:
    """Locate a file inside ``megatron.<subpackage>`` (e.g. ``core``, ``training``).

    Returns ``None`` if megatron isn't importable so callers can treat that
    as "nothing to patch". Raises if the package is present but the
    expected file is missing (signals a megatron version mismatch).
    """
    full_pkg = f"megatron.{subpackage}"
    try:
        spec = find_spec(full_pkg)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None

    base_dir = next(iter(spec.submodule_search_locations))
    file_path = os.path.join(base_dir, *relative_path.split("/"))
    if not os.path.exists(file_path):
        raise RuntimeError(
            f"Expected megatron file '{full_pkg}/{relative_path}' not found at "
            f"'{file_path}'. The megatron version may have moved this file; "
            "compat patch cannot be applied."
        )
    return file_path


def _patch_megatron_hook_mode_in(file_path: str) -> None:
    """Comment out ``torch_memory_saver.hook_mode = "torch"`` in a megatron file.

    Megatron sets ``tms.hook_mode = "torch"`` at module import time on the
    global ``torch_memory_saver`` singleton. That mutation breaks sglang's
    pauseable CUDA graph path, which asserts ``_hook_mode == "preload"``
    inside ``TorchMemorySaver.cuda_graph(...)``. Commenting the line out
    leaves the singleton at its default ``"preload"`` mode that sglang
    expects.
    """
    with open(file_path, "r") as f:
        content = f.read()

    sentinel = '# torch_memory_saver.hook_mode = "torch"'
    if sentinel in content:
        return

    anchor = '    torch_memory_saver.hook_mode = "torch"\n'
    if anchor not in content:
        raise RuntimeError(
            f"Megatron hook_mode anchor '{anchor.strip()}' not found in "
            f"{file_path}; the megatron version may have moved or removed it."
        )

    replacement = (
        '    # torch_memory_saver.hook_mode = "torch"  '
        "# patched by nemo_rl: conflicts with sglang pauseable CUDA Graph\n"
    )
    content = content.replace(anchor, replacement, 1)
    _write_and_verify(file_path, content, sentinel)
    logger.info("Patched megatron tms.hook_mode mutation in %s.", file_path)


def _patch_megatron_dynamic_context_hook_mode() -> None:
    file_path = _get_megatron_file("core", "inference/contexts/dynamic_context.py")
    if file_path is None:
        return
    _patch_megatron_hook_mode_in(file_path)


def _patch_megatron_training_hook_mode() -> None:
    file_path = _get_megatron_file("training", "training.py")
    if file_path is None:
        return
    _patch_megatron_hook_mode_in(file_path)


def _apply_sglang_compat_patches() -> None:
    # Keep while nemo_rl.transformers_compat installs any patch; the torch-FX helper outlives Transformers 5.13.
    _patch_sglang_transformers_compat_bootstrap()
    _patch_sglang_safe_unpickler()
    _patch_sglang_inkling2_bucketed_refit()
    _override_sglang_imbalance_check_env()
    _patch_megatron_dynamic_context_hook_mode()
    _patch_megatron_training_hook_mode()
