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

"""Compare vLLM vs Megatron last-token logits captured by ``vllm_forward.py``.

Captures are produced by ``vllm_forward.py`` and ``megatron_forward.py`` over a
batch of real prompts.

Prints per-prompt summary stats and writes a scatter plot of every logit
value (megatron on the x-axis, vllm on the y-axis) to ``--plot``.

Run with either env:
    uv run --extra vllm  python my_script/compare.py
    uv run --extra mcore python my_script/compare.py
"""

import argparse
import os

import torch

DEFAULT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
NEMOTRON3_NANO_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
MODEL_ALIASES = {
    "llama": DEFAULT_MODEL,
    "llama3.1-8b": DEFAULT_MODEL,
    "llama-3.1-8b": DEFAULT_MODEL,
    "nemotron3-nano": NEMOTRON3_NANO_MODEL,
    "nemotron-3-nano": NEMOTRON3_NANO_MODEL,
    "nemotron3-nano-30b-a3b": NEMOTRON3_NANO_MODEL,
}


def resolve_model_ref(value: str) -> str:
    return MODEL_ALIASES.get(value, value)


def is_nemotron3_nano_ref(value: str | None) -> bool:
    if value is None:
        return False
    value = resolve_model_ref(value)
    normalised = value.lower().replace("_", "-")
    return (
        value == NEMOTRON3_NANO_MODEL or "nvidia-nemotron-3-nano-30b-a3b" in normalised
    )


def model_output_tag(model: str | None) -> str:
    if is_nemotron3_nano_ref(model):
        return "_nemotron3_nano"
    return ""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model alias/id used to choose default capture paths. "
        "Use 'nemotron3-nano' for "
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.",
    )
    p.add_argument("--vllm", default=None)
    p.add_argument(
        "--megatron",
        default=None,
    )
    p.add_argument(
        "--batch-invariant",
        action="store_true",
        help="Use *_bi.pt files for both sides",
    )
    p.add_argument(
        "--mxfp8", action="store_true", help="Use *_mxfp8*.pt files for both sides"
    )
    p.add_argument(
        "--plot",
        default=None,
        help="Output path for the scatter PNG "
        "(default: my_script/compare_logits_scatter.png).",
    )
    p.add_argument(
        "--max-points",
        type=int,
        default=200_000,
        help="Subsample logit pairs to at most this many points "
        "in the scatter plot (default: 200000). Set to 0 for all.",
    )
    p.add_argument(
        "--compare-modules",
        action="store_true",
        help="Print debug tensor diffs when the capture payloads "
        "contain module_inputs_by_layer/module_outputs_by_layer.",
    )
    p.add_argument(
        "--log-softmax-device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device used to normalize Megatron logits before "
        "comparing against vLLM logprobs.",
    )
    p.add_argument(
        "--target-token-ids-file",
        default=None,
        help=(
            "Optional torch .pt token dump containing the target token id for "
            "each compared prompt prefix."
        ),
    )
    p.add_argument(
        "--target-token-ids-key",
        default="offline_target_token_ids",
        help=(
            "Payload key to read from --target-token-ids-file "
            "(default: offline_target_token_ids)."
        ),
    )
    args = p.parse_args()
    args.model = resolve_model_ref(args.model)
    suffix = model_output_tag(args.model)
    if args.mxfp8:
        suffix += "_mxfp8"
    if args.batch_invariant:
        suffix += "_bi"
    if args.vllm is None:
        args.vllm = os.path.join(DEFAULT_DIR, f"vllm_capture{suffix}.pt")
    if args.megatron is None:
        args.megatron = os.path.join(DEFAULT_DIR, f"megatron_capture{suffix}.pt")
    if args.plot is None:
        args.plot = os.path.join(DEFAULT_DIR, f"compare_logits_scatter{suffix}.png")
    return args


def load_target_token_ids(path, key):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        if key not in payload:
            raise KeyError(f"{path} does not contain target token key {key!r}")
        value = payload[key]
        metadata = payload.get("offline_metadata", [])
    else:
        value = payload
        metadata = []

    if isinstance(value, torch.Tensor):
        target_token_ids = [int(token_id) for token_id in value.flatten().tolist()]
    else:
        target_token_ids = [int(token_id) for token_id in value]
    return target_token_ids, metadata


def diff_stats(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    n = min(a.numel(), b.numel())
    a, b = a[:n], b[:n]
    diff = (a - b).abs()
    return {
        "n": n,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "a_norm": float(a.norm()),
        "b_norm": float(b.norm()),
        "cos_sim": float(
            torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
        ),
    }


def first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for key in ("hidden_states", "x", "input", "residual"):
            if key in value:
                found = first_tensor(value[key])
                if found is not None:
                    return found
        for item in value.values():
            found = first_tensor(item)
            if found is not None:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            found = first_tensor(item)
            if found is not None:
                return found
    return None


def get_by_selector(entry, selector):
    if "+" in selector:
        terms = []
        for part in selector.split("+"):
            term = get_by_selector(entry, part)
            if not isinstance(term, torch.Tensor):
                return None
            terms.append(term)
        result = terms[0]
        for term in terms[1:]:
            if term.shape != result.shape:
                return None
            result = result + term
        return result
    if selector == "first":
        return first_tensor(entry)
    if isinstance(entry, dict) and "args" in entry and selector.startswith("arg"):
        idx = int(selector[3:])
        args = entry.get("args", ())
        if idx < len(args):
            return first_tensor(args[idx])
        return None
    if isinstance(entry, dict) and "kwargs" in entry and selector.startswith("kw."):
        key = selector[3:]
        return first_tensor(entry.get("kwargs", {}).get(key))
    if isinstance(entry, (list, tuple)) and selector.startswith("item"):
        idx = int(selector[4:])
        if idx < len(entry):
            return first_tensor(entry[idx])
        return None
    if selector == "output":
        return first_tensor(entry)
    return None


def select_tensor(entry, selectors):
    if isinstance(entry, dict) and "__calls__" in entry:
        calls = entry["__calls__"]
        for selector in selectors:
            tensors = []
            for call in calls:
                tensor = get_by_selector(call, selector)
                if not isinstance(tensor, torch.Tensor):
                    tensors = []
                    break
                tensors.append(tensor)
            tensor = concatenate_call_tensors(tensors)
            if tensor is not None:
                return tensor, f"{selector}x{len(tensors)}"
        return None, None

    for selector in selectors:
        tensor = get_by_selector(entry, selector)
        if isinstance(tensor, torch.Tensor):
            return tensor, selector
    return None, None


def concatenate_call_tensors(tensors):
    if not tensors:
        return None

    first = tensors[0]
    if first.dim() == 0:
        return torch.stack(tensors, dim=0)
    for tensor in tensors:
        if tensor.dim() != first.dim() or tensor.shape[1:] != first.shape[1:]:
            return None
    return torch.cat(tensors, dim=0)


def normalize_token_layout(tensor, seq_lens):
    if not isinstance(tensor, torch.Tensor):
        return tensor
    if tensor.dim() < 2 or not seq_lens:
        return tensor

    total_tokens = sum(seq_lens)
    batch = len(seq_lens)
    max_seq = max(seq_lens)

    if tensor.shape[0] == total_tokens:
        return tensor
    if tensor.shape[0] == batch and tensor.shape[1] >= max_seq:
        return torch.cat(
            [tensor[i, :seq_len] for i, seq_len in enumerate(seq_lens)],
            dim=0,
        )
    if tensor.shape[1] == batch and tensor.shape[0] >= max_seq:
        return torch.cat(
            [tensor[:seq_len, i] for i, seq_len in enumerate(seq_lens)],
            dim=0,
        )
    return tensor


def compare_selected_tensor(
    label,
    v_entry,
    m_entry,
    v_selectors,
    m_selectors,
    seq_lens,
):
    v_t, v_selector = select_tensor(v_entry, v_selectors)
    m_t, m_selector = select_tensor(m_entry, m_selectors)
    if v_t is None or m_t is None:
        print(f"  {label:<68s} {'MISSING':>11s}")
        return

    v_t = normalize_token_layout(v_t, seq_lens)
    m_t = normalize_token_layout(m_t, seq_lens)
    stats = diff_stats(v_t, m_t)
    print(
        f"  {label:<68s} {stats['max_abs_diff']:>11.6f} "
        f"{stats['mean_abs_diff']:>11.4e} {stats['cos_sim']:>10.6f} "
        f"{str(tuple(v_t.shape)):>22s} {str(tuple(m_t.shape)):>22s} "
        f"{v_selector:>10s} {m_selector:>10s}"
    )


def module_inputs(payload):
    return payload.get("module_inputs_by_layer", {})


def module_outputs(payload):
    return payload.get("module_outputs_by_layer", {})


def module_input_calls(payload):
    return payload.get("module_input_calls_by_layer", {})


def module_output_calls(payload):
    return payload.get("module_output_calls_by_layer", {})


def layer_map(source, layer_idx):
    layer = source.get(layer_idx)
    if layer is None:
        layer = source.get(str(layer_idx), {})
    return layer or {}


def get_layer_entry(payload, layer_idx, module_name, outputs=False):
    call_source = (
        module_output_calls(payload) if outputs else module_input_calls(payload)
    )
    calls = layer_map(call_source, layer_idx).get(module_name)
    if calls:
        return {"__calls__": calls}

    source = module_outputs(payload) if outputs else module_inputs(payload)
    return layer_map(source, layer_idx).get(module_name, {})


def captured_module_layers(payload):
    layers = payload.get("captured_module_layers")
    if isinstance(layers, torch.Tensor):
        layers = layers.flatten().tolist()
    if isinstance(layers, (list, tuple)):
        return sorted({int(layer_idx) for layer_idx in layers})

    # Older captures only fully hooked layer 0. Other layers usually contain
    # only the synthetic <layer> entrypoint hook.
    detail_layers = []
    for key, layer_entries in module_inputs(payload).items():
        if (
            isinstance(layer_entries, dict)
            and len(set(layer_entries) - {"<layer>"}) > 0
        ):
            detail_layers.append(int(key))
    return sorted(set(detail_layers or [0]))


def print_common_exact_name_report(v, m, seq_lens, layer_indices):
    print("\n" + "=" * 110)
    print("Common exact-name input diffs for captured module-detail layers")
    print("=" * 110)
    v_inputs = module_inputs(v)
    m_inputs = module_inputs(m)
    for layer_idx in layer_indices:
        v_layer = v_inputs.get(layer_idx) or v_inputs.get(str(layer_idx)) or {}
        m_layer = m_inputs.get(layer_idx) or m_inputs.get(str(layer_idx)) or {}
        common = sorted(set(v_layer) & set(m_layer))
        if not common:
            print(f"\n  layer {layer_idx}: no common captured module names")
            continue
        print(f"\n  layer {layer_idx}:")
        for name in common:
            compare_selected_tensor(
                name,
                get_layer_entry(v, layer_idx, name),
                get_layer_entry(m, layer_idx, name),
                (
                    "kw.hidden_states+kw.residual",
                    "arg1+arg2",
                    "kw.hidden_states",
                    "arg1",
                    "arg0",
                    "first",
                ),
                ("arg0", "kw.hidden_states", "first"),
                seq_lens,
            )
        print(f"    vLLM-only names: {sorted(set(v_layer) - set(m_layer))}")
        print(f"    Megatron-only names: {sorted(set(m_layer) - set(v_layer))}")


def layer_entry_stream(payload, layer_idx, seq_lens, is_vllm):
    entry = get_layer_entry(payload, layer_idx, "<layer>")
    if is_vllm:
        selectors = (
            "kw.hidden_states+kw.residual",
            "arg1+arg2",
            "kw.hidden_states",
            "arg1",
            "arg0",
            "first",
        )
    else:
        selectors = ("kw.hidden_states", "arg0", "first")
    tensor, selector = select_tensor(entry, selectors)
    if tensor is None:
        return None, selector
    return normalize_token_layout(tensor, seq_lens), selector


def last_token_offsets(seq_lens):
    offsets = []
    total = 0
    for seq_len in seq_lens:
        total += seq_len
        offsets.append(total - 1)
    return torch.tensor(offsets, dtype=torch.long)


def print_layer_entry_last_token_report(v, m, seq_lens, num_layers):
    if not seq_lens:
        return

    offsets = last_token_offsets(seq_lens)
    print("\n" + "=" * 110)
    print("Layer-entry last-token residual-stream diffs")
    print("=" * 110)
    print(f"  {'layer':>5s} {'top prompt max/mean hidden diffs':<90s}")
    for layer_idx in range(num_layers):
        v_t, _ = layer_entry_stream(v, layer_idx, seq_lens, is_vllm=True)
        m_t, _ = layer_entry_stream(m, layer_idx, seq_lens, is_vllm=False)
        if v_t is None or m_t is None:
            print(f"  {layer_idx:>5d} MISSING")
            continue
        if v_t.shape != m_t.shape or v_t.size(0) <= int(offsets[-1]):
            print(
                f"  {layer_idx:>5d} shape mismatch v={tuple(v_t.shape)} "
                f"m={tuple(m_t.shape)}"
            )
            continue

        diff = (v_t[offsets] - m_t[offsets]).abs().float()
        max_by_prompt = diff.max(dim=1).values
        mean_by_prompt = diff.mean(dim=1)
        top = torch.topk(max_by_prompt, k=min(5, len(seq_lens)))
        parts = [
            f"p{int(prompt_idx)}:{float(max_by_prompt[prompt_idx]):.6f}/"
            f"{float(mean_by_prompt[prompt_idx]):.2e}"
            for prompt_idx in top.indices
        ]
        print(f"  {layer_idx:>5d} {' '.join(parts):<90s}")


def print_module_capture_report(v, m):
    v_inputs = module_inputs(v)
    m_inputs = module_inputs(m)
    if not v_inputs or not m_inputs:
        return
    v_outputs = module_outputs(v)
    m_outputs = module_outputs(m)

    seq_lens = v.get("seq_lens") or m.get("seq_lens") or []
    num_layers = min(
        int(v.get("num_layers", len(v_inputs))),
        int(m.get("num_layers", len(m_inputs))),
    )

    print("\n" + "=" * 110)
    print("Layer-entry hidden-state input diffs")
    print("=" * 110)
    print(
        f"  {'layer':>5s} {'max_abs':>11s} {'mean_abs':>11s} {'cos_sim':>10s} "
        f"{'v_shape':>22s} {'m_shape':>22s} {'v_sel':>10s} {'m_sel':>10s}"
    )
    for layer_idx in range(num_layers):
        v_entry = get_layer_entry(v, layer_idx, "<layer>")
        m_entry = get_layer_entry(m, layer_idx, "<layer>")
        compare_selected_tensor(
            str(layer_idx),
            v_entry,
            m_entry,
            (
                "kw.hidden_states+kw.residual",
                "arg1+arg2",
                "kw.hidden_states",
                "arg1",
                "arg0",
                "first",
            ),
            ("kw.hidden_states", "arg0", "first"),
            seq_lens,
        )

    module_layers = sorted(
        set(captured_module_layers(v)) & set(captured_module_layers(m))
    )
    if not module_layers:
        module_layers = [0]

    print("\n" + "=" * 110)
    print("Semantic module input diffs for captured module-detail layers")
    print("=" * 110)
    print(
        f"  {'pair':<68s} {'max_abs':>11s} {'mean_abs':>11s} {'cos_sim':>10s} "
        f"{'v_shape':>22s} {'m_shape':>22s} {'v_sel':>10s} {'m_sel':>10s}"
    )
    pairs = [
        (
            "layer residual stream",
            "<layer>",
            "<layer>",
            ("kw.hidden_states+kw.residual", "arg1+arg2", "kw.hidden_states", "arg1"),
            ("kw.hidden_states", "arg0"),
        ),
        (
            "input_layernorm input",
            "input_layernorm",
            "input_layernorm",
            ("arg0", "kw.hidden_states", "first"),
            ("arg0", "kw.hidden_states", "first"),
        ),
        (
            "post-norm hidden -> qkv",
            "self_attn.qkv_proj",
            "self_attention.linear_qkv.linear",
            ("arg0", "first"),
            ("arg0", "first"),
        ),
        (
            "SDPA q",
            "self_attn.attn",
            "self_attention.core_attention",
            ("arg0", "first"),
            ("arg0", "first"),
        ),
        (
            "SDPA k",
            "self_attn.attn",
            "self_attention.core_attention",
            ("arg1",),
            ("arg1",),
        ),
        (
            "SDPA v",
            "self_attn.attn",
            "self_attention.core_attention",
            ("arg2",),
            ("arg2",),
        ),
        (
            "attention output -> o_proj",
            "self_attn.o_proj",
            "self_attention.linear_proj",
            ("arg0", "first"),
            ("arg0", "first"),
        ),
        (
            "post-norm hidden -> fc1",
            "mlp.gate_up_proj",
            "mlp.linear_fc1.linear",
            ("arg0", "first"),
            ("arg0", "first"),
        ),
        (
            "post-activation -> down/fc2",
            "mlp.down_proj",
            "mlp.linear_fc2",
            ("arg0", "first"),
            ("arg0", "first"),
        ),
    ]
    for layer_idx in module_layers:
        print(f"\n  layer {layer_idx}:")
        for label, v_name, m_name, v_selectors, m_selectors in pairs:
            v_entry = get_layer_entry(v, layer_idx, v_name)
            m_entry = get_layer_entry(m, layer_idx, m_name)
            compare_selected_tensor(
                label, v_entry, m_entry, v_selectors, m_selectors, seq_lens
            )

    if v_outputs and m_outputs:
        print("\n" + "=" * 110)
        print("Semantic module output diffs for captured module-detail layers")
        print("=" * 110)
        print(
            f"  {'pair':<68s} {'max_abs':>11s} {'mean_abs':>11s} "
            f"{'cos_sim':>10s} {'v_shape':>22s} {'m_shape':>22s} "
            f"{'v_sel':>10s} {'m_sel':>10s}"
        )
        output_pairs = [
            ("layer residual stream output", "<layer>", "<layer>"),
            ("input_layernorm output", "input_layernorm", "input_layernorm"),
            (
                "qkv projection output",
                "self_attn.qkv_proj",
                "self_attention.linear_qkv.linear",
            ),
            (
                "attention kernel output",
                "self_attn.attn",
                "self_attention.core_attention",
            ),
            (
                "attention projection output",
                "self_attn.o_proj",
                "self_attention.linear_proj",
            ),
            ("mlp fc1/gate_up output", "mlp.gate_up_proj", "mlp.linear_fc1.linear"),
            ("mlp down/fc2 output", "mlp.down_proj", "mlp.linear_fc2"),
        ]
        for layer_idx in module_layers:
            print(f"\n  layer {layer_idx}:")
            for label, v_name, m_name in output_pairs:
                v_entry = get_layer_entry(v, layer_idx, v_name, outputs=True)
                m_entry = get_layer_entry(m, layer_idx, m_name, outputs=True)
                v_selectors = ("item0+item1", "output", "first")
                m_selectors = ("item0+item1", "output", "first")
                compare_selected_tensor(
                    label, v_entry, m_entry, v_selectors, m_selectors, seq_lens
                )

    print_common_exact_name_report(v, m, seq_lens, module_layers)
    print_layer_entry_last_token_report(v, m, seq_lens, num_layers)


def plot_scatter(v_logits, m_logits, out_path, max_points):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v_flat = v_logits.reshape(-1).numpy()
    m_flat = m_logits.reshape(-1).numpy()
    assert v_flat.shape == m_flat.shape, (
        f"shape mismatch: vllm {v_flat.shape} vs megatron {m_flat.shape}"
    )

    # Compute similarity stats on the full tensors before any sub-sampling.
    full_stats = diff_stats(v_logits, m_logits)
    v_t = v_logits.reshape(-1).float()
    m_t = m_logits.reshape(-1).float()
    pearson = float(torch.corrcoef(torch.stack([v_t, m_t]))[0, 1].item())

    n_total = v_flat.size
    if max_points and n_total > max_points:
        rng = torch.Generator().manual_seed(0)
        sel = torch.randperm(n_total, generator=rng)[:max_points].numpy()
        v_pts = v_flat[sel]
        m_pts = m_flat[sel]
        sub_note = f" (sub-sampled {max_points}/{n_total})"
    else:
        v_pts = v_flat
        m_pts = m_flat
        sub_note = f" ({n_total} pts)"

    lo = float(min(v_pts.min(), m_pts.min()))
    hi = float(max(v_pts.max(), m_pts.max()))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(m_pts, v_pts, s=1, alpha=0.2, rasterized=True)
    ax.plot([lo, hi], [lo, hi], color="red", linewidth=0.5, label="y = x")
    ax.set_xlabel("Megatron next-token logprobs (log_softmax of logits)")
    ax.set_ylabel("vLLM generation logprobs")
    ax.set_title(f"Next-token logprobs: vLLM vs Megatron{sub_note}")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left")

    sim_text = (
        f"cos_sim   = {full_stats['cos_sim']:.6f}\n"
        f"pearson r = {pearson:.6f}\n"
        f"max |Δ|   = {full_stats['max_abs_diff']:.4g}\n"
        f"mean |Δ|  = {full_stats['mean_abs_diff']:.4g}\n"
        f"|vllm|    = {full_stats['a_norm']:.4g}\n"
        f"|mcore|   = {full_stats['b_norm']:.4g}"
    )
    ax.text(
        0.98,
        0.02,
        sim_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="gray"
        ),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    v = torch.load(args.vllm, map_location="cpu", weights_only=False)
    m = torch.load(args.megatron, map_location="cpu", weights_only=False)

    v_logprobs = v.get("next_token_logprobs")
    m_logits = m.get("last_token_logits")
    if v_logprobs is None or m_logits is None:
        raise RuntimeError(
            "expected 'next_token_logprobs' in vllm capture and "
            "'last_token_logits' in megatron capture; re-run both scripts."
        )
    v_logprobs = v_logprobs.float()
    m_logits = m_logits.float()

    # Megatron's logits dim may pad past vLLM's tokenizer length (e.g.,
    # 128256 vs 128256 — usually equal, but truncate to the common range
    # just in case).
    v_vocab = v_logprobs.shape[-1]
    m_vocab = m_logits.shape[-1]
    vocab = min(v_vocab, m_vocab)
    if v_vocab != m_vocab:
        print(
            f"[warn] vocab mismatch: vllm={v_vocab} megatron={m_vocab}; "
            f"truncating to {vocab}"
        )
    v_logprobs = v_logprobs[..., :vocab]
    m_logits = m_logits[..., :vocab]

    # Convert Megatron's raw logits to logprobs so both sides are directly
    # comparable. vLLM normalizes on GPU, so CUDA is useful for separating
    # transformer-body drift from log-softmax implementation differences.
    if args.log_softmax_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--log-softmax-device cuda requires CUDA")
        m_logprobs = torch.log_softmax(m_logits.cuda(), dim=-1).cpu()
    else:
        m_logprobs = torch.log_softmax(m_logits, dim=-1)

    v_prompts = v.get("prompts", [])
    m_prompts = m.get("prompts", [])

    print("=" * 70)
    print(f"vllm     capture : {args.vllm}")
    print(f"megatron capture : {args.megatron}")
    print(f"num prompts      : vllm={len(v_prompts)} megatron={len(m_prompts)}")
    print(
        f"logprob shape    : vllm={tuple(v_logprobs.shape)} "
        f"megatron(logits)={tuple(m_logits.shape)}"
    )
    print(f"log_softmax dev  : {args.log_softmax_device}")
    print(
        f"model family     : vllm={v.get('model_family', '?')} "
        f"megatron={m.get('model_family', '?')}"
    )
    print(
        f"module layers    : vllm={v.get('captured_module_layers', '?')} "
        f"megatron={m.get('captured_module_layers', '?')}"
    )

    n = min(v_logprobs.shape[0], m_logprobs.shape[0])
    v_logprobs = v_logprobs[:n]
    m_logprobs = m_logprobs[:n]

    if v_prompts != m_prompts:
        print(
            "[warn] prompt lists differ between captures — comparing by index "
            "anyway, but results may be meaningless"
        )

    print("\nPer-prompt next-token logprobs:")
    print(
        f"  {'idx':>3} | {'max_abs':>10} | {'mean_abs':>10} | {'cos_sim':>9} | "
        f"{'|vllm|':>10} | {'|mcore|':>10}"
    )
    for i in range(n):
        stats = diff_stats(v_logprobs[i], m_logprobs[i])
        print(
            f"  {i:>3} | {stats['max_abs_diff']:>10.6f} | "
            f"{stats['mean_abs_diff']:>10.4e} | {stats['cos_sim']:>9.6f} | "
            f"{stats['a_norm']:>10.4f} | {stats['b_norm']:>10.4f}"
        )

    print("\nAggregate (all prompts, all vocab):")
    for k, val in diff_stats(v_logprobs, m_logprobs).items():
        print(f"  {k}: {val}")

    if args.target_token_ids_file:
        target_token_ids, target_metadata = load_target_token_ids(
            args.target_token_ids_file,
            args.target_token_ids_key,
        )
        target_count = min(n, len(target_token_ids))
        if target_count < n:
            print(
                "[warn] fewer target token ids than compared rows: "
                f"targets={len(target_token_ids)} rows={n}; using {target_count}"
            )
        selected_rows = torch.arange(target_count)
        selected_targets = torch.tensor(
            target_token_ids[:target_count], dtype=torch.long
        )
        v_selected = v_logprobs[selected_rows, selected_targets]
        m_selected = m_logprobs[selected_rows, selected_targets]
        selected_diff = m_selected - v_selected
        selected_abs = selected_diff.abs()
        selected_rel = selected_abs / torch.maximum(
            torch.maximum(v_selected.abs(), m_selected.abs()),
            torch.full_like(selected_abs, 1e-12),
        )

        print("\nSelected target-token logprobs:")
        print(f"  target file: {args.target_token_ids_file}")
        print(f"  n: {target_count}")
        print(f"  mean_abs_diff: {float(selected_abs.mean()):.8e}")
        print(f"  max_abs_diff: {float(selected_abs.max()):.8e}")
        print(f"  mean_rel_diff: {float(selected_rel.mean()):.8e}")
        print(f"  max_rel_diff: {float(selected_rel.max()):.8e}")
        print(f"  mean_signed_diff: {float(selected_diff.mean()):.8e}")
        print("  worst selected-token diffs:")
        worst_count = min(10, target_count)
        top_abs, top_indices = torch.topk(selected_abs, worst_count)
        for rank, (abs_value, index) in enumerate(
            zip(top_abs.tolist(), top_indices.tolist()),
            start=1,
        ):
            meta = (
                target_metadata[index]
                if isinstance(target_metadata, list) and index < len(target_metadata)
                else {}
            )
            print(
                "    "
                f"#{rank} row={index} "
                f"sample={meta.get('sample_idx', '?')} "
                f"pos={meta.get('position', '?')} "
                f"token={int(selected_targets[index].item())} "
                f"vllm={float(v_selected[index]):.8e} "
                f"megatron={float(m_selected[index]):.8e} "
                f"abs={float(abs_value):.8e} "
                f"rel={float(selected_rel[index]):.8e}"
            )

    plot_scatter(v_logprobs, m_logprobs, args.plot, args.max_points)
    print(f"\nscatter plot -> {args.plot}")

    if args.compare_modules or (
        "module_inputs_by_layer" in v and "module_inputs_by_layer" in m
    ):
        print_module_capture_report(v, m)


if __name__ == "__main__":
    main()
