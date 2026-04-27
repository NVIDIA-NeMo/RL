# !!! Note: I haven't gotten the HF converted weights to train correctly,
# !!! so use the InternVL native checkpoints instead.

import json
import os
import re
import shutil
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from safetensors.torch import save_file, safe_open
from torchvision.transforms.functional import InterpolationMode
from transformers import (
    AutoModel,
    AutoTokenizer,
    InternVLForConditionalGeneration,
    AutoProcessor,
)

### CONFIG ###
WEIGHT_INPUT_DIR = (
    "/lustre/fs1/portfolios/llmservice/users/jseppanen/checkpoints/InternVL3_5-4B-MPO"
)
CONFIG_INPUT_DIR = (
    "/lustre/fs1/portfolios/llmservice/users/jseppanen/checkpoints/InternVL3_5-4B-HF"
)
OUTPUT_DIR = "/lustre/fs1/portfolios/llmservice/users/jseppanen/checkpoints/InternVL3_5-4B-MPO-HF"

ORIGINAL_TO_CONVERTED_KEY_MAPPING_VISION = {
    r"vision_model": r"model.vision_tower",
    r"layers": r"layer",
    r"class_embedding": r"cls_token",
    r"position_embedding": r"position_embeddings",
    r"patch_embedding": r"patch_embeddings.projection",
    r"ls(\d+)": r"lambda_\1",
    r"attn.proj": r"attention.projection_layer",
    r"attn.dropout": r"attention.projection_dropout",
    r"attn": r"attention",
    r"norm1": r"layernorm_before",
    r"norm2": r"layernorm_after",
}

ORIGINAL_TO_CONVERTED_KEY_MAPPING_TEXT_QWEN2 = {
    r"language_model.model.": r"model.language_model.",
    r"language_model.lm_head": r"lm_head",
}

ORIGINAL_TO_CONVERTED_KEY_MAPPING_MULTI = {
    r"mlp1.0": r"model.multi_modal_projector.layer_norm",
    r"mlp1.1": r"model.multi_modal_projector.linear_1",
    r"mlp1.3": r"model.multi_modal_projector.linear_2",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def read_index(path: str) -> dict:
    """Load a safetensors index JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    """Create directory if missing."""
    Path(path).mkdir(parents=True, exist_ok=True)


def split_qkv(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a concatenated QKV tensor into q,k,v along dim 0."""
    size = t.shape[0] // 3
    q = t[:size]
    k = t[size : 2 * size]
    v = t[2 * size :]
    return q, k, v


def convert_old_keys_to_new_keys(state_dict_keys: list[str]) -> dict[str, str]:
    """Create a mapping from original keys to converted keys using regex mappings."""
    output: dict[str, str] = {}
    if not state_dict_keys:
        return output
    old_text_vision = "\n".join(
        [k for k in state_dict_keys if k.startswith("vision_model")]
    )
    if old_text_vision:
        new_text = old_text_vision
        for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING_VISION.items():
            new_text = re.sub(pattern, replacement, new_text)
        output.update(dict(zip(old_text_vision.split("\n"), new_text.split("\n"))))
    old_text_language = "\n".join(
        [k for k in state_dict_keys if k.startswith("language_model")]
    )
    if old_text_language:
        new_text = old_text_language
        for (
            pattern,
            replacement,
        ) in ORIGINAL_TO_CONVERTED_KEY_MAPPING_TEXT_QWEN2.items():
            new_text = re.sub(pattern, replacement, new_text)
        output.update(dict(zip(old_text_language.split("\n"), new_text.split("\n"))))
    old_text_multi = "\n".join(
        [
            k
            for k in state_dict_keys
            if not (k.startswith("vision_model") or k.startswith("language_model"))
        ]
    )
    if old_text_multi:
        new_text = old_text_multi
        for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING_MULTI.items():
            new_text = re.sub(pattern, replacement, new_text)
        output.update(dict(zip(old_text_multi.split("\n"), new_text.split("\n"))))
    return output


def copy_configs(src_dir: str, dst_dir: str) -> None:
    """Copy non-weight config files to destination."""
    for name in os.listdir(src_dir):
        src_path = os.path.join(src_dir, name)
        if not os.path.isfile(src_path):
            continue
        if name == "model.safetensors.index.json":
            continue
        if Path(name).suffix in {".json", ".jinja", ".txt"}:
            shutil.copy2(src_path, os.path.join(dst_dir, name))


def write_index(dst_dir: str, weight_map: dict[str, str]) -> None:
    """Write HF-style index JSON."""
    meta = {
        "total_size": int(
            sum(
                os.path.getsize(os.path.join(dst_dir, f))
                for f in set(weight_map.values())
            )
        )
    }
    try:
        total_params = 0
        seen = set()
        for fname, group in {}.items():
            pass
        total_params = None
    except Exception:
        total_params = None
    index = {
        "metadata": meta
        if total_params is None
        else {"total_parameters": total_params, **meta},
        "weight_map": weight_map,
    }
    with open(os.path.join(dst_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)


def convert(weights_dir: str, config_src_dir: str, dst_dir: str) -> None:
    """Run MPO to HF conversion using regex key mappings."""
    ensure_dir(dst_dir)
    copy_configs(config_src_dir, dst_dir)
    index = read_index(os.path.join(weights_dir, "model.safetensors.index.json"))
    shard_files = sorted(set(index["weight_map"].values()))
    new_weight_map: dict[str, str] = {}
    for shard in shard_files:
        src_path = os.path.join(weights_dir, shard)
        dst_path = os.path.join(dst_dir, shard)
        out_tensors: dict[str, torch.Tensor] = {}
        with safe_open(src_path, framework="pt") as f:
            keys = list(f.keys())
            key_map = convert_old_keys_to_new_keys(keys)
            for key in keys:
                if key not in key_map:
                    continue
                new_key = key_map[key]
                t = f.get_tensor(key)
                if "attn.qkv" in key and (
                    key.endswith(".weight") or key.endswith(".bias")
                ):
                    q, k, v = split_qkv(t)
                    base_q = new_key.replace("attention.qkv", "attention.q_proj")
                    base_k = new_key.replace("attention.qkv", "attention.k_proj")
                    base_v = new_key.replace("attention.qkv", "attention.v_proj")
                    out_tensors[base_q] = q
                    out_tensors[base_k] = k
                    out_tensors[base_v] = v
                    new_weight_map[base_q] = shard
                    new_weight_map[base_k] = shard
                    new_weight_map[base_v] = shard
                else:
                    out_tensors[new_key] = t
                    new_weight_map[new_key] = shard
        if out_tensors:
            save_file(out_tensors, dst_path)
    write_index(dst_dir, new_weight_map)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def hf_inference(model_path, image_path, prompt):
    print(f"Loading {model_path}...")
    model = InternVLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)
    print(f"Running {model_path}...")
    output_ids = model.generate(**inputs, max_new_tokens=1024)
    response = processor.batch_decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    response = response.split("assistant\n")[1]
    print(response)
    return response


def internvl_inference(model_path, image_path, prompt):
    print(f"Loading {model_path}...")
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=1024)
    print(f"Running {model_path}...")
    response = model.chat(
        tokenizer, pixel_values, "<image>\n" + prompt, generation_config
    )
    print(response)
    return response


if __name__ == "__main__":
    convert(WEIGHT_INPUT_DIR, CONFIG_INPUT_DIR, OUTPUT_DIR)
    image_path = f"{CONFIG_INPUT_DIR}/examples/image1.jpg"
    prompt = "Please describe the image shortly."
    output = hf_inference(OUTPUT_DIR, image_path, prompt)
    expected = internvl_inference(WEIGHT_INPUT_DIR, image_path, prompt)
    assert output == expected
