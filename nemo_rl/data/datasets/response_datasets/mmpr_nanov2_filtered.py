import json
from typing import Any, Optional

from datasets import Dataset, Features, Sequence, Value

from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.data.interfaces import TaskDataSpec


PYTHON_LIST_VERIFIER = """
mmpr-1.2-inat_train2018_merge_en_20240811_sr0.50_wo_image
""".strip().split()

MATH_VERIFIER = """
mmpr-1.2-CLEVR_math_en_20240402_extracted_pairs_vqa_correctness_rules
mmpr-1.2-CLEVR_math_en_20240402_extracted_prefix_pair_sr0.0_with_image
mmpr-1.2-CLEVR_math_en_20240402_extracted_prefix_pair_sr0.5_with_image
mmpr-1.2-CLEVR_math_en_20240402_extracted_prefix_pair_sr0.5_wo_image
mmpr-1.2-dvqa_en_20240402_extracted_int_only_pairs_vqa_correctness_rules
mmpr-1.2-dvqa_en_20240402_extracted_int_only_pairs_vqa_format_rules
mmpr-1.2-geo170k_extracted_full_pairs_vqa_correctness_rules
mmpr-1.2-geo170k_extracted_full_pairs_vqa_format_rules
mmpr-1.2-geo170k_extracted_pairs_vqa_correctness_rules
mmpr-1.2-geo170k_extracted_pairs_vqa_format_rules
mmpr-1.2-geometry3k_en_20240402_extracted_open_ended_only_pairs_vqa_correctness_rules
mmpr-1.2-geometry3k_en_20240402_extracted_open_ended_only_pairs_vqa_format_rules
mmpr-1.2-geometry3k_en_20240402_extracted_pairs_vqa_correctness_rules
mmpr-1.2-geometry3k_en_20240402_extracted_pairs_vqa_format_rules
mmpr-1.2-geomverse_extracted_pairs_vqa_correctness_rules
mmpr-1.2-geomverse_extracted_pairs_vqa_format_rules
mmpr-1.2-geoqa+_en_20240402_extracted_open_ended_only_pairs_vqa_correctness_rules
mmpr-1.2-geoqa+_en_20240402_extracted_open_ended_only_pairs_vqa_format_rules
mmpr-1.2-geoqa+_extracted_en_version_pairs_vqa_correctness_rules
mmpr-1.2-geoqa+_extracted_en_version_pairs_vqa_format_rules
mmpr-1.2-geos_en_20240402_extracted_open_ended_only_pairs_vqa_correctness_rules
mmpr-1.2-geos_en_20240402_extracted_open_ended_only_pairs_vqa_format_rules
mmpr-1.2-geos_en_20240402_extracted_pairs_vqa_correctness_rules
mmpr-1.2-geos_en_20240402_extracted_pairs_vqa_format_rules
mmpr-1.2-infographics_20240403_qa_20240407_v2_extracted_pairs_vqa_correctness_rules
mmpr-1.2-MathV360K_prompts_pairs_vqa_correctness_rules
mmpr-1.2-MathV360K_prompts_pairs_vqa_format_rules
mmpr-1.2-mavis_function_abs_pairs_vqa_correctness_rules
mmpr-1.2-mavis_function_abs_pairs_vqa_direct_rules
mmpr-1.2-mavis_function_abs_pairs_vqa_format_rules
mmpr-1.2-mavis_function_cos_pairs_vqa_correctness_rules
mmpr-1.2-mavis_function_cos_pairs_vqa_direct_rules
mmpr-1.2-mavis_function_cos_pairs_vqa_format_rules
mmpr-1.2-mavis_function_log_pairs_vqa_correctness_rules
mmpr-1.2-mavis_function_log_pairs_vqa_direct_rules
mmpr-1.2-mavis_function_log_pairs_vqa_format_rules
mmpr-1.2-mavis_function_poly_pairs_vqa_correctness_rules
mmpr-1.2-mavis_function_poly_pairs_vqa_direct_rules
mmpr-1.2-mavis_function_poly_pairs_vqa_format_rules
mmpr-1.2-mavis_function_sin_pairs_vqa_correctness_rules
mmpr-1.2-mavis_function_sin_pairs_vqa_direct_rules
mmpr-1.2-mavis_function_sin_pairs_vqa_format_rules
mmpr-1.2-mavis_function_tan_pairs_vqa_correctness_rules
mmpr-1.2-mavis_function_tan_pairs_vqa_direct_rules
mmpr-1.2-mavis_function_tan_pairs_vqa_format_rules
mmpr-1.2-super_clevr_en_20240402_int_pairs_vqa_correctness_rules
mmpr-1.2-super_clevr_en_20240402_int_pairs_vqa_format_rules
mmpr-1.2-tallyqa_vg_en_20240816_cot_pairs_vqa_correctness_rules
mmpr-1.2-unigeo_calc_en_20240402_extracted_open_ended_only_pairs_vqa_correctness_rules
mmpr-1.2-unigeo_calc_en_20240402_extracted_open_ended_only_pairs_vqa_format_rules
mmpr-1.2-vqav2_en_20240402_int_pairs_vqa_correctness_rules
mmpr-1.2-vqav2_en_20240402_int_pairs_vqa_format_rules
""".strip().split()

MULTIPLE_CHOICE_VERIFIER = """
mmpr-1.2-geo170k_extracted_pairs_vqa_correctness_rules
mmpr-1.2-koniq10k_en_20240403_pairs_vqa_correctness_rules
mmpr-1.2-koniq10k_en_20240403_pairs_vqa_format_rules
mmpr-1.2-scienceqa_multi_choice_en_20240402_extracted_pairs_vqa_correctness_rules
mmpr-1.2-scienceqa_multi_choice_en_20240402_extracted_pairs_vqa_format_rules
mmpr-1.2-nlvr2_en_20240910_ov_pairs_vqa_correctness_rules
mmpr-1.2-nlvr2_en_20240910_ov_pairs_vqa_format_rules
mmpr-1.2-m3cot_train_extracted_pairs_vqa_direct_rules
""".strip().split()


class MmprNanov2FilteredDataset(RawDataset):
    def __init__(
        self,
        train_data_path: Optional[str] = None,
        prompt_file: Optional[str] = None,
        val_size: int = 500,
    ):
        self.task_name = "mmpr_nanov2_filtered"
        if not train_data_path:
            raise ValueError("MmprNanov2FilteredDataset requires a JSONL path")
        full_dataset = self._load_jsonl(train_data_path)
        if val_size > 0:
            # take last (most difficult) samples
            cutoff = len(full_dataset) - val_size
            val_dataset = full_dataset.select(range(cutoff, len(full_dataset)))
            train_dataset = full_dataset.select(range(cutoff))
        else:
            train_dataset = full_dataset
            val_dataset = None
        self.formatted_ds = {
            "train": train_dataset,
            "validation": val_dataset,
        }
        self.task_spec = TaskDataSpec(task_name="mmpr_nanov2_filtered", prompt_file=prompt_file)

    def _load_jsonl(self, path: str) -> Dataset:
        """Load a JSONL with image path and conversations into a Dataset."""
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                images = row.get("images", [])
                if not images:
                    # mmpr has also text-only samples, but nemo-rl VLM code path
                    # doesn't support mixed text and image samples
                    images.append("__noimage__")

                question = row["question"].replace("<image>", "").strip()
                verifier = get_verifier(row)
                if verifier != "gui-coordinate":
                    question = unify_answer_format(question)
                rows.append(
                    {
                        "images": images,
                        "question": question,
                        "answer": row["answer"],
                        "verifier": verifier,
                        "task_name": self.task_name,
                    }
                )
        features = Features(
            {
                "images": Sequence(Value("string")),
                "question": Value("string"),
                "answer": Value("string"),
                "verifier": Value("string"),
                "task_name": Value("string"),
            }
        )
        return Dataset.from_list(rows, features=features)


def format_mmpr_nanov2_filtered_dataset(example: dict[str, Any]) -> dict[str, Any]:
    """Format MmprNanov2FilteredDataset into an OpenAI-API-like message log."""
    user_content = []
    for image in example["images"]:
        user_content.append({"type": "image", "image": image})
    user_content.append({
        "type": "text",
        "text": example["question"].replace("<image>", "").strip(),
    })

    assistant_content = f"{example['verifier']}:{example['answer']}"

    ret = {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "task_name": "mmpr_nanov2_filtered",
    }
    return ret


def unify_answer_format(question: str) -> str:
    # FIXME(jseppanen): hacky heuristics to unify answer format in MMPR
    if "\\boxed{" in question:
        return question
    elif "\"Final answer: ..\"" in question:
        return question.replace("\"Final answer: ..\"", "\"\\boxed{...}\"")
    else:
        return question + "\nPlease put the final answer within \\boxed{...}."


def get_verifier(sample) -> str:
    if sample.get("verifier"):
        return sample["verifier"]
    if sample["dataset"] in MATH_VERIFIER:
        return "mathruler"
    elif sample["dataset"] in MULTIPLE_CHOICE_VERIFIER:
        return "multiple-choice"
    elif sample["dataset"] in PYTHON_LIST_VERIFIER:
        return "python-list"
    else:
        return "string-match"
