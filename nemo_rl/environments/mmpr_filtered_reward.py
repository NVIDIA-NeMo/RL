import ast
import logging
import re
import warnings

import numpy as np
from mathruler.grader import grade_answer


# mmpr-1.2-ai2d_train_12k_en_20240410_extracted_pairs_vqa_correctness_rules
# mmpr-1.2-ai2d_train_12k_en_20240410_extracted_pairs_vqa_format_rules
# mmpr-1.2-ai2d_train_12k_en_20240410_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-ai2d_train_12k_en_20240410_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-ai2d_train_12k_en_20240410_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-chartqa_trainval_30k_w_csv_en_20240402_extracted_pairs_vqa_correctness_rules
# mmpr-1.2-chartqa_trainval_30k_w_csv_en_20240402_extracted_pairs_vqa_format_rules
# mmpr-1.2-chartqa_trainval_30k_w_csv_en_20240402_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-chartqa_trainval_30k_w_csv_en_20240402_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-chartqa_trainval_30k_w_csv_en_20240402_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-CLEVR_math_en_20240402_extracted_pairs_vqa_correctness_rules
# mmpr-1.2-CLEVR_math_en_20240402_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-CLEVR_math_en_20240402_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-CLEVR_math_en_20240402_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-cocorem_exist_yorn_en_20241016_pairs_vqa_correctness_rules
# mmpr-1.2-cocorem_exist_yorn_en_20241016_pairs_vqa_format_rules
# mmpr-1.2-docvqa_train_56k_en_20240402_extracted_pairs_vqa_correctness_rules
# mmpr-1.2-docvqa_train_56k_en_20240402_extracted_pairs_vqa_format_rules
# mmpr-1.2-docvqa_train_56k_en_20240402_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-docvqa_train_56k_en_20240402_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-docvqa_train_56k_en_20240402_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-dvqa_en_20240402_extracted_int_only_pairs_vqa_correctness_rules
# mmpr-1.2-dvqa_en_20240402_extracted_int_only_pairs_vqa_format_rules
# mmpr-1.2-figureqa_en_20240402_extracted_pairs_vqa_correctness_rules
# mmpr-1.2-figureqa_en_20240402_extracted_pairs_vqa_format_rules
# mmpr-1.2-figureqa_en_20240402_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-figureqa_en_20240402_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-figureqa_en_20240402_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-geo170k_extracted_full_pairs_vqa_correctness_rules
# mmpr-1.2-geo170k_extracted_full_pairs_vqa_format_rules
# mmpr-1.2-geo170k_extracted_pairs_vqa_correctness_rules
# mmpr-1.2-geo170k_extracted_pairs_vqa_format_rules
# mmpr-1.2-geometry3k_en_20240402_extracted_open_ended_only_pairs_vqa_correctness_rules
# mmpr-1.2-geometry3k_en_20240402_extracted_open_ended_only_pairs_vqa_format_rules
# mmpr-1.2-geometry3k_en_20240402_extracted_pairs_vqa_correctness_rules
# mmpr-1.2-geometry3k_en_20240402_extracted_pairs_vqa_direct_rules
# mmpr-1.2-geometry3k_en_20240402_extracted_pairs_vqa_format_rules
# mmpr-1.2-geomverse_extracted_pairs_vqa_correctness_rules
# mmpr-1.2-geomverse_extracted_pairs_vqa_format_rules
# mmpr-1.2-geoqa+_en_20240402_extracted_open_ended_only_pairs_vqa_correctness_rules
# mmpr-1.2-geoqa+_en_20240402_extracted_open_ended_only_pairs_vqa_format_rules
# mmpr-1.2-geoqa+_extracted_en_version_pairs_vqa_correctness_rules
# mmpr-1.2-geoqa+_extracted_en_version_pairs_vqa_format_rules
# mmpr-1.2-geos_en_20240402_extracted_open_ended_only_pairs_vqa_correctness_rules
# mmpr-1.2-geos_en_20240402_extracted_open_ended_only_pairs_vqa_format_rules
# mmpr-1.2-geos_en_20240402_extracted_pairs_vqa_correctness_rules
# mmpr-1.2-geos_en_20240402_extracted_pairs_vqa_format_rules
# mmpr-1.2-gqa_train_en_20240402_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-gqa_train_en_20240402_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-gqa_train_en_20240402_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-iconqa_train_extracted_pairs_vqa_correctness_rules
# mmpr-1.2-iconqa_train_extracted_pairs_vqa_format_rules
# mmpr-1.2-inat_train2018_merge_en_20240811_sr0.50_wo_image
# mmpr-1.2-infographics_20240403_qa_20240407_v2_extracted_pairs_vqa_correctness_rules
# mmpr-1.2-infographics_20240403_qa_20240407_v2_extracted_pairs_vqa_format_rules
# mmpr-1.2-infographics_20240403_qa_20240407_v2_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-infographics_20240403_qa_20240407_v2_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-infographics_20240403_qa_20240407_v2_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-koniq10k_en_20240403_pairs_vqa_correctness_rules
# mmpr-1.2-koniq10k_en_20240403_pairs_vqa_format_rules
# mmpr-1.2-m3cot_train_extracted_pairs_vqa_correctness_rules
# mmpr-1.2-m3cot_train_extracted_pairs_vqa_direct_rules
# mmpr-1.2-m3cot_train_extracted_pairs_vqa_format_rules
# mmpr-1.2-m3cot_train_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-m3cot_train_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-m3cot_train_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-mapqa_suv_en_20240402_extracted_pairs_vqa_correctness_rules
# mmpr-1.2-mapqa_suv_en_20240402_extracted_pairs_vqa_format_rules
# mmpr-1.2-mapqa_suv_en_20240402_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-mapqa_suv_en_20240402_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-mapqa_suv_en_20240402_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-MathV360K_prompts_pairs_vqa_correctness_rules
# mmpr-1.2-MathV360K_prompts_pairs_vqa_format_rules
# mmpr-1.2-mavis_function_abs_pairs_vqa_correctness_rules
# mmpr-1.2-mavis_function_abs_pairs_vqa_direct_rules
# mmpr-1.2-mavis_function_abs_pairs_vqa_format_rules
# mmpr-1.2-mavis_function_cos_pairs_vqa_correctness_rules
# mmpr-1.2-mavis_function_cos_pairs_vqa_format_rules
# mmpr-1.2-mavis_function_log_pairs_vqa_correctness_rules
# mmpr-1.2-mavis_function_log_pairs_vqa_format_rules
# mmpr-1.2-mavis_function_poly_pairs_vqa_correctness_rules
# mmpr-1.2-mavis_function_poly_pairs_vqa_format_rules
# mmpr-1.2-mavis_function_sin_pairs_vqa_correctness_rules
# mmpr-1.2-mavis_function_sin_pairs_vqa_format_rules
# mmpr-1.2-mavis_function_tan_pairs_vqa_correctness_rules
# mmpr-1.2-mavis_function_tan_pairs_vqa_format_rules
# mmpr-1.2-mavis_geo_depth0_text_dominant_vision_dominant_en_pairs_vqa_correctness_rules
# mmpr-1.2-mavis_geo_depth0_text_dominant_vision_dominant_en_pairs_vqa_format_rules
# mmpr-1.2-mavis_geo_depth1_text_dominant_vision_dominant_en_pairs_vqa_correctness_rules
# mmpr-1.2-mavis_geo_depth1_text_dominant_vision_dominant_en_pairs_vqa_format_rules
# mmpr-1.2-mavis_geo_depth2_text_dominant_vision_dominant_en_pairs_vqa_correctness_rules
# mmpr-1.2-mavis_geo_depth2_text_dominant_vision_dominant_en_pairs_vqa_format_rules
# mmpr-1.2-mavis_geo_depth3_text_dominant_vision_dominant_en_pairs_vqa_correctness_rules
# mmpr-1.2-mavis_geo_depth3_text_dominant_vision_dominant_en_pairs_vqa_format_rules
# mmpr-1.2-nlvr2_en_20240910_ov_pairs_vqa_correctness_rules
# mmpr-1.2-nlvr2_en_20240910_ov_pairs_vqa_format_rules
# mmpr-1.2-okvqa_train_9k_en_20240402_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-okvqa_train_9k_en_20240402_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-okvqa_train_9k_en_20240402_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-scienceqa_multi_choice_en_20240402_extracted_pairs_vqa_correctness_rules
# mmpr-1.2-scienceqa_multi_choice_en_20240402_extracted_pairs_vqa_direct_rules
# mmpr-1.2-scienceqa_multi_choice_en_20240402_extracted_pairs_vqa_format_rules
# mmpr-1.2-scienceqa_multi_choice_en_20240402_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-scienceqa_multi_choice_en_20240402_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-scienceqa_multi_choice_en_20240402_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-SROIE_information_extraction_multi_turn_20240620_extracted_pairs_vqa_correctness_rules
# mmpr-1.2-SROIE_information_extraction_multi_turn_20240620_extracted_pairs_vqa_format_rules
# mmpr-1.2-SROIE_information_extraction_multi_turn_20240620_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-SROIE_information_extraction_multi_turn_20240620_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-SROIE_information_extraction_multi_turn_20240620_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-super_clevr_en_20240402_int_pairs_vqa_correctness_rules
# mmpr-1.2-super_clevr_en_20240402_int_pairs_vqa_format_rules
# mmpr-1.2-super_clevr_en_20240402_yorn_pairs_vqa_correctness_rules
# mmpr-1.2-super_clevr_en_20240402_yorn_pairs_vqa_format_rules
# mmpr-1.2-tabmwp_en_20240402_cot_pairs_vqa_correctness_rules
# mmpr-1.2-tallyqa_vg_en_20240816_cot_pairs_vqa_correctness_rules
# mmpr-1.2-textvqa_train_21k_wo_ocr_en_20240611_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-textvqa_train_21k_wo_ocr_en_20240611_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-textvqa_train_21k_wo_ocr_en_20240611_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-unigeo_calc_en_20240402_extracted_open_ended_only_pairs_vqa_correctness_rules
# mmpr-1.2-unigeo_calc_en_20240402_extracted_open_ended_only_pairs_vqa_format_rules
# mmpr-1.2-vqav2_en_20240402_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-vqav2_en_20240402_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-vqav2_en_20240402_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-vqav2_en_20240402_int_pairs_vqa_correctness_rules
# mmpr-1.2-vqav2_en_20240402_int_pairs_vqa_format_rules
# mmpr-1.2-vsr_en_20240402_cot_ques_pairs_vqa_correctness_rules

# EXACT_VERIFIER = """
# mmpr-1.2-ai2d_train_12k_en_20240410_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-ai2d_train_12k_en_20240410_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-ai2d_train_12k_en_20240410_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-chartqa_trainval_30k_w_csv_en_20240402_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-chartqa_trainval_30k_w_csv_en_20240402_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-chartqa_trainval_30k_w_csv_en_20240402_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-CLEVR_math_en_20240402_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-CLEVR_math_en_20240402_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-CLEVR_math_en_20240402_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-docvqa_train_56k_en_20240402_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-docvqa_train_56k_en_20240402_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-docvqa_train_56k_en_20240402_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-figureqa_en_20240402_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-figureqa_en_20240402_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-figureqa_en_20240402_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-gqa_train_en_20240402_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-gqa_train_en_20240402_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-gqa_train_en_20240402_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-infographics_20240403_qa_20240407_v2_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-infographics_20240403_qa_20240407_v2_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-infographics_20240403_qa_20240407_v2_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-m3cot_train_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-m3cot_train_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-m3cot_train_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-mapqa_suv_en_20240402_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-mapqa_suv_en_20240402_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-mapqa_suv_en_20240402_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-okvqa_train_9k_en_20240402_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-okvqa_train_9k_en_20240402_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-okvqa_train_9k_en_20240402_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-scienceqa_multi_choice_en_20240402_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-scienceqa_multi_choice_en_20240402_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-scienceqa_multi_choice_en_20240402_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-SROIE_information_extraction_multi_turn_20240620_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-SROIE_information_extraction_multi_turn_20240620_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-SROIE_information_extraction_multi_turn_20240620_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-textvqa_train_21k_wo_ocr_en_20240611_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-textvqa_train_21k_wo_ocr_en_20240611_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-textvqa_train_21k_wo_ocr_en_20240611_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-vqav2_en_20240402_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-vqav2_en_20240402_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-vqav2_en_20240402_extracted_prefix_pair_sr0.5_wo_image
# """.strip().split()

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
mmpr-1.2-koniq10k_en_20240403_pairs_vqa_correctness_rules
mmpr-1.2-koniq10k_en_20240403_pairs_vqa_format_rules
mmpr-1.2-scienceqa_multi_choice_en_20240402_extracted_pairs_vqa_correctness_rules
mmpr-1.2-scienceqa_multi_choice_en_20240402_extracted_pairs_vqa_format_rules
mmpr-1.2-nlvr2_en_20240910_ov_pairs_vqa_correctness_rules
mmpr-1.2-nlvr2_en_20240910_ov_pairs_vqa_format_rules
mmpr-1.2-m3cot_train_extracted_pairs_vqa_direct_rules
""".strip().split()


def mmpr_filtered_reward(
    ground_truth: str,
    response: str,
    format_weight: float = 0.1,
):
    """Aggregate reward function for MMPR dataset.
    Partial reward for format compliance and accuracy; penalized for extra think/boxed blocks.
    """
    if "</think>" in response and "<think>" not in response:
        # XXX nano-v2 tokenizer adds <think> in reasoning mode, but that is omitted from prediction
        response = "<think>\n" + response

    think_blocks = list(re.finditer(r"<think>.*?</think>", response, re.DOTALL))
    if think_blocks:
        # remove first think block
        response = response[think_blocks[0].end() :].strip()

    verifier_name = ground_truth.split(":", 1)[0]
    if verifier_name == "gui-coordinate":
        return _grade_gui_response(ground_truth, response, think_blocks, format_weight)

    boxed_answers = extract_all_boxed(response)
    format_reward_value = (
        1.0 if len(think_blocks) == 1 and len(boxed_answers) == 1 else 0.0
    )

    verifier, ground_truth = ground_truth.split(":", 1)

    # Accuracy check
    if boxed_answers:
        grades = []
        for boxed_answer in boxed_answers[:5]:
            grades.append(grade_mmpr(verifier, ground_truth, boxed_answer))
        if len(boxed_answers) > 5:
            grades.extend([0.0] * (len(boxed_answers) - 5))
        acc_reward_value = float(np.mean(grades))
        is_correct = any(x > 0.0 for x in grades)
    else:
        acc_reward_value = 0.0
        is_correct = False

    # penalize 10% for any additional syntax error
    extra_thinks = response.count("<think>") + response.count("</think>")
    extra_boxeds = max(0, len(boxed_answers) - 1)
    acc_reward_value = acc_reward_value * (0.9 ** (extra_thinks + extra_boxeds))

    # Weighted combination (verl's exact formula)
    final_reward = (
        1.0 - format_weight
    ) * acc_reward_value + format_weight * format_reward_value
    return final_reward, is_correct


def extract_all_boxed(text: str) -> list[str]:
    if "\\boxed{" not in text:
        return []
    results = []
    # require that boxed can't be nested
    parts = text.split("\\boxed{")[1:]
    for part in parts:
        depth = 1
        for i, char in enumerate(part):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            if depth == 0:
                results.append(part[:i])
                break
        # if parens are not balanced, the answer is ignored
    return results


def grade_mmpr(verifier: str, gt_answer: str, pred_answer: str) -> float:
    try:
        if verifier == "mathruler":
            return grade_math(gt_answer, pred_answer)
        elif verifier == "multiple-choice":
            return grade_multiple_choice(gt_answer, pred_answer)
        elif verifier == "python-list":
            return grade_python_list(gt_answer, pred_answer)
        elif verifier == "string-match":
            return grade_string_match(gt_answer, pred_answer)
        elif verifier == "gui-coordinate":
            return grade_gui_coordinate(gt_answer, pred_answer)
        else:
            raise ValueError(f"Unknown verifier: {verifier}")
    except Exception as e:
        logging.exception(
            "MMPR verification failed for %s: %s vs %s",
            verifier,
            gt_answer,
            pred_answer,
        )
        return 0.0


def _grade_gui_response(ground_truth, response, think_blocks, format_weight):
    """Reward for GUI coordinate samples using <point>[(x,y)]</point> format."""
    _, gt_answer = ground_truth.split(":", 1)
    gt_parts = gt_answer.split(",")
    if len(gt_parts) != 2:
        return 0.0, False
    try:
        gt_x, gt_y = float(gt_parts[0]), float(gt_parts[1])
    except ValueError:
        return 0.0, False

    m = re.search(
        r"<point>\s*\[\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*\]\s*</point>", response
    )
    if m:
        pred_x = int(m.group(1)) / 1000.0
        pred_y = int(m.group(2)) / 1000.0
        dist = ((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2) ** 0.5
        acc_reward_value = (1.0 - dist / 0.15) ** 2 if dist < 0.15 else 0.0
    else:
        acc_reward_value = 0.0

    point_tags = len(re.findall(r"<point>", response))
    format_reward_value = (
        1.0 if len(think_blocks) == 1 and point_tags == 1 else 0.0
    )

    extra_thinks = response.count("<think>") + response.count("</think>")
    extra_points = max(0, point_tags - 1)
    acc_reward_value = acc_reward_value * (0.9 ** (extra_thinks + extra_points))

    final_reward = (
        1.0 - format_weight
    ) * acc_reward_value + format_weight * format_reward_value
    is_correct = acc_reward_value > 0
    return final_reward, is_correct


def grade_gui_coordinate(gt_answer: str, pred_answer: str, max_dist: float = 0.15) -> float:
    """GUI coordinate verifier: smooth quadratic reward based on Euclidean distance."""
    gt_parts = gt_answer.split(",")
    if len(gt_parts) != 2:
        return 0.0
    try:
        gt_x, gt_y = float(gt_parts[0]), float(gt_parts[1])
    except ValueError:
        return 0.0

    m = re.search(r"<point>\s*\[?\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*\]?\s*</point>", pred_answer)
    if m:
        pred_x, pred_y = int(m.group(1)) / 1000.0, int(m.group(2)) / 1000.0
    else:
        return 0.0

    dist = ((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2) ** 0.5
    if dist >= max_dist:
        return 0.0
    return (1.0 - dist / max_dist) ** 2


def _normalize_unicode_math(text: str) -> str:
    return (
        text
        .replace("°", "^\\circ")
        .replace("²", "^2")
        .replace("³", "^3")
        .replace("⁴", "^4")
        .replace("⁵", "^5")
        .replace("⁶", "^6")
        .replace("⁷", "^7")
        .replace("⁸", "^8")
        .replace("⁹", "^9")
        .replace("√", "\\sqrt")
        .replace("﹣", "-")
        .replace("﹢", "+")
        .replace("﹦", "=")
        .replace("﹤", "<")
        .replace("﹥", ">")
        .replace("：", ":")
        .replace("π", "\\pi")
    )


def _normalize_ratio(text: str) -> str:
    m = re.match(r"^(\d+)\s*[:/]\s*(\d+)$", text.strip())
    if m:
        return f"\\frac{{{m.group(1)}}}{{{m.group(2)}}}"
    return text


def grade_math(gt_answer: str, pred_answer: str) -> float:
    """Mathematical equality verifier.
    Binary 0/1 via mathruler symbolic equivalence.
    """
    pred_answer = _normalize_unicode_math(pred_answer)
    gt_answer = _normalize_unicode_math(gt_answer)
    pred_answer = _normalize_ratio(pred_answer)
    gt_answer = _normalize_ratio(gt_answer)
    try:
        float(pred_answer)
    except ValueError:
        pass
    else:
        gt_answer = (
            gt_answer
            .replace("cm²", "")
            .replace("cm2", "")
            .replace("cm", "")
            .replace("m³", "")
            .replace("m3", "")
            .replace("m²", "")
            .replace("m2", "")
            .replace("m", "")
            .replace("kg", "")
            .replace("克", "")
        )
    return float(grade_answer(pred_answer, gt_answer))


def grade_multiple_choice(gt_answer: str, pred_answer: str) -> float:
    """Multiple choice answer verifier.
    Partial reward for correct letter with trailing punctuation; 0 for wrong or unparseable.
    """
    pred_answer = pred_answer.upper()
    gt_answer = "".join(
        ch for ch in gt_answer.upper() if ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )
    assert len(gt_answer) == 1, f"gt_answer: {gt_answer}"
    if len(pred_answer) == 0:
        return 0.0
    elif len(pred_answer) > 1:
        if pred_answer[1] in (".", ",", ":", ";"):
            pred_answer = pred_answer[:1]
        else:
            return 0.0
    score = float(pred_answer[0] == gt_answer[0])
    # penalize for additional characters
    score = score * (0.99 ** (len(pred_answer) - 1))
    return score


def grade_python_list(gt_answer: str, pred_answer: str) -> float:
    """Python list equality verifier.
    Partial reward for correct items; string items scored by word F1, others by equality.
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SyntaxWarning)
            pred_list = ast.literal_eval(pred_answer)
            gt_list = ast.literal_eval(gt_answer)
        correct = sum(
            _word_f1(p, g) if isinstance(p, str) and isinstance(g, str) else float(p == g)
            for p, g in zip(pred_list, gt_list)
        )
        return correct / max(len(pred_list), len(gt_list))
    except Exception:
        return 0.0


def _word_f1(pred: str, gt: str) -> float:
    pred_words = set(pred.lower().split())
    gt_words = set(gt.lower().split())
    if not pred_words or not gt_words:
        return 0.0
    common = pred_words & gt_words
    if not common:
        return 0.0
    precision = len(common) / len(pred_words)
    recall = len(common) / len(gt_words)
    return 2 * precision * recall / (precision + recall)


def grade_string_match(gt_answer: str, pred_answer: str) -> float:
    """Generic heuristic-based string equality verifier.
    Binary 0/1 with cascading normalizations (float, numbers, latex, lists, US states).
    """
    pred_answer = _strip_outer_quotes(pred_answer).lower()
    gt_answer = _strip_outer_quotes(gt_answer).lower()
    pred_answer = _strip_trailing_punctuation(pred_answer)
    gt_answer = _strip_trailing_punctuation(gt_answer)
    if pred_answer == gt_answer:
        return 1.0
    elif _normalize_float(pred_answer) == _normalize_float(gt_answer):
        return 1.0
    elif _normalize_numbers(pred_answer) == _normalize_numbers(gt_answer):
        return 1.0
    elif _normalize_latex(pred_answer) == _normalize_latex(gt_answer):
        return 1.0
    elif _normalize_lists(pred_answer) == _normalize_lists(gt_answer):
        return 1.0
    elif _normalize_states(pred_answer) == _normalize_states(gt_answer):
        return 1.0
    else:
        return 0.0


def _strip_outer_quotes(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    return text


def _strip_trailing_punctuation(text: str) -> str:
    stripped = text.rstrip(".!?")
    return stripped or text


def _normalize_float(text: str) -> str:
    cleaned = text.replace("\\%", "").replace("\\$", "").replace("$", "").strip()
    try:
        return str(float(cleaned))
    except ValueError:
        return text


def _normalize_numbers(text: str) -> str:
    text = re.sub(r"(\d+),(\d+)", r"\1\2", text)
    text = text.replace("\\%", "%")
    return text


def _normalize_lists(text: str) -> str:
    orig_text = text
    orig_len = len(text)
    text = text.replace(",", " ").replace(";", " ")
    text = text.replace("and", " ").replace("or", " ")
    if len(text) < 0.9 * orig_len:
        return orig_text
    text = " ".join(text.split())
    return text


def _normalize_latex(text: str) -> str:
    for cmd in (
        "\\boxed",
        "\\text",
        "\\textbf",
        "\\textit",
        "\\texttt",
        "\\mathrm",
        "\\mathbf",
        "\\mathit",
        "\\mathsf",
        "\\mathbb",
        "\\mathcal",
        "\\emph",
        "\\url",
    ):
        text = _remove_latex_command(cmd, text)
    for token in ("\\(", "\\)", "\\[", "\\]"):
        text = text.replace(token, "")
    text = text.replace("$", "")
    text = " ".join(text.split())
    return text


def _remove_latex_command(cmd: str, text: str) -> str:
    assert cmd.startswith("\\"), f"command must start with \\: {cmd}"
    while cmd + "{" in text:
        prefix, suffix = text.split(cmd + "{", 1)
        depth = 1
        for i, char in enumerate(suffix):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            if depth == 0:
                text = prefix + suffix[:i] + suffix[i+1:]
                break
        if depth != 0:
            # unbalanced braces
            text = prefix + suffix
    return text


def _normalize_states(text: str) -> str:
    # multi-word states must be replaced before single-word ones to avoid
    # e.g. "west virginia" -> "west VA" instead of "WV"
    text = (
        text.replace("district of columbia", "DC")
        .replace("new hampshire", "NH")
        .replace("new jersey", "NJ")
        .replace("new mexico", "NM")
        .replace("new york", "NY")
        .replace("north carolina", "NC")
        .replace("north dakota", "ND")
        .replace("rhode island", "RI")
        .replace("south carolina", "SC")
        .replace("south dakota", "SD")
        .replace("west virginia", "WV")
        .replace("alabama", "AL")
        .replace("alaska", "AK")
        .replace("arizona", "AZ")
        .replace("arkansas", "AR")
        .replace("california", "CA")
        .replace("colorado", "CO")
        .replace("connecticut", "CT")
        .replace("delaware", "DE")
        .replace("florida", "FL")
        .replace("georgia", "GA")
        .replace("hawaii", "HI")
        .replace("idaho", "ID")
        .replace("illinois", "IL")
        .replace("indiana", "IN")
        .replace("iowa", "IA")
        .replace("kansas", "KS")
        .replace("kentucky", "KY")
        .replace("louisiana", "LA")
        .replace("maine", "ME")
        .replace("maryland", "MD")
        .replace("massachusetts", "MA")
        .replace("michigan", "MI")
        .replace("minnesota", "MN")
        .replace("mississippi", "MS")
        .replace("missouri", "MO")
        .replace("montana", "MT")
        .replace("nebraska", "NE")
        .replace("nevada", "NV")
        .replace("ohio", "OH")
        .replace("oklahoma", "OK")
        .replace("oregon", "OR")
        .replace("pennsylvania", "PA")
        .replace("tennessee", "TN")
        .replace("texas", "TX")
        .replace("utah", "UT")
        .replace("vermont", "VT")
        .replace("virginia", "VA")
        .replace("washington", "WA")
        .replace("wisconsin", "WI")
        .replace("wyoming", "WY")
    )
    return _normalize_lists(text)
