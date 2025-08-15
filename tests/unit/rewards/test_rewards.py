import pytest
import numpy as np
from numpy.testing import assert_allclose
from nemo_rl.environments.rewards import (
    math_expression_reward,
    format_reward,
    exact_answer_alphanumeric_reward,
    bbox_giou_reward,
    combine_reward_functions,
)

def test_math_expression_reward():
    # Test correct math expression
    ground_truth = "2x + 3"
    response = "Let me solve for y ... <think>5x + 5 = 3x + 2 + y \implies y = 2x + 3</think> <answer>2x + 3</answer>"
    reward, is_correct = math_expression_reward(ground_truth, response)
    assert_allclose(reward, 1.0, atol=1e-6)
    assert is_correct is True

    # Test incorrect math expression
    response = "Let me solve this... <think>I'm a dumb LLM so I have no reasoning trace to actuallysolve this</think> <answer>3x + 2</answer>"
    reward, is_correct = math_expression_reward(ground_truth, response)
    assert_allclose(reward, 0.0, atol=1e-6)
    assert is_correct is False

    # Test for missing answer tags
    response = "Let me solve this... The answer is 2x + 3"
    reward, is_correct = math_expression_reward(ground_truth, response)
    assert_allclose(reward, 0.0, atol=1e-6)
    assert is_correct is False

def test_format_reward():
    ground_truth = "any_ground_truth"  # Format reward doesn't use ground truth
    
    # Test complete format
    response = "<think>My thinking</think> <answer>My answer</answer>"
    reward, is_correct = format_reward(ground_truth, response)
    assert reward == 1.0
    assert is_correct is None

    # Test only think tags
    response = "<think>My thinking</think>"
    reward, is_correct = format_reward(ground_truth, response)
    assert reward == 0.25
    assert is_correct is None

    # Test only answer tags
    response = "<answer>My answer</answer>"
    reward, is_correct = format_reward(ground_truth, response)
    assert reward == 0.75
    assert is_correct is None

    # Test no tags
    response = "Just plain text"
    reward, is_correct = format_reward(ground_truth, response)
    assert reward == 0.0
    assert is_correct is None

def test_format_reward_custom_tags():
    ground_truth = "does_not_matter"

    # Both tags in response and reward function match
    response = "<think_trace>Reasoning here</think_trace> <solution>42</solution>"
    reward, is_correct = format_reward(ground_truth, response, think_tag="think_trace", answer_tag="solution")
    assert reward == 1.0
    assert is_correct is None

    # Only think tag present, tags match
    response = "<think_trace>Reasoning here</think_trace>"
    reward, is_correct = format_reward(ground_truth, response, think_tag="think_trace", answer_tag="solution")
    assert reward == 0.25
    assert is_correct is None

    # Only answer tag present, tags match
    response = "<solution>42</solution>"
    reward, is_correct = format_reward(ground_truth, response, think_tag="think_trace", answer_tag="solution")
    assert reward == 0.75
    assert is_correct is None

    # Neither tag present, tags match
    response = "No tags here"
    reward, is_correct = format_reward(ground_truth, response, think_tag="think_trace", answer_tag="solution")
    assert reward == 0.0
    assert is_correct is None

    # Tags in response do not match those in reward function (should yield 0.0)
    response = "<think>Reasoning here</think> <answer>42</answer>"
    reward, is_correct = format_reward(ground_truth, response, think_tag="think_trace", answer_tag="solution")
    assert reward == 0.0
    assert is_correct is None

    # Mixed: one tag matches, one does not (should yield 0.25 for think_trace, 0 for solution)
    response = "<think_trace>Reasoning here</think_trace> <answer>42</answer>"
    reward, is_correct = format_reward(ground_truth, response, think_tag="think_trace", answer_tag="solution")
    assert reward == 0.25
    assert is_correct is None

    # Mixed: one tag matches, one does not (should yield 0.75 for solution, 0 for think_trace)
    response = "<think>Reasoning here</think> <solution>42</solution>"
    reward, is_correct = format_reward(ground_truth, response, think_tag="think_trace", answer_tag="solution")
    assert reward == 0.75
    assert is_correct is None

def test_exact_answer_alphanumeric_reward():
    ground_truth = "Hello123"
    
    # Test exact match
    response = "<answer>Hello123</answer>"
    reward, is_correct = exact_answer_alphanumeric_reward(ground_truth, response)
    assert_allclose(reward, 1.0, atol=1e-6)
    assert is_correct is True

    # Test case insensitive match
    response = "<answer>HELLO123</answer>"
    reward, is_correct = exact_answer_alphanumeric_reward(ground_truth, response)
    assert_allclose(reward, 1.0, atol=1e-6)
    assert is_correct is True

    # Test with special characters
    response = "<answer>Hello-123!</answer>"
    reward, is_correct = exact_answer_alphanumeric_reward(ground_truth, response)
    assert_allclose(reward, 1.0, atol=1e-6)
    assert is_correct is True

    # Test incorrect answer
    response = "<answer>Hello124</answer>"
    reward, is_correct = exact_answer_alphanumeric_reward(ground_truth, response)
    assert_allclose(reward, 0.0, atol=1e-6)
    assert is_correct is False

def test_bbox_giou_reward():
    ground_truth = "[0.1, 0.1, 0.5, 0.5]"
    
    # Test perfect match
    response = "<answer>[0.1, 0.1, 0.5, 0.5]</answer>"
    reward, is_correct = bbox_giou_reward(ground_truth, response)
    print(f"reward: {reward}, is_correct: {is_correct}")
    assert_allclose(reward, 1.0, atol=1e-6)
    assert is_correct is True

    # Test partial overlap
    response = "<answer>[0.2, 0.2, 0.6, 0.6]</answer>"
    reward, is_correct = bbox_giou_reward(ground_truth, response)
    print(f"reward: {reward}, is_correct: {is_correct}")
    assert 0 < reward < 1.0
    assert is_correct is False

    # Test no overlap
    response = "<answer>[0.6, 0.6, 0.9, 0.9]</answer>"
    reward, is_correct = bbox_giou_reward(ground_truth, response)
    print(f"reward: {reward}, is_correct: {is_correct}")
    assert reward < 0.0  # GIoU can be negative when boxes don't overlap
    assert is_correct is False

    # test bad bounding box format (5 numbers)
    response = "<answer>[0.6, 0.6, 0.9, 0.9, 0.1]</answer>"
    reward, is_correct = bbox_giou_reward(ground_truth, response)
    print(f"reward: {reward}, is_correct: {is_correct}")
    assert_allclose(reward, 0.0, atol=1e-6)
    assert is_correct is False

    # Test invalid format
    response = "<answer>invalid bbox format</answer>"
    reward, is_correct = bbox_giou_reward(ground_truth, response)
    print(f"reward: {reward}, is_correct: {is_correct}")
    assert_allclose(reward, 0.0, atol=1e-6)
    assert is_correct is False


def test_exact_answer_alphanumeric_reward_combined():
    # Define test cases
    ground_truth = "test123"
    good_response = "<think>thinking</think> <answer>test123</answer>"
    bad_response = "<think>thinking</think> <answer>wrong</answer>"
    incorrect_format_response = "here is a bbox: [0.1, 0.1, 0.5, 0.5] without any tags"

    # Create reward function combinations with weights
    reward_functions = [
        (format_reward, 0.3),
        (exact_answer_alphanumeric_reward, 0.7)
    ]
    combined_reward = combine_reward_functions(reward_functions)

    # Test good response
    reward, is_correct = combined_reward(ground_truth, good_response)
    assert_allclose(reward, 1.0, atol=1e-6)
    assert is_correct is True

    # Test bad response
    reward, is_correct = combined_reward(ground_truth, bad_response)
    assert_allclose(reward, 0.3, atol=1e-6)
    assert is_correct is False

    # test bad format
    reward, is_correct = combined_reward(ground_truth, incorrect_format_response)
    assert_allclose(reward, 0.0, atol=1e-6)
    assert is_correct is False


def test_bbox_giou_reward_combined():
    # Test combining all reward functions
    ground_truth_bbox = "[0.1, 0.1, 0.5, 0.5]"
    good_response = "<think>The bounding box coordinates are [0.1, 0.1, 0.5, 0.5]</think> <answer>[0.1, 0.1, 0.5, 0.5]</answer>"
    no_think_response = "<answer>[0.1, 0.1, 0.5, 0.5]</answer>"
    no_answer_response = "<think>thinking</think>"
    no_think_no_answer_response = "here is a bbox: [0.1, 0.1, 0.5, 0.5] without any tags"
    
    reward_functions = [
        (format_reward, 0.2),
        (bbox_giou_reward, 0.8)
    ]

    combined_reward = combine_reward_functions(reward_functions)
    
    # Test perfect response
    reward, is_correct = combined_reward(ground_truth_bbox, good_response)
    assert_allclose(reward, 1.0, atol=1e-6)
    assert is_correct is True

    # Test partially correct response (correct format, wrong bbox)
    reward, is_correct = combined_reward(ground_truth_bbox, no_think_response)
    assert_allclose(reward, 0.75 * 0.2 + 0.8, atol=1e-6)
    assert is_correct is True

    reward, is_correct = combined_reward(ground_truth_bbox, no_answer_response)
    assert_allclose(reward, 0.2 * 0.25, atol=1e-6)
    assert is_correct is False

    reward, is_correct = combined_reward(ground_truth_bbox, no_think_no_answer_response)
    assert_allclose(reward, 0.0, atol=1e-6)
    assert is_correct is False
