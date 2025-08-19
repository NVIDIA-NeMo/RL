#!/usr/bin/env python3
"""
Direct GSM8K evaluation script for LLaDA model using NeMo-Skills components.

This script directly calls the generation and evaluation components to avoid
environment isolation issues with the full pipeline.

Usage:
    python eval_llada_direct.py
"""

import os
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import openai

# Set dummy API key for local server
os.environ["OPENAI_API_KEY"] = "dummy-key-for-local-server"

def load_gsm8k_problems(limit: int = 50) -> List[Dict]:
    """Load GSM8K test problems. Using a subset for faster testing."""
    try:
        # Try to find GSM8K dataset in NeMo-Skills
        import nemo_skills
        dataset_path = Path(nemo_skills.__file__).parent / "dataset" / "gsm8k" / "test.jsonl"
        
        if not dataset_path.exists():
            # Fallback: create a few sample problems
            return [
                {
                    "problem": "James writes a 3-page letter to 2 different friends. He uses both sides of the paper. How many pieces of paper did he use?",
                    "expected": "6"
                },
                {
                    "problem": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes 4 into muffins for her friends. How many eggs does she have left?",
                    "expected": "9"
                },
                {
                    "problem": "Tom has 10 apples. He gives 3 to his friend and eats 2 himself. How many apples does he have left?",
                    "expected": "5"
                }
            ]
        
        problems = []
        with open(dataset_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= limit:  # Limit for faster testing
                    break
                problems.append(json.loads(line))
        
        return problems
    except Exception as e:
        print(f"Warning: Could not load GSM8K dataset: {e}")
        # Return sample problems
        return [
            {
                "problem": "James writes a 3-page letter to 2 different friends. He uses both sides of the paper. How many pieces of paper did he use?",
                "expected": "6"
            },
            {
                "problem": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes 4 into muffins for her friends. How many eggs does she have left?",
                "expected": "9"
            },
            {
                "problem": "Tom has 10 apples. He gives 3 to his friend and eats 2 himself. How many apples does he have left?",
                "expected": "5"
            }
        ]

def create_gsm8k_prompt(problem: str) -> str:
    """Create a proper GSM8K evaluation prompt."""
    few_shot_examples = """Given the following problem, reason and give a final answer to the problem.

Problem: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Your response should end with "The final answer is \\boxed{[answer]}" where [answer] is the response to the problem.

There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The final answer is \\boxed{6}.

Problem: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Your response should end with "The final answer is \\boxed{[answer]}" where [answer] is the response to the problem.

There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The final answer is \\boxed{5}.

Problem: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Your response should end with "The final answer is \\boxed{[answer]}" where [answer] is the response to the problem.

Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The final answer is \\boxed{39}.

"""
    
    return f"""{few_shot_examples}
Problem: {problem}
Your response should end with "The final answer is \\boxed{{[answer]}}" where [answer] is the response to the problem."""

def extract_answer(response: str) -> str:
    """Extract the numerical answer from the model response."""
    import re
    
    # Look for \\boxed{answer} pattern
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # Fallback: look for "answer is X" patterns
    answer_patterns = [
        r'answer is ([0-9]+(?:\.[0-9]+)?)',
        r'Answer: ([0-9]+(?:\.[0-9]+)?)',
        r'= ([0-9]+(?:\.[0-9]+)?)',
        r'([0-9]+(?:\.[0-9]+)?)\s*$'  # Number at end of string
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return "No answer found"

async def evaluate_llada_gsm8k():
    """Run direct GSM8K evaluation on LLaDA model."""
    
    print("=" * 60)
    print("Direct GSM8K Evaluation with LLaDA Model")
    print("=" * 60)
    
    # Initialize OpenAI client for local server
    client = openai.AsyncOpenAI(
        api_key="dummy-key-for-local-server",
        base_url="http://localhost:8000/v1"
    )
    
    # Load problems
    print("Loading GSM8K problems...")
    problems = load_gsm8k_problems(limit=3)   # Start with just 3 problems for testing
    print(f"Loaded {len(problems)} problems")
    
    results = []
    correct = 0
    total = 0
    
    print("\nRunning evaluation...")
    print("=" * 60)
    
    for i, problem in enumerate(problems):
        problem_text = problem.get("problem", problem.get("question", ""))
        if not problem_text:
            continue
            
        expected = problem.get("expected", problem.get("answer", ""))
        
        print(f"\nProblem {i+1}/{len(problems)}:")
        print(f"Q: {problem_text[:100]}{'...' if len(problem_text) > 100 else ''}")
        
        try:
            # Create prompt
            prompt = create_gsm8k_prompt(problem_text)
            
            # Get model response
            response = await client.chat.completions.create(
                model="llada-8b-instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,  # Short responses for testing
                temperature=0.7,
                top_p=0.95
            )
            
            generated_text = response.choices[0].message.content
            predicted_answer = extract_answer(generated_text)
            
            # Check if answer is correct (basic string matching)
            is_correct = str(predicted_answer).strip() == str(expected).strip()
            if is_correct:
                correct += 1
            
            total += 1
            
            print(f"A: {predicted_answer} (Expected: {expected}) {'✓' if is_correct else '✗'}")
            
            results.append({
                "problem": problem_text,
                "expected": expected,
                "predicted": predicted_answer,
                "correct": is_correct,
                "full_response": generated_text
            })
            
        except Exception as e:
            print(f"Error processing problem {i+1}: {e}")
            continue
    
    # Calculate metrics
    accuracy = (correct / total * 100) if total > 0 else 0
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Problems: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("=" * 60)
    
    # Save detailed results
    output_dir = Path("./eval_results_direct")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "detailed_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump({
            "total_problems": total,
            "correct_answers": correct,
            "accuracy": accuracy,
            "model": "llada-8b-instruct",
            "server": "http://localhost:8000"
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_dir}/detailed_results.json")
    print(f"Summary saved to: {output_dir}/summary.json")
    
    return accuracy

async def main():
    """Main entry point."""
    try:
        accuracy = await evaluate_llada_gsm8k()
        print(f"\nFinal GSM8K Accuracy: {accuracy:.2f}%")
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
