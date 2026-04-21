import os
import random
import re
from datasets import Dataset

# --- CONFIGURATION ---
ARROW_FILE_PATH = "/lustre/fsw/portfolios/llmservice/users/sdiao/data/climb_nm5.5_phase3_400b_shuffled_text_only_global_shuffle/data-01090-of-02476.arrow"
NUM_SAMPLES = 20

def parse_and_clean(text):
    if not text or len(text) < 50: return None

    # --- 1. Standard Chat (High Quality) ---
    if "<extra_id_1>User" in text and "<extra_id_1>Assistant" in text:
        parts = text.split("<extra_id_1>Assistant")
        return {"type": "Chat (NeMo)", "input": parts[0].replace("<extra_id_1>User", "").strip(), "output": parts[1].strip()}

    # --- 2. Input/Output Format (Alpaca Style) ---
    # Case insensitive check for "input:" at start
    if text.lower().startswith("input:"):
        # Look for "output:" or "response:"
        if "output:" in text.lower():
            parts = re.split(r'output:', text, flags=re.IGNORECASE, maxsplit=1)
            return {"type": "Input/Output", "input": parts[0].replace("input:", "").strip(), "output": parts[1].strip()}
        
        return {"type": "PARTIAL (Found input, no output)", "input": "DEBUG", "output": "DEBUG"}

    # --- 3. Question/Answer Format ---
    if "Question:" in text:
        # Check for Answer or Solution
        if "Answer:" in text:
            parts = text.split("Answer:", 1)
            return {"type": "Q&A (Answer)", "input": parts[0].strip(), "output": parts[1].strip()}
        elif "Solution:" in text:
            parts = text.split("Solution:", 1)
            return {"type": "Q&A (Solution)", "input": parts[0].strip(), "output": parts[1].strip()}

    # --- 4. Multi-turn Dialogue (Speaker names) ---
    speaker_pattern = re.compile(r'\n\*\*(.*?):\*\*')
    matches = list(speaker_pattern.finditer(text))
    if len(matches) >= 2:
        last_turn_start = matches[-1].start()
        return {"type": "Dialogue (Multi-turn)", "input": text[:last_turn_start].strip(), "output": text[last_turn_start:].strip()}

    # --- 5. Raw Instruction Heuristic (Coding/Math) ---
    # If it starts with an instruction verb but has no clear tags
    start_words = ["Given", "Write", "Create", "Implement", "Calculate", "Imagine"]
    if any(text.startswith(w) for w in start_words):
        # Heuristic: If there is a code block or "Solution:", split there
        if "Solution:" in text:
             parts = text.split("Solution:", 1)
             return {"type": "Implicit Instruction", "input": parts[0].strip(), "output": parts[1].strip()}
        
    return None

def main():
    print(f"--- Deep Mining {ARROW_FILE_PATH} ---")
    try:
        ds = Dataset.from_file(ARROW_FILE_PATH)
        all_text = ds["text"]
    except Exception as e:
        print(f"Error: {e}")
        return

    samples = random.sample(list(all_text), min(NUM_SAMPLES, len(all_text)))
    
    stats = {"kept": 0, "skipped": 0, "partial": 0}

    for i, text in enumerate(samples):
        result = parse_and_clean(text)
        print(f"\n[Sample {i+1}]")
        
        if result:
            if result['type'].startswith("PARTIAL"):
                print(f"\033[93m{result['type']}\033[0m") # Yellow
                print(f"Full Text Preview: {text[:150].replace(chr(10), ' ')}...")
                stats["partial"] += 1
            else:
                print(f"\033[92mKEPT ({result['type']})\033[0m") # Green
                print(f"IN:  {result['input'][:80].replace(chr(10), ' ')}...")
                print(f"OUT: {result['output'][:80].replace(chr(10), ' ')}...")
                stats["kept"] += 1
        else:
            print(f"\033[91mSKIPPED (Raw/Article)\033[0m") # Red
            print(f"Preview: {text[:100].replace(chr(10), ' ')}...")
            stats["skipped"] += 1

    print("\n" + "="*40)
    print(f"SUMMARY: Kept {stats['kept']} | Partial {stats['partial']} | Skipped {stats['skipped']}")

if __name__ == "__main__":
    main()