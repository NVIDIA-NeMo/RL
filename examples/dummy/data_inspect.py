import os
import random
import re
from datasets import Dataset # using Hugging Face datasets library

# --- CONFIGURATION ---
# The path you found earlier
ARROW_FILE_PATH = "/lustre/fsw/portfolios/llmservice/users/sdiao/data/climb_nm5.5_phase3_400b_shuffled_text_only_global_shuffle/data-01090-of-02476.arrow"
NUM_SAMPLES = 20

def parse_and_clean(text):
    """
    Tries to extract (input, output) pairs. Returns a dict or None.
    """
    if not text or len(text) < 50: return None

    # STRATEGY 1: Explicit Chat Format (<extra_id_1>)
    if "<extra_id_1>User" in text and "<extra_id_1>Assistant" in text:
        try:
            parts = text.split("<extra_id_1>Assistant")
            return {
                "type": "Chat (Clean)",
                "input": parts[0].replace("<extra_id_1>User", "").strip(), 
                "output": parts[1].strip()
            }
        except: pass

    # STRATEGY 2: Multi-turn Dialogue (Participant/Student/Teacher)
    # Regex for lines starting with "**Name:**"
    speaker_pattern = re.compile(r'\n\*\*(.*?):\*\*')
    matches = list(speaker_pattern.finditer(text))
    
    if len(matches) >= 2:
        last_turn_start = matches[-1].start()
        return {
            "type": "Dialogue (Multi-turn)",
            "input": text[:last_turn_start].strip(),
            "output": text[last_turn_start:].strip()
        }

    # STRATEGY 3: START/END Blocks (Math/Reasoning)
    if "START:" in text and "END:" in text:
        try:
            content = text.split("START:", 1)[1].split("END:", 1)[0].strip()
            
            if "Question:" in content or "Problem:" in content:
                if "Answer:" in content:
                    parts = content.split("Answer:", 1)
                    return {"type": "Math/Reasoning", "input": parts[0].strip(), "output": "Answer: " + parts[1].strip()}
                elif "Solution:" in content:
                    parts = content.split("Solution:", 1)
                    return {"type": "Math/Reasoning", "input": parts[0].strip(), "output": "Solution: " + parts[1].strip()}
            
            return {"type": "Raw Text (Matched START/END but Rejected)", "input": "SKIPPED", "output": "SKIPPED"}
        except: pass

    return None

def main():
    if not os.path.exists(ARROW_FILE_PATH):
        print(f"ERROR: File not found at {ARROW_FILE_PATH}")
        return

    print(f"--- Inspecting {ARROW_FILE_PATH} ---")
    
    # Load dataset using Hugging Face Library
    try:
        # Try loading specific file
        ds = Dataset.from_file(ARROW_FILE_PATH)
        # If the arrow file has no metadata, sometimes we need to just select the column
        all_text = ds["text"]
    except Exception as e:
        print(f"Error loading file with HF datasets: {e}")
        return

    # Randomly sample
    samples = random.sample(all_text, min(NUM_SAMPLES, len(all_text)))

    print(f"Total Rows in File: {len(all_text)}")
    print(f"Inspecting {len(samples)} random samples...\n")
    print("="*60)

    stats = {"kept": 0, "skipped": 0}

    for i, text in enumerate(samples):
        result = parse_and_clean(text)
        
        print(f"\n[Sample {i+1}] Length: {len(text)}")
        
        if result:
            if result['input'] == "SKIPPED":
                print(f"Status: SKIPPED (Raw Text)") 
                stats["skipped"] += 1
                # print(f"Preview: {text[:100].replace(chr(10), ' ')}...")
            else:
                print(f"Status: KEPT ({result['type']})") 
                stats["kept"] += 1
                print(f"INPUT PREVIEW:  {result['input'][:100].replace(chr(10), ' ')}...")
                print(f"OUTPUT PREVIEW: {result['output'][:100].replace(chr(10), ' ')}...")
        else:
            print(f"Status: SKIPPED (Unknown Format)") 
            stats["skipped"] += 1
            print(f"Preview: {text[:100].replace(chr(10), ' ')}...")
        
        print("-" * 30)

    print("\n" + "="*60)
    print(f"SUMMARY: Kept {stats['kept']} / Skipped {stats['skipped']}")

if __name__ == "__main__":
    main()