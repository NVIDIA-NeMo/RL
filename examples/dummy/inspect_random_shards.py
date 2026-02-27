import os
import random
import glob
import re
from datasets import Dataset

# --- CONFIGURATION ---
# The directory containing all your arrow files
DATA_DIR = "/lustre/fsw/portfolios/llmservice/users/sdiao/data/climb_nm5.5_phase3_400b_shuffled_text_only_global_shuffle/"
FILES_TO_INSPECT = 3
SAMPLES_PER_FILE = 5

def parse_and_clean(text):
    if not text or len(text) < 50: return None

    # --- 1. Standard Chat (<extra_id_1>) ---
    if "<extra_id_1>User" in text and "<extra_id_1>Assistant" in text:
        try:
            parts = text.split("<extra_id_1>Assistant")
            return {"type": "Chat (NeMo)", "input": parts[0].replace("<extra_id_1>User", "").strip(), "output": parts[1].strip()}
        except: pass

    # --- 2. Input/Output (Alpaca Style) ---
    if text.lower().startswith("input:"):
        if "output:" in text.lower():
            parts = re.split(r'output:', text, flags=re.IGNORECASE, maxsplit=1)
            return {"type": "Input/Output", "input": parts[0].replace("input:", "").strip(), "output": parts[1].strip()}

    # --- 3. Question/Answer & Problem/Solution ---
    if "Question:" in text or "Problem:" in text or text.startswith("Solve "):
        if "Answer:" in text:
            parts = text.split("Answer:", 1)
            return {"type": "Q&A (Answer)", "input": parts[0].strip(), "output": parts[1].strip()}
        elif "Solution:" in text:
            parts = text.split("Solution:", 1)
            return {"type": "Q&A (Solution)", "input": parts[0].strip(), "output": parts[1].strip()}

    # --- 4. Multi-turn Dialogue ---
    speaker_pattern = re.compile(r'\n\*\*(.*?):\*\*')
    matches = list(speaker_pattern.finditer(text))
    if len(matches) >= 2:
        last_turn_start = matches[-1].start()
        return {"type": "Dialogue (Multi-turn)", "input": text[:last_turn_start].strip(), "output": text[last_turn_start:].strip()}

    return None

def main():
    print(f"--- Scanning Directory: {DATA_DIR} ---")
    
    # Get all arrow files
    all_files = glob.glob(os.path.join(DATA_DIR, "*.arrow"))
    print(f"Found {len(all_files)} files total.")
    
    if len(all_files) == 0:
        print("Error: No files found. Check path.")
        return

    # Pick random files
    selected_files = random.sample(all_files, min(FILES_TO_INSPECT, len(all_files)))

    for file_path in selected_files:
        print("\n" + "="*80)
        print(f"INSPECTING FILE: {os.path.basename(file_path)}")
        print("="*80)
        
        try:
            ds = Dataset.from_file(file_path)
            # Use streaming-like access to avoid loading everything if possible
            # or just take the first N
            indices = random.sample(range(len(ds)), min(SAMPLES_PER_FILE, len(ds)))
            samples = [ds[i]['text'] for i in indices]
        except Exception as e:
            print(f"Error reading file: {e}")
            continue

        for i, text in enumerate(samples):
            result = parse_and_clean(text)
            print(f"\n[Sample {i+1}]")
            
            if result:
                print(f"\033[92mKEPT ({result['type']})\033[0m") # Green
                print(f"IN:  {result['input'][:100].replace(chr(10), ' ')}...")
                print(f"OUT: {result['output'][:100].replace(chr(10), ' ')}...")
            else:
                print(f"\033[91mSKIPPED (Raw/Article)\033[0m") # Red
                print(f"Preview: {text[:100].replace(chr(10), ' ')}...")

if __name__ == "__main__":
    main()