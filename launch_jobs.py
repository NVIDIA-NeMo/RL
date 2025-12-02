import os
import subprocess

# List of experiment scripts to run
scripts = [
    "exp_qwen32b.sh",
    "exp_qwen30b_a3b.sh",
    "exp_qwen235b_a22b.sh",
    "exp_deepseek_v3.sh",
    "exp_llama3_8b.sh"
]

def submit_job(script_name):
    if not os.path.exists(script_name):
        print(f"⚠️  Script not found: {script_name}")
        return

    print(f"🚀 Submitting {script_name}...")
    try:
        # Ensure executable
        subprocess.run(["chmod", "+x", script_name], check=True)
        # Execute the script (which contains sbatch)
        result = subprocess.run([f"./{script_name}"], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Success: {result.stdout.strip()}")
        else:
            print(f"❌ Failed: {result.stderr.strip()}")
            
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")

if __name__ == "__main__":
    print("=== Launching GB200 Benchmark Experiments ===")
    for script in scripts:
        submit_job(script)
