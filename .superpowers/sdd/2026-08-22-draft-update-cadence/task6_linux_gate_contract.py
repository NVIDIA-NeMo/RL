from pathlib import Path


GATE = Path(__file__).with_name("task6_linux_gate.sbatch")


def main() -> None:
    text = GATE.read_text()
    combined = "uv sync --locked --extra mcore --extra vllm --group test --group dev"
    assert combined not in text, "RED: mutually conflicting extras are combined"
    assert text.count("uv sync --locked --extra mcore --group test --group dev") == 1
    assert text.count("uv sync --locked --extra vllm --group test --group dev") == 1
    assert "mcore-venv" in text and "vllm-venv" in text
    assert "expected_head=${EXPECTED_HEAD:?" in text
    assert "/tmp/nr${SLURM_JOB_ID}-m" in text
    assert "/tmp/nr${SLURM_JOB_ID}-v" in text
    assert text.count("unset RAY_ADDRESS") >= 2
    assert "import megatron.core" in text
    assert "import vllm" in text
    assert "phase-mcore-${SLURM_JOB_ID}.txt" in text
    assert "phase-vllm-${SLURM_JOB_ID}.txt" in text
    assert text.count("phase_exit_code=%s") >= 2
    assert "TASK6_MCORE_PHASE_PASS" in text
    assert "TASK6_VLLM_PHASE_PASS" in text
    assert "TASK6_LINUX_GATE_PASS" in text
    print("TASK6_SPLIT_GATE_CONTRACT_GREEN")


if __name__ == "__main__":
    main()
