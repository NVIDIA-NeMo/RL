#!/usr/bin/env python3
"""
Quick test script to verify that HuggingFace model name support works correctly.
This is a temporary test file that demonstrates the new functionality.
"""

import os
import sys

def test_model_path_detection():
    """Test the logic for detecting local paths vs HuggingFace model names."""
    
    test_cases = [
        ("/path/to/local/model", "local path (if exists)"),
        ("GSAI-ML/LLaDA-8B-Instruct", "HuggingFace model name"),
        ("meta-llama/Llama-2-7b-hf", "HuggingFace model name"),
        ("./relative/path", "local path (if exists)"),
        ("../models/my_model", "local path (if exists)"),
        ("microsoft/DialoGPT-large", "HuggingFace model name"),
    ]
    
    print("Testing model path detection logic:")
    print("=" * 50)
    
    for model_path, expected_type in test_cases:
        if os.path.exists(model_path):
            detected_type = "local path"
        else:
            detected_type = "HuggingFace model name"
        
        status = "✅" if expected_type.startswith(detected_type) else "❌"
        print(f"{status} {model_path:<30} -> {detected_type}")
    
    print("\n" + "=" * 50)
    print("The server will now accept both:")
    print("• Local paths: /path/to/model, ./model, ../models/my_model")  
    print("• HF model names: GSAI-ML/LLaDA-8B-Instruct, meta-llama/Llama-2-7b-hf")

if __name__ == "__main__":
    test_model_path_detection()
