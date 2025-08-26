#!/usr/bin/env python3
"""
Test script to verify Fast-dLLM integration with the LLaDA OpenAI server.
"""

import json
import requests
import sys
import time
import asyncio
import aiohttp


def test_fast_dllm_integration():
    """Test the Fast-dLLM integration by making API calls with different cache configurations."""
    
    base_url = "http://localhost:8000"
    
    # Test configurations
    test_configs = [
        {
            "name": "Basic generation (no cache)",
            "config": {
                "generation_algorithm": "basic",
                "steps": 64,
                "block_length": 32,
                "max_tokens": 64
            }
        },
        {
            "name": "Prefix cache",
            "config": {
                "generation_algorithm": "prefix_cache",
                "steps": 64,
                "block_length": 32,
                "max_tokens": 64
            }
        },
        {
            "name": "Dual cache (optimized)",
            "config": {
                "generation_algorithm": "dual_cache",
                "steps": 64,
                "block_length": 32,
                "max_tokens": 64
            }
        },
        {
            "name": "Parallel decoding with threshold",
            "config": {
                "generation_algorithm": "dual_cache",
                "threshold": 0.8,
                "steps": 64,
                "block_length": 32,
                "max_tokens": 64
            }
        },
        {
            "name": "Dynamic parallel with factor",
            "config": {
                "generation_algorithm": "dual_cache",
                "factor": 2.0,
                "steps": 64,
                "block_length": 32,
                "max_tokens": 64
            }
        }
    ]
    
    test_message = {
        "model": "llada-8b-instruct",
        "messages": [
            {"role": "user", "content": "What is 2 + 2?"}
        ],
        "temperature": 0.0
    }
    
    print("Testing Fast-dLLM integration...")
    print("=" * 50)
    
    # Check if server is running
    try:
        health_response = requests.get(f"{base_url}/health", timeout=5)
        health_data = health_response.json()
        print(f"Server status: {health_data['status']}")
        print(f"Model loaded: {health_data['model_loaded']}")
        print(f"Device: {health_data['device']}")
        print()
    except Exception as e:
        print(f"❌ Server not accessible: {e}")
        print("Please start the server first:")
        print("python llada_openai_server.py --model-path GSAI-ML/LLaDA-8B-Instruct")
        return False
    
    results = []
    
    for test_config in test_configs:
        print(f"Testing: {test_config['name']}")
        print("-" * 30)
        
        # Prepare request
        request_data = {**test_message, **test_config['config']}
        
        try:
            # Make request and time it
            start_time = time.time()
            response = requests.post(
                f"{base_url}/v1/chat/completions",
                json=request_data,
                timeout=60
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                response_time = end_time - start_time
                generated_text = result['choices'][0]['message']['content']
                token_count = result['usage']['completion_tokens']
                
                print(f"✅ Success!")
                print(f"   Response time: {response_time:.2f}s")
                print(f"   Tokens generated: {token_count}")
                print(f"   Tokens/sec: {token_count/response_time:.2f}")
                print(f"   Generated text: {generated_text[:100]}...")
                
                results.append({
                    'name': test_config['name'],
                    'success': True,
                    'response_time': response_time,
                    'tokens': token_count,
                    'tokens_per_sec': token_count / response_time,
                    'config': test_config['config']
                })
                
            else:
                print(f"❌ Failed with status {response.status_code}")
                print(f"   Error: {response.text}")
                results.append({
                    'name': test_config['name'],
                    'success': False,
                    'error': response.text,
                    'config': test_config['config']
                })
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            results.append({
                'name': test_config['name'],
                'success': False,
                'error': str(e),
                'config': test_config['config']
            })
        
        print()
    
    # Summary
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    successful_tests = [r for r in results if r['success']]
    failed_tests = [r for r in results if not r['success']]
    
    print(f"Total tests: {len(results)}")
    print(f"Successful: {len(successful_tests)}")
    print(f"Failed: {len(failed_tests)}")
    print()
    
    if successful_tests:
        print("Performance comparison:")
        print("-" * 30)
        for result in successful_tests:
            print(f"{result['name']:30} | {result['tokens_per_sec']:6.2f} tokens/sec | {result['response_time']:6.2f}s")
        print()
    
    if failed_tests:
        print("Failed tests:")
        print("-" * 30)
        for result in failed_tests:
            print(f"{result['name']:30} | {result['error'][:50]}...")
        print()
    
    return len(failed_tests) == 0


async def test_algorithm_endpoints():
    """Test the algorithm information endpoints."""
    base_url = "http://localhost:8000"
    
    print("Testing algorithm endpoints...")
    print("=" * 50)
    
    try:
        async with aiohttp.ClientSession() as session:
            # Test health endpoint
            async with session.get(f"{base_url}/health") as response:
                if response.status == 200:
                    health = await response.json()
                    print("✅ Health endpoint working")
                    print(f"   Available algorithms: {health.get('available_generation_algorithms', [])}")
                else:
                    print(f"❌ Health endpoint error: {response.status}")
            
            # Test algorithms endpoint  
            async with session.get(f"{base_url}/generation/algorithms") as response:
                if response.status == 200:
                    algorithms = await response.json()
                    print("✅ Algorithms endpoint working")
                    print(f"   Registered algorithms: {len(algorithms.get('algorithms', []))}")
                    for algo in algorithms.get('algorithms', []):
                        status = "✅" if algo.get('available') else "❌"
                        print(f"     {status} {algo['name']}: {algo['description']}")
                else:
                    print(f"❌ Algorithms endpoint error: {response.status}")
                    
    except Exception as e:
        print(f"❌ Error testing endpoints: {e}")


async def main():
    """Main test function that runs both sync and async tests."""
    print("🚀 **LLADA FAST-dLLM INTEGRATION TEST**")
    print("=" * 80)
    
    # Test algorithm endpoints first
    await test_algorithm_endpoints()
    print()
    
    # Run the main integration test
    success = test_fast_dllm_integration()
    
    print("=" * 80)
    if success:
        print("🎉 All Fast-dLLM integration tests passed!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    print("=" * 80)
    
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
