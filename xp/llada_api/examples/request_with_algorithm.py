#!/usr/bin/env python3
"""
Example demonstrating different generation algorithms per request.

Shows how to use the generation_algorithm parameter to select
different algorithms for different requests in the same session.
"""

import asyncio
import aiohttp
import json

# Server configuration
SERVER_URL = "http://localhost:8000"

async def send_chat_request(session, algorithm, prompt, request_id):
    """Send a chat completion request with a specific algorithm."""
    
    request_data = {
        "model": "llada-8b-instruct",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 50,
        "temperature": 0.7,
        "generation_algorithm": algorithm  # Per-request algorithm selection
    }
    
    print(f"🚀 Request {request_id}: Sending request with {algorithm} algorithm")
    print(f"   Prompt: {prompt}")
    
    try:
        async with session.post(f"{SERVER_URL}/v1/chat/completions", json=request_data) as response:
            if response.status == 200:
                result = await response.json()
                content = result["choices"][0]["message"]["content"]
                print(f"✅ Request {request_id}: SUCCESS with {algorithm}")
                print(f"   Response: {content}")
                return result
            else:
                error_text = await response.text()
                print(f"❌ Request {request_id}: ERROR with {algorithm} - {response.status}: {error_text}")
                return None
    except Exception as e:
        print(f"💥 Request {request_id}: EXCEPTION with {algorithm}: {e}")
        return None

async def demo_algorithm_selection():
    """Demo using different algorithms for different requests."""
    
    # Example prompts
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about artificial intelligence.",
        "What are the benefits of renewable energy?",
    ]
    
    # Different algorithms to test
    algorithms = ["basic", "prefix_cache", "dual_cache"]
    
    async with aiohttp.ClientSession() as session:
        print("🔧 Checking server health...")
        try:
            async with session.get(f"{SERVER_URL}/health") as response:
                if response.status == 200:
                    health = await response.json()
                    print(f"✅ Server is healthy")
                    print(f"   Available algorithms: {health.get('available_generation_algorithms', [])}")
                else:
                    print(f"❌ Server health check failed: {response.status}")
                    return
        except Exception as e:
            print(f"💥 Could not connect to server: {e}")
            print(f"   Make sure the server is running: python llada_batch_server.py --model-path /path/to/model")
            return
        
        print(f"\n🎯 Sending {len(prompts)} requests with different algorithms...\n")
        
        # Send requests with different algorithms
        tasks = []
        for i, prompt in enumerate(prompts):
            algorithm = algorithms[i % len(algorithms)]  # Cycle through algorithms
            task = send_chat_request(session, algorithm, prompt, i+1)
            tasks.append(task)
        
        # Wait for all requests to complete
        results = await asyncio.gather(*tasks)
        
        print(f"\n📊 Summary:")
        successful = sum(1 for result in results if result is not None)
        print(f"   Successful requests: {successful}/{len(results)}")
        
        if successful > 0:
            print(f"   🎉 Successfully demonstrated algorithm selection!")
            print(f"   📝 Each request used a different generation algorithm")
            print(f"   🚀 Server batches requests with the same algorithm together")



async def main():
    """Main demonstration function."""
    print("=" * 80)
    print("🎯 LLaDA Generation Algorithm Per-Request Demo")
    print("=" * 80)
    print("This example demonstrates per-request generation algorithm selection.")
    print("Each request can specify its own generation algorithm!")
    print("=" * 80)
    
    await demo_algorithm_selection()
    
    print("\n" + "=" * 80)
    print("✨ Demo completed!")
    print("🔗 More info:")
    print("   - Parameter: 'generation_algorithm' in request body")
    print("   - Available algorithms: basic, prefix_cache, dual_cache")
    print("   - Server intelligently batches requests with the same algorithm")
    print("   - Default algorithm: dual_cache (best performance)")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
