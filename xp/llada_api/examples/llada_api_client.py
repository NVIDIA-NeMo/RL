#!/usr/bin/env python3
"""
Example client for the LLaDA OpenAI API server.

This script demonstrates how to use the LLaDA API server with both
streaming and non-streaming requests.
"""

import asyncio
import json
import time
from typing import AsyncGenerator

import aiohttp
import requests


class LLaDAAPIClient:
    """Client for interacting with the LLaDA OpenAI API server."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        
    def chat_completion(
        self, 
        messages, 
        model="llada-8b-instruct",
        temperature=0.0,
        max_tokens=128,
        steps=64,
        block_length=64,
        cfg_scale=0.0,
        remasking="low_confidence",
        stream=False
    ):
        """Send a chat completion request to the API."""
        
        url = f"{self.base_url}/v1/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "steps": steps,
            "block_length": block_length,
            "cfg_scale": cfg_scale,
            "remasking": remasking,
            "stream": stream
        }
        
        if stream:
            return self._stream_request(url, payload)
        else:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
    
    def _stream_request(self, url: str, payload: dict) -> AsyncGenerator[dict, None]:
        """Handle streaming requests."""
        async def _async_stream():
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    response.raise_for_status()
                    
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        
                        if line.startswith('data: '):
                            data = line[6:]  # Remove 'data: ' prefix
                            
                            if data == '[DONE]':
                                break
                                
                            try:
                                yield json.loads(data)
                            except json.JSONDecodeError:
                                continue
        
        return _async_stream()
    
    def list_models(self):
        """List available models."""
        url = f"{self.base_url}/v1/models"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    
    def health_check(self):
        """Check server health."""
        url = f"{self.base_url}/health"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()


def example_basic_chat():
    """Example: Basic chat completion."""
    print("=== Basic Chat Completion Example ===")
    
    client = LLaDAAPIClient()
    
    # Check if server is healthy
    try:
        health = client.health_check()
        print(f"Server health: {health}")
        
        if not health.get("model_loaded"):
            print("❌ Model not loaded on server!")
            return
            
    except Exception as e:
        print(f"❌ Server not accessible: {e}")
        return
    
    messages = [
        {"role": "user", "content": "What is the future of artificial intelligence?"}
    ]
    
    print(f"Input: {messages[0]['content']}")
    print("\nGenerating response...")
    
    try:
        start_time = time.time()
        response = client.chat_completion(
            messages=messages,
            temperature=0.0,
            max_tokens=100,
            steps=32,  # Fewer steps for faster generation
            block_length=50
        )
        
        end_time = time.time()
        
        print(f"\n✅ Response (took {end_time - start_time:.2f}s):")
        print(f"📝 {response['choices'][0]['message']['content']}")
        print(f"\n📊 Token usage: {response['usage']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_streaming_chat():
    """Example: Streaming chat completion."""
    print("\n=== Streaming Chat Completion Example ===")
    
    client = LLaDAAPIClient()
    
    messages = [
        {"role": "user", "content": "Write a short story about a robot learning emotions."}
    ]
    
    print(f"Input: {messages[0]['content']}")
    print("\n🔄 Streaming response:")
    
    async def run_streaming():
        try:
            stream = client.chat_completion(
                messages=messages,
                temperature=0.3,  # More creative
                max_tokens=150,
                steps=64,
                stream=True
            )
            
            print("📝 ", end="", flush=True)
            
            async for chunk in stream:
                if chunk.get("choices") and chunk["choices"][0].get("delta", {}).get("content"):
                    content = chunk["choices"][0]["delta"]["content"]
                    print(content, end="", flush=True)
            
            print("\n\n✅ Streaming completed!")
            
        except Exception as e:
            print(f"\n❌ Streaming error: {e}")
    
    asyncio.run(run_streaming())


def example_different_parameters():
    """Example: Testing different LLaDA-specific parameters."""
    print("\n=== LLaDA Parameter Comparison ===")
    
    client = LLaDAAPIClient()
    
    messages = [
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ]
    
    # Test different parameter combinations
    parameter_sets = [
        {
            "name": "Fast (Low Steps)",
            "steps": 16,
            "temperature": 0.0,
            "cfg_scale": 0.0,
            "max_tokens": 80
        },
        {
            "name": "Quality (High Steps)",
            "steps": 128,
            "temperature": 0.0,
            "cfg_scale": 0.0,
            "max_tokens": 80
        },
        {
            "name": "Creative (High Temp)",
            "steps": 64,
            "temperature": 0.8,
            "cfg_scale": 0.0,
            "max_tokens": 80
        },
        {
            "name": "Guided (CFG)",
            "steps": 64,
            "temperature": 0.2,
            "cfg_scale": 1.5,
            "max_tokens": 80
        }
    ]
    
    print(f"Input: {messages[0]['content']}\n")
    
    for params in parameter_sets:
        print(f"🔧 Testing: {params['name']}")
        print(f"   Steps: {params['steps']}, Temp: {params['temperature']}, CFG: {params['cfg_scale']}")
        
        try:
            start_time = time.time()
            response = client.chat_completion(
                messages=messages,
                temperature=params['temperature'],
                max_tokens=params['max_tokens'],
                steps=params['steps'],
                cfg_scale=params['cfg_scale']
            )
            end_time = time.time()
            
            content = response['choices'][0]['message']['content']
            print(f"   ⏱️  Time: {end_time - start_time:.2f}s")
            print(f"   📝 Response: {content[:100]}{'...' if len(content) > 100 else ''}")
            print()
            
        except Exception as e:
            print(f"   ❌ Error: {e}\n")


def example_list_models():
    """Example: List available models."""
    print("=== Available Models ===")
    
    client = LLaDAAPIClient()
    
    try:
        models = client.list_models()
        print("📋 Available models:")
        
        for model in models["data"]:
            print(f"   • {model['id']} (owned by: {model['owned_by']})")
            
    except Exception as e:
        print(f"❌ Error listing models: {e}")


def main():
    """Run all examples."""
    print("🚀 LLaDA OpenAI API Client Examples")
    print("=" * 50)
    
    # Test server connectivity first
    example_list_models()
    
    # Run examples
    example_basic_chat()
    example_streaming_chat()
    example_different_parameters()


if __name__ == "__main__":
    main()
