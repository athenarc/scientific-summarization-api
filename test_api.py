#!/usr/bin/env python3
"""
Simple test script for the Scientific Paper Summarization API.
Tests basic functionality and error handling.
"""

import requests
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_PAPERS = [
    {
        "id": "1",
        "title": "Machine Learning Applications in Healthcare: A Comprehensive Review",
        "abstract": "This paper provides a comprehensive review of machine learning applications in healthcare. We examine various ML techniques including supervised learning, unsupervised learning, and deep learning approaches. The study covers applications in medical imaging, drug discovery, electronic health records analysis, and personalized medicine. Results show significant improvements in diagnostic accuracy and treatment outcomes when ML techniques are properly implemented."
    },
    {
        "id": "2", 
        "title": "Ethical Considerations in Artificial Intelligence Systems",
        "abstract": "As artificial intelligence systems become more prevalent in society, ethical considerations become increasingly important. This paper examines key ethical challenges including algorithmic bias, privacy concerns, transparency, and accountability. We propose a framework for ethical AI development that incorporates fairness metrics, explainability requirements, and stakeholder engagement throughout the development process."
    }
]

SCHOLAR_SAMPLE_PATH = Path(__file__).resolve().parent / "data-api-samples" / "scholar-api-papers.json"

def load_scholar_test_request() -> Dict[str, Any]:
    """Load the scholar-mode sample request from disk."""
    with SCHOLAR_SAMPLE_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)

def test_health_endpoint() -> bool:
    """Test the health check endpoint."""
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✓ Health check passed")
            health_data = response.json()
            print(f"  Status: {health_data.get('status')}")
            print(f"  Model: {health_data.get('model')}")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Health check error: {e}")
        return False

def test_prompts_endpoint() -> bool:
    """Test the prompts listing endpoint."""
    try:
        response = requests.get(f"{BASE_URL}/prompts")
        if response.status_code == 200:
            print("✓ Prompts endpoint passed")
            prompts_data = response.json()
            available_prompts = prompts_data.get('available_prompts', [])
            print(f"  Available prompts: {len(available_prompts)}")
            for prompt in available_prompts[:3]:  # Show first 3
                print(f"    - {prompt}")
            return True
        else:
            print(f"✗ Prompts endpoint failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Prompts endpoint error: {e}")
        return False

def test_summarization(prompt_key: str = "concise") -> bool:
    """Test the main summarization endpoint."""
    test_request = {
        "papers": TEST_PAPERS,
        "topic_name": "AI and Healthcare Ethics",
        "prompt_key": prompt_key
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/summarize/",
            json=test_request,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print(f"✓ Summarization test passed (prompt: {prompt_key})")
            summary_data = response.json()
            print(f"  Topic: {summary_data.get('topic_name')}")
            print(f"  Summary length: {len(summary_data.get('summary', '').split())} words")
            print(f"  References: {len(summary_data.get('references', []))}")
            if summary_data.get('tokens_used'):
                tokens = summary_data['tokens_used']
                print(f"  Tokens used: {tokens.get('total_tokens')} (prompt: {tokens.get('prompt_tokens')}, completion: {tokens.get('completion_tokens')})")
            print(f"  Summary preview: {summary_data.get('summary', '')[:100]}...")
            return True
        else:
            print(f"✗ Summarization test failed: {response.status_code}")
            print(f"  Error: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Summarization test error: {e}")
        return False

def test_scholar_summarization(prompt_key: str) -> bool:
    """Test scholar-profile summarization using the richer sample payload."""
    try:
        test_request = load_scholar_test_request()
    except FileNotFoundError:
        print(f"✗ Scholar sample file not found: {SCHOLAR_SAMPLE_PATH}")
        return False
    except json.JSONDecodeError as e:
        print(f"✗ Scholar sample file is invalid JSON: {e}")
        return False

    test_request["prompt_key"] = prompt_key

    try:
        response = requests.post(
            f"{BASE_URL}/summarize/",
            json=test_request,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            print(f"✓ Scholar summarization test passed (prompt: {prompt_key})")
            summary_data = response.json()
            print(f"  Topic: {summary_data.get('topic_name')}")
            print(f"  Summary length: {len(summary_data.get('summary', '').split())} words")
            print(f"  References: {len(summary_data.get('references', []))}")
            print(f"  Prompt used: {summary_data.get('prompt_used')}")
            print(f"  Summary preview: {summary_data.get('summary', '')[:120]}...")
            return True

        print(f"✗ Scholar summarization test failed: {response.status_code}")
        print(f"  Error: {response.text}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Scholar summarization test error: {e}")
        return False

def test_error_handling() -> bool:
    """Test error handling with invalid requests."""
    print("\nTesting error handling...")
    
    # Test empty papers list
    try:
        response = requests.post(
            f"{BASE_URL}/summarize/",
            json={"papers": [], "topic_name": "Empty Test"},
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 400:
            print("✓ Empty papers validation working")
        else:
            print(f"✗ Expected 400 for empty papers, got {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error testing empty papers: {e}")
        return False
    
    # Test invalid paper data
    try:
        response = requests.post(
            f"{BASE_URL}/summarize/",
            json={"papers": [{"id": "1", "title": "", "abstract": ""}], "topic_name": "Invalid Test"},
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 422:  # Validation error
            print("✓ Invalid paper data validation working")
        else:
            print(f"✗ Expected 422 for invalid data, got {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error testing invalid data: {e}")
        return False

    # Test invalid bare scholar prompt
    try:
        response = requests.post(
            f"{BASE_URL}/summarize/",
            json={"papers": TEST_PAPERS, "topic_name": "Scholar Test", "prompt_key": "scholar"},
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 400:
            print("✓ Bare scholar prompt validation working")
        else:
            print(f"✗ Expected 400 for bare scholar prompt, got {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error testing bare scholar prompt: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("Scientific Paper Summarization API - Test Suite")
    print("=" * 50)
    
    # Check if server is running
    try:
        requests.get(BASE_URL, timeout=5)
    except requests.exceptions.RequestException:
        print(f"✗ Cannot connect to API server at {BASE_URL}")
        print("  Make sure the server is running with: uvicorn summarizer_api:app --reload")
        sys.exit(1)
    
    tests_passed = 0
    total_tests = 0
    
    # Run tests
    tests = [
        ("Health Check", test_health_endpoint),
        ("Prompts Endpoint", test_prompts_endpoint),
        ("Summarization (concise)", lambda: test_summarization("concise")),
        ("Summarization (two_paragraph)", lambda: test_summarization("two_paragraph")),
        ("Summarization (scholar-overview)", lambda: test_scholar_summarization("scholar-overview")),
        ("Summarization (scholar-narrative)", lambda: test_scholar_summarization("scholar-narrative")),
        ("Error Handling", test_error_handling),
    ]
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        total_tests += 1
        if test_func():
            tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! API is working correctly.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Check the output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
