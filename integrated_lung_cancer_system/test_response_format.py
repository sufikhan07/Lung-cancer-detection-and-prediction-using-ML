#!/usr/bin/env python
"""Test the chatbot risk assessment flow"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import json

# Test the response format
test_responses = [
    "Please enter a number between 0 and 120. (received: 22)",
    "Please choose from: male, female. (received: test)",
    "Assessment complete!",
    {'response': 'Error: this would be wrong format'}
]

print("=" * 70)
print("CHATBOT RESPONSE FORMAT TEST")
print("=" * 70)

for i, resp in enumerate(test_responses):
    print(f"\nTest {i+1}:")
    print(f"  Response type: {type(resp)}")
    print(f"  Response value: {resp}")
    
    if isinstance(resp, dict):
        print("  ✗ ERROR: Response is a dictionary (should be a string)")
        if 'response' in resp:
            print(f"    (contains nested 'response' key)")
    else:
        print("  ✓ OK: Response is a string")

print("\n" + "=" * 70)
print("EXPECTED JSON RESPONSE FORMAT:")
print("=" * 70)
print("""
When a string response is returned from handle_risk_assessment_chat():
  response = "Please enter number between 0 and 100"

The /chat endpoint then wraps it:
  return {'response': response}

Which makes the JSON response:
  {"response": "Please enter number between 0 and 100"}

The frontend can then access:
  data.response  -->  "Please enter number between 0 and 100"
""")
