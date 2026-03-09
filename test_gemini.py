#!/usr/bin/env python3
"""
Simple test script to verify Gemini API connectivity.
Run: 
     python test_gemini.py && echo "Gemini OK" || echo "Gemini failed"
     python test_gemini.py          # connectivity test
     python test_gemini.py --chat   # interactive chat

Requires GOOGLE_API_KEY or GEMINI_API_KEY in .env or environment.
"""

import os
import sys
from pathlib import Path

# Load .env if present (optional)
_env = Path(__file__).parent / ".env"
if _env.exists():
    for line in _env.read_text().strip().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"\''))

def get_client_and_model():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Missing API key. Set GOOGLE_API_KEY or GEMINI_API_KEY in .env or environment.")
        return None, None
    model = os.getenv("MODEL", "gemini-2.0-flash")
    return api_key, model

def test_connectivity():
    api_key, model = get_client_and_model()
    if not api_key:
        return 1

    print(f"Testing Gemini connectivity (model: {model})...")
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents="Reply with exactly: OK",
        )
        text = response.text.strip() if response.text else "(empty)"
        print(f"✅ Connected. Response: {text}")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

def chat():
    api_key, model = get_client_and_model()
    if not api_key:
        return 1

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"❌ Failed to create client: {e}")
        return 1

    history = []
    print(f"Chat with Gemini ({model}). Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        history.append({"role": "user", "parts": [{"text": user_input}]})
        try:
            response = client.models.generate_content(
                model=model,
                contents=history,
            )
            text = response.text.strip() if response.text else "(no response)"
            print(f"Gemini: {text}\n")
            history.append({"role": "model", "parts": [{"text": text}]})
        except Exception as e:
            print(f"❌ Error: {e}\n")

    return 0

def main():
    if "--chat" in sys.argv or "-c" in sys.argv:
        return chat()
    return test_connectivity()

if __name__ == "__main__":
    exit(main())