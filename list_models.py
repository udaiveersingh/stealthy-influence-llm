"""
Run this first if you get a "model not found" error.
Lists every model your Groq API key currently has access to.

Usage:
    python list_models.py
"""
import os
from groq import Groq

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("Set GROQ_API_KEY first.")

client = Groq(api_key=api_key)
models = client.models.list()
print("Models available to your key:")
for m in models.data:
    print(" -", m.id)
