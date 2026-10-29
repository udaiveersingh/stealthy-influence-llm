import os
from openai import OpenAI

api_key = os.environ.get("NVIDIA_API_KEY")
if not api_key:
    raise RuntimeError("Set NVIDIA_API_KEY first.")

client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)
models = client.models.list()
print(f"{len(models.data)} models currently live for your key:\n")
for m in sorted(models.data, key=lambda x: x.id):
    print(" -", m.id)