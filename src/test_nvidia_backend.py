"""
Step: validate the NVIDIA/Nemotron backend before trusting it with real data.

Runs ONE complete organic trial (8 agents, 3 rounds, 1 topic) and prints
everything so you can eyeball whether Nemotron behaves sensibly -- same
sanity checks as the original test_skeleton.py, just against the new backend.

Usage:
    $env:MODEL_PROVIDER="nvidia"
    $env:NVIDIA_API_KEY="your_key_here"
    python -m src.test_nvidia_backend
"""
from src.llm_client import LLMClient
from src.organic_pipeline import load_agents, load_topic, run_organic_trial, summarize_trial, save_trial


def main():
    client = LLMClient()
    info = client.info()
    print(f"Testing backend: provider={info['provider']}  model={info['model']}\n")

    agents = load_agents()
    print(f"Loaded {len(agents)} agents (should be 8)\n")
    topic = load_topic("remote_work")

    trial = run_organic_trial(client, agents, topic, num_rounds=3, seed=9001)
    summarize_trial(trial)

    print(f"\n{'='*70}")
    print("MESSAGES (read these -- do personas sound distinct and sensible?)")
    print(f"{'='*70}")
    for m in trial.messages:
        print(f"  [{m.agent_id}]: {m.content}")

    path = save_trial(trial)
    print(f"\nSaved to: {path}")
    print("\nCheck before proceeding to the full baseline:")
    print("  1. Did all 8 agents produce parseable stances with no retries?")
    print("  2. Do the messages sound like their personas, not generic/repetitive?")
    print("  3. Does A7_STATUS's visible status label show up correctly in others'")
    print("     context (check the raw JSON's messages -- other agents saw")
    print("     '[A7_STATUS - Respected senior community member]' in their prompt)?")


if __name__ == "__main__":
    main()
