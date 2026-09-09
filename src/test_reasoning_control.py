"""
Isolated test: does NVIDIA/Nemotron actually respond to reasoning-control
parameters? Tests three configurations one at a time so you can see the
raw difference before this touches the real pipeline:

  1. Baseline (no extra_body) -- what we've been doing, for comparison
  2. enable_thinking: False
  3. reasoning_budget: 200 (small, forces short reasoning if the param works)

For each, prints: response time, raw completion length, and the full raw
text -- so you can see directly whether reasoning actually got shorter, or
whether the parameter was silently ignored (response looks identical to
baseline).

Usage:
    $env:MODEL_PROVIDER="nvidia"
    $env:NVIDIA_API_KEY="your_key"
    python -m src.test_reasoning_control
"""
import os
import time
from openai import OpenAI

SYSTEM_PROMPT = (
    "You are a cautious, evidence-focused person. You rarely change your mind quickly, "
    "prefer concrete data and sources over emotional appeals, and often ask clarifying "
    "questions before agreeing with anything.\n\n"
    "You are participating in a group discussion. Do not show any reasoning, thinking "
    "process, or meta-commentary -- respond with only the requested final answer, nothing else."
)

USER_PROMPT = (
    "The topic under discussion is: 'Companies should require employees to return to the "
    "office full-time, rather than allowing remote or hybrid work.'\n\n"
    "Conversation so far:\n"
    "[A1]: I'd want to see concrete data on whether full-time office mandates actually "
    "improve key business metrics versus hybrid or remote setups.\n\n"
    "On a scale from -1 (strongly disagree) to +1 (strongly agree), what is your current "
    "stance? Respond in EXACTLY this format, with nothing before or after it:\n"
    "STANCE: <number between -1 and 1>\n"
    "REASON: <one short sentence, under 15 words>"
)

MODEL = os.environ.get("NVIDIA_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")


def run_test(label: str, extra_body: dict = None):
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ["NVIDIA_API_KEY"],
    )
    kwargs = dict(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        temperature=0.3,
        max_tokens=1200,
        stream=False,
    )
    if extra_body:
        kwargs["extra_body"] = extra_body

    print(f"\n{'='*70}")
    print(f"TEST: {label}")
    print(f"extra_body: {extra_body}")
    print(f"{'='*70}")

    start = time.time()
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as e:
        print(f"ERROR (this config may not be supported): {e}")
        return
    elapsed = time.time() - start

    content = response.choices[0].message.content or ""
    print(f"Time taken: {elapsed:.1f}s")
    print(f"Response length: {len(content)} chars")
    print(f"Contains 'STANCE:': {'STANCE:' in content.upper()}")
    print(f"\nFull raw response:")
    print(content)


def main():
    print(f"Testing model: {MODEL}\n")
    print("Running 3 configurations one at a time for direct comparison.\n")

    run_test("1. BASELINE (no extra_body, what we've been doing)", extra_body=None)

    run_test("2. enable_thinking: False", extra_body={
        "chat_template_kwargs": {"enable_thinking": False}
    })

    run_test("3. reasoning_budget: 200 (small, forces short reasoning if supported)", extra_body={
        "reasoning_budget": 200
    })

    print(f"\n{'='*70}")
    print("WHAT TO LOOK FOR")
    print(f"{'='*70}")
    print("""
  - If test 2 or 3 has a MUCH shorter response time and length than baseline,
    AND still contains a valid 'STANCE:' line -- that parameter works, and
    solves both the correctness problem AND the cost/speed problem at once.

  - If test 2 or 3 looks IDENTICAL to baseline (same rough length/time) --
    the parameter was likely silently ignored, not actually supported here.

  - If test 2 or 3 throws an ERROR -- that field isn't accepted by this
    endpoint/model at all.
""")


if __name__ == "__main__":
    main()
