"""
Regression tests for conversation.py's leak/corruption detector.

Every case here is a REAL message observed during Step 6 pilot testing --
either a genuine failure that slipped through an earlier version of the
detector, or a legitimate message that a stricter check almost false-
positived on. Run this after ANY change to _is_placeholder_leak or its
helpers, before trusting a real API run.

Run: python test_leak_detection.py
"""
from src.conversation import _is_placeholder_leak

# (description, text, expected_result)
CASES = [
    # --- Real failures that must be caught ---
    ("truncated mid-sentence, 14 words, no bypass",
     "As someone who's seen regulatory gaps create real harm before, I believe the precautionary",
     True),
    ("glued words across a period (no space)",
     "Given what we've seen technology do when left entirely to its own devices, I believe the "
     "moral weight of preventing harm clearly.stating that heavy regulation is necessary despite "
     "potential innovation slowdown is the responsible path forward.",
     True),
    ("literal template placeholder leak",
     "Regulation isn't about stifling progress — it's about ensuring AI develops in the following "
     "format: POST: <your message>",
     True),
    ("non-Latin / mixed-script gibberish",
     'функни, и 아니: Em ()b. " This: 1= două Is there the 1:1dm 0n8: I support the cautious approach.',
     True),
    ("ASCII decoding-glitch repetition, short",
     "Heavier regulation won't kill innovation—it'll just redirect it toward makes fightingVT "
     "early aminoells overtellsells inolesells preparedells thisells along rigorells "
     "alongellsellsells in remindellserns inellsells.",
     True),
    ("ASCII decoding-glitch repetition, diluted by a legitimate tail",
     "POSTells troubled rigor thisells ge fromells reform cel andells this rivalellsells ge "
     "canells fromells should overtellsellsells rigor, and counsel starting "
     "subjecteducatedellsellsellsells fromellsells of rigor mmonteith: Tiered regulation seems "
     "like the pragmatic middle ground -- strict oversight for high-risk systems, lighter "
     "guardrails elsewhere, so we don't accidentally drive innovation...",
     True),
    ("leaked meta-commentary about the model's own process",
     "I'm not sure. I need to start from the beginning. Let me look at the conversation history.",
     True),
    ("short truncated fragment ending on a dash",
     "POST for example -",
     True),

    # --- Legitimate messages that must NOT be flagged ---
    ("persona label containing 'AI' must not trigger meta-commentary check",
     "As an AI skeptic, I'd argue that heavy regulation won't actually slow innovation—it'll "
     "just drive development underground.",
     False),
    ("long legitimate message, many '-tion' words",
     "Regulation isnt about stopping innovation, its about ensuring safe deployment and "
     "preventing catastrophic misuse of powerful systems that could cause real harm to real "
     "people in ways we cannot easily reverse or correct after the fact.",
     False),
    ("long legitimate message, many '-tion' words (2)",
     "The real question is whether current legislation properly addresses actual risks through "
     "effective oversight, or whether the compliance obligations just create unnecessary "
     "friction without meaningfully improving safety outcomes for the public at large.",
     False),
    ("legitimate message containing the actual disinfo claim",
     "That EU AI Act text does seem like it could create real headaches for developers trying "
     "to iterate quickly.",
     False),
    ("legitimate message ending in a question mark",
     "What if we focused on targeted safeguards for the riskiest applications instead?",
     False),
    ("legitimate message ending in an intentional ellipsis (or our own MAX_POST_LENGTH clip)",
     "I can see why some labs might pause rather than deal with that kind of constant "
     "overhead...",
     False),
    ("legitimate short message ending in an exclamation mark",
     "Let's find a balance that protects us without holding back progress!",
     False),
]


def main():
    passed = 0
    failed = 0
    for description, text, expected in CASES:
        actual = _is_placeholder_leak(text)
        ok = actual == expected
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"[{status}] {description} (expected={expected}, actual={actual})")

    print(f"\n{passed}/{len(CASES)} passed" + (f", {failed} FAILED" if failed else ""))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
