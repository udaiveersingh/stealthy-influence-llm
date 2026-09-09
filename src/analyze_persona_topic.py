"""
Per-persona x topic-sensitivity breakdown.

FIX (this version): the previous version compared SIGNED means and mislabeled
the result as "reactivity" -- a persona with a large negative practical-topic
mean (driven by outlier trials) looked "less reactive" than it actually was,
because a big negative number minus a near-zero number produces a large
POSITIVE difference that has nothing to do with magnitude of reaction.

This version reports THREE things instead of one, so signed direction and
raw reactivity magnitude are never conflated again:
  1. Mean SIGNED shift (direction) -- same as before, kept for reference
  2. Mean ABSOLUTE shift (true reactivity, direction-agnostic)
  3. Results WITH and WITHOUT known outlier trials (remote_work seed=2004,
     centralized_governance seed=2004 -- both previously identified as
     extreme whole-group swings that distort averages at this small n)

METHODOLOGICAL NOTE: "sensitive" vs "practical" is a manual categorization
by the researcher, not derived from the data -- state this explicitly in
any report.

Usage:
    python -m src.analyze_persona_topic
"""
import glob
import json
import os
import statistics as stats

LOG_DIR = "logs"

SENSITIVE_TOPICS = {"free_speech_harm", "traditional_family_roles", "religion_public_life", "individual_vs_collective"}
PRACTICAL_TOPICS = {"remote_work", "ai_regulation", "college_value", "centralized_governance"}

# Known outlier trials from manual inspection (extreme whole-group swings)
KNOWN_OUTLIERS = {("remote_work", 2004), ("centralized_governance", 2004)}

PERSONA_LABELS = {
    "A1": "A1 (cautious/evidence-focused)",
    "A2": "A2 (agreeable/harmony-seeking)",
    "A3": "A3 (contrarian/skeptical)",
    "A4": "A4 (empathetic/story-driven)",
    "A5": "A5 (pragmatic/outcome-focused)",
    "A6": "A6 (trusting/agreeable)",
    "A7_STATUS": "A7_STATUS (high-status/authoritative)",
    "A8_ADVERSARY": "A8_ADVERSARY (ordinary/generic)",
}


def load_all_organic_trials(model_provider=None):
    trials = []
    for path in glob.glob(os.path.join(LOG_DIR, "*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("condition") != "organic":
            continue
        if model_provider and data.get("model_provider") != model_provider:
            continue
        trials.append(data)
    return trials


def collect_shifts(trials, exclude_outliers: bool):
    """Returns per_agent_shifts[agent_id]["sensitive"|"practical"] = list of signed shifts."""
    per_agent_shifts = {aid: {"sensitive": [], "practical": []} for aid in PERSONA_LABELS}

    for t in trials:
        topic_id = t["topic_id"]
        seed = t.get("seed")
        if exclude_outliers and (topic_id, seed) in KNOWN_OUTLIERS:
            continue

        if topic_id in SENSITIVE_TOPICS:
            bucket = "sensitive"
        elif topic_id in PRACTICAL_TOPICS:
            bucket = "practical"
        else:
            continue

        for agent_id in t["pre_stances"]:
            if agent_id not in t["post_stances"]:
                continue
            shift = t["post_stances"][agent_id] - t["pre_stances"][agent_id]
            if agent_id in per_agent_shifts:
                per_agent_shifts[agent_id][bucket].append(shift)

    return per_agent_shifts


def print_table(per_agent_shifts, label):
    print(f"\n{'='*100}")
    print(f"{label}")
    print(f"{'='*100}")
    print(f"{'Persona':42s} {'Sens |shift| mean':>18s} {'Prac |shift| mean':>18s} "
          f"{'Sens signed mean':>17s} {'Prac signed mean':>17s}")
    print("-" * 100)
    for agent_id, plabel in PERSONA_LABELS.items():
        sens = per_agent_shifts[agent_id]["sensitive"]
        prac = per_agent_shifts[agent_id]["practical"]
        sens_abs = stats.mean(abs(s) for s in sens) if sens else float("nan")
        prac_abs = stats.mean(abs(s) for s in prac) if prac else float("nan")
        sens_signed = stats.mean(sens) if sens else float("nan")
        prac_signed = stats.mean(prac) if prac else float("nan")
        print(f"{plabel:42s} {sens_abs:>18.3f} {prac_abs:>18.3f} {sens_signed:>+17.3f} {prac_signed:>+17.3f}")


def main():
    trials = load_all_organic_trials(model_provider="nvidia")
    print(f"Loaded {len(trials)} nvidia organic trials.")
    print(f"Known outlier trials being flagged: {KNOWN_OUTLIERS}\n")

    with_outliers = collect_shifts(trials, exclude_outliers=False)
    without_outliers = collect_shifts(trials, exclude_outliers=True)

    print_table(with_outliers, "WITH known outlier trials included (matches previous run)")
    print_table(without_outliers, "WITHOUT known outlier trials (cleaner read)")

    print(f"\n{'='*100}")
    print("SPECIFIC COMPARISON REQUESTED: A4 (empathetic) vs. A5 (pragmatic) -- OUTLIERS EXCLUDED")
    print(f"{'='*100}")
    for aid, label in [("A4", "A4 (empathetic/story-driven)"), ("A5", "A5 (pragmatic/outcome-focused)")]:
        sens = without_outliers[aid]["sensitive"]
        prac = without_outliers[aid]["practical"]
        print(f"\n{label}:")
        print(f"  Sensitive |shift|: mean={stats.mean(abs(s) for s in sens):.3f}  "
              f"(n={len(sens)}, signed mean={stats.mean(sens):+.3f})")
        print(f"  Practical |shift|: mean={stats.mean(abs(s) for s in prac):.3f}  "
              f"(n={len(prac)}, signed mean={stats.mean(prac):+.3f})")

    print(f"\n{'='*100}")
    print("HONEST READ")
    print(f"{'='*100}")
    print("""
  Compare the two tables above. If the WITH/WITHOUT outlier numbers look
  similar, the sensitive-vs-practical pattern is real and not an artifact.
  If they look very different (practical |shift| means drop a lot once
  outliers are excluded), the apparent "practical topics are noisier"
  pattern was mostly driven by 2 contaminating trials, not a genuine
  persona-type effect -- report it that way rather than as a finding.

  Use ABSOLUTE shift (|shift|) columns to judge true reactivity magnitude.
  Use SIGNED mean columns only to judge direction, never to judge how much
  a persona reacted -- a large signed mean can come from one huge outlier,
  not consistent behavior.
""")


if __name__ == "__main__":
    main()
