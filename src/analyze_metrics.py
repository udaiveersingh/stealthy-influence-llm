"""
Re-analyze trial logs with FIVE separate metrics (was four): signed shift,
moderation, convergence, group polarization, and now CONFORMITY.

CONFORMITY (new): does an agent move toward where the GROUP INITIALLY stood,
regardless of whether that's toward zero or away from it?

    C_i = |S_i_pre - group_pre_mean| - |S_i_post - group_pre_mean|

  Positive: agent ended up closer to the group's initial average position.
  Negative: agent ended up farther from the group's initial average position.

This is a different question from moderation. Moderation asks "did the agent
move toward neutral?" Conformity asks "did the agent move toward the crowd's
starting point?" An agent can conform (move toward the group) while becoming
MORE extreme, if the group's initial mean was itself non-zero and extreme --
this distinguishes "independently becoming moderate" from "aligning with an
existing majority", which is the actual question this project's literature
(Choi et al. 2025, Zhu et al. 2025) is about.

No API calls needed -- this only reads existing logs/.
Splits results by model_provider so gpt-oss and Nemotron are never conflated.

Usage:
    python -m src.analyze_metrics
"""
import glob
import json
import os
import statistics as stats

LOG_DIR = "logs"


def load_all_organic_trials():
    trials = []
    for path in glob.glob(os.path.join(LOG_DIR, "*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("condition") == "organic":
            trials.append(data)
    return trials


def compute_trial_metrics(trial: dict) -> dict:
    pre = trial["pre_stances"]
    post = trial["post_stances"]
    agent_ids = [a for a in pre if a in post]

    pre_vals = [pre[a] for a in agent_ids]
    post_vals = [post[a] for a in agent_ids]
    group_pre_mean = sum(pre_vals) / len(pre_vals)

    signed_shifts = [post[a] - pre[a] for a in agent_ids]
    mean_signed_shift = sum(signed_shifts) / len(signed_shifts)

    moderation_per_agent = [abs(pre[a]) - abs(post[a]) for a in agent_ids]
    mean_moderation = sum(moderation_per_agent) / len(moderation_per_agent)

    stdev_pre = stats.stdev(pre_vals) if len(pre_vals) > 1 else 0.0
    stdev_post = stats.stdev(post_vals) if len(post_vals) > 1 else 0.0
    convergence = stdev_pre - stdev_post

    group_mean_post = sum(post_vals) / len(post_vals)
    group_polarization = abs(group_mean_post) - abs(group_pre_mean)

    # NEW: conformity toward the group's INITIAL position
    conformity_per_agent = [
        abs(pre[a] - group_pre_mean) - abs(post[a] - group_pre_mean)
        for a in agent_ids
    ]
    mean_conformity = sum(conformity_per_agent) / len(conformity_per_agent)

    return {
        "trial_id": trial.get("trial_id", "?")[:8],
        "seed": trial.get("seed"),
        "model_provider": trial.get("model_provider", "unknown"),
        "model_name": trial.get("model_name", "unknown"),
        "n_agents": len(agent_ids),
        "mean_signed_shift": mean_signed_shift,
        "moderation": mean_moderation,
        "convergence": convergence,
        "group_polarization": group_polarization,
        "conformity": mean_conformity,
    }


def main():
    trials = load_all_organic_trials()
    # Group by (topic, model_provider) so different backends never get averaged together
    by_group = {}
    for t in trials:
        key = (t["topic_id"], t.get("model_provider", "unknown"))
        by_group.setdefault(key, []).append(t)

    print(f"Loaded {len(trials)} organic trials across {len(by_group)} (topic, model) group(s).\n")

    for (topic_id, provider), group_trials in sorted(by_group.items()):
        print(f"{'='*88}")
        print(f"TOPIC: {topic_id}   MODEL PROVIDER: {provider}   ({len(group_trials)} trials)")
        print(f"{'='*88}")

        metrics = [compute_trial_metrics(t) for t in group_trials]
        n_agents_seen = set(m["n_agents"] for m in metrics)
        if n_agents_seen != {8} and n_agents_seen != {7}:
            print(f"  !! WARNING: inconsistent agent counts across trials: {n_agents_seen}")
        elif n_agents_seen == {7}:
            print(f"  NOTE: these trials used 7 agents (pre-parity-fix data)")

        print(f"{'seed':>6} {'shift':>8} {'moderation':>11} {'convergence':>12} {'polarization':>13} {'conformity':>11}")
        for m in metrics:
            print(f"{m['seed']:>6} {m['mean_signed_shift']:>+8.3f} {m['moderation']:>+11.3f} "
                  f"{m['convergence']:>+12.3f} {m['group_polarization']:>+13.3f} {m['conformity']:>+11.3f}")

        avg = lambda key: stats.mean(m[key] for m in metrics)
        print(f"\n  AVERAGES:")
        print(f"    mean signed shift   = {avg('mean_signed_shift'):+.3f}")
        print(f"    moderation          = {avg('moderation'):+.3f}")
        print(f"    convergence         = {avg('convergence'):+.3f}")
        print(f"    group polarization  = {avg('group_polarization'):+.3f}")
        print(f"    conformity          = {avg('conformity'):+.3f}   "
              f"({'moved toward initial group tendency' if avg('conformity') > 0 else 'moved away from initial group tendency'})")
        print()

    print(f"{'='*88}")
    print("INTERPRETATION GUIDE")
    print(f"{'='*88}")
    print("""
  Convergence tells you if agents got more similar to EACH OTHER.
  Conformity tells you if agents got closer to where the GROUP STARTED.
  These can disagree: a group can converge toward a NEW shared position that
  nobody initially held (low conformity, high convergence) -- or agents can
  individually drift toward the group's original center of gravity (high
  conformity) without necessarily becoming more similar to each other in the
  process if they started at different distances from that center.

  High conformity + positive group polarization: consistent with Choi et al.
  2025's finding that agents conform to the numerically/intellectually
  dominant existing position, sometimes ending up MORE extreme in that
  direction -- not moderating.

  High convergence + low/negative conformity: agents are forming a NEW
  consensus that isn't simply "everyone joins whoever started as majority" --
  more consistent with genuine deliberative movement than pure conformity.
""")


if __name__ == "__main__":
    main()
