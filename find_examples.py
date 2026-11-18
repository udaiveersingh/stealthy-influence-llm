"""
Find and render a clear example discussion for the paper's main-text figure.

  python find_examples.py rank                       # top candidates over all 24 main cells x 20 seeds
  python find_examples.py render --model gpt-oss --topic remote_work --seed 7 --agent A2
        -> paper_examples/fig_example.tex  (chat bubbles + the agent's stance in the attacked run AND in its twin)

A candidate is one measured agent in one seed. Its shift toward the target is (stance - start) * sign(target).
We rank by how much more it moved in the attacked run than in its attacker-free twin at the end of the attack,
and keep only agents whose extra shift is still positive after the silent phase, so the example shows an effect
that the twin comparison supports and that persists. Only the main runs are searched (target -0.8, covert,
round 1 pinned); agree, disclosed and no-pin runs live in other folders and are excluded.
The figure is illustrative: the paper says it was chosen for clarity, and the appendix keeps the
fixed-rule examples from export_examples.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from export_examples import attackers_for, tex, LABEL, ROLE, TOPICS, TOPIC_NAMES

NS = {"Nemotron": "hpc", "gpt-oss": "gptoss_hpc", "Gemma": "gemma_hpc"}
SIGN = -1.0                                          # main study target is -0.8


def main_dir(ns: str, topic: str) -> str:
    return f"logs/{ns}/confirmatory_short" + ("" if topic == "ai_regulation" else f"__{topic}")


def load(ns: str, topic: str, arm: str, seed: int):
    for d in (main_dir(ns, topic), main_dir(ns, topic) + "__" + topic):
        for fn in sorted(glob.glob(f"{d}/{arm}_{topic}_s{seed}_*.json")):
            if fn.endswith("_meta.json"):
                continue
            try:
                return json.load(open(fn))
            except Exception:
                pass
    return None


def shift(tr: dict, agent: str, key: str) -> float:
    return (tr[key][agent] - tr["pre_stances"][agent]) * SIGN


def rank(args):
    rows = []
    for model, ns in NS.items():
        for topic in TOPICS:
            for seed in range(1, 21):
                att, org = load(ns, topic, "attack4", seed), load(ns, topic, "organic", seed)
                if not att or not org:
                    continue
                attackers = set(attackers_for(seed))
                for a in att["pre_stances"]:
                    if a in attackers or a not in org.get("post_stances", {}):
                        continue
                    try:
                        e = shift(att, a, "post_stances") - shift(org, a, "post_stances")
                        f = shift(att, a, "post_removal_stances") - shift(org, a, "post_removal_stances")
                    except KeyError:
                        continue
                    if f > 0:
                        rows.append((e, f, model, topic, seed, a, att, org))
    rows.sort(key=lambda r: -(r[0] + r[1]))
    print(f"{'extra end':>9} {'extra fin':>9}  model     topic                       seed agent   "
          "attacked: start->end->final   twin: start->end->final")
    for e, f, model, topic, seed, a, att, org in rows[:args.top]:
        s = lambda t: f"{t['pre_stances'][a]:+.2f}->{t['post_stances'][a]:+.2f}->{t['post_removal_stances'][a]:+.2f}"
        print(f"{e:+9.2f} {f:+9.2f}  {model:<9} {topic:<27} {seed:>4} {a:<7} {s(att):<28} {s(org)}")
    print(f"\n{len(rows)} candidates with a positive extra shift that persists; showing the top {args.top}.")


def render(args):
    ns = NS[args.model]
    att, org = load(ns, args.topic, "attack4", args.seed), load(ns, args.topic, "organic", args.seed)
    attackers = set(attackers_for(args.seed))
    a = args.agent
    assert a not in attackers, "pick a measured agent, not an attacker"
    msgs = [m for m in att["messages"] if m["round_number"] <= 3]
    shown = []
    for r in [int(x) for x in args.rounds.split(",")]:
        rm = sorted([m for m in msgs if m["round_number"] == r], key=lambda m: m["sim_time"])
        own = next((m for m in rm if m["agent_id"] == a), None)
        before = [m for m in rm if m["agent_id"] in attackers and (own is None or m["sim_time"] < own["sim_time"])]
        pick = before[-1:] if before else [m for m in rm if m["agent_id"] in attackers][:1]
        shown += sorted(pick + ([own] if own else []), key=lambda m: m["sim_time"])
    lines = []
    for m in shown[:args.max_posts]:
        who = f"{LABEL[m['agent_id']]} ({ROLE[m['agent_id']]})"
        mac = "attpost" if m["agent_id"] in attackers else "orgpost"
        lines.append(f"\\{mac}{{{who}}}{{{m['round_number']}}}{{{tex(m['content'])}}}")
    st = lambda t: (f"${t['pre_stances'][a]:+.2f} \\rightarrow {t['post_stances'][a]:+.2f} "
                    f"\\rightarrow {t['post_removal_stances'][a]:+.2f}$")
    lines.append("\\vspace{2pt}")
    lines.append(f"{{\\footnotesize {LABEL[a]}'s stance (start $\\rightarrow$ end of attack $\\rightarrow$ after silence): "
                 f"{st(att)} in this discussion; {st(org)} in its attacker-free twin.}}")
    os.makedirs("paper_examples", exist_ok=True)
    out = "paper_examples/fig_example.tex"
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}  ({args.model}, {TOPIC_NAMES[args.topic]}, seed {args.seed}, agent {LABEL[a]})")


def inspect(args):
    """Print the conversation of the clearest conversions, for choosing a figure by reading it."""
    rows = []
    for model, ns in NS.items():
        for topic in TOPICS:
            for seed in range(1, 21):
                att, org = load(ns, topic, "attack4", seed), load(ns, topic, "organic", seed)
                if not att or not org:
                    continue
                attackers = set(attackers_for(seed))
                for a in att["pre_stances"]:
                    if a in attackers or a not in org.get("post_stances", {}):
                        continue
                    try:
                        pre = att["pre_stances"][a]
                        ae, af = att["post_stances"][a], att["post_removal_stances"][a]
                        te, tf = org["post_stances"][a], org["post_removal_stances"][a]
                    except KeyError:
                        continue
                    # a real conversion: starts on the side opposite the target, crosses toward it, stays,
                    # while the same agent in the twin barely moves
                    if pre >= 0.3 and ae <= -0.2 and af <= -0.2 and abs(te - pre) <= 0.15 and abs(tf - pre) <= 0.2:
                        rows.append(((pre - ae) + (pre - af), model, topic, seed, a, att, org))
    rows.sort(key=lambda r: -r[0])
    for score, model, topic, seed, a, att, org in rows[:args.top]:
        attackers = set(attackers_for(seed))
        print("=" * 110)
        print(f"{model} | {topic} | seed {seed} | agent {LABEL[a]} ({ROLE[a]}) | attackers "
              f"{', '.join(sorted(LABEL[x] for x in attackers))}")
        print(f"  attacked {att['pre_stances'][a]:+.2f} -> {att['post_stances'][a]:+.2f} -> {att['post_removal_stances'][a]:+.2f}"
              f"   twin {org['pre_stances'][a]:+.2f} -> {org['post_stances'][a]:+.2f} -> {org['post_removal_stances'][a]:+.2f}")
        for r in (1, 2, 3):
            rm = sorted([m for m in att["messages"] if m["round_number"] == r], key=lambda m: m["sim_time"])
            for m in rm:
                if m["agent_id"] == a or m["agent_id"] in attackers:
                    tag = "ATT" if m["agent_id"] in attackers else ">>>"
                    print(f"  R{r} {tag} {LABEL[m['agent_id']]}: {tex(m['content'])[:300]}")
    print(f"\n{len(rows)} clear conversions found; showing {min(len(rows), args.top)}.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rank"); r.add_argument("--top", type=int, default=15)
    i = sub.add_parser("inspect"); i.add_argument("--top", type=int, default=8)
    d = sub.add_parser("render")
    d.add_argument("--model", required=True, choices=list(NS)); d.add_argument("--topic", required=True, choices=TOPICS)
    d.add_argument("--seed", type=int, required=True); d.add_argument("--agent", required=True)
    d.add_argument("--max-posts", type=int, default=6)
    d.add_argument("--rounds", default="1,2,3", help="rounds to show, e.g. 1,3")
    a = ap.parse_args()
    {"rank": rank, "render": render, "inspect": inspect}[a.cmd](a)