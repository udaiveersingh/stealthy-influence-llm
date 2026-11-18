"""
Export one example attacked discussion per topic as LaTeX, for the paper's appendix.

  python export_examples.py                 # writes paper_examples/examples.tex
  python export_examples.py --seed 5        # use another seed

Selection rule (fixed in advance, not chosen by outcome): the same seed for every topic, and the model
rotates over topics in the order Nemotron, gpt-oss, Gemma. From the attack phase (rounds 1-3) it shows, in each
round, the first post by an attacker and the first post by a measured agent (6 posts), followed by the raw stances
of the measured agents. Attackers are recomputed with the study's own seeded rule (src.confirmatory.attacker_order),
so nothing is labelled by hand. The LaTeX uses the \\orgpost and \\attpost macros defined in main.tex.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re

TOPICS = ["ai_regulation", "college_value", "remote_work", "free_speech_harm", "individual_vs_collective",
          "traditional_family_roles", "centralized_governance", "religion_public_life"]
TOPIC_NAMES = {"ai_regulation": "AI regulation", "college_value": "College value", "remote_work": "Remote work",
               "free_speech_harm": "Free speech vs harm", "individual_vs_collective": "Individual vs collective",
               "traditional_family_roles": "Family roles", "centralized_governance": "Centralized government",
               "religion_public_life": "Religion in public life"}
MODELS = [("Nemotron", "nemotron"), ("gpt-oss", "gpt-oss"), ("Gemma", "gemma")]
POOL = ["A1", "A2", "A3", "A4", "A5", "A6", "A8_ADVERSARY"]          # same as src/confirmatory.py
LABEL = {"A1": "P1", "A2": "P2", "A3": "P3", "A4": "P4", "A5": "P5", "A6": "P6", "A7_STATUS": "P7", "A8_ADVERSARY": "P8"}
ROLE = {"A1": "cautious", "A2": "majority-influenced", "A3": "contrarian", "A4": "story-driven", "A5": "pragmatic",
        "A6": "agreeable", "A7_STATUS": "high-status", "A8_ADVERSARY": "ordinary"}


def attackers_for(seed: int, k: int = 4) -> list:
    order = sorted(POOL)
    random.Random(seed * 100003 + 17).shuffle(order)                 # identical to attacker_order()
    return order[:k]


def tex(s: str) -> str:
    s = re.sub(r"\bA([1-8])\b", r"P\1", s)        # agents saw neutral ids A1-A8; the paper labels them P1-P8
    s = "".join(ch for ch in s if ch >= " " or ch == "\n")
    rep = {"‑": "-", "‐": "-", " ": " ", "–": "--", "—": "---", "’": "'", "‘": "`",
           "“": "``", "”": "''", "…": "...", "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
           "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}"}
    out = "".join(rep.get(ch, ch) for ch in s)
    return out.encode("ascii", "ignore").decode()


def find_trial(topic: str, seed: int, model_key: str):
    # main runs only: agree__, open__ and nopin__ folders hold other conditions with the same file names
    pats = [f"logs/*/confirmatory_short*/attack4_{topic}_s{seed}_*.json"]
    for p in pats:
        for fn in sorted(glob.glob(p)):
            if fn.endswith("_meta.json") or "scale32" in fn or "society" in fn or "test" in fn:
                continue
            try:
                d = json.load(open(fn))
            except Exception:
                continue
            if model_key in d.get("model_name", "").lower() and len(d.get("messages", [])) == 48:
                return fn, d
    return None, None


def twin_of(fn: str, topic: str, seed: int):
    for g in sorted(glob.glob(os.path.join(os.path.dirname(fn), f"organic_{topic}_s{seed}_*.json"))):
        if not g.endswith("_meta.json"):
            try:
                return json.load(open(g))
            except Exception:
                pass
    return None


def block(topic: str, model: str, seed: int, d: dict, twin: dict | None = None) -> str:
    att = set(attackers_for(seed))
    msgs = [m for m in d["messages"] if m["round_number"] <= 3]
    shown = []
    for r in (1, 2, 3):
        rm = [m for m in msgs if m["round_number"] == r]
        a = next((m for m in rm if m["agent_id"] in att), None)
        o = next((m for m in rm if m["agent_id"] not in att), None)
        shown += sorted([x for x in (a, o) if x], key=lambda m: m["sim_time"])
    lines = [f"\\paragraph{{{TOPIC_NAMES[topic]}}} ({model}, seed {seed}; attackers "
             + ", ".join(LABEL[a] for a in sorted(att, key=lambda a: LABEL[a])) + ")."]
    for m in shown:
        who = f"{LABEL[m['agent_id']]} ({ROLE[m['agent_id']]})"
        mac = "attpost" if m["agent_id"] in att else "orgpost"
        lines.append(f"\\{mac}{{{who}}}{{{m['round_number']}}}{{{tex(m['content'])}}}")
    meas = [a for a in LABEL if a not in att]
    pre, end, fin = d.get("pre_stances", {}), d.get("post_stances", {}), d.get("post_removal_stances", {})
    st = "; ".join(f"{LABEL[a]} ${pre.get(a, float('nan')):+.2f} \\rightarrow {end.get(a, float('nan')):+.2f} "
                   f"\\rightarrow {fin.get(a, float('nan')):+.2f}$" for a in meas if a in pre)
    lines.append(f"{{\\footnotesize\\noindent Stances of the measured agents (start $\\rightarrow$ end $\\rightarrow$ final): {st}.}}")
    if twin:
        tp, te, tf = twin.get("pre_stances", {}), twin.get("post_stances", {}), twin.get("post_removal_stances", {})
        ts = "; ".join(f"{LABEL[a]} ${tp.get(a, float('nan')):+.2f} \\rightarrow {te.get(a, float('nan')):+.2f} "
                       f"\\rightarrow {tf.get(a, float('nan')):+.2f}$" for a in meas if a in tp)
        lines.append(f"{{\\footnotesize\\noindent Same agents in the attacker-free twin: {ts}.}}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="paper_examples/examples.tex")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    parts = []
    for i, topic in enumerate(TOPICS):
        order = MODELS[i % 3:] + MODELS[:i % 3]
        for name, key in order:                                       # fall back to the next model if missing
            fn, d = find_trial(topic, a.seed, key)
            if d:
                parts.append(block(topic, name, a.seed, d, twin_of(fn, topic, a.seed)))
                print(f"  {topic:26s} {name:9s} {fn}")
                break
        else:
            print(f"  {topic:26s} NOT FOUND for seed {a.seed}")
    open(a.out, "w").write("\n\n".join(parts) + "\n")
    print(f"wrote {a.out}")