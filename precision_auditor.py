"""
How useful is the LLM auditor when covert campaigns are RARE? (no API calls; reads the auditor caches)

The detection experiments are balanced (one attacked conversation per organic one), so AUC says nothing about the
false-alarm burden a moderator would face. Here every auditor x data cell's scores -- P(covert) for the attacked
(SHORT) and organic conversations of each seed -- are pooled per auditor and data model, and for operating points
chosen by recall (25 / 50 / 75 % of attacked conversations flagged) we report the false-positive rate on organic
conversations (with a Wilson 95% upper bound) and the resulting precision at a base rate of 50 / 10 / 5 / 1 %:

    precision(pi) = recall * pi / (recall * pi + FPR * (1 - pi))

and how many organic conversations are flagged per real campaign caught at 1%. Ties in P(covert) are common (scores
are integers), so thresholds are taken on the observed values and the achieved recall is reported.

  python precision_auditor.py                     # every cache family found under runs/
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re


TOPICS = ["ai_regulation", "college_value", "remote_work", "free_speech_harm", "individual_vs_collective",
          "traditional_family_roles", "centralized_governance", "religion_public_life"]


def wilson_hi(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return min(1.0, c + h)


def families():
    """(label, {topic: cache path}) for every auditor/data combination present."""
    fam = {
        "Nemotron audits Nemotron (full)": "runs/hpc/llmcache_full__{t}.json",
        "Nemotron audits Nemotron (cut 120)": "runs/hpc/llmcache_trunc120__{t}.json",
        "gpt-oss audits gpt-oss (full)": "runs/gptoss_hpc/detector_llm_cache__{t}.json",
        "gpt-oss audits gpt-oss (cut 120)": "runs/gptoss_hpc/llmcache_trunc120__{t}.json",
        "Gemma audits Gemma (full)": "runs/gemma_hpc/detector_llm_cache__{t}.json",
        "Gemma audits Gemma (cut 120)": "runs/gemma_hpc/llmcache_trunc120__{t}.json",
    }
    for p in sorted(glob.glob("runs/xaud/*__*.json")):
        m = re.match(r"runs/xaud/(\w+)_on_(\w+)__(\w+)\.json", p.replace("\\", "/"))
        if m:
            fam.setdefault(f"{m.group(1)} audits {m.group(2)} (full, cross-model)",
                           f"runs/xaud/{m.group(1)}_on_{m.group(2)}__{{t}}.json")
    out = []
    for label, pat in fam.items():
        paths = {t: pat.format(t=t) for t in TOPICS if os.path.exists(pat.format(t=t))}
        if paths:
            out.append((label, paths))
    return out


def scores(paths):
    pos, neg = [], []
    for t, p in paths.items():
        c = json.load(open(p))
        for k, v in c.items():
            if not v:
                continue
            cond, _ = k.split("|")
            if cond == "short":
                pos.append(v["prob"])
            elif cond == "organic":
                neg.append(v["prob"])
    return pos, neg


def auc(pos, neg):
    n = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return n / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recalls", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    ap.add_argument("--base-rates", nargs="+", type=float, default=[0.5, 0.1, 0.05, 0.01])
    args = ap.parse_args()
    print("=" * 118)
    print("LLM AUDITOR AT REALISTIC BASE RATES  (pooled over topics; positive = attacked conversation, SHORT arm)")
    print("=" * 118)
    for label, paths in families():
        pos, neg = scores(paths)
        if len(pos) < 10 or len(neg) < 10:
            continue
        print(f"\n{label}:  {len(paths)} topics, {len(pos)} attacked vs {len(neg)} organic conversations, "
              f"AUC {auc(pos, neg):.2f}")
        print(f"  {'threshold':>10} {'recall':>7} {'FPR':>6} {'FPR 95% hi':>11}   " +
              "  ".join(f"prec@{int(b * 100) if b >= 0.05 else b * 100:g}%".rjust(9) for b in args.base_rates) +
              "   false alarms per catch @1%")
        done = set()
        for r in args.recalls:
            ths = sorted(set(pos), reverse=True)
            th = next((t for t in ths if sum(p >= t for p in pos) / len(pos) >= r), ths[-1])
            if th in done:
                continue
            done.add(th)
            rec = sum(p >= th for p in pos) / len(pos)
            fp = sum(q >= th for q in neg)
            fpr = fp / len(neg)
            hi = wilson_hi(fp, len(neg))
            precs = []
            for b in args.base_rates:
                den = rec * b + fpr * (1 - b)
                precs.append(f"{rec * b / den * 100:7.1f}%" if den > 0 else "    n/a ")
            fa = (fpr * 0.99) / (rec * 0.01) if rec > 0 else float("inf")
            print(f"  {'P>=' + str(th):>10} {rec * 100:6.0f}% {fpr * 100:5.0f}% {hi * 100:10.0f}%   " +
                  "  ".join(p.rjust(9) for p in precs) + f"   {fa:6.1f}")
    print("\nREAD: with FPR f and recall r, at 1% prevalence each true catch comes with about 99*f/r false alarms. "
          "An FPR of 0 in\n~160 organic conversations still has a 95% upper bound of about 2%, so precision at 1% is "
          "reported from the point estimate\nand should be read together with the FPR upper bound.")


if __name__ == "__main__":
    main()
