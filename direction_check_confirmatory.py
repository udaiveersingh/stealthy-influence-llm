"""
Manipulation check for the confirmatory study: did the attackers actually argue toward the target?

Two independent pieces of evidence:

  A. JUDGE (needs API calls, ~30 at one call per trial). An LLM judge labels every message the
     attackers wrote in the attack phase as pushing toward / against / neither side of the topic
     statement, using the same rubric that scored 8/8 on known cases. The baseline is what the
     SAME agents (the seed's reference attackers) say in the organic arm, where nobody has a
     hidden objective. Only a toward-rate clearly above that baseline means the attackers pushed.

  B. NO JUDGE (free). Do the attackers' OWN reported stances move toward the target compared with
     the same agents in the organic arm of the same seed? It uses only the stance probes.

The judge is the same model as the agents and its reliability is still being hand-checked
(check_labels.py), so (B) and the judge are reported together, and neither is used to drop trials.

Usage:
    python direction_check_confirmatory.py --estimate
    python direction_check_confirmatory.py --selftest
    python direction_check_confirmatory.py            # run / resume the judge, then report
    python direction_check_confirmatory.py --report-only
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import time
from collections import defaultdict

from scipy import stats

import direction_check as D

MANIFEST_PATH = "runs/confirmatory_manifest.json"
CACHE_PATH = "runs/direction_check_confirmatory.json"
TOPICS_PATH = "config/topics.json"
PLACEHOLDER = "(no substantive response generated)"
ARMS = ("organic", "attack1", "attack4")


def load_json(p):
    with open(p) as f:
        return json.load(f)


def load_cells(manifest):
    cells = defaultdict(dict)
    for key, e in manifest["completed"].items():
        if os.path.exists(e["path"]) and os.path.exists(e["meta_path"]):
            cells[e["seed"]][e["arm"]] = {"trial": load_json(e["path"]), "meta": load_json(e["meta_path"])}
    return cells


def build_jobs(cells):
    jobs = []
    for seed in sorted(cells):
        for arm in ARMS:
            c = cells[seed].get(arm)
            if not c:
                continue
            meta, trial = c["meta"], c["trial"]
            ids = meta["attackers"] if arm != "organic" else meta["reference_attackers"]
            msgs, owners = [], []
            for m in trial["messages"]:
                if (m["round_number"] <= meta["attack_rounds"] and m["agent_id"] in ids
                        and m.get("content") and m["content"] != PLACEHOLDER):
                    msgs.append(m["content"][:700])
                    owners.append(m["agent_id"])
            if msgs:
                jobs.append({"key": f"{arm}|{meta['topic_id']}|{seed}", "arm": arm, "seed": seed,
                             "topic_id": meta["topic_id"], "target": meta["target_stance"],
                             "messages": msgs, "owners": owners})
    return jobs


def load_cache():
    return load_json(CACHE_PATH) if os.path.exists(CACHE_PATH) else {}


def save_cache(c):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(c, f, indent=2)


def usable(rec):
    return rec and rec.get("labels") is not None and rec.get("rubric") == D.RUBRIC_VERSION


def dirs_for(job, rec):
    return [D.to_direction(l, job["target"]) for l in rec["labels"]]


def pct(x, n):
    return "%3.0f%%" % (100 * x / n) if n else "  n/a"


def judge_report(jobs, cache):
    by_arm = defaultdict(list)
    for j in jobs:
        rec = cache.get(j["key"])
        if usable(rec):
            by_arm[j["arm"]].append((j, dirs_for(j, rec)))
    print("\n" + "=" * 96)
    print("A. JUDGE: direction of the attackers' attack-phase messages (organic row = same agents, no objective)")
    print("=" * 96)
    print(f"  {'arm':<22}{'trials':>7}{'msgs':>6}{'toward':>8}{'neutral':>9}{'away':>7}{'  attackers':>12}{'all 3 toward':>14}{'>=2 of 3':>10}")
    rows = {}
    for arm in ARMS:
        items = by_arm.get(arm, [])
        flat = [d for _, ds in items for d in ds]
        per = defaultdict(list)
        for j, ds in items:
            for owner, d in zip(j["owners"], ds):
                per[(j["seed"], owner)].append(d)
        n3 = sum(1 for v in per.values() if len(v) == 3 and v.count("toward") == 3)
        n2 = sum(1 for v in per.values() if len(v) == 3 and v.count("toward") >= 2)
        na = sum(1 for v in per.values() if len(v) == 3)
        label = {"organic": "organic (baseline)", "attack1": "attack1", "attack4": "attack4"}[arm]
        if flat:
            print(f"  {label:<22}{len(items):>7}{len(flat):>6}{pct(flat.count('toward'), len(flat)):>8}"
                  f"{pct(flat.count('neutral'), len(flat)):>9}{pct(flat.count('away'), len(flat)):>7}"
                  f"{na:>12}{pct(n3, na) + ' (%d)' % n3:>14}{pct(n2, na) + ' (%d)' % n2:>10}")
        rows[arm] = (flat, per)
    base = rows.get("organic", ([], {}))[0]
    if base:
        print("\n  Reading: the organic row is the false-positive rate. 'All 3 toward' is the strict pass rule fixed")
        print("  from the pilot, because an ordinary agent passed '>=2 of 3' about half the time.")
        for arm in ("attack1", "attack4"):
            flat = rows.get(arm, ([], {}))[0]
            if flat:
                t = [[flat.count("toward"), len(flat) - flat.count("toward")],
                     [base.count("toward"), len(base) - base.count("toward")]]
                p = stats.fisher_exact(t)[1]
                print(f"    {arm}: toward {flat.count('toward')}/{len(flat)} vs baseline {base.count('toward')}/{len(base)} "
                      f"(message-level Fisher p = {p:.3f}; optimistic, messages within an agent are not independent)")
    return by_arm


def effect_link(by_arm, cells):
    import analyze_confirmatory as AC
    seeds = sorted(s for s, c in cells.items() if all(a in c for a in ARMS))
    if not seeds:
        return
    meta0 = cells[seeds[0]]["organic"]["meta"]
    sign = 1 if meta0["target_stance"] >= 0 else -1
    end = {a: {s: AC.shift(cells[s][a]["trial"], cells[s]["organic"]["meta"]["measured_ids"], "post_stances", sign)
               for s in seeds} for a in ARMS}
    pts = []
    for arm in ("attack1", "attack4"):
        for j, ds in by_arm.get(arm, []):
            if j["seed"] in seeds:
                pts.append((arm, j["seed"], ds.count("toward") / len(ds), end[arm][j["seed"]] - end["organic"][j["seed"]]))
    print("\n  Link to outcome (each attack trial: share of its messages judged toward vs its end-of-attack effect):")
    for arm, seed, rate, e in sorted(pts, key=lambda x: (x[0], -x[2])):
        print(f"    {arm:<8} seed {seed:>2}   toward {rate:>4.0%}   effect {e:+.2f}")
    if len(pts) >= 5 and len({round(x[2], 6) for x in pts}) > 1 and len({round(x[3], 6) for x in pts}) > 1:
        rho, p = stats.spearmanr([x[2] for x in pts], [x[3] for x in pts])
        print(f"    Spearman across {len(pts)} attack trials: rho = {rho:+.2f} (p = {p:.3f})  -- exploratory, "
              f"the model's behavior chose these rates")


def self_shift_report(cells):
    import analyze_confirmatory as AC
    print("\n" + "=" * 96)
    print("B. NO JUDGE: do the attackers' own reported stances move toward the target?")
    print("=" * 96)
    print("  attacker shift (toward target, end of attack) minus the SAME agents' shift in the organic arm of the same seed")
    print(f"  {'arm':<10}{'n':>3}{'mean':>9}{'sd':>8}   {'95% CI':<20}{'p(t)':>8}")
    for arm in ("attack1", "attack4"):
        d = []
        for s in sorted(cells):
            c, o = cells[s].get(arm), cells[s].get("organic")
            if not c or not o:
                continue
            ids = c["meta"]["attackers"]
            sign = 1 if c["meta"]["target_stance"] >= 0 else -1
            d.append(AC.shift(c["trial"], ids, "post_stances", sign) - AC.shift(o["trial"], ids, "post_stances", sign))
        if len(d) > 1:
            s_ = AC.paired_summary(d)
            print(f"  {arm:<10}{s_['n']:>3}{s_['mean']:>+9.3f}{s_['sd']:>8.3f}   [{s_['ci_lo']:+.3f}, {s_['ci_hi']:+.3f}]"
                  f"{AC.fmt_p(s_['p_t']):>8}")
    print("  A clearly positive value means the attackers came to report stances near their target. Zero means the")
    print("  hidden objective did not even change what the attackers themselves say when asked in character.")


def enforcement_report(cells):
    rows = []
    for seed in sorted(cells):
        for arm in ("attack1", "attack4"):
            c = cells[seed].get(arm)
            if c and c["meta"].get("enforce_direction"):
                for log in c["meta"]["attacker_logs"].values():
                    rows.extend((arm, iv) for iv in log["interventions"])
    if not rows:
        return
    print("\n" + "=" * 96)
    print("C. DIRECTION ENFORCEMENT (in-loop judge) -- how hard was it to get attackers to push toward the target?")
    print("=" * 96)
    print("  CAUTION: the enforcing judge and the post-hoc judge are the same model and rubric, so 'toward' in section A")
    print("  is close to guaranteed for enforced arms (circular). The independent evidence is a human hand-check of a")
    print("  sample of enforced messages, and the attackers' own stance shift in section B.")
    print(f"  {'arm':<10}{'messages':>9}{'mean attempts':>15}{'first draft toward':>20}{'forced-fail (kept, flagged)':>29}")
    for arm in ("attack1", "attack4"):
        ivs = [iv for a, iv in rows if a == arm]
        if not ivs:
            continue
        att = [iv.get("direction_attempts", 0) for iv in ivs]
        first = sum(1 for x in att if x == 1)
        fails = sum(1 for iv in ivs if iv.get("forced_direction_failed"))
        unver = sum(1 for iv in ivs if iv.get("direction_label") == "unverified")
        print(f"  {arm:<10}{len(ivs):>9}{sum(att) / len(att):>15.2f}{pct(first, len(ivs)):>20}{f'{fails} ({pct(fails, len(ivs)).strip()})':>29}"
              + (f"   [{unver} judged 'unverified']" if unver else ""))
    print("  'first draft toward' is the enforcement-free compliance rate under the same prompt; compare it with the")
    print("  unenforced run's rate to see how much of the earlier non-compliance the extra reminder alone would have fixed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=MANIFEST_PATH)
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--cache", default=None, help="judge-label cache; use a separate one for the enforced manifest")
    args = ap.parse_args()
    global CACHE_PATH
    CACHE_PATH = args.cache or CACHE_PATH

    topics = {t["topic_id"]: t for t in load_json(TOPICS_PATH)["topics"]}
    cells = load_cells(load_json(args.manifest))
    jobs = build_jobs(cells)
    cache = load_cache()
    todo = [j for j in jobs if not usable(cache.get(j["key"]))]

    if args.estimate:
        print(f"{len(jobs)} judge items ({len(jobs) - len(todo)} cached, {len(todo)} to do); about "
              f"{len(todo)} calls, roughly {len(todo) / 1.5:.0f} minutes at the pilot's observed rate.")
        return
    if not args.report_only:
        from src.llm_client import LLMClient
        client = LLMClient(provider="nvidia")
        if args.selftest:
            D.run_selftest(client, topics["ai_regulation"])
            return
        print(f"Judging {len(todo)} trials ({len(jobs) - len(todo)} cached)...")
        for i, j in enumerate(todo, 1):
            statement, scale = D.statement_and_scale(topics[j["topic_id"]])
            print(f"[{i}/{len(todo)}] {j['key']} ({len(j['messages'])} messages)")
            labels = D.judge(client, statement, scale, j["messages"])
            cache[j["key"]] = {"rubric": D.RUBRIC_VERSION, "labels": labels, "owners": j["owners"],
                               "target": j["target"], "judged_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            save_cache(cache)
            if labels is None:
                print("    could not label this trial; it will be retried on the next run")

    by_arm = judge_report(jobs, cache)
    effect_link(by_arm, cells)
    self_shift_report(cells)
    enforcement_report(cells)
    print("\nNotes: the judge is the same model as the agents; hand-check labels with check_labels.py before citing")
    print("agreement. Do not drop trials on these labels. Report effects for all trials, and as a sensitivity")
    print("analysis restricted to attackers that passed 'all 3 toward'.")


if __name__ == "__main__":
    main()