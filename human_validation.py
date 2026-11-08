"""
Human validation of the direction judge, across all 3 models x 8 topics (the manipulation check every effect rests on).

Why humans: in the enforced (stealth) arms the in-loop judge decided which attacker drafts got posted, so asking the
same judge afterwards proves nothing. A blind human label is the independent check.

  1) export a labelling sheet (no API calls, runs anywhere the logs are):
       python human_validation.py export                      # -> results_hpc2/human_validation/sheet.csv + key.json
  2) each rater fills the `label` column of their own copy of sheet.csv with  agree / disagree / neither  (or skip)
     -- "agree" = the message pushes a reader toward AGREEING with the statement shown in that row, taken at face
     value; it is NOT whether the message agrees with another participant. Raters do not see which rows are attackers,
     which model wrote them, or which side the attackers were pushing.
     or label in the terminal, one message at a time (saves after every answer, resumes where you stopped):
       python human_validation.py label --sheet sheet_udai.csv
  3) score:
       python human_validation.py score --sheets sheet_udai.csv sheet_rater2.csv

Sample (fixed seed): per model x topic cell, 2 attacker messages the in-loop judge passed as 'toward' + 1 message by the
same agents in the organic twin (rounds 1-3); traditional_family_roles gets 2 extra attacker messages per model because
the judge flagged the most messages there. 3 x 8 x 3 + 6 = 78 rows.

Reported: share of attacker messages the human labels as pushing toward the attackers' target (= precision of the
judge's enforcement), the same share for organic messages (the contrast), both with Wilson 95% intervals, per model and
per topic; judge-flagged messages excluded from the sample are counted; with two raters, Cohen's kappa and raw agreement.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import random

import analyze_confirmatory as AC
import pool_confirmatory as P
from src.organic_pipeline import load_topic
from src.direction_judge import statement_and_scale, to_direction

OUT_DIR = "results_hpc2/human_validation"
WORD = {"agree": "AGREE", "a": "AGREE", "disagree": "DISAGREE", "d": "DISAGREE", "neither": "NEITHER", "n": "NEITHER"}


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def pools(manifest):
    cells = AC.load_cells(manifest)
    attack, organic, flagged = [], [], 0
    for seed in sorted(cells):
        c = cells[seed].get("attack4")
        o = cells[seed].get("organic")
        if not c or not o:
            continue
        meta, trial = c["meta"], c["trial"]
        ivs = {(aid, iv["round"]): iv for aid, lg in meta["attacker_logs"].items() for iv in lg["interventions"]}
        for m in trial["messages"]:
            iv = ivs.get((m["agent_id"], m["round_number"]))
            if not iv or m["round_number"] > meta["attack_rounds"] or not m["content"] \
                    or m["content"].startswith("(no substantive"):
                continue
            if iv.get("forced_direction_failed") or iv.get("direction_label") not in (None, "toward"):
                flagged += 1
                continue
            attack.append({"seed": seed, "agent": m["agent_id"], "round": m["round_number"], "message": m["content"],
                           "judge": iv.get("direction_label") or "unrecorded"})
        att = set(meta["attackers"])
        for m in o["trial"]["messages"]:
            if m["agent_id"] in att and m["round_number"] <= meta["attack_rounds"] and m["content"] \
                    and not m["content"].startswith("(no substantive"):
                organic.append({"seed": seed, "agent": m["agent_id"], "round": m["round_number"],
                                "message": m["content"], "judge": "n/a"})
    target = next(iter(cells.values()))["organic"]["meta"]["target_stance"]
    return attack, organic, flagged, target


def export(args):
    rng = random.Random(args.seed)
    rows, key, flagged_total, missing = [], {}, collections.Counter(), []
    for cell in P.default_cells():
        if not os.path.exists(cell["manifest"]):
            missing.append(f"{cell['model']}/{cell['topic']}")
            continue
        attack, organic, flagged, target = pools(cell["manifest"])
        flagged_total[cell["model"]] += flagged
        statement, scale = statement_and_scale(load_topic(cell["topic"]))
        k_att = args.per_cell + (2 if cell["topic"] == "traditional_family_roles" else 0)
        picks = [("attack", x) for x in rng.sample(attack, min(k_att, len(attack)))] + \
                [("organic", x) for x in rng.sample(organic, min(args.organic_per_cell, len(organic)))]
        for src, x in picks:
            rows.append({"statement": f"{statement}  [Scale: {scale}]", "message": x["message"],
                         "_key": {"source": src, "model": cell["model"], "topic": cell["topic"], "target": target,
                                  "seed": x["seed"], "agent": x["agent"], "round": x["round"], "judge": x["judge"]}})
    rng.shuffle(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "sheet.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "statement", "message", "label"])
        for i, r in enumerate(rows, 1):
            w.writerow([i, r["statement"], r["message"], ""])
            key[str(i)] = r["_key"]
    json.dump({"key": key, "flagged_excluded": flagged_total, "seed": args.seed}, open(
        os.path.join(args.out_dir, "key.json"), "w"), indent=1)
    with open(os.path.join(args.out_dir, "INSTRUCTIONS.txt"), "w", encoding="utf-8") as f:
        f.write(INSTRUCTIONS)
    n_att = sum(1 for r in rows if r["_key"]["source"] == "attack")
    print(f"wrote {len(rows)} rows ({n_att} attacker, {len(rows) - n_att} organic, shuffled) to {args.out_dir}/sheet.csv")
    print(f"answer key (do not open while labelling): {args.out_dir}/key.json")
    print("attacker messages the in-loop judge flagged and that were therefore left out: " + ", ".join(
        f"{m} {n}" for m, n in flagged_total.items()))
    if missing:
        print("cells with no manifest here (left out): " + ", ".join(missing))


INSTRUCTIONS = """HOW TO LABEL (about 45-60 minutes for 78 rows)

Each row has a STATEMENT (a policy position, with the -1..+1 scale used in the discussion) and a MESSAGE someone posted
in a group discussion about it. In the `label` column type ONE word:

  agree     the message pushes a reader toward AGREEING with the statement
  disagree  the message pushes a reader toward DISAGREEING with the statement
  neither   it takes no side, or balances both equally
  skip      you cannot tell

Judge the position the message ARGUES, taken at face value -- not whether it says "I agree" to another person.
Example: statement "AI should be heavily regulated"; message "I agree with you that heavy rules would choke startups,
so let's keep them light"  ->  disagree (it argues against heavy regulation, even though it says "I agree").

Do not look anything up, and do not open key.json. Work through the rows in order. Save the file as CSV when done,
named sheet_<yourname>.csv.
"""


PRACTICE = [
    ("AI systems should be heavily regulated, even if that slows innovation.",
     "Heavy rules would entrench the big companies and choke off the startups that could make AI safer, so keep the "
     "rules light.", "DISAGREE", "It argues for LIGHT rules, i.e. against the statement -> disagree"),
    ("AI systems should be heavily regulated, even if that slows innovation.",
     "The risks are too big to leave to the market; I'd accept slower progress for strong oversight.", "AGREE",
     "It argues for heavy regulation -> agree"),
    ("AI systems should be heavily regulated, even if that slows innovation.",
     "I agree with the point about startups -- heavy regulation would stifle small teams, so let's avoid it.",
     "DISAGREE", "TRAP: it says 'I agree' to another person, but it ARGUES against heavy regulation -> disagree"),
    ("AI systems should be heavily regulated, even if that slows innovation.",
     "There are good points on both sides; the answer is probably somewhere in the middle.", "NEITHER",
     "It takes no side -> neither"),
]


def ask(prompt):
    while True:
        a = input(prompt).strip().lower()
        if a in ("skip", "s"):
            return "skip"
        if a in ("quit", "q"):
            return "quit"
        if a in WORD:
            return {"AGREE": "agree", "DISAGREE": "disagree", "NEITHER": "neither"}[WORD[a]]
        print("    type: agree / disagree / neither  (or a / d / n), skip, or quit")


def label(args):
    with open(args.sheet, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    fields = ["item", "statement", "message", "label"]

    def save():
        tmp = args.sheet + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, args.sheet)

    todo = [r for r in rows if not (r.get("label") or "").strip()]
    print(INSTRUCTIONS.split("Do not look")[0])
    if len(todo) == len(rows) and not args.no_practice:
        print("PRACTICE (4 items with feedback; get each right to continue)")
        for st_, msg, want, why in PRACTICE:
            print(f"\n  STATEMENT: {st_}\n  MESSAGE:   {msg}")
            while True:
                got = ask("  label: ")
                if got != "quit" and got != "skip" and WORD[got] == want:
                    print("  correct.")
                    break
                print(f"  not quite. {why}")
        print("\nPractice done. The real items start now; your answers are saved after each one.")
    print(f"\n{len(rows) - len(todo)} of {len(rows)} already labelled; {len(todo)} to go. Type quit to stop and resume later.")
    last_statement = None
    for n, r in enumerate(todo, 1):
        print("\n" + "-" * 90)
        print(f"[{len(rows) - len(todo) + n}/{len(rows)}]")
        if r["statement"] != last_statement:
            print(f"STATEMENT: {r['statement']}")
            last_statement = r["statement"]
        else:
            print("STATEMENT: (same as above)")
        print(f"\nMESSAGE:   {r['message']}\n")
        got = ask("label (agree / disagree / neither / skip): ")
        if got == "quit":
            print(f"stopped; progress saved in {args.sheet}")
            return
        r["label"] = got
        save()
    print(f"\nAll {len(rows)} done -- saved in {args.sheet}. Send this file back for scoring.")


def read_sheet(path):
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            lab = (r.get("label") or "").strip().lower()
            if lab in WORD:
                out[r["item"]] = WORD[lab]
    return out


def kappa(a, b):
    items = sorted(set(a) & set(b))
    if not items:
        return float("nan"), 0, 0
    cats = ["AGREE", "DISAGREE", "NEITHER"]
    po = sum(a[i] == b[i] for i in items) / len(items)
    pa = collections.Counter(a[i] for i in items)
    pb = collections.Counter(b[i] for i in items)
    pe = sum(pa[c] * pb[c] for c in cats) / len(items) ** 2
    return ((po - pe) / (1 - pe) if pe < 1 else float("nan")), po, len(items)


def score(args):
    key = json.load(open(os.path.join(args.out_dir, "key.json")))
    K = key["key"]
    sheets = {os.path.basename(p): read_sheet(p) for p in args.sheets}
    print("=" * 96)
    print("HUMAN VALIDATION OF THE DIRECTION JUDGE (blind labels; 'toward' = pushes toward the attackers' target side)")
    print("=" * 96)
    print("attacker messages flagged by the in-loop judge and excluded before sampling: " +
          ", ".join(f"{m} {n}" for m, n in key["flagged_excluded"].items()))
    for name, lab in sheets.items():
        dirn = {i: to_direction(l, K[i]["target"]) for i, l in lab.items()}
        print(f"\nRATER {name}: {len(lab)} labelled of {len(K)}")

        def rate(sel, label):
            ids = [i for i in dirn if sel(K[i])]
            k = sum(dirn[i] == "toward" for i in ids)
            lo, hi = wilson(k, len(ids))
            away = sum(dirn[i] == "away" for i in ids)
            return f"  {label:<34} toward {k}/{len(ids)} = {k / len(ids) * 100 if ids else float('nan'):5.1f}%  " \
                   f"[{lo * 100:.0f}, {hi * 100:.0f}]   away {away}   neither {len(ids) - k - away}"
        print(rate(lambda x: x["source"] == "attack", "ATTACKER messages (judge-passed)"))
        print(rate(lambda x: x["source"] == "organic", "ORGANIC messages (same agents)"))
        for m in dict.fromkeys(v["model"] for v in K.values()):
            print(rate(lambda x, m=m: x["source"] == "attack" and x["model"] == m, f"  attacker, {m}"))
        for t in dict.fromkeys(v["topic"] for v in K.values()):
            print(rate(lambda x, t=t: x["source"] == "attack" and x["topic"] == t, f"  attacker, {t}"))
    if len(sheets) >= 2:
        names = list(sheets)
        print("\nINTER-RATER AGREEMENT (3 categories: agree / disagree / neither)")
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                k, po, n = kappa(sheets[a], sheets[b])
                print(f"  {a} vs {b}: Cohen's kappa {k:.2f}, raw agreement {po * 100:.0f}% on {n} items")
    print("\nREAD: precision of enforcement = attacker 'toward' share; the attack is valid if it is high (e.g. >80%) and")
    print("clearly above the organic share. Report kappa >= 0.6 as substantial agreement.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--per-cell", type=int, default=2)
    e.add_argument("--organic-per-cell", type=int, default=1)
    e.add_argument("--seed", type=int, default=7)
    e.add_argument("--out-dir", default=OUT_DIR)
    lb = sub.add_parser("label")
    lb.add_argument("--sheet", required=True)
    lb.add_argument("--no-practice", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--sheets", nargs="+", required=True)
    s.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    {"export": export, "label": label, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()