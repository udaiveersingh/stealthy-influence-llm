"""
Replace the human-validation rows that came from superseded runs, without touching any other row.

The original sheet (results_hpc2/human_validation/) sampled 2 attacker + 1 organic message per model x topic cell.
Two Nemotron cells (college_value, remote_work) were later re-run, so their 6 rows are replaced here by rows sampled
the same way from the new runs. Item numbers stay the same, so every label already given to the other 72 rows is kept.

  1) prepare   python replace_rows.py prepare
       -> results_hpc2/human_validation_v2/sheet.csv          blank 78-row sheet for raters (6 messages replaced)
       -> results_hpc2/human_validation_v2/key.json           answer key (do not open while labelling)
       -> results_hpc2/human_validation_v2/redo_sheet.csv     12 rows for the author: the 6 new messages mixed with
                                                              6 decoys, shuffled, so the author labels them blind
       -> results_hpc2/human_validation_v2/sheet_judge_<tag>.csv   copy of the old judge sheet with the 6 rows blanked
  2) the author labels redo_sheet.csv:   python human_validation.py label --sheet results_hpc2/human_validation_v2/redo_sheet.csv --no-practice
  3) merge     python replace_rows.py merge --author-sheet <old labelled sheet> --name udai
       -> results_hpc2/human_validation_v2/sheet_udai.csv     the author's 78 labels with the 6 replaced rows filled in
  4) the judge fills only the 6 blank rows (needs the model server):
       python human_validation.py judge --tag nemotron --out-dir results_hpc2/human_validation_v2
  5) score     python human_validation.py score --out-dir results_hpc2/human_validation_v2 \
                   --sheets results_hpc2/human_validation_v2/sheet_udai.csv results_hpc2/human_validation_v2/sheet_judge_nemotron.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil

import human_validation as HV
import pool_confirmatory as P
from src.organic_pipeline import load_topic
from src.direction_judge import statement_and_scale

OLD = "results_hpc2/human_validation"
NEW = "results_hpc2/human_validation_v2"
REPLACE = {("Nemotron", "college_value"), ("Nemotron", "remote_work")}
FIELDS = ["item", "statement", "message", "label"]


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def statement_for(topic):
    st_, sc = statement_and_scale(load_topic(topic))
    return f"{st_}  [Scale: {sc}]"


def prepare(args):
    rng = random.Random(args.seed)
    old_key = json.load(open(os.path.join(OLD, "key.json")))
    K = old_key["key"]
    rows = read_rows(os.path.join(OLD, "sheet.csv"))
    for r in rows:
        r["label"] = ""
    targets = [i for i, v in K.items() if (v["model"], v["topic"]) in REPLACE]
    print(f"rows to replace: {len(targets)} -> items {sorted(targets, key=int)}")

    cells = {(c["model"], c["topic"]): c for c in P.default_cells()}
    pools = {}
    for mt in REPLACE:
        man = cells[mt]["manifest"]
        if "runs/hpc/" not in man.replace("\\", "/"):
            raise SystemExit(f"{mt} still points at {man}; update pool_confirmatory.default_cells first")
        attack, organic, flagged, target = HV.pools(man)
        pools[mt] = {"attack": attack, "organic": organic, "target": target}
        print(f"  {mt[0]}/{mt[1]}: {len(attack)} attacker, {len(organic)} organic candidates from {man}")

    new_key = json.loads(json.dumps(K))
    by_item = {r["item"]: r for r in rows}
    redo = []
    for i in sorted(targets, key=int):
        v = K[i]
        mt = (v["model"], v["topic"])
        x = pools[mt][v["source"]].pop(rng.randrange(len(pools[mt][v["source"]])))
        by_item[i]["message"] = x["message"]
        new_key[i] = {"source": v["source"], "model": v["model"], "topic": v["topic"], "target": pools[mt]["target"],
                      "seed": x["seed"], "agent": x["agent"], "round": x["round"], "judge": x["judge"], "replaced": True}
        redo.append({"statement": by_item[i]["statement"], "message": x["message"], "_item": i})

    # decoys: 6 messages from other new-run cells, half attacker half organic, so the author cannot tell which count
    others = [c for c in P.default_cells() if (c["model"], c["topic"]) not in REPLACE and os.path.exists(c["manifest"])]
    used = {r["message"] for r in rows}
    for src in ["attack"] * 3 + ["organic"] * 3:
        while True:
            c = rng.choice(others)
            a, o, _, _ = HV.pools(c["manifest"])
            cand = [m for m in (a if src == "attack" else o) if m["message"] not in used]
            if cand:
                x = rng.choice(cand)
                used.add(x["message"])
                redo.append({"statement": statement_for(c["topic"]), "message": x["message"], "_item": None})
                break
    rng.shuffle(redo)

    os.makedirs(NEW, exist_ok=True)
    write_rows(os.path.join(NEW, "sheet.csv"), rows)
    json.dump({"key": new_key, "flagged_excluded": old_key["flagged_excluded"], "seed": old_key.get("seed"),
               "replaced_items": sorted(targets, key=int)}, open(os.path.join(NEW, "key.json"), "w"), indent=1)
    redo_rows = [{"item": f"R{n}", "statement": r["statement"], "message": r["message"], "label": ""}
                 for n, r in enumerate(redo, 1)]
    write_rows(os.path.join(NEW, "redo_sheet.csv"), redo_rows)
    json.dump({f"R{n}": r["_item"] for n, r in enumerate(redo, 1)}, open(os.path.join(NEW, "redo_map.json"), "w"))
    if os.path.exists(os.path.join(OLD, "INSTRUCTIONS.txt")):
        shutil.copy(os.path.join(OLD, "INSTRUCTIONS.txt"), NEW)

    for name in os.listdir(OLD):                       # carry over judge sheets with the replaced rows blanked
        if name.startswith("sheet_judge_") and name.endswith(".csv"):
            jr = read_rows(os.path.join(OLD, name))
            for r in jr:
                if r["item"] in targets:
                    r["message"], r["label"] = by_item[r["item"]]["message"], ""
            write_rows(os.path.join(NEW, name), jr)
            print(f"  copied {name} with {len(targets)} rows blanked for the judge to fill")
    print(f"\nwrote {NEW}/sheet.csv (blank, for raters), key.json, redo_sheet.csv (12 rows for the author)")
    print("next: label redo_sheet.csv, then run  python replace_rows.py merge --author-sheet <your old sheet> --name <you>")


def merge(args):
    old = read_rows(args.author_sheet)
    redo = {r["item"]: (r.get("label") or "").strip() for r in read_rows(os.path.join(NEW, "redo_sheet.csv"))}
    rmap = json.load(open(os.path.join(NEW, "redo_map.json")))
    new_rows = {r["item"]: r for r in read_rows(os.path.join(NEW, "sheet.csv"))}
    fill = {item: redo[r] for r, item in rmap.items() if item}
    missing = [i for i, lab in fill.items() if not lab]
    if missing:
        raise SystemExit(f"redo_sheet.csv still has blank labels for {len(missing)} of the rows that count; finish it first")
    out = []
    for r in old:
        row = dict(new_rows[r["item"]])
        row["label"] = fill.get(r["item"], r.get("label", ""))
        out.append(row)
    path = os.path.join(NEW, f"sheet_{args.name}.csv")
    write_rows(path, out)
    print(f"wrote {path}: {len(out)} rows, {len(fill)} replaced rows filled from redo_sheet.csv")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--seed", type=int, default=11)
    m = sub.add_parser("merge")
    m.add_argument("--author-sheet", required=True, help="your labelled 78-row sheet from the original round")
    m.add_argument("--name", required=True)
    args = ap.parse_args()
    {"prepare": prepare, "merge": merge}[args.cmd](args)


if __name__ == "__main__":
    main()