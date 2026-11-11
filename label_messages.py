"""
Standalone labelling tool for the human-validation sheet (no other files needed; Python 3.8+, standard library only).

    python label_messages.py sheet_yourname.csv

Shows one message at a time; type agree / disagree / neither (or a / d / n), skip if you cannot tell, quit to stop.
Your answer is saved into the CSV after every message, and running the same command again resumes where you stopped.
When finished, send the CSV back.
"""
import csv
import os
import sys
from types import SimpleNamespace

WORD = {"agree": "AGREE", "a": "AGREE", "disagree": "DISAGREE", "d": "DISAGREE", "neither": "NEITHER", "n": "NEITHER"}

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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python label_messages.py sheet_yourname.csv")
    label(SimpleNamespace(sheet=sys.argv[1], no_practice=False))
