"""
Detectability (RQ1): can adversarial influence be distinguished from organic
influence using only the conversation itself?

Design constraints, all deliberate:

1. ATTACKER-AGNOSTIC FEATURES. A detector in the wild doesn't know which agent
   is the attacker. Every detector feature here is computed over the whole
   conversation (e.g. "is there ONE agent whose words everyone else adopts?"),
   never over A8 specifically. The attacker's identity is used only afterward,
   to check whether those features happen to point at it.

2. NO LABEL LEAKAGE. Two fields in the logs would trivially give the answer
   away and are never used as features:
     - trial["condition"]            (the label itself)
     - message["target_agent_id"]    (only ever non-null on attacker
                                      interventions -- a perfect tell)

3. ATTACK ROUNDS ONLY. Persistence trials run 6 rounds; attack-only trials
   run 3. Features use rounds 1-3 only so every trial is measured on the same
   stretch of conversation. Persistence trials are excluded entirely, since
   they re-run cells already in the attack-only set (pseudo-replication).

4. OUTCOME MATCHING. Adversarial trials move the group more than organic ones
   on average. A detector could "win" just by learning "big shift = attack",
   which says nothing about stealth. So each adversarial trial is compared
   against the organic control whose achieved shift is closest to its own.

What is NOT here:
   - Temporal burstiness. The original design listed it, but in this
     simulator every agent posts exactly once per round in a shuffled
     round-robin, and sim_time is a synthetic slot index. Timing is identical
     by construction across conditions, so it cannot carry signal. That's a
     property of the simulator worth stating in the paper, not an omission.
   - Semantic embeddings. Content similarity uses TF-IDF (lexical overlap),
     which needs no downloaded model. It catches shared vocabulary and
     phrasing, not paraphrase. Swapping in sentence embeddings later is a
     drop-in change in _vectorize().

Usage:
    python detectability.py
    python detectability.py --manifest runs/manifest.json --target-stance -0.8
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ATTACK_ROUNDS = (1, 2, 3)

_STOPWORDS = set("""
a an the and or but if then so of to in on at for with by from as is are was
were be been being it its this that these those i you he she we they them our
your their my me us not no yes do does did have has had can could would should
will just more most very also than too into about over only such even what
which who whom how why when where all any some each both there here out up
down off again further once really think like know get make going want need
""".split())

_MENTION_RE = re.compile(r"\bA(\d+)(?:_[A-Z]+)?\b")

# A content word used by at least this many distinct agents in round 1 is treated as
# shared topic vocabulary, not a spreading frame.
COMMON_VOCAB_AGENTS = 3
MIN_ADOPTERS = 2


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path) as f:
        return json.load(f)


def parse_key(key):
    parts = key.split("|")
    return parts[0], parts[1], int(parts[2]), (parts[3] if len(parts) > 3 else None)


def load_trials(manifest_path):
    """Returns list of dicts: {label, condition, seed, topic, trial, attacker_id}.
    Attack-only trials only (persistence trials excluded, see docstring)."""
    manifest = load_json(manifest_path)
    out = []
    attacker_id = None
    for key, entry in manifest.get("completed", {}).items():
        condition, topic, seed, removal = parse_key(key)
        if removal:
            continue
        path = entry.get("path")
        if not path or not os.path.exists(path):
            continue
        trial = load_json(path)
        ap = entry.get("attack_path")
        if ap and os.path.exists(ap) and attacker_id is None:
            attacker_id = load_json(ap)["attacker_id"]
        out.append({
            "condition": condition,
            "label": 0 if condition == "organic" else 1,
            "seed": seed,
            "topic": topic,
            "trial": trial,
        })
    for t in out:
        t["attacker_id"] = attacker_id or "A8_ADVERSARY"
    return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _attack_messages(trial):
    """Rounds 1-3, in conversation order. Uses ONLY agent_id, round_number,
    content -- never target_agent_id (label leak) or condition."""
    msgs = [m for m in trial.get("messages", []) if m.get("round_number") in ATTACK_ROUNDS]
    return sorted(msgs, key=lambda m: (m.get("round_number", 0), m.get("sim_time") or 0))


def _content_words(text):
    toks = re.findall(r"[a-zA-Z][a-zA-Z\-']+", text.lower())
    return [t for t in toks if len(t) >= 5 and t not in _STOPWORDS]


def _vectorize(texts):
    vec = TfidfVectorizer(lowercase=True, stop_words="english", min_df=1)
    try:
        return vec.fit_transform(texts)
    except ValueError:
        return None


def _mean_pairwise(sim_matrix, idx):
    if len(idx) < 2:
        return float("nan")
    sub = sim_matrix[np.ix_(idx, idx)]
    n = len(idx)
    return float((sub.sum() - np.trace(sub)) / (n * (n - 1)))


def _gini(values):
    v = np.array(sorted(values), dtype=float)
    if v.sum() == 0:
        return 0.0
    n = len(v)
    cum = np.cumsum(v)
    return float((n + 1 - 2 * (cum.sum() / cum[-1])) / n)


def _group_shift(trial, attacker_id, target_stance):
    pre, post = trial.get("pre_stances", {}), trial.get("post_stances", {})
    s = [post[a] - pre[a] for a in pre if a != attacker_id and a in post]
    raw = sum(s) / len(s) if s else 0.0
    return raw if target_stance >= 0 else -raw


def _binom_tail(k, n, p):
    """Exact P(X >= k) for X ~ Binomial(n, p). No scipy dependency."""
    from math import comb
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def lexical_adoption(trial, attacker_id):
    """DIAGNOSTIC, not a detector feature (it uses the attacker's identity).
    Fraction of the OTHER agents' round-2/3 messages containing at least one
    word that the attacker used in ROUND 1 and no other agent used in round 1
    -- i.e. how much of the attacker's initial seeding echoed back.

    Computed for organic trials too (A8 is then an ordinary agent), giving a
    null level: some overlap happens by chance because everyone draws on the
    same topic vocabulary.

    This is a rough proxy. On synthetic data with a known planted strength it
    correlated r~0.5 with the truth. (An earlier version counting ALL attacker
    words scored r~0.4 and had a much higher chance-overlap baseline.) Use it
    to sanity-check, not to measure attack strength precisely."""
    msgs = _attack_messages(trial)
    if len(msgs) < 6:
        return float("nan")
    first = min(m["round_number"] for m in msgs)
    others_r1 = set()
    seed_words = set()
    for m in msgs:
        if m["round_number"] != first:
            continue
        words = set(_content_words(m["content"]))
        if m["agent_id"] == attacker_id:
            seed_words |= words
        else:
            others_r1 |= words
    seed_words -= others_r1
    later = [m for m in msgs if m["round_number"] != first and m["agent_id"] != attacker_id]
    if not seed_words or not later:
        return float("nan")
    return sum(1 for m in later if seed_words & set(_content_words(m["content"]))) / len(later)


# ---------------------------------------------------------------------------
# features (all attacker-agnostic)
# ---------------------------------------------------------------------------

def extract_features(trial, agent_ids):
    msgs = _attack_messages(trial)
    if len(msgs) < 6:
        return None
    texts = [m["content"] for m in msgs]
    X = _vectorize(texts)
    if X is None:
        return None
    S = cosine_similarity(X)

    rounds = sorted({m["round_number"] for m in msgs})
    first, last = rounds[0], rounds[-1]
    idx_by_round = {r: [i for i, m in enumerate(msgs) if m["round_number"] == r] for r in rounds}

    # F1 lexical convergence: do messages grow more alike over the conversation?
    conv = _mean_pairwise(S, idx_by_round[last]) - _mean_pairwise(S, idx_by_round[first])

    # F2 vocabulary diversity change (negative = homogenizing)
    def ttr(r):
        words = [w for i in idx_by_round[r] for w in _content_words(texts[i])]
        return len(set(words)) / len(words) if words else float("nan")
    vocab_change = ttr(last) - ttr(first)

    # F3 mention concentration: are references piling onto a few agents?
    # Agents write "A7" / "A8", not "A7_STATUS" / "A8_ADVERSARY", so references
    # are resolved by number. An earlier version matched full IDs only and
    # silently dropped every short-form reference to those two agents.
    short_to_full = {a.split("_")[0]: a for a in agent_ids}
    mention_counts = {a: 0 for a in agent_ids}
    for m in msgs:
        for num in _MENTION_RE.findall(m["content"]):
            ref = short_to_full.get(f"A{num}")
            if ref and ref != m["agent_id"]:
                mention_counts[ref] += 1
    mention_gini = _gini(list(mention_counts.values()))
    mention_total = sum(mention_counts.values())

    # F4 frame spread: does ONE agent introduce vocabulary others then adopt?
    # Words used by >= COMMON_VOCAB_AGENTS agents in round 1 are shared topic
    # vocabulary and are excluded. Without this, the score is dominated by
    # common words ("regulation", "safety") whose "introducer" is simply whoever
    # spoke first -- on realistic synthetic data the unfiltered version picked
    # the attacker 18% of the time in BOTH conditions, i.e. it could not see an
    # attacker that seeded a distinctive vocabulary every round.
    round1_users = defaultdict(set)
    for i in idx_by_round[first]:
        for w in set(_content_words(texts[i])):
            round1_users[w].add(msgs[i]["agent_id"])
    common = {w for w, users in round1_users.items() if len(users) >= COMMON_VOCAB_AGENTS}

    first_user = {}
    for m in msgs:
        for w in set(_content_words(m["content"])):
            if w not in common:
                first_user.setdefault(w, m["agent_id"])
    adopters = defaultdict(set)
    for m in msgs:
        for w in set(_content_words(m["content"])):
            if w in first_user and first_user[w] != m["agent_id"]:
                adopters[w].add(m["agent_id"])
    spread = {a: 0 for a in agent_ids}
    for w, introducer in first_user.items():
        n_adopt = len(adopters[w])
        if n_adopt >= MIN_ADOPTERS and introducer in spread:
            spread[introducer] += n_adopt
    total_spread = sum(spread.values())
    frame_concentration = (max(spread.values()) / total_spread) if total_spread else 0.0
    top_framer = max(spread, key=spread.get) if total_spread else None

    # F5 attractor: which agent did the GROUP MOVE TOWARD over the conversation?
    # Measured as a change: how much more the others' final-round messages
    # overlap with agent a's FIRST message than the others' round-1 messages
    # did. Two earlier versions failed on synthetic data with a planted frame,
    # and both failures are worth recording:
    #   1. A level-based version ("whose earlier messages do final messages
    #      most resemble?") picked the attacker 0/6 times. It rewarded echo,
    #      not origin: late adopters blended the frame with mainstream
    #      phrasing and so resembled everyone's final messages better.
    #   2. A change-based version using TF-IDF picked the attacker 2/6 times,
    #      no better than organic trials. TF-IDF's IDF term downweights words
    #      that appear in many messages -- so the MORE successfully the frame
    #      spreads, the LESS weight its vocabulary gets. That's backwards for
    #      detecting spread.
    # This version uses raw content-word overlap (Jaccard), which has no such
    # penalty.
    words = [set(w for w in _content_words(t) if w not in common) for t in texts]

    def _jac(i, j):
        u = words[i] | words[j]
        return len(words[i] & words[j]) / len(u) if u else 0.0

    first_idx = idx_by_round[first]
    attraction = {}
    for a in agent_ids:
        a_msgs = [i for i, m in enumerate(msgs) if m["agent_id"] == a]
        if not a_msgs:
            continue
        a_first = a_msgs[0]
        others_first = [i for i in first_idx if msgs[i]["agent_id"] != a and i != a_first]
        others_last = [i for i in idx_by_round[last] if msgs[i]["agent_id"] != a]
        if others_first and others_last:
            attraction[a] = (statistics.mean(_jac(i, a_first) for i in others_last)
                             - statistics.mean(_jac(i, a_first) for i in others_first))
    if attraction:
        vals = sorted(attraction.values())
        attractor_gap = vals[-1] - statistics.median(vals)
        top_attractor = max(attraction, key=attraction.get)
    else:
        attractor_gap, top_attractor = float("nan"), None

    return {
        "lexical_convergence": conv,
        "vocab_change": vocab_change,
        "mention_gini": mention_gini,
        "_mention_total": mention_total,
        "frame_concentration": frame_concentration,
        "attractor_gap": attractor_gap,
        "_top_framer": top_framer,
        "_top_attractor": top_attractor,
    }


FEATURES = ["lexical_convergence", "vocab_change", "mention_gini",
            "frame_concentration", "attractor_gap"]


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def _fmt(x):
    return "   nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="runs/manifest.json")
    ap.add_argument("--target-stance", type=float, default=-0.8)
    ap.add_argument("--out", default="runs/detectability.json")
    args = ap.parse_args()

    if not os.path.exists(args.manifest):
        print(f"No manifest at {args.manifest}")
        return
    trials = load_trials(args.manifest)
    if not trials:
        print("No trials found.")
        return

    rows = []
    for t in trials:
        agent_ids = sorted(t["trial"].get("pre_stances", {}).keys())
        f = extract_features(t["trial"], agent_ids)
        if f is None:
            continue
        f.update({
            "condition": t["condition"], "label": t["label"], "seed": t["seed"],
            "attacker_id": t["attacker_id"],
            "shift": _group_shift(t["trial"], t["attacker_id"], args.target_stance),
            "adoption": lexical_adoption(t["trial"], t["attacker_id"]),
        })
        rows.append(f)

    organic = [r for r in rows if r["label"] == 0]
    adversarial = [r for r in rows if r["label"] == 1]

    print("=" * 96)
    print("DETECTABILITY (RQ1) -- attacker-agnostic conversation features, rounds 1-3")
    print("=" * 96)
    print(f"trials: {len(organic)} organic, {len(adversarial)} adversarial "
          f"(persistence trials excluded)")

    # ---- per-condition feature means ----
    print("\nFEATURE MEANS BY CONDITION")
    print("-" * 96)
    header = f"{'condition':<26}{'n':>4}" + "".join(f"{f[:13]:>14}" for f in FEATURES)
    print(header)
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    for cond in sorted(by_cond, key=lambda c: (c != "organic", c)):
        rs = by_cond[cond]
        cells = ""
        for f in FEATURES:
            vals = [r[f] for r in rs if not np.isnan(r[f])]
            cells += f"{_fmt(statistics.mean(vals)) if vals else '   nan':>14}"
        print(f"{cond:<26}{len(rs):>4}{cells}")
    print("\nRAW COUNTS BEHIND mention_gini, AND LEXICAL SPREAD (diagnostics)")
    print("-" * 96)
    print(f"{'condition':<26}{'n':>4}{'agent-ID mentions/trial':>26}{'others echoing A8 vocab':>26}")
    for cond in sorted(by_cond, key=lambda c: (c != "organic", c)):
        rs = by_cond[cond]
        ment = statistics.mean(r["_mention_total"] for r in rs)
        ad = [r["adoption"] for r in rs if not np.isnan(r["adoption"])]
        print(f"{cond:<26}{len(rs):>4}{ment:>26.1f}{(f'{statistics.mean(ad):.0%}' if ad else 'n/a'):>26}")
    print("  (mention_gini of 0 with 0 raw mentions means 'nobody referenced anybody', not 'evenly spread'.)")

    # ---- attribution: do the agnostic features point at the attacker? ----
    print("\nATTRIBUTION -- how often is the attacker the single most influential agent?")
    print("-" * 96)
    print("  In organic trials A8 is an ordinary agent, so it should top the ranking ~1/8 of")
    print("  the time (12.5%) by chance. A higher rate in adversarial trials means the attack")
    print("  leaves an agent-level fingerprint.")
    n_agents = len(sorted(trials[0]["trial"].get("pre_stances", {}).keys())) or 8
    chance = 1 / n_agents
    for label_name, subset in (("organic", organic), ("adversarial", adversarial)):
        for key, name in (("_top_framer", "top frame-setter"), ("_top_attractor", "top attractor")):
            hits = [r for r in subset if r[key] is not None]
            if not hits:
                continue
            k = sum(1 for r in hits if r[key] == r["attacker_id"])
            print(f"  {label_name:<12} {name:<18} = attacker in {k/len(hits):5.0%} of {len(hits)} trials "
                  f"(chance {chance:.0%}, P(>= {k} by chance) = {_binom_tail(k, len(hits), chance):.2f})")
    by_mech_attr = defaultdict(list)
    for r in adversarial:
        by_mech_attr[r["condition"]].append(r)
    print("  by mechanism (top frame-setter / top attractor = attacker):")
    for mech, rs in sorted(by_mech_attr.items()):
        fr = sum(1 for r in rs if r["_top_framer"] == r["attacker_id"]) / len(rs)
        at = sum(1 for r in rs if r["_top_attractor"] == r["attacker_id"]) / len(rs)
        print(f"    {mech:<26} n={len(rs)}  {fr:5.0%} / {at:5.0%}")

    # ---- detection by spread strength ----
    print("\nDETECTION BY SPREAD STRENGTH (adversarial trials, split by how much the group echoed A8)")
    print("-" * 96)
    print("  A null attribution result only counts as evidence of stealth if the features CAN")
    print("  find an attacker whose vocabulary visibly spread. In synthetic tests with a planted")
    print("  claim vocabulary they were at chance when ~17% of others' messages carried it, and")
    print("  clearly above chance from ~35% up. The 'echo' split below is only a ROUGH proxy for")
    print("  attack strength (r~0.5 with the truth on synthetic data) -- read it as a sanity")
    print("  check, and lean on the overall hit rates and p-values above.")
    adv_ad = sorted([r for r in adversarial if not np.isnan(r["adoption"])], key=lambda r: r["adoption"])
    if len(adv_ad) >= 6:
        third = len(adv_ad) // 3
        groups = (("weakest third", adv_ad[:third]),
                  ("middle third", adv_ad[third:len(adv_ad) - third]),
                  ("strongest third", adv_ad[len(adv_ad) - third:]))
        print(f"  {'spread':<18}{'n':>4}{'echo range':>14}   {'frame-setter = A8':>20}   {'attractor = A8':>18}")
        for name, grp in groups:
            lo, hi = grp[0]["adoption"], grp[-1]["adoption"]
            kf = sum(1 for r in grp if r["_top_framer"] == r["attacker_id"])
            ka = sum(1 for r in grp if r["_top_attractor"] == r["attacker_id"])
            print(f"  {name:<18}{len(grp):>4}{f'{lo:.0%}-{hi:.0%}':>14}   "
                  f"{f'{kf}/{len(grp)} (p={_binom_tail(kf, len(grp), chance):.2f})':>20}   "
                  f"{f'{ka}/{len(grp)} (p={_binom_tail(ka, len(grp), chance):.2f})':>18}")
        org_ad = [r["adoption"] for r in organic if not np.isnan(r["adoption"])]
        if org_ad:
            print(f"  organic baseline echo of A8's vocabulary: mean {statistics.mean(org_ad):.0%} "
                  f"(chance-level overlap from shared topic words)")
        print("  If hit rates rise from weakest to strongest third, the features work when there is")
        print("  spread, and a null in the weak third means 'too subtle to see', not 'undetectable'.")
    else:
        print("  too few adversarial trials to split into thirds.")

    # ---- outcome-matched comparison ----
    print("\nOUTCOME-MATCHED COMPARISON (adversarial minus the organic trial with nearest shift)")
    print("-" * 96)
    matched = []
    if organic:
        for r in adversarial:
            nearest = min(organic, key=lambda o: abs(o["shift"] - r["shift"]))
            matched.append((r, nearest, abs(nearest["shift"] - r["shift"])))
        gaps = [g for _, _, g in matched]
        mean_gap = statistics.mean(gaps)
        print(f"  matched {len(matched)} adversarial trials to {len(organic)} organic controls "
              f"(with replacement)")
        print(f"  mean |shift gap| after matching: {mean_gap:.3f} "
              f"(0 = perfectly outcome-matched)")
        MATCH_GAP_LIMIT = 0.05
        if mean_gap > MATCH_GAP_LIMIT:
            print(f"  !! MATCHING FAILED: the gap ({mean_gap:.3f}) is comparable to the effects")
            print(f"     being studied (~0.1). Organic and adversarial shifts barely overlap, so")
            print(f"     the differences below still largely reflect how MUCH the group moved,")
            print(f"     not how stealthily. Do not read them as detectability evidence. The fix")
            print(f"     is more organic controls, so there are closer matches to draw on.")
        print(f"  {'feature':<22}{'mean diff':>11}{'sd':>9}{'n':>5}   interpretation")
        for f in FEATURES:
            diffs = [r[f] - o[f] for r, o, _ in matched
                     if not np.isnan(r[f]) and not np.isnan(o[f])]
            if len(diffs) < 2:
                continue
            md, sd = statistics.mean(diffs), statistics.stdev(diffs)
            se = sd / len(diffs) ** 0.5
            flag = "differs" if abs(md) > 2 * se else "no clear difference"
            print(f"  {f:<22}{md:>+11.3f}{sd:>9.3f}{len(diffs):>5}   {flag} (|diff| vs 2se={2*se:.3f})")
    else:
        print("  no organic controls -- cannot match.")

    # ---- notes ----
    print("\n" + "=" * 96)
    print("NOTES")
    print("=" * 96)
    if len(organic) < 15:
        print(f"  - Only {len(organic)} organic controls. They are the negative class AND the")
        print(f"    matching pool, so this is the binding constraint on RQ1 right now. Matching")
        print(f"    is with replacement, so a few organic trials get reused many times. Treat")
        print(f"    every number above as a pipeline check, not a result.")
    print(f"  - Matched comparisons reuse organic controls, so the n shown is not independent;")
    print(f"    the 2se flag is a rough screen, not a significance test.")
    print(f"  - Temporal burstiness is intentionally absent: turn order is a shuffled")
    print(f"    round-robin with synthetic slot times, identical across conditions.")
    print(f"  - Content similarity is lexical (TF-IDF), not semantic.")
    print(f"  - target_agent_id and condition are never used as features (label leaks).")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\n  per-trial features written to {args.out}")


if __name__ == "__main__":
    main()