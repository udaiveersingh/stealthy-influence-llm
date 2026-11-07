"""
Society-scale run: 512 agents on a social network (design: src/society.py).

  python run_society.py --topic ai_regulation                 # organic, attack2pct, attack10pct; seed 1
  python run_society.py --topic ai_regulation --estimate
  python run_society.py --topic ai_regulation --mock --n-variants 2   # offline smoke test, 64 agents, no GPU

The organic twin runs first; the attack arms start from its frozen pre-stances. Every finished arm is recorded in
runs/<ns>/society_manifest__<topic>.json and skipped on re-run; an unfinished arm resumes from its per-round
checkpoint, so a preempted job loses at most one round.

Isolation (learned the hard way): this runner refuses to start if any of the 32-agent scale-up variables are set
(PERSONAS_PATH, SCALE_ATTACKER_POOL, SCALE_ARMS), and refuses a non-Nemotron model unless --allow-model is given.
Its files live under runs/<ns>/ and logs/<ns>/ with ns = $RUN_NS or "society", never in the 8-agent manifests.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from src.organic_pipeline import load_topic
from src.society import (ARM_FRACTIONS, FEED_SIZE, arm_sizes, build_society, society_to_json, run_society_trial)

TOPIC_PLAN_PATH = "config/topic_plan.json"
ARM_ORDER = ["organic", "attack2pct", "attack10pct"]


def ns():
    return os.environ.get("RUN_NS", "").strip() or "society"


def paths(topic, seed, n_variants):
    tag = "" if n_variants == 16 else f"_v{n_variants}"
    root_r, root_l = f"runs/{ns()}", f"logs/{ns()}"
    return {
        "manifest": f"{root_r}/society_manifest{tag}__{topic}.json",
        "society": f"{root_r}/society_def_s{seed}{tag}.json",
        "ckpt_dir": f"{root_r}/ckpt{tag}",
        "log_dir": f"{root_l}/society{tag}__{topic}",
    }


def load_json(p, default):
    return json.load(open(p)) if os.path.exists(p) else default


def save_json(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="several societies one after another (each seed = a new network, persona-intensity draw and "
                         "attacker set); overrides --seed")
    ap.add_argument("--arms", nargs="+", default=ARM_ORDER, choices=ARM_ORDER)
    ap.add_argument("--n-variants", type=int, default=16, help="variants per base persona (16 x 32 = 512 agents)")
    ap.add_argument("--target", default="auto")
    ap.add_argument("--length-limit", type=int, default=320)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("LLM_MAX_CONCURRENCY", "8")))
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--mock", action="store_true", help="offline test with mock_llm_client (no model, random stances)")
    ap.add_argument("--memory", choices=["short", "matched"], required=True,
                    help="short = feed only (the first pilot); matched = round-1 feed pinned + own last posts + feed, "
                         "the small-group memory rule. Required, so a run can never pick one up by default.")
    ap.add_argument("--allow-model", action="store_true", help="permit a model other than the default Nemotron")
    args = ap.parse_args()
    for seed in (args.seeds or [args.seed]):
        args.seed = seed
        run_one(args)


def run_one(args):

    bad = [v for v in ("PERSONAS_PATH", "SCALE_ATTACKER_POOL", "SCALE_ARMS") if os.environ.get(v, "").strip()]
    if bad:
        raise SystemExit(f"{', '.join(bad)} set in this shell: these belong to the 32-agent runs. Unset them "
                         f"(or open a fresh terminal). Nothing was run.")
    if isinstance(args.target, str) and args.target == "auto":
        plan = json.load(open(TOPIC_PLAN_PATH))["topics"]
        args.target = float(plan[args.topic]["target"])
    else:
        args.target = float(args.target)

    soc = build_society(seed=args.seed, n_variants=args.n_variants)
    n = len(soc["agents"])
    sizes = arm_sizes(n)
    arms = [a for a in ARM_ORDER if a in args.arms]
    calls_org = n * 2 + n * 6 + 2 * n * 2
    calls_att = {a: n * 6 + 2 * n * 2 + sizes[a] * 3 * 4 for a in arms if a != "organic"}
    degs = sorted(len(v) for v in soc["adj"].values())
    print("=" * 70)
    print(f"SOCIETY RUN  topic={args.topic}  seed={args.seed}  agents={n}  target={args.target:+.2f}")
    print(f"  network: degree median {degs[n // 2]}, mean {sum(degs) / n:.1f}, max {degs[-1]}; feed size {FEED_SIZE}")
    print(f"  arms: " + ", ".join(f"{a} ({sizes[a]} attackers)" for a in arms))
    print(f"  calls: organic ~{calls_org}; " + "; ".join(f"{a} ~{c}" for a, c in calls_att.items())
          + "  (plus retries)")
    print("=" * 70)
    if args.estimate:
        return

    topic = load_topic(args.topic)
    P = paths(args.topic + ("_mock" if args.mock else ""), args.seed, args.n_variants)
    if args.mock:
        P = {k: v.replace(f"runs/{ns()}", f"runs/{ns()}_mock").replace(f"logs/{ns()}", f"logs/{ns()}_mock")
             for k, v in P.items()}

    if args.mock:
        from mock_llm_client import MockLLMClient
        make_client = lambda: MockLLMClient(seed=args.seed)
        model = "mock"
    else:
        if not os.environ.get("NVIDIA_API_KEY"):
            raise SystemExit("NVIDIA_API_KEY is not set (use NVIDIA_API_KEY=local with the HPC vLLM server).")
        from src.llm_client import LLMClient, DEFAULT_NVIDIA_MODEL
        model = os.environ.get("NVIDIA_MODEL", DEFAULT_NVIDIA_MODEL)
        if model != DEFAULT_NVIDIA_MODEL and not args.allow_model:
            raise SystemExit(f"NVIDIA_MODEL is {model!r}, not the default Nemotron. Pass --allow-model if intended.")
        make_client = lambda: LLMClient(provider="nvidia")
    print(f"MODEL: {model}   namespace: {ns()}   workers: {args.workers}")
    print(f"manifest: {P['manifest']}\nlogs: {P['log_dir']}")

    soc_json = society_to_json(soc)
    existing = load_json(P["society"], None)
    if existing and existing != json.loads(json.dumps(soc_json)):
        raise SystemExit(f"{P['society']} exists but differs from the society this code builds; refusing to mix.")
    if not existing:
        save_json(P["society"], soc_json)

    manifest = load_json(P["manifest"], {"completed": {}})
    other = {e.get("memory", "short") for e in manifest["completed"].values()} - {args.memory}
    if other:
        raise SystemExit(f"{P['manifest']} already holds memory={other.pop()!r} runs; use another RUN_NS for "
                         f"memory={args.memory!r} (e.g. society_mem_hpc). Nothing was run.")
    print(f"memory rule: {args.memory}")
    started = time.time()
    for arm in arms:
        key = f"{arm}|{args.topic}|{args.seed}"
        if key in manifest["completed"]:
            print(f"--- {arm}: already done, skipping")
            continue
        org_key = f"organic|{args.topic}|{args.seed}"
        if arm != "organic" and org_key not in manifest["completed"]:
            print(f"--- {arm}: organic twin not finished yet, skipping")
            continue
        frozen = manifest["completed"][org_key]["pre_stances"] if arm != "organic" else None
        print(f"\n--- {arm.upper()} | {args.topic} | seed {args.seed} ---")
        t0 = time.time()
        client = make_client()
        os.makedirs(P["ckpt_dir"], exist_ok=True)
        ckpt = os.path.join(P["ckpt_dir"], f"{arm}_{args.topic}_s{args.seed}.pkl")
        trial, meta = run_society_trial(client, soc, topic, arm, args.seed, frozen_initial_stances=frozen,
                                        target_stance=args.target, length_limit=args.length_limit,
                                        workers=args.workers, checkpoint_path=ckpt, memory=args.memory)
        meta["model"] = getattr(client, "model", model)
        meta["society_path"] = P["society"]
        os.makedirs(P["log_dir"], exist_ok=True)
        tp = os.path.join(P["log_dir"], f"{arm}_{args.topic}_s{args.seed}_{trial.trial_id}.json")
        mp = os.path.join(P["log_dir"], f"{trial.trial_id}_meta.json")
        save_json(tp, trial.to_dict())
        save_json(mp, meta)
        manifest = load_json(P["manifest"], {"completed": {}})
        manifest["completed"][key] = {"trial_id": trial.trial_id, "arm": arm, "seed": args.seed,
                                      "topic_id": args.topic, "path": tp, "meta_path": mp,
                                      "pre_stances": trial.pre_stances, "model": meta["model"],
                                      "attackers": meta["attackers"], "measured_ids": meta["measured_ids"],
                                      "memory": args.memory}
        save_json(P["manifest"], manifest)
        os.remove(ckpt)
        print(f"  saved -> {tp}   ({(time.time() - t0) / 60:.1f} min)")

    done = [a for a in arms if f"{a}|{args.topic}|{args.seed}" in load_json(P["manifest"], {"completed": {}})["completed"]]
    print("\n" + "=" * 70)
    print(f"{'SOCIETY COMPLETE' if len(done) == len(arms) else 'SOCIETY INCOMPLETE'} -- {len(done)}/{len(arms)} arms, "
          f"{(time.time() - started) / 60:.1f} min this session")
    print("=" * 70)


if __name__ == "__main__":
    main()
