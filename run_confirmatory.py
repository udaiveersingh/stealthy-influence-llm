"""
Confirmatory study runner -- one run for effectiveness, persistence and detectability.

  python run_confirmatory.py --estimate
  python run_confirmatory.py                     # seeds 1-10, arms organic / attack1 / attack4
  python run_confirmatory.py --seeds 1 2 3       # a subset (resumes, skips finished trials)

Always run with the rate limit set for the session, e.g.:
  $env:LLM_MAX_RPM="2"

Design: see src/confirmatory.py. Everything is written to its own manifest
(runs/confirmatory_manifest.json) and its own folder (logs/confirmatory/), so nothing from the
Step 6 pilot can be mixed in by accident. Run `python audit_logs.py logs/confirmatory/*.json`
afterwards (the pilot's audit globs logs/*.json and does not recurse).

Execution is sequential on purpose. The API allowance (~2 requests/min) is the bottleneck, and
parallel cells only produced rate-limit storms. Finished trials are skipped on re-run.
Per seed the organic arm runs first; the attack arms start from its frozen pre-stances.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from src.confirmatory import ARMS, PERSONA_ARMS, angles_for, run_confirmatory_trial, save_confirmatory_trial
from src.organic_pipeline import load_agents, load_topic

MANIFEST_PATH = "runs/confirmatory_manifest.json"
LOG_DIR = "logs/confirmatory"
ENF_MANIFEST_PATH = "runs/confirmatory_enf_manifest.json"
ENF_LOG_DIR = "logs/confirmatory_enf"
SHORT_MANIFEST_PATH = "runs/confirmatory_short_manifest.json"
SHORT_LOG_DIR = "logs/confirmatory_short"
DIVERSE_MANIFEST_PATH = "runs/confirmatory_diverse_manifest.json"
DIVERSE_LOG_DIR = "logs/confirmatory_diverse"
TOPIC_PLAN_PATH = "config/topic_plan.json"
DEFAULT_TOPIC = "ai_regulation"
PERSONA_MANIFEST_PATH = "runs/persona_manifest.json"
PERSONA_LOG_DIR = "logs/persona"

# Per-run paths. Kept separate from the DEFAULT constants above so that a second main() call in the same process cannot
# inherit the previous run's paths (the constants used to be overwritten).
_PATHS = {"manifest": MANIFEST_PATH, "log_dir": LOG_DIR}
MAX_CONSECUTIVE_FAILURES = 3
FAILURE_COOLDOWN_SECONDS = 90
STANCE_SAMPLES = 2


def topic_path(path, topic):
    """The default topic keeps its existing file names. Every other topic gets its own manifest / log folder, because the
    analysis tools key trials by seed and arm only, so two topics in one manifest would silently overwrite each other."""
    if topic == DEFAULT_TOPIC or not path:
        return path
    return path.replace(".json", f"__{topic}.json") if path.endswith(".json") else f"{path}__{topic}"


def resolve_target(raw, topic):
    if str(raw).lower() != "auto":
        return float(raw)
    if not os.path.exists(TOPIC_PLAN_PATH):
        raise SystemExit(f"--target auto needs {TOPIC_PLAN_PATH}. Run `python make_topic_plan.py` first.")
    plan = json.load(open(TOPIC_PLAN_PATH))["topics"]
    if topic not in plan:
        raise SystemExit(f"topic {topic!r} is not in {TOPIC_PLAN_PATH} (topics there: {', '.join(plan)}).")
    return float(plan[topic]["target"])


def load_manifest():
    if os.path.exists(_PATHS["manifest"]):
        with open(_PATHS["manifest"]) as f:
            return json.load(f)
    return {"completed": {}}


def save_manifest(m):
    os.makedirs(os.path.dirname(_PATHS["manifest"]), exist_ok=True)
    tmp = _PATHS["manifest"] + ".tmp"
    with open(tmp, "w") as f:
        json.dump(m, f, indent=2)
    os.replace(tmp, _PATHS["manifest"])


def key_for(arm, topic_id, seed):
    return f"{arm}|{topic_id}|{seed}"


def estimate(seeds, arms, n_agents=8, attack_rounds=3, removal_rounds=3, enforce=False):
    rounds = attack_rounds + removal_rounds
    conv = n_agents * rounds
    stance = n_agents * STANCE_SAMPLES
    organic = stance + conv + stance + stance          # fresh pre-stances + mid + final
    attack = conv + stance + stance                    # pre-stances are frozen: no calls
    per_seed = 0
    for a in arms:
        if a == "organic":
            per_seed += organic
        else:
            extra = ARMS[a] * attack_rounds * 3 if enforce else 0   # judge call + ~1 regeneration + judge, per message
            per_seed += attack + extra
    total = per_seed * len(seeds)
    print(f"  organic trial: {organic} calls; attack trial: {attack} calls (frozen pre-stances)"
          + (f"; with direction enforcement roughly +{3} calls per attacker message "
             f"(1-attacker ~{attack + ARMS['attack1'] * attack_rounds * 3}, 4-attacker ~{attack + ARMS['attack4'] * attack_rounds * 3})"
             if enforce else ""))
    print(f"  {len(seeds)} seeds x {len(arms)} arms = {len(seeds) * len(arms)} trials, ~{total} calls "
          f"(plus retries, typically +10-20%)")
    for rate in (1.8, 3.0, 4.0):
        print(f"    at {rate:.1f} calls/min: ~{total / rate / 60:.0f} hours")
    print("  (1.8/min is what the confirmatory run actually achieved at LLM_MAX_RPM=2; raise LLM_MAX_RPM to go faster)")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
    ap.add_argument("--arms", nargs="+", default=None, choices=list(ARMS),
                    help="default: all arms; with --match-length the default is attack4 only")
    ap.add_argument("--topic", default="ai_regulation")
    ap.add_argument("--target", default="-0.8",
                    help='a number such as -0.8, or "auto" to read the per-topic target from config/topic_plan.json')
    ap.add_argument("--attack-rounds", type=int, default=3)
    ap.add_argument("--removal-rounds", type=int, default=3)
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--diversify-angles", action="store_true",
                    help="each attacker argues from ONE distinct angle (no shared talking points). Requires "
                         "--match-length --enforce-direction; default arm attack4.")
    ap.add_argument("--persona-sweep", action="store_true",
                    help="one attacker per trial from A3 / A6 / A7(high status) / A8, length-matched and direction-enforced; "
                         "implies --match-length --enforce-direction")
    ap.add_argument("--match-length", action="store_true",
                    help="length-matched attackers: same 1-2 short sentences instruction as ordinary agents, drafts over "
                         "--length-limit characters rejected. Requires --enforce-direction.")
    ap.add_argument("--length-limit", type=int, default=320)
    ap.add_argument("--enforce-direction", action="store_true",
                    help="judge every attacker message before posting; regenerate until judged toward the target")
    ap.add_argument("--manifest", default=None,
                    help="default runs/confirmatory_manifest.json, or runs/confirmatory_enf_manifest.json with --enforce-direction")
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--organic-from", default=None,
                    help="manifest to copy the organic-arm entries from (default with --enforce-direction: the unenforced manifest)")
    args = ap.parse_args()

    args.target = resolve_target(args.target, args.topic)
    if args.persona_sweep:
        args.enforce_direction = True
        args.match_length = True
        if args.arms is None:
            args.arms = list(PERSONA_ARMS)
    persona_run = any(a in PERSONA_ARMS for a in (args.arms or []))
    if persona_run:
        if not (args.enforce_direction and args.match_length):
            raise SystemExit("persona arms must be run with --persona-sweep (or --enforce-direction --match-length): "
                             "every persona is length-matched and direction-enforced so personas are compared like for like.")
        if any(a not in PERSONA_ARMS and a != "organic" for a in args.arms):
            raise SystemExit("do not mix persona arms with attack1/attack4 in one run: they use different manifests.")
    if args.diversify_angles and not (args.match_length and args.enforce_direction):
        raise SystemExit("--diversify-angles requires --match-length --enforce-direction, so it is compared with the "
                         "length-matched, direction-enforced arm like for like.")
    if args.diversify_angles:
        try:
            angles_for(args.topic, args.target)
        except ValueError as e:
            raise SystemExit(str(e))
    if args.match_length and not args.enforce_direction:
        raise SystemExit("--match-length requires --enforce-direction (the length-matched arm is compared with the "
                         "direction-enforced arm, so it must be enforced too).")

    default_manifest = PERSONA_MANIFEST_PATH if persona_run else DIVERSE_MANIFEST_PATH if args.diversify_angles else SHORT_MANIFEST_PATH if args.match_length else (ENF_MANIFEST_PATH if args.enforce_direction else MANIFEST_PATH)
    default_logs = PERSONA_LOG_DIR if persona_run else DIVERSE_LOG_DIR if args.diversify_angles else SHORT_LOG_DIR if args.match_length else (ENF_LOG_DIR if args.enforce_direction else LOG_DIR)
    _PATHS["manifest"] = args.manifest or topic_path(default_manifest, args.topic)
    _PATHS["log_dir"] = args.log_dir or topic_path(default_logs, args.topic)
    organic_from = args.organic_from or (topic_path("runs/confirmatory_manifest.json", args.topic) if args.enforce_direction else None)
    if args.arms is None:
        args.arms = ["attack4"] if (args.match_length or args.diversify_angles) else list(ARMS)
    if args.enforce_direction and os.path.abspath(_PATHS["manifest"]) in (os.path.abspath(topic_path("runs/confirmatory_manifest.json", args.topic)),):
        raise SystemExit("--enforce-direction / --match-length must not write into the unenforced manifest (same keys would be "
                         "silently skipped as 'already done'). Use the default enforced manifest.")

    # organic must come first within a seed: the attack arms start from its pre-stances
    arms = sorted(args.arms, key=lambda a: ARMS[a])
    print("=" * 70)
    print("CONFIRMATORY STUDY")
    print("=" * 70)
    estimate(args.seeds, arms, attack_rounds=args.attack_rounds, removal_rounds=args.removal_rounds,
             enforce=args.enforce_direction)
    if args.estimate:
        return

    # Preflight. Environment variables belong to ONE terminal window in PowerShell, so a fresh window has neither.
    if not os.environ.get("NVIDIA_API_KEY"):
        raise SystemExit("NVIDIA_API_KEY is not set in THIS terminal window (variables do not carry over to a new "
                         "window). In PowerShell:   $env:NVIDIA_API_KEY=\"<your key>\"   then run again. Nothing was run.")
    if "LLM_MAX_RPM" not in os.environ:
        import src.llm_client as _lc
        if hasattr(_lc, "set_max_rpm"):
            _lc.set_max_rpm(4)
        print("WARNING: LLM_MAX_RPM is not set in this terminal; using a safe default of 4 requests/minute for this run "
              "(the library default of 40 is far above what your account allows).")

    from src.llm_client import LLMClient
    agents, topic = load_agents(), load_topic(args.topic)
    manifest = load_manifest()
    if organic_from and os.path.exists(organic_from):
        src_m = json.load(open(organic_from))["completed"]
        copied = 0
        for seed in args.seeds:
            k = key_for("organic", args.topic, seed)
            if k not in manifest["completed"] and k in src_m:
                manifest["completed"][k] = dict(src_m[k]); copied += 1
        if copied:
            save_manifest(manifest)
            print(f"Reused {copied} organic control(s) from {organic_from} (same pre-stances, same seeds).")
    print(f"Manifest: {_PATHS['manifest']}   logs: {_PATHS['log_dir']}   direction enforcement: {'ON' if args.enforce_direction else 'off'}"
          f"   length matching: {('ON (limit %d chars)' % args.length_limit) if args.match_length else 'off'}"
          f"   distinct angles: {'ON' if args.diversify_angles else 'off'}")
    print(f"\nAlready completed (will skip): {len(manifest['completed'])} trials")
    print(f"LLM_MAX_RPM={os.environ.get('LLM_MAX_RPM', '(default 40 -- set 2 for your account)')}\n")

    started, consecutive, done, failed = time.time(), 0, 0, 0
    for seed in args.seeds:
        for arm in arms:
            key = key_for(arm, args.topic, seed)
            if key in manifest["completed"]:
                print(f"--- {arm} | seed {seed}: already done, skipping")
                continue
            if arm != "organic" and key_for("organic", args.topic, seed) not in manifest["completed"]:
                print(f"--- {arm} | seed {seed}: no organic control for this seed yet, skipping "
                      f"(run the organic arm first)")
                continue
            frozen = None
            if arm != "organic":
                frozen = manifest["completed"][key_for("organic", args.topic, seed)]["pre_stances"]

            print(f"\n--- {arm.upper()} | {args.topic} | seed {seed} ---")
            try:
                trial, meta = run_confirmatory_trial(
                    LLMClient(provider="nvidia"), agents, topic, arm, seed,
                    frozen_initial_stances=frozen, attack_rounds=args.attack_rounds,
                    removal_rounds=args.removal_rounds, target_stance=args.target,
                    enforce_direction=(args.enforce_direction and arm != 'organic'),
                    match_length=(args.match_length and arm != 'organic'), length_limit=args.length_limit,
                    diversify_angles=(args.diversify_angles and arm != 'organic'))
                trial_path, meta_path = save_confirmatory_trial(trial, meta, _PATHS["log_dir"])
            except Exception as e:                       # noqa: BLE001 -- keep the run alive, count it
                failed += 1
                consecutive += 1
                print(f"\n!!! TRIAL FAILED: {arm} seed {seed}: {e}", file=sys.stderr)
                if consecutive >= MAX_CONSECUTIVE_FAILURES:
                    print(f"\n!!! {consecutive} consecutive failures -- stopping instead of burning through "
                          f"the remaining trials. READ THE ERROR ABOVE: a missing/invalid NVIDIA_API_KEY, or rate limiting "
                          f"(lower LLM_MAX_RPM and wait a few minutes). Finished trials are skipped on re-run.", file=sys.stderr)
                    break
                time.sleep(FAILURE_COOLDOWN_SECONDS)
                continue

            consecutive = 0
            done += 1
            manifest = load_manifest()
            manifest["completed"][key] = {
                "trial_id": trial.trial_id, "arm": arm, "seed": seed, "topic_id": args.topic,
                "path": trial_path, "meta_path": meta_path, "pre_stances": trial.pre_stances,
                "attackers": meta["attackers"], "measured_ids": meta["measured_ids"],
                "enforce_direction": meta.get("enforce_direction", False), "match_length": meta.get("match_length", False), "diversify_angles": meta.get("diversify_angles", False),
            }
            save_manifest(manifest)
            print(f"  saved -> {trial_path}")
        else:
            continue
        break

    print("\n" + "=" * 70)
    total = len(load_manifest()["completed"])
    status = "RUN COMPLETE" if not failed and consecutive < MAX_CONSECUTIVE_FAILURES else "RUN INCOMPLETE"
    print(f"{status} -- {done} trials this session, {failed} failed, {total} in manifest, "
          f"{(time.time() - started) / 3600:.1f} h")
    print("=" * 70)


if __name__ == "__main__":
    main()