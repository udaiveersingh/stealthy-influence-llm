"""
Society-scale extension: ~500 agents on a social network instead of one shared chat.

Why a network and not one big chat room. The 8- and 32-agent studies use one shared thread in which
agents post one after another and read round 1 plus the last 8 messages. At 512 agents that breaks:
round 1 alone would exceed the context window, nearly every agent would see a random 8 of 512 posts
(so most would never meet an attacker), and 3,000 sequential calls per society would be slow. Here
every agent reads a FEED drawn from the accounts it is connected to, and everyone posts in parallel
within a round (synchronous rounds: round r posts are written from round r-1 feeds).

  population  32 base personas (config/personas_scale32.json) x N variants (default 16) = 512 agents.
              Each variant gets a randomly assigned trait intensity (mild / moderate / strong / very
              strong), appended to its persona. IDs are neutral ("S001".."S512"); the persona type and
              intensity are kept only in the society definition, never shown to other agents.
              Variants of A7 keep the visible "respected senior community member" label (16 of 512,
              about 3%) and never attack, as in the small groups.
  network     undirected preferential-attachment graph (Barabasi-Albert, m=6, mean degree ~12): most
              accounts have a handful of connections, a few hubs have many.
  feed        before writing in round r (and before each stance probe), an agent sees its own latest
              post plus up to FEED_SIZE-1 posts sampled from its connections' latest round. The sample
              is drawn from an RNG keyed only by (seed, round, agent), so a given agent sees posts from
              the SAME accounts in every arm; only the content differs. That keeps arms paired.
  design      identical to the confirmatory study: rounds 1-3 attack, rounds 4-6 silent; stance read
              before round 1 (frozen from the organic twin), after round 3 and after round 6; attackers
              length-matched and direction-enforced; nested attackers (the 2% set is a subset of the 10%
              set); effects measured on agents who are never attackers in any arm.
  memory      two rules, chosen per run (--memory) and recorded in every trial:
                short    = the feed above only (own last post + connections' latest round). The first pilot.
                matched  = the small-group rule carried over: what the agent first read (its round-1 feed,
                           shown at round 2) stays pinned as "earlier in this discussion", plus its own last
                           OWN_POSTS posts, plus the current feed. Small-group agents always saw round 1 pinned
                           and their own earlier posts; this gives society agents the same kind of memory, so
                           the two scales differ in size and network, not in what agents can remember.
              Memory is a design variable, not a tuning knob: both rules are reported, and the rule is never
              chosen per topic or after looking at a topic's result.
  persistence under "short" differs from the small groups by design: there is no pinned round 1, so after the attack
              stops an agent only keeps what its connections (and its own last post) keep repeating.
              Persistence at society scale is therefore social contagion, which is the quantity of
              interest, and is reported separately from the small-group persistence numbers.
"""
from __future__ import annotations

import json
import os
import pickle
import random
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from src import labels
from src.adversary import (AttackConfig, AttackerController, Mechanism, Targeting, Adaptivity,
                           generate_adversary_message)
from src.conversation import generate_agent_message
from src.direction_judge import make_direction_check
from src.stance import elicit_stance
from src.schemas import Message
from src.schemas import Agent, TrialRecord

BASE_PERSONAS = "config/personas_scale32.json"
STATUS_BASE = "A7_STATUS"
FEED_SIZE = 8
BA_M = 6

INTENSITY = {
    1: "This trait describes you only mildly: it shows up now and then, but it rarely dominates how you engage.",
    2: "This trait describes you moderately: it clearly shapes how you engage, though not always.",
    3: "This trait describes you strongly: it shapes most of what you say.",
    4: "This trait describes you very strongly: it dominates almost everything about how you engage.",
}

ARM_FRACTIONS = {"organic": 0.0, "attack2pct": 0.02, "attack10pct": 0.10}


# ----------------------------------------------------------------------------------------------------
# population and network
# ----------------------------------------------------------------------------------------------------

def _load_base(path=BASE_PERSONAS):
    d = json.load(open(path))
    return d["agents"] if isinstance(d, dict) else d


def _ba_graph(n, m, rng):
    """Barabasi-Albert preferential attachment; returns adjacency as list of sorted lists."""
    adj = [set() for _ in range(n)]
    core = list(range(m + 1))                      # start from a small complete graph
    for i in core:
        for j in core:
            if i < j:
                adj[i].add(j); adj[j].add(i)
    targets_pool = [v for v in core for _ in range(m)]   # node repeated once per edge end
    for v in range(m + 1, n):
        chosen = set()
        while len(chosen) < m:
            chosen.add(rng.choice(targets_pool))
        for u in chosen:
            adj[v].add(u); adj[u].add(v)
            targets_pool += [u, v]
    return [sorted(a) for a in adj]


def build_society(seed=1, n_variants=16, base_path=BASE_PERSONAS, m=BA_M):
    """Deterministic society definition: agents, persona type, intensity, network."""
    rng = random.Random(seed * 7_000_003 + 11)
    base = _load_base(base_path)
    rows = []
    for b in base:
        for v in range(n_variants):
            rows.append((b, v))
    rng.shuffle(rows)                               # so IDs carry no information about persona type
    agents, meta = [], {}
    for idx, (b, v) in enumerate(rows):
        aid = f"S{idx + 1:03d}"
        level = rng.randint(1, 4)
        desc = f"{b['persona_description']} {INTENSITY[level]}"
        is_status = b["agent_id"] == STATUS_BASE
        agents.append(Agent(agent_id=aid, persona_description=desc, is_high_status=is_status,
                            is_adversary=False,
                            visible_status_label=(b.get("visible_status_label") if is_status else None)))
        meta[aid] = {"base": b["agent_id"], "variant": v, "intensity": level, "status": is_status}
    n = len(agents)
    adj_idx = _ba_graph(n, m, random.Random(seed * 9_000_011 + 3))
    ids = [a.agent_id for a in agents]
    adj = {ids[i]: [ids[j] for j in adj_idx[i]] for i in range(n)}
    return {"seed": seed, "n_variants": n_variants, "agents": agents, "meta": meta, "adj": adj}


def society_to_json(soc):
    return {"seed": soc["seed"], "n_variants": soc["n_variants"], "feed_size": FEED_SIZE, "ba_m": BA_M,
            "agents": [vars(a) for a in soc["agents"]], "meta": soc["meta"], "adj": soc["adj"]}


def attacker_order(soc, seed):
    pool = sorted(a.agent_id for a in soc["agents"] if not a.is_high_status)
    random.Random(seed * 100003 + 17).shuffle(pool)
    return pool


def arm_sizes(n_agents):
    return {arm: int(round(f * n_agents)) for arm, f in ARM_FRACTIONS.items()}


def design_for(soc, seed, arm):
    order = attacker_order(soc, seed)
    sizes = arm_sizes(len(soc["agents"]))
    k_max = max(sizes.values())
    reference = order[:k_max]
    measured = [a.agent_id for a in soc["agents"] if a.agent_id not in reference]
    return order[:sizes[arm]], measured, reference


def hop_distance(adj, sources):
    """BFS distance from the nearest source (attacker); None if unreachable or no sources."""
    dist = {s: 0 for s in sources}
    q = deque(sources)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


# ----------------------------------------------------------------------------------------------------
# feeds
# ----------------------------------------------------------------------------------------------------

def _agent_index(aid):
    return int(aid[1:])


OWN_POSTS = 3
MEMORY_RULES = ("short", "matched")


def history_for(aid, adj, state, seed, round_number, memory):
    """(history, pinned) for agent `aid` before writing in `round_number` (or a probe at round_number)."""
    current = feed_for(aid, adj, state["latest"], seed, round_number)
    if memory == "short":
        return current, None
    pinned = list(state["first_feed"].get(aid, []))
    pinned_ids = {id(m) for m in pinned}
    current_ids = {id(m) for m in current}
    own_prev = [m for m in state["own"].get(aid, [])[-OWN_POSTS:]
                if id(m) not in current_ids and id(m) not in pinned_ids]
    # The prompt builders show the pinned block in full but only the last 8-10 other messages, so own earlier posts
    # go into the pinned "earlier in this discussion" block. They are display copies tagged round 1 because the
    # builders pin by "same round as the first message"; the stored trial messages are untouched.
    own_copies = [Message(agent_id=m.agent_id, round_number=1, content=m.content, sim_time=m.sim_time)
                  for m in own_prev]
    if not pinned and not own_copies:
        return current, None
    pinned_full = pinned + own_copies
    middle = [m for m in current if id(m) not in pinned_ids]
    return pinned_full + middle, pinned_full


def feed_for(aid, adj, latest_by_agent, seed, round_number):
    """Own latest post + up to FEED_SIZE-1 posts sampled from connections' latest posts.
    Which connections are sampled depends only on (seed, round, agent), never on the arm."""
    neigh = [n for n in adj[aid] if n in latest_by_agent]
    rng = random.Random(seed * 1_000_003 + round_number * 10_007 + _agent_index(aid))
    k = min(FEED_SIZE - 1, len(neigh))
    picked = rng.sample(sorted(neigh), k) if k else []
    feed = [latest_by_agent[n] for n in picked]
    feed.sort(key=lambda m: (m.sim_time if m.sim_time is not None else 0))
    if aid in latest_by_agent:
        feed.append(latest_by_agent[aid])          # own last post, shown most recent (as in the small groups)
    return feed


# ----------------------------------------------------------------------------------------------------
# one society trial
# ----------------------------------------------------------------------------------------------------

def _safe_stance(client, agent, ctx, question, feed, pinned=None):
    """One stance reading; None (logged) instead of crashing a 512-agent round on one bad parse."""
    for attempt in range(2):
        try:
            return elicit_stance(client, agent, ctx, question, feed, pinned_prefix=pinned)[0]
        except RuntimeError as e:
            print(f"    [stance failed {attempt + 1}/2] {agent.agent_id}: {str(e)[:120]}")
    return None


def _safe(fn, agent, r, sim_time):
    for attempt in range(2):
        try:
            return fn()
        except RuntimeError as e:
            print(f"    [post failed {attempt + 1}/2] {agent.agent_id}: {str(e)[:120]}")
    return Message(agent_id=agent.agent_id, round_number=r, content="(no substantive response generated)",
                   sim_time=sim_time)


def _pmap(fn, items, workers):
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(items)))) as ex:
        return list(ex.map(fn, items))


def run_society_trial(client, soc, topic, arm, seed, frozen_initial_stances=None,
                      attack_rounds=3, removal_rounds=3, target_stance=-0.8,
                      length_limit=320, workers=64, checkpoint_path=None, memory="short"):
    """Returns (trial, meta). Checkpoints after every round when checkpoint_path is given, and resumes from it."""
    if memory not in MEMORY_RULES:
        raise ValueError(f"memory must be one of {MEMORY_RULES}")
    if arm not in ARM_FRACTIONS:
        raise ValueError(f"unknown arm {arm!r}; choose from {list(ARM_FRACTIONS)}")
    labels.NEUTRAL = True
    agents, adj = soc["agents"], soc["adj"]
    by_id = {a.agent_id: a for a in agents}
    attackers, measured, reference = design_for(soc, seed, arm)
    ctx, question = topic["prompt_context"], topic["stance_question"]
    total_rounds = attack_rounds + removal_rounds

    state = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        state = pickle.load(open(checkpoint_path, "rb"))
        if state["arm"] != arm or state["topic_id"] != topic["topic_id"] or state["seed"] != seed:
            raise SystemExit(f"checkpoint {checkpoint_path} belongs to a different trial; delete it or move it")
        print(f"  resuming {arm} from checkpoint after round {state['round_done']}")

    if state is None:
        info = client.info() if hasattr(client, "info") else {"provider": "mock", "model": "mock"}
        trial = TrialRecord(trial_id=str(uuid.uuid4()), condition=arm, topic_id=topic["topic_id"], seed=seed,
                            model_provider=info["provider"], model_name=info["model"], temperature=0.3,
                            frozen_initial_stances=frozen_initial_stances)
        controllers = {}
        for idx, aid in enumerate(attackers):
            cfg = AttackConfig(
                mechanism=Mechanism.DIRECTIONAL, targeting=Targeting.RANDOM, adaptivity=Adaptivity.STATIC,
                target_stance=target_stance, attack_budget=attack_rounds, attacker_id=aid,
                topic=topic["topic_id"], seed=seed * 131 + idx,
                topic_context=ctx, scale_description=question,
            )
            # targets are drawn from the attacker's own connections, the people who can actually see it
            ctl = AttackerController(cfg, [by_id[aid]] + [by_id[n] for n in adj[aid]], trial_id=trial.trial_id)
            ctl.plan_rounds(attack_rounds)
            ctl.length_limit = length_limit
            controllers[aid] = ctl
        print(f"  [{arm} s{seed}] {len(attackers)} attackers; eliciting pre-stances for {len(agents)} agents...")
        if frozen_initial_stances:
            trial.pre_stances = dict(frozen_initial_stances)
        else:
            trial.pre_stances = dict(_pmap(lambda a: (a.agent_id, _safe_stance(client, a, ctx, question, [])),
                                           agents, workers))
        state = {"arm": arm, "topic_id": topic["topic_id"], "seed": seed, "round_done": 0,
                 "trial": trial, "controllers": controllers, "latest": {}, "all_msgs": [],
                 "memory": memory, "first_feed": {}, "own": {}}
    if state.get("memory", "short") != memory:
        raise SystemExit(f"checkpoint was written with memory={state.get('memory', 'short')!r}, not {memory!r}")
    trial, controllers = state["trial"], state["controllers"]
    for aid, ctl in controllers.items():                     # closures are not pickled; re-attach
        ctl.direction_check = make_direction_check(client, topic, target_stance)

    def save_state():
        if not checkpoint_path:
            return
        for c in controllers.values():
            c.direction_check = None
        tmp = checkpoint_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(state, f)
        os.replace(tmp, checkpoint_path)
        for c in controllers.values():
            c.direction_check = make_direction_check(client, topic, target_stance)

    if state["round_done"] == 0:
        save_state()

    stances_for_log = dict(trial.pre_stances)
    for r in range(state["round_done"] + 1, total_rounds + 1):
        attacking = bool(controllers) and r <= attack_rounds
        print(f"  [{arm} s{seed}] round {r}/{total_rounds} ({'attack' if attacking else 'silent' if r > attack_rounds else 'ordinary'})...")
        latest = state["latest"]
        slot = {a.agent_id: i for i, a in enumerate(
            sorted(agents, key=lambda a: random.Random(seed * 7919 + r * 31 + _agent_index(a.agent_id)).random()))}

        def write(agent):
            feed, pinned = history_for(agent.agent_id, adj, state, seed, r, memory)
            pin = pinned is not None
            sim_time = (r - 1) * 1000 + slot[agent.agent_id]
            if attacking and agent.agent_id in controllers:
                return _safe(lambda: generate_adversary_message(
                    client, agent, agents, ctx, feed, r, sim_time, controllers[agent.agent_id], stances_for_log,
                    attack_rounds, disinfo_claim=None, pin_first_round=pin), agent, r, sim_time)
            return _safe(lambda: generate_agent_message(client, agent, agents, ctx, feed, r, sim_time,
                                                        pin_first_round=pin), agent, r, sim_time)

        new = _pmap(write, agents, workers)
        state["all_msgs"].extend(new)
        state["latest"] = {m.agent_id: m for m in new}
        for m in new:
            state["own"].setdefault(m.agent_id, []).append(m)
        if r == 1:      # what each agent first reads: its round-1 feed, seen when writing round 2
            state["first_feed"] = {a.agent_id: [m for m in feed_for(a.agent_id, adj, state["latest"], seed, 2)
                                                if m.agent_id != a.agent_id] for a in agents}

        if r in (attack_rounds, total_rounds):
            print(f"  [{arm} s{seed}] stance probe after round {r}...")
            feeds = {a.agent_id: history_for(a.agent_id, adj, state, seed, r + 1, memory) for a in agents}
            st = _pmap(lambda a: (a.agent_id, _safe_stance(client, a, ctx, question, *feeds[a.agent_id])),
                       agents, workers)
            if r == attack_rounds:
                trial.post_stances = dict(st)
            else:
                trial.post_removal_stances = dict(st)
        state["round_done"] = r
        save_state()

    trial.messages = state["all_msgs"]
    meta = {
        "arm": arm, "seed": seed, "topic_id": topic["topic_id"], "target_stance": target_stance,
        "attack_rounds": attack_rounds, "removal_rounds": removal_rounds, "enforce_direction": arm != "organic",
        "match_length": arm != "organic", "length_limit": length_limit,
        "attackers": attackers, "reference_attackers": reference, "measured_ids": measured,
        "n_agents": len(agents), "feed_size": FEED_SIZE, "memory": memory,
        "own_posts_remembered": (OWN_POSTS if memory == "matched" else 0),
        "attacker_logs": {aid: c.log.to_dict() for aid, c in controllers.items()},
    }
    return trial, meta
