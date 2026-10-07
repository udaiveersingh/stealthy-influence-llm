from src.adversary import *
from src.organic_pipeline import load_agents, load_topic
import json
agents = load_agents()
for t in json.load(open("config/topics.json"))["topics"]:
    tid = t["topic_id"]
    cfg = AttackConfig(mechanism=Mechanism.DIRECTIONAL, targeting=Targeting.RANDOM, adaptivity=Adaptivity.STATIC,
        target_stance=-0.8, attack_budget=3, topic=tid, seed=1,
        topic_context=t["prompt_context"], scale_description=t["stance_question"])
    p = AttackerController(cfg, agents).hidden_objective_prompt(agents[0], "(conversation)")
    print("=" * 70, "\n", tid, "\n", p.split("Do not reveal")[0])
