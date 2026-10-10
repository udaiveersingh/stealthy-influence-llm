import json, sys
topic = sys.argv[1] if len(sys.argv) > 1 else "remote_work"
def load(p): return json.load(open(p))
org = load(f"runs/confirmatory_manifest__{topic}.json")["completed"]
div = load(f"runs/confirmatory_diverse_manifest__{topic}.json")["completed"]
sign = -1  # target -0.8
for name, man, arm in (("ORGANIC", org, "organic"), ("DIVERSE", div, "attack4")):
    print(name)
    for seed in range(1, 11):
        e = man.get(f"{arm}|{topic}|{seed}")
        if not e: continue
        t = load(e["path"]); ids = load(e["meta_path"])["measured_ids"]
        pre = t["pre_stances"]; fin = t["post_removal_stances"]
        moves = {a: round(sign*(fin[a]-pre[a]), 2) for a in ids if a in fin}
        print(f"  seed {seed:>2} mean {sum(moves.values())/len(moves):+.2f}  {moves}")