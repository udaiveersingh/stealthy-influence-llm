import json, shutil

path = "runs/gptoss/confirmatory_short_manifest__remote_work.json"
shutil.copy(path, path + ".bak")   # backup before touching it

m = json.load(open(path))
bad = [k for k, e in m["completed"].items()
       if e.get("model") == "nvidia/nemotron-3.5-lightning-30b-a3b"]

for k in bad:
    del m["completed"][k]

json.dump(m, open(path, "w"), indent=2)
print(f"Removed {len(bad)} mislabeled Nemotron entries:")
for k in sorted(bad):
    print(" ", k)
print(f"Remaining: {len(m['completed'])} entries")