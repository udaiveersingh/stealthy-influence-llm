import json

path = "runs/manifest.json"
m = json.load(open(path))
removed = [k for k in m["completed"] if k.startswith("disinformation|")]
for k in removed:
    del m["completed"][k]
json.dump(m, open(path, "w"), indent=2)
print(f"Removed {len(removed)} stale disinformation entries:", removed)
print(f"Remaining: {len(m['completed'])} entries")