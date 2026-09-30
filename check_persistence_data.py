import json
import glob

for f in sorted(glob.glob("logs/manufactured_consensus_*.json")) + sorted(glob.glob("logs/hedged_disinformation_*.json")):
    d = json.load(open(f))
    n_msgs = len(d["messages"])
    has_removal = bool(d.get("post_removal_stances"))
    print(f"{n_msgs:3d} messages | has post_removal: {has_removal} | {f}")
