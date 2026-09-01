import json, os, glob, datetime

for path in glob.glob("logs/*.json"):
    with open(path) as f:
        data = json.load(f)
    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y%m%d_%H%M%S")
    trial_short = data["trial_id"].split("-")[0]
    new_name = f"{mtime}__{data['condition']}__{data['topic_id']}__seed{data['seed']:04d}__{trial_short}.json"
    new_path = os.path.join("logs", new_name)
    os.rename(path, new_path)
    print(f"{path} -> {new_path}")