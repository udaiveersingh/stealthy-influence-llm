import json
m = json.load(open('runs/gptoss/confirmatory_short_manifest__remote_work.json'))
by_model = {}
for key, e in m['completed'].items():
    model = e.get('model')
    by_model.setdefault(model, []).append((e.get('arm'), e.get('seed'), key))

for model, entries in by_model.items():
    print(f"\n=== model: {model}  ({len(entries)} entries) ===")
    for arm, seed, key in sorted(entries, key=lambda x: (x[0], x[1])):
        print(f"  {arm:10s} seed {seed:3d}   key={key}")