import json
import glob
import os

folder = r"./data_local/SolDef_AI/Labeled"
files = glob.glob(os.path.join(folder, "*.json"))

all_groups = set()
for path in files:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print("skip", path, "error:", e)
        continue

    # Candidate arrays of annotations
    candidates = []
    if isinstance(data, dict):
        # common fields in annotation JSONs
        for key in ["labels", "annotation", "objects", "regions", "shapes"]:
            if key in data and isinstance(data[key], list):
                candidates.append(data[key])
        for val in data.values():
            if isinstance(val, list):
                candidates.append(val)
    elif isinstance(data, list):
        candidates.append(data)

    for items in candidates:
        for item in items:
            if isinstance(item, dict):
                for group_key in ["group", "label_group", "labelGroup", "category", "class", "label"]:
                    if group_key in item:
                        all_groups.add(str(item[group_key]))
            elif isinstance(item, str):
                all_groups.add(item)

print("json files:", len(files))
print("distinct label groups:", len(all_groups))
for group in sorted(all_groups):
    print(group)