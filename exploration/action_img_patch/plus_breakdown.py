import json, glob, collections, re, sys
O = sys.argv[1]
s = collections.defaultdict(lambda: [0, 0]); pert = collections.defaultdict(lambda: [0, 0]); tot = [0, 0]
KEYS = ["view", "light", "noise", "add", "moved", "table", "tb", "level", "texture", "background", "initstate"]
for f in glob.glob(O + "/*/gpu0_task*_results.json"):
    d = json.load(open(f)); k = d["task_suite"]
    s[k][0] += d["successes"]; s[k][1] += d["total_episodes"]; tot[0] += d["successes"]; tot[1] += d["total_episodes"]
    desc = d["task_description"]; tags = [key for key in KEYS if re.search(r"\b" + key + r"\b", desc)]
    tag = tags[0] if tags else "other"
    if tag == "initstate" and "view" in tags: tag = "view"
    pert[tag][0] += d["successes"]; pert[tag][1] += d["total_episodes"]
print("=== by suite")
for k, v in sorted(s.items()): print(f"  {k:15s} {v[0]:3d}/{v[1]:3d} = {100*v[0]/v[1]:5.1f}%")
print(f"  OVERALL         {tot[0]:3d}/{tot[1]:3d} = {100*tot[0]/tot[1]:5.1f}%")
print("=== by perturbation keyword")
for k, v in sorted(pert.items(), key=lambda x: -x[1][1]): print(f"  {k:10s} {v[0]:3d}/{v[1]:3d} = {100*v[0]/v[1]:5.1f}%")
