"""Tabulate eval results: python table.py runs/p1 [runs/p2 ...]"""
import glob, json, sys
for R in sys.argv[1:]:
    rows = {}
    for f in sorted(glob.glob(f"{R}/*/eval_*/results.json")):
        d = json.load(open(f)); rows[f.split("/")[-3] + "/" + f.split("/")[-2]] = {r["tid"]: r["sr"] for r in d["results"]}
    if not rows:
        continue
    tids = sorted(set().union(*rows.values()))
    print(f"== {R}\n" + "run".ljust(26) + " ".join(f"t{t:02d}" for t in tids) + "  mean")
    for n, r in rows.items():
        print(n.ljust(26) + " ".join(f"{r[t]:.2f}" if t in r else " -- " for t in tids) + f"  {sum(r.values()) / len(r):.3f}")
