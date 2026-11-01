"""C1: does the policy go to the colour written in the (swapped) note? closest-approached cube per episode."""
import glob, json, sys
import numpy as np
def load(p): return {x['tid']: x for x in json.load(open(p))['results']}
def count(x, swap):
    tr = wr = ot = 0
    for i, p in enumerate(x['progress']):
        keys = list(range(len(p['mind']))); a = int(np.argmin(p['mind']))
        k = np.random.RandomState(i).choice([c for c in keys if c != p['true']]) if swap else None
        if a == p['true']: tr += 1
        elif a == k: wr += 1
        else: ot += 1
    return tr, wr, ot
ref = {}
for f in sorted(glob.glob('logs/eval_*t5x_v6*.json')):
    try: ref.update(load(f))
    except Exception: pass
for tag in sys.argv[1:]:
    for t, x in sorted(load(f'logs/ctl_{tag}.json').items()):
        tr, wr, ot = count(x, 'wrong' in tag)
        s = f"{tag} t{t} sr={x['sr']:.2f} closest: true {tr}, note colour {wr}, other {ot}"
        if t in ref: s += f" | normal notes: sr={ref[t]['sr']:.2f} closest true {count(ref[t], False)[0]}/{len(ref[t]['progress'])}"
        print(s)
