"""check: CSS4 name of the measured colour of the first 'gone' region in each demo vs the cue cluster label"""
import sys, collections, numpy as np
sys.path.insert(0, "../seqpol")
from notes import episode_events, ChangeNoteWriter, DPIX, measured_color
from cue_labels import color_clusters
from matplotlib.colors import CSS4_COLORS, to_rgb
N = {k: np.array(to_rgb(v)) * 255 for k, v in CSS4_COLORS.items()}
def css(c): return min(N, key=lambda n: ((N[n] - c) ** 2).sum())
def measured(before, after, mask):
    m = ChangeNoteWriter.main_region(mask & (np.abs(after.astype(np.int16) - before.astype(np.int16)).max(-1) > DPIX))
    px = before[m].astype(np.float32)
    s = px.max(1) - px.min(1)
    return px[s >= np.quantile(s, 0.75)].mean(0)
tid = int(sys.argv[1]); k = {50: 3, 51: 5, 52: 9}[tid]; n = int(sys.argv[2])
mm = f"/data/workspaces/mikasa/data_mikasa_robo/mm/t{tid:02d}"
top = np.load(f"{mm}/top.npy", mmap_mode="r"); state = np.load(f"{mm}/state.npy")
st, L = np.load(f"{mm}/ep_start.npy"), np.load(f"{mm}/ep_len.npy")
lab = color_clusters(tid, k)[0]
tab = collections.defaultdict(collections.Counter)
for e in range(n):
    T, X = top[st[e]:st[e] + L[e]], state[st[e]:st[e] + L[e]]
    ev = [x for x in episode_events(T, X, 3) if x[0] > 0]
    if not ev: tab[int(lab[e])]["<none>"] += 1; continue
    t, tb, m = ev[0]
    tab[int(lab[e])][measured_color(np.asarray(T[t]), np.asarray(T[tb]), m, 'html')] += 1
for c in sorted(tab): print(c, tab[c].most_common(5))
