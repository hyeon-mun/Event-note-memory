"""Figures for paper/main.tex (vector PDF). Run with the MIKASA venv (has matplotlib):
   /data/workspaces/mikasa/MIKASA-Robo/.venv/bin/python paper/make_figs.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.image as mpimg

H = os.path.dirname(os.path.abspath(__file__))
FIG = f"{H}/../docs/figures"
OUT = f"{H}/figs"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    if os.environ.get("PREVIEW"):
        fig.savefig(f"{os.environ['PREVIEW']}/{name}.png", dpi=200, bbox_inches="tight")


def pipeline():
    fig, ax = plt.subplots(figsize=(7.0, 2.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 27); ax.axis("off")

    def box(x, y, w, h, title, sub, hl=False, ts=5.9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.0",
                                    fc="#e4ecfb" if hl else "#f4f5f7", ec="#2f6fd6" if hl else "#8a93a3", lw=0.8))
        ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=ts, weight="bold")
        ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=5.1, color="#4a5262", linespacing=1.1)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#4a5262"))

    y, w, g = 15.5, 14.2, 2.6
    items = [("Top camera", "+ proprio\nevery step"), ("(1) Event detection", "robot still\n+ pixel burst"),
             ("(2) Change crop", "largest changed region\nzoomed to 384 px"), ("(3) VLM caption", "SmolVLM2-2.2B\nbefore / after crop"),
             ("(4) Selection", "keep removed side\n\"Gone: X.\""), ("(5) Representation", "measured colour\nHTML16 + orange")]
    xs = [i * (w + g) for i in range(len(items))]
    for x, (t, sub) in zip(xs, items):
        box(x + 0.3, y, w, 10, t, sub)
    for x in xs[:-1]:
        arrow(x + w + 0.4, y + 5, x + w + g + 0.2, y + 5)
    box(xs[4] + 0.3, 1.5, 2 * w + g, 10, "(6) Note memory", "write order, drop repeats\nkeep last K = 8")
    arrow(xs[5] + w / 2, y - 0.2, xs[5] + w / 2, 11.7)
    box(0.3, 1.5, 45.5, 10, "instruction + \"Notes: (1) a blue cube. (2) Gone: a aqua cube.\"",
        "+ current top / wrist images + proprio", hl=True, ts=5.3)
    box(49.3, 1.5, 15.2, 10, "$\\pi_{0.5}$, fine-tuned", "-> action chunk")
    arrow(xs[4] + 0.1, 6.5, 64.4, 6.5)
    arrow(49.1, 6.5, 46.1, 6.5)
    save(fig, "fig_pipeline")


def example():
    """RC9-Long demo episode 3 (cue = aqua cube): frames, zoomed crops, notes of v3 / v4 / v6."""
    ims = [("rc9_example_3_0_frame.png", "t=0: cue shown"), ("rc9_example_3_1_frame.png", "t=99: before"),
           ("rc9_example_3_1_after.png", "t=101: cue gone"), ("rc9_example_3_1_cb.png", "crop before (zoom)"),
           ("rc9_example_3_1_ca.png", "crop after (zoom)"), ("rc9_example_3_3_frame.png", "t=384: candidates")]
    fig, axs = plt.subplots(1, 6, figsize=(7.0, 1.35))
    for a, (f, t) in zip(axs, ims):
        a.imshow(mpimg.imread(f"{FIG}/{f}"), interpolation="nearest"); a.set_xticks([]); a.set_yticks([])
        a.set_title(t, fontsize=7, pad=2)
    txt = ("v3: \"Change: a cyan cube, a maroon cube -> brown.\"    "
           "v4: \"Gone: a cyan cube, a maroon cube.\"    "
           "v6: \"Gone: a aqua cube.\"")
    save(fig, "fig_example")


def errors():
    rows = [("no memory", 5, 2, 43), ("v2", 32, 11, 7), ("v3", 22, 8, 20), ("v4", 33, 13, 4),
            ("v5", 28, 4, 18), ("v6", 33, 5, 12), ("oracle", 42, 1, 7)]
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    for i, (n, r, a, o) in enumerate(rows):
        y = len(rows) - 1 - i
        for left, w, c, h in ((0, r, "#1f8a5a", ""), (r, a, "#d9a520", "//"), (r + a, o, "#c2453a", "")):
            ax.barh(y, w, left=left, color=c, hatch=h, edgecolor="white", lw=0.4, height=0.72)
            if w >= 4:
                ax.text(left + w / 2, y, str(w), ha="center", va="center", fontsize=6.5, color="white")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_xlim(0, 50); ax.set_xlabel("episodes (RC9-Long, n = 50)")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#1f8a5a", label="right cube"), Patch(fc="#d9a520", hatch="//", ec="white", label="similar-colour pair"),
                       Patch(color="#c2453a", label="other cube")], fontsize=6, ncol=3, loc="lower center",
              bbox_to_anchor=(0.45, 1.0), frameon=False, handlelength=1.2, columnspacing=0.8)
    save(fig, "fig_errors")


if __name__ == "__main__":
    pipeline(); example(); errors()
    print(sorted(os.listdir(OUT)))
