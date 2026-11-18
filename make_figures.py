"""
Figures for the paper. Run from the repo root after pool_confirmatory.py:
    python make_figures.py --cells results_hpc2/pooled_cells.csv --out paper_figures
Without --cells (or if the file is missing) Fig. 2 is drawn from the cell means below WITHOUT per-cell
confidence intervals (provisional); with it, per-cell 95% CIs (mean +/- t(0.975, n-1) * se) are added.
All other numbers are the final pooled results (pool_confirmatory.py, analyze_society.py).
"""
import argparse, csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
from scipy import stats

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
COL = {"Nemotron": "#4950BC", "gpt-oss": "#D9822B", "Gemma": "#2E8B6E"}
MODELS = ["Nemotron", "gpt-oss", "Gemma"]
TOPICS = ["ai_regulation", "college_value", "remote_work", "free_speech_harm", "individual_vs_collective",
          "traditional_family_roles", "centralized_governance", "religion_public_life"]
TLAB = {"ai_regulation": "AI regulation", "college_value": "College value", "remote_work": "Remote work",
        "free_speech_harm": "Free speech vs harm", "individual_vs_collective": "Individual vs collective",
        "traditional_family_roles": "Traditional family roles", "centralized_governance": "Centralized governance",
        "religion_public_life": "Religion in public life"}
END = {"Nemotron": [.201, .106, .056, .206, .140, .077, .104, .097], "gpt-oss": [.123, .163, .212, .177, .033, .218, .240, .319],
       "Gemma": [.182, .180, .194, .121, .171, .355, .148, .090]}
FIN = {"Nemotron": [.124, .047, -.220, .114, .113, .016, -.074, .032], "gpt-oss": [.032, .085, .171, .094, .006, .101, .081, .126],
       "Gemma": [.180, .078, .081, .091, .052, .104, .070, .011]}
POOL = {"end": {"Nemotron": (.149, .101, .197), "gpt-oss": (.184, .117, .250), "Gemma": (.182, .121, .242), "All": (.170, .141, .199)},
        "final": {"Nemotron": (.054, -.014, .123), "gpt-oss": (.087, .051, .122), "Gemma": (.087, .048, .126), "All": (.075, .050, .100)}}


def load_cells(path):
    if not path or not os.path.exists(path):
        return None
    out = {}
    for r in csv.DictReader(open(path)):
        if r["arm"] != "attack4":
            continue
        n = int(r["n_seeds"]); tc = stats.t.ppf(0.975, n - 1)
        out[(r["model"], r["topic"])] = {w: (float(r[f"{w}_mean"]), tc * float(r[f"{w}_se"])) for w in ("end", "final")}
    return out


def fig1(out):
    fig = plt.figure(figsize=(6.3, 2.05))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 63); ax.set_ylim(0, 20.5); ax.axis("off")
    # timeline
    ax.text(0.6, 19.4, "(a) Protocol for each attacked group and its twin", fontsize=8, weight="bold", va="top")
    blocks = [(0.9, 4.2, "Pre-stance\n(frozen,\nshared)", "#EEEEF6"), (5.4, 10.8, "Rounds 1-3\nattackers active", "#DADBF1"),
              (16.6, 3.6, "Stance\n(end)", "#EEEEF6"), (20.6, 10.8, "Rounds 4-6\nattackers silent", "#F4F4F8"), (31.8, 3.6, "Stance\n(final)", "#EEEEF6")]
    for x, w, t, c in blocks:
        ax.add_patch(FancyBboxPatch((x, 9.6), w, 6.6, boxstyle="round,pad=0.02,rounding_size=0.6", fc=c, ec="#9A9AAE", lw=0.6))
        ax.text(x + w / 2, 12.9, t, ha="center", va="center", fontsize=6.2)
    ax.text(0.6, 7.6, "Effect = shift toward the attackers' target in the attacked group\n          minus the same shift in its twin "
            "(same seed, personas, pre-stances).\nRetention = final effect / end effect.", fontsize=6.6, va="top")
    # setups
    ax.text(36.5, 19.4, "(b) Three population settings", fontsize=8, weight="bold", va="top")
    import math, random
    def ring(cx, cy, n, natt, rx, ry, r):
        ax.add_patch(FancyBboxPatch((cx - 2.1, cy - 0.9), 4.2, 1.8, boxstyle="round,pad=0.02,rounding_size=0.4", fc="#DADBF1", ec="#9A9AAE", lw=0.5))
        ax.text(cx, cy, "shared\nthread", ha="center", va="center", fontsize=5)
        step = max(1, round(n / natt))
        for k in range(n):
            a = -math.pi / 2 + 2 * math.pi * k / n
            x, y = cx + rx * math.cos(a), cy + ry * math.sin(a)
            ax.plot([cx, x], [cy, y], color="#D5D6E6", lw=0.4, zorder=0)
            ax.add_patch(plt.Circle((x, y), r, fc="#4950BC" if k % step == 0 else "white", ec="#4950BC" if k % step == 0 else "#888", lw=0.5, zorder=2))
    ring(40.0, 11.5, 8, 4, 3.0, 3.8, 0.5); ax.text(40.0, 4.6, "8 agents\n4 attackers", ha="center", va="top", fontsize=6)
    ring(48.2, 11.5, 32, 8, 3.5, 4.4, 0.28); ax.text(48.2, 4.6, "32 agents\n4 or 8 attackers", ha="center", va="top", fontsize=6)
    rnd = random.Random(3); pts = []
    for k in range(30):
        g = k; c, rr = g % 6, g // 6
        pts.append((53.4 + c * 1.45 + rnd.uniform(-.35, .35), 7.6 + rr * 1.8 + rnd.uniform(-.4, .4)))
    deg = [0] * 30; edges = []
    for k in range(1, 30):
        cand = sorted(range(k), key=lambda j: -((deg[j] + 1) ** .5) / (math.dist(pts[k], pts[j]) + .5) - rnd.random() * .2)[:2 if k > 2 else 1]
        for j in cand:
            edges.append((k, j)); deg[k] += 1; deg[j] += 1
    for a, b in edges:
        ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]], color="#C9CAD9", lw=0.4, zorder=0)
    for k, (x, y) in enumerate(pts):
        att = k in (4, 17, 26)
        ax.add_patch(plt.Circle((x, y), 0.25 + min(deg[k], 6) * 0.05, fc="#4950BC" if att else "white", ec="#4950BC" if att else "#888", lw=0.5, zorder=2))
    ax.text(57.6, 4.6, "512 agents\nnetwork feeds", ha="center", va="top", fontsize=6)
    ax.add_patch(plt.Circle((37.0, 1.0), 0.4, fc="#4950BC", ec="#4950BC")); ax.text(37.8, 1.0, "covert attacker", va="center", fontsize=6.2)
    ax.add_patch(plt.Circle((45.0, 1.0), 0.4, fc="white", ec="#888", lw=0.5)); ax.text(45.8, 1.0, "ordinary agent (network schematic)", va="center", fontsize=6.2)
    fig.savefig(os.path.join(out, "fig1_design.pdf")); plt.close(fig)


def fig2(out, cells):
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 4.2), sharey=True)
    rows, labels = [], []
    y = 0
    for t in TOPICS:
        for m in MODELS:
            rows.append((y, m, t)); y += 1
        labels.append((y - 2, TLAB[t])); y += 0.6
    ypool = y + 0.6
    for ax, w, src, title in ((axes[0], "end", END, "End of attack (after round 3)"), (axes[1], "final", FIN, "After 3 silent rounds (round 6)")):
        ax.axvline(0, color="#777", lw=0.6)
        for yy, m, t in rows:
            v = src[m][TOPICS.index(t)]
            if cells and (m, t) in cells:
                v, h = cells[(m, t)][w]
                ax.plot([v - h, v + h], [yy, yy], color=COL[m], lw=0.9)
            ax.plot(v, yy, "o", ms=3.2, color=COL[m])
        for k, m in enumerate(MODELS + ["All"]):
            mu, lo, hi = POOL[w][m]; yy = ypool + k * 1.5
            c = COL.get(m, "black")
            ax.add_patch(Polygon([[lo, yy], [mu, yy - 0.42], [hi, yy], [mu, yy + 0.42]], closed=True, fc=c, ec=c))
        ax.set_title(title); ax.set_xlabel("Effect toward target (attacked - twin)")
        ax.set_xlim(-0.45, 0.6) if cells else ax.set_xlim(-0.2, 0.4)
        ax.grid(axis="x", color="#E6E6EE", lw=0.5)
    axes[0].set_yticks([l[0] for l in labels] + [ypool + k * 1.5 for k in range(4)])
    axes[0].set_yticklabels([l[1] for l in labels] + ["Pooled: Nemotron", "Pooled: gpt-oss", "Pooled: Gemma", "Pooled: all 24"])
    axes[0].invert_yaxis()
    hs = [plt.Line2D([], [], marker="o", ls="", color=COL[m], label=m) for m in MODELS]
    axes[1].legend(handles=hs, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_forest.pdf")); plt.close(fig)


def fig3(out):
    import numpy as np
    M = np.array([FIN[m] for m in MODELS]).T
    fig, ax = plt.subplots(figsize=(3.05, 2.6))
    im = ax.imshow(M, cmap="PuOr_r", vmin=-0.25, vmax=0.25, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(MODELS)
    ax.set_yticks(range(8)); ax.set_yticklabels([TLAB[t] for t in TOPICS])
    for i in range(8):
        for j in range(3):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=6.8, color="white" if abs(M[i, j]) > 0.16 else "black")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig3_persistence.pdf")); plt.close(fig)


def fig4(out):
    # Column-width version: (a) on top, (b) below.
    fig, axes = plt.subplots(2, 1, figsize=(3.15, 2.55), gridspec_kw={"height_ratios": [1, 1]})
    ax = axes[0]
    labs = ["4/8\n50%", "8/32\n25%", "4/32\n12.5%", "51/512\n10%", "10/512\n2%"]
    e = [0.128, 0.149, 0.096, 0.030, 0.006]; f = [0.036, 0.094, 0.052, 0.023, 0.012]
    x = range(len(labs))
    ax.bar([i - 0.2 for i in x], e, 0.38, color="#4950BC", label="End of attack")
    ax.bar([i + 0.2 for i in x], f, 0.38, color="#A9ACE0", label="After silence")
    ax.set_xticks(list(x)); ax.set_xticklabels(labs, fontsize=6.8); ax.set_ylabel("Effect")
    ax.set_title("(a) By attacker share (Nemotron)", loc="left"); ax.legend(frameon=False, fontsize=6.8, loc="upper right")
    ax.grid(axis="y", color="#E6E6EE", lw=0.5); ax.set_axisbelow(True)
    ax = axes[1]
    labs2 = ["Direct contacts\n(1 link)", "Two links\naway", "All measured\nagents"]
    e2 = [0.049, -0.004, 0.030]; f2 = [0.034, 0.002, 0.023]
    x2 = range(3)
    ax.bar([i - 0.2 for i in x2], e2, 0.38, color="#4950BC"); ax.bar([i + 0.2 for i in x2], f2, 0.38, color="#A9ACE0")
    ax.axhline(0, color="#777", lw=0.6)
    ax.set_xticks(list(x2)); ax.set_xticklabels(labs2, fontsize=6.8); ax.set_ylabel("Effect")
    ax.set_title("(b) 512-agent network, 10% attackers", loc="left")
    ax.grid(axis="y", color="#E6E6EE", lw=0.5); ax.set_axisbelow(True)
    fig.tight_layout(h_pad=0.6); fig.savefig(os.path.join(out, "fig4_scale.pdf")); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="results_hpc2/pooled_cells.csv")
    ap.add_argument("--out", default="paper_figures")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    cells = load_cells(a.cells)
    print("per-cell CIs:", "yes" if cells else "no (provisional Fig. 2)")
    fig1(a.out); fig2(a.out, cells); fig3(a.out); fig4(a.out)