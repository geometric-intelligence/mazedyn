"""
make_fig_ft_landmarks.py
========================
Identifies landmark nodes ground-up from Flow Tree statistics alone,
then compares that set to structural, data-driven, and semantic definitions.

Two panels:
  A. Scatter: structural out-degree (x) vs Flow Tree score (y),
     coloured by which definition(s) catch each node.
     Nodes in the upper-left quadrant are "FT-only" landmarks —
     high landmark-like dynamics but low graph centrality.

  B. UpSet plot: intersection sizes across all four definitions,
     with the FT-unique set highlighted.

Run *after* make_fig3.py (needs figures/fig3_cache.pkl):
    conda activate mazedyn-test
    python make_fig_ft_landmarks.py

Output: figures/fig_ft_landmarks.pdf / .png
"""

import os
import pickle
from itertools import combinations

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from wordfreq import zipf_frequency

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
    }
)

os.makedirs("figures", exist_ok=True)
CACHE_FILE = "figures/fig3_cache.pkl"

# ── palette: one colour per definition ───────────────────────────────────────
C_STRUCT = "#4A7FB5"  # slate blue    – structural (out-degree)
C_PLAYER = "#C75D4B"  # brick red     – data-driven (player freq)
C_SEM = "#7BA05B"  # sage green    – semantic (word freq)
C_FT = "#9B6BAA"  # muted purple  – Flow Tree score
C_NONE = "#CCCCCC"  # light grey    – not a landmark by any definition
C_MULTI = "#E8A838"  # amber         – caught by multiple definitions

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
data_path = "data/wikispeedia_paths-and-graph"

edges_df = pd.read_csv(
    f"{data_path}/links.tsv",
    names=["linkSource", "linkTarget"],
    sep="\t",
    skiprows=13,
)
wikiG = nx.DiGraph()
wikiG.add_edges_from(zip(edges_df.linkSource, edges_df.linkTarget))


def get_nth_article(s, n):
    return s.split(";")[n]


def process_path(path):
    assert ";<" * 23 not in path
    for i in range(21, 0, -1):
        seq = "<" + ";<" * i
        if seq in path:
            path = path.replace(seq, f"<{i + 1}")
    nodes = []
    for i, node in enumerate(path.split(";")):
        if "<" in node:
            bk = 1 if node == "<" else int(node[1:])
            node = nodes[i - bk - 1]
        nodes.append(node)
    return nodes


finished_df = pd.read_csv(
    f"{data_path}/paths_finished.tsv",
    names=["hashedIpAddress", "timestamp", "durationInSec", "path", "rating"],
    sep="\t",
    skiprows=16,
)
finished_df["StartAt"] = finished_df["path"].apply(lambda x: get_nth_article(x, 0))
finished_df["EndAt"] = finished_df["path"].apply(lambda x: get_nth_article(x, -1))
finished_df["finished"] = True

unfinished_df = pd.read_csv(
    f"{data_path}/paths_unfinished.tsv",
    names=["hashedIpAddress", "timestamp", "durationInSec", "path", "target", "type"],
    sep="\t",
    skiprows=17,
)
unfinished_df["StartAt"] = unfinished_df["path"].apply(lambda x: get_nth_article(x, 0))
unfinished_df["EndAt"] = unfinished_df["target"]
unfinished_df["finished"] = False

trajs_df = pd.concat([finished_df, unfinished_df]).reset_index(drop=True)
trajs_df = trajs_df.drop_duplicates(subset=["hashedIpAddress", "StartAt", "target"])
trajs_df["traj"] = trajs_df["path"].apply(process_path)
print(f"  Trajectories: {len(trajs_df)}")

# ── 2. Node properties ────────────────────────────────────────────────────────
STOP_WORDS = {"a", "an", "the", "and", "for", "of"}
node_counts, node_outdegrees, node_zipf = {}, {}, {}
for traj in trajs_df.traj:
    for n in traj:
        node_counts[n] = node_counts.get(n, 0) + 1
for n in node_counts:
    if n in wikiG:
        node_outdegrees[n] = wikiG.out_degree(n)
        words = [w for w in n.split("_") if w.lower() not in STOP_WORDS]
        node_zipf[n] = np.mean([zipf_frequency(w, "en") for w in words]) if words else 0


def top_k(d, k, restrict=None):
    keys = list(restrict or d.keys())
    return set(sorted(keys, key=lambda x: d.get(x, 0), reverse=True)[:k])


most_common_nodes = top_k(node_counts, 500)
most_common_nodes.discard("Wikipedia_Text_of_the_GNU_Free_Documentation_License")

K = 50  # landmark set size — same as paper
outdegree_landmarks = top_k(node_outdegrees, K, restrict=most_common_nodes)
player_landmarks = top_k(node_counts, K, restrict=most_common_nodes)
semantic_landmarks = top_k(node_zipf, K, restrict=most_common_nodes)

# ── 3. Load cached flow trees ─────────────────────────────────────────────────
if not os.path.exists(CACHE_FILE):
    raise FileNotFoundError(f"Cache not found at {CACHE_FILE}. Run make_fig3.py first.")
print(f"Loading cache from {CACHE_FILE}...")
with open(CACHE_FILE, "rb") as f:
    cache = pickle.load(f)
ft_dict = cache["ft_dict"]

# ── 4. Compute Flow Tree landmark score for each node ─────────────────────────
# Score = mean branching factor / mean depth
# High score → tree fans out quickly from root → landmark-like behaviour
print("Computing FT landmark scores...")


def tree_depth(G, root=0):
    if G.number_of_nodes() < 2:
        return 0
    return max(nx.single_source_shortest_path_length(G, root).values())


def mean_bf(G):
    bfs = [G.degree(n) for n in G.nodes() if G.degree(n) != 1]
    return np.mean(bfs) if bfs else 0.0


node_records = []
for node in most_common_nodes:
    trees = ft_dict.get(node, {}).get("human", [])
    if not trees:
        continue
    depths = [tree_depth(G) for G in trees]
    bfs = [mean_bf(G) for G in trees]
    d = max(np.mean(depths), 0.5)  # floor to avoid div-by-zero
    node_records.append(
        {
            "node": node,
            "ft_score": np.mean(bfs) / d,
            "mean_bf": np.mean(bfs),
            "mean_depth": np.mean(depths),
            "out_degree": node_outdegrees.get(node, 0),
            "player_count": node_counts.get(node, 0),
            "zipf": node_zipf.get(node, 0),
        }
    )

ndf = pd.DataFrame(node_records).sort_values("ft_score", ascending=False)

# Top-K by FT score → FT landmark set
ft_landmarks = set(ndf.head(K)["node"])

print(f"\n  FT landmark set size: {len(ft_landmarks)}")
for name, other in [
    ("structural", outdegree_landmarks),
    ("data-driven", player_landmarks),
    ("semantic", semantic_landmarks),
]:
    print(f"  FT ∩ {name}: {len(ft_landmarks & other)}")
ft_only = ft_landmarks - outdegree_landmarks - player_landmarks - semantic_landmarks
print(f"  FT only (not caught by any other): {len(ft_only)}")
print(f"    → {sorted(ft_only)[:10]}{'...' if len(ft_only) > 10 else ''}")

# Membership flags
ndf["in_struct"] = ndf["node"].isin(outdegree_landmarks)
ndf["in_player"] = ndf["node"].isin(player_landmarks)
ndf["in_sem"] = ndf["node"].isin(semantic_landmarks)
ndf["in_ft"] = ndf["node"].isin(ft_landmarks)
ndf["n_sets"] = ndf[["in_struct", "in_player", "in_sem", "in_ft"]].sum(axis=1)


# ── 5. Assign display colour by membership ───────────────────────────────────
def node_colour(row):
    sets = (row["in_struct"], row["in_player"], row["in_sem"], row["in_ft"])
    n = sum(sets)
    if n == 0:
        return C_NONE
    if n > 1:
        return C_MULTI
    if sets[0]:
        return C_STRUCT
    if sets[1]:
        return C_PLAYER
    if sets[2]:
        return C_SEM
    return C_FT


ndf["colour"] = ndf.apply(node_colour, axis=1)

# ── 6. Figure layout ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(9.0, 4.0))
gs = gridspec.GridSpec(
    1,
    2,
    width_ratios=[1.05, 1.55],
    left=0.08,
    right=0.97,
    top=0.90,
    bottom=0.18,
    wspace=0.38,
)
ax_scatter = fig.add_subplot(gs[0])
ax_upset = fig.add_subplot(gs[1])

# ── Panel A: scatter – out-degree vs FT score ────────────────────────────────
log_outdeg = np.log1p(ndf["out_degree"])

# All non-landmark nodes first (grey, background)
mask_none = ndf["colour"] == C_NONE
ax_scatter.scatter(
    log_outdeg[mask_none],
    ndf.loc[mask_none, "ft_score"],
    s=9,
    color=C_NONE,
    alpha=0.5,
    linewidths=0,
    zorder=1,
)

# Landmark nodes coloured
for col, label in [
    (C_STRUCT, "structural"),
    (C_PLAYER, "data-driven"),
    (C_SEM, "semantic"),
    (C_FT, "FT only"),
    (C_MULTI, "≥2 definitions"),
]:
    mask = ndf["colour"] == col
    ax_scatter.scatter(
        log_outdeg[mask],
        ndf.loc[mask, "ft_score"],
        s=18 if col != C_FT else 28,
        color=col,
        alpha=0.85,
        linewidths=0,
        zorder=3,
        label=label,
    )

# Annotate FT-only nodes
# ft_only_df = ndf[ndf["node"].isin(ft_only)].copy()
# ft_only_df = ft_only_df.sort_values("ft_score", ascending=False)
# for _, row in ft_only_df.head(8).iterrows():
#     lbl = row["node"].replace("_", " ")
#     ax_scatter.annotate(
#         lbl,
#         (np.log1p(row["out_degree"]), row["ft_score"]),
#         textcoords="offset points", xytext=(5, 2),
#         fontsize=5.5, color=C_FT, alpha=0.9,
#         arrowprops=dict(arrowstyle="-", color=C_FT, lw=0.5, alpha=0.6),
#     )

# Threshold lines showing where each definition's cutoff falls
# struct_thresh = np.log1p(sorted(node_outdegrees.values(), reverse=True)[K - 1])
# ft_thresh = ndf.iloc[K - 1]["ft_score"]
# ax_scatter.axvline(struct_thresh, color=C_STRUCT, lw=0.7, ls="--", alpha=0.55)
# ax_scatter.axhline(ft_thresh, color=C_FT, lw=0.7, ls="--", alpha=0.55)

ax_scatter.set_xlabel("log(out-degree + 1)", labelpad=4)
ax_scatter.set_ylabel("FT landmark score  (BF / depth)", labelpad=4)
ax_scatter.set_title("A", loc="left", fontsize=10, fontweight="bold", pad=6)

leg = ax_scatter.legend(
    frameon=False,
    fontsize=6.5,
    loc="upper left",
    handletextpad=0.4,
    labelspacing=0.35,
)

# ── Panel B: UpSet plot ───────────────────────────────────────────────────────
# The four sets (columns in the dot matrix)
SET_NAMES = ["structural", "data-driven", "semantic", "FT score"]
SET_COLOURS = [C_STRUCT, C_PLAYER, C_SEM, C_FT]


# Enumerate all non-empty intersections, sorted descending by count
def count_intersection(flags):
    """Count nodes where exactly the given flags are True."""
    mask = np.ones(len(ndf), dtype=bool)
    for col, val in zip(["in_struct", "in_player", "in_sem", "in_ft"], flags):
        mask &= ndf[col] == val
    return mask.sum()


all_combos = []
for r in range(1, 5):
    for idx_combo in combinations(range(4), r):
        flags = [False, False, False, False]
        for i in idx_combo:
            flags[i] = True
        cnt = count_intersection(flags)
        if cnt > 0:
            all_combos.append((tuple(flags), cnt))

# Sort by descending count
all_combos.sort(key=lambda x: -x[1])

n_combos = len(all_combos)
combo_flags = [c[0] for c in all_combos]
combo_counts = [c[1] for c in all_combos]

# Split the ax_upset into top (bars) and bottom (dot matrix) using manual axes
upset_bbox = ax_upset.get_position()
ax_upset.remove()

bar_height = 0.52
dot_height = 0.28
gap = 0.04
y_dot_bot = upset_bbox.y0
y_dot_top = y_dot_bot + dot_height * (
    upset_bbox.height / (bar_height + dot_height + gap)
)
y_bar_bot = y_dot_top + gap * (upset_bbox.height / (bar_height + dot_height + gap))
y_bar_top = upset_bbox.y1

ax_bars = fig.add_axes(
    [upset_bbox.x0, y_bar_bot, upset_bbox.width, y_bar_top - y_bar_bot]
)
ax_dots = fig.add_axes(
    [upset_bbox.x0, y_dot_bot, upset_bbox.width, y_dot_top - y_dot_bot]
)

x_pos = np.arange(n_combos)

# Bar chart (top): intersection size
bar_cols = []
for flags in combo_flags:
    active = [i for i, f in enumerate(flags) if f]
    if len(active) == 1:
        bar_cols.append(SET_COLOURS[active[0]])
    else:
        bar_cols.append(C_MULTI)

# Highlight the FT-only bar
ft_only_idx = next(
    (i for i, f in enumerate(combo_flags) if f == (False, False, False, True)), None
)

for i, (cnt, col) in enumerate(zip(combo_counts, bar_cols)):
    alpha = 1.0 if i == ft_only_idx else 0.78
    ax_bars.bar(i, cnt, color=col, width=0.65, alpha=alpha, linewidth=0)
    if i == ft_only_idx:
        ax_bars.bar(i, cnt, color="none", width=0.65, edgecolor=C_FT, linewidth=1.2)
    ax_bars.text(
        i, cnt + 0.3, str(cnt), ha="center", va="bottom", fontsize=6, color="#444"
    )

ax_bars.set_xlim(-0.6, n_combos - 0.4)
ax_bars.set_xticks([])
ax_bars.set_ylabel("intersection\nsize", labelpad=4, fontsize=7)
ax_bars.spines["bottom"].set_visible(False)
ax_bars.tick_params(axis="y", labelsize=6.5)
ax_bars.set_title("B", loc="left", fontsize=10, fontweight="bold", pad=6)

# Dot matrix (bottom)
ax_dots.set_xlim(-0.6, n_combos - 0.4)
ax_dots.set_ylim(-0.6, 3.6)
ax_dots.set_yticks(range(4))
ax_dots.set_yticklabels(SET_NAMES, fontsize=7)
ax_dots.set_xticks([])
ax_dots.spines["left"].set_visible(False)
ax_dots.spines["bottom"].set_visible(False)
ax_dots.tick_params(left=False)

# Grey background dots for all
for row_i in range(4):
    for col_i in range(n_combos):
        ax_dots.scatter(col_i, row_i, s=28, color="#E0E0E0", zorder=1, linewidths=0)

# Filled/coloured dots for active sets
for col_i, flags in enumerate(combo_flags):
    active_rows = [row_i for row_i, f in enumerate(flags) if f]
    col = bar_cols[col_i]
    for row_i in active_rows:
        ax_dots.scatter(
            col_i, row_i, s=40, color=col, zorder=3, linewidths=0, alpha=0.9
        )
    # Vertical connector line between topmost and bottommost active dot
    if len(active_rows) > 1:
        ax_dots.plot(
            [col_i, col_i],
            [min(active_rows), max(active_rows)],
            color=col,
            lw=1.4,
            zorder=2,
            alpha=0.7,
        )

# Tick labels coloured to match definitions
for tick, col in zip(ax_dots.get_yticklabels(), SET_COLOURS):
    tick.set_color(col)
    tick.set_fontweight("bold" if col == C_FT else "normal")

# ── 7. Save ───────────────────────────────────────────────────────────────────
plt.savefig("figures/fig_ft_landmarks.pdf", bbox_inches="tight")
plt.savefig("figures/fig_ft_landmarks.png", bbox_inches="tight", dpi=300)
print("\nSaved figures/fig_ft_landmarks.pdf and .png")

# ── 8. Print the FT-only node list ───────────────────────────────────────────
print("\nFT-only landmark nodes (in FT top-50 but not structural/player/semantic):")
ft_only_sorted = ndf[ndf["node"].isin(ft_only)].sort_values("ft_score", ascending=False)
for _, row in ft_only_sorted.iterrows():
    print(
        f"  {row['node']:40s}  ft_score={row['ft_score']:.2f}  "
        f"out_degree={row['out_degree']:3.0f}  "
        f"player_count={row['player_count']:5.0f}"
    )
