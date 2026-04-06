"""
make_fig_landmark_gallery.py
============================
Shows all landmark flow trees from the union of the top-K nodes across all
four landmark definitions, arranged in a grid.

Each panel shows one forward Flow Tree. Nodes in multiple definitions get a
bold coloured border. Small definition-membership dots sit below each tree.
A faint ellipse is drawn around groups that share an intersection.

Run after make_fig3.py (needs figures/fig3_cache.pkl):
    conda activate mazedyn-test
    python make_fig_landmark_gallery.py

Output: figures/fig_landmark_gallery.pdf / .png
"""

import os
import pickle

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from flow_tree_utils import plot_flow_tree
from wordfreq import zipf_frequency

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 7,
        "figure.dpi": 300,
    }
)

os.makedirs("figures", exist_ok=True)
CACHE_FILE = "figures/fig3_cache.pkl"

# ── one colour per definition ─────────────────────────────────────────────────
DEF_NAMES = ["structural", "data-driven", "semantic", "FT score"]
DEF_COLOURS = ["#4A7FB5", "#C75D4B", "#7BA05B", "#9B6BAA"]
C_MULTI = "#E8A838"  # amber — in 2+ definitions
C_PANEL_BG = "#F7F7F7"

K = 5  # top-K per definition

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


def get_nth(s, n):
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
finished_df["StartAt"] = finished_df["path"].apply(lambda x: get_nth(x, 0))
finished_df["EndAt"] = finished_df["path"].apply(lambda x: get_nth(x, -1))
finished_df["finished"] = True

unfinished_df = pd.read_csv(
    f"{data_path}/paths_unfinished.tsv",
    names=["hashedIpAddress", "timestamp", "durationInSec", "path", "target", "type"],
    sep="\t",
    skiprows=17,
)
unfinished_df["StartAt"] = unfinished_df["path"].apply(lambda x: get_nth(x, 0))
unfinished_df["EndAt"] = unfinished_df["target"]
unfinished_df["finished"] = False

trajs_df = pd.concat([finished_df, unfinished_df]).reset_index(drop=True)
trajs_df = trajs_df.drop_duplicates(subset=["hashedIpAddress", "StartAt", "target"])
trajs_df["traj"] = trajs_df["path"].apply(process_path)

# ── 2. Node properties ────────────────────────────────────────────────────────
STOP = {"a", "an", "the", "and", "for", "of"}
node_counts, node_outdegrees, node_zipf = {}, {}, {}
for traj in trajs_df.traj:
    for n in traj:
        node_counts[n] = node_counts.get(n, 0) + 1
for n in node_counts:
    if n in wikiG:
        node_outdegrees[n] = wikiG.out_degree(n)
        words = [w for w in n.split("_") if w.lower() not in STOP]
        node_zipf[n] = np.mean([zipf_frequency(w, "en") for w in words]) if words else 0


def top_k(d, k, restrict=None):
    keys = list(restrict or d.keys())
    return sorted(keys, key=lambda x: d.get(x, 0), reverse=True)[:k]


most_common = set(top_k(node_counts, 500))
most_common.discard("Wikipedia_Text_of_the_GNU_Free_Documentation_License")

struct_lm = set(top_k(node_outdegrees, K, restrict=most_common))
player_lm = set(top_k(node_counts, K, restrict=most_common))
semantic_lm = set(top_k(node_zipf, K, restrict=most_common))

# ── 3. Load cache + compute FT score ─────────────────────────────────────────
if not os.path.exists(CACHE_FILE):
    raise FileNotFoundError("Cache not found. Run make_fig3.py first.")
print("Loading cache...")
with open(CACHE_FILE, "rb") as f:
    cache = pickle.load(f)
ft_dict = cache["ft_dict"]


def tree_depth(G, root=0):
    if G.number_of_nodes() < 2:
        return 0
    return max(nx.single_source_shortest_path_length(G, root).values())


def mean_bf(G):
    bfs = [G.degree(n) for n in G.nodes() if G.degree(n) != 1]
    return np.mean(bfs) if bfs else 0.0


ft_scores = {}
for node in most_common:
    trees = ft_dict.get(node, {}).get("human", [])
    if not trees:
        continue
    depths = [tree_depth(G) for G in trees]
    bfs = [mean_bf(G) for G in trees]
    d = max(np.mean(depths), 0.5)
    ft_scores[node] = np.mean(bfs) / d

ft_lm = set(top_k(ft_scores, K))

# ── 4. Build union + membership table ────────────────────────────────────────
all_sets = [struct_lm, player_lm, semantic_lm, ft_lm]
union_nodes = struct_lm | player_lm | semantic_lm | ft_lm
print(f"Union of top-{K} per definition: {len(union_nodes)} unique nodes")

membership = {}  # node -> list of definition indices (0-3)
for node in union_nodes:
    membership[node] = [i for i, s in enumerate(all_sets) if node in s]


# Sort: most definitions first, then by FT score (fan-like nodes to the front)
def sort_key(node):
    n_defs = len(membership[node])
    score = ft_scores.get(node, 0)
    return (-n_defs, -score)


sorted_nodes = sorted(union_nodes, key=sort_key)
n_nodes = len(sorted_nodes)
print(f"Nodes to plot: {n_nodes}")

# ── 5. Build figure ───────────────────────────────────────────────────────────
N_COLS = 8
N_ROWS = int(np.ceil(n_nodes / N_COLS))

CELL_W = 1.55  # inches per cell
CELL_H = 1.80
DOT_ROW = 0.28  # extra height below tree for membership dots

fig_w = N_COLS * CELL_W + 0.6
fig_h = N_ROWS * (CELL_H + DOT_ROW) + 0.8

fig = plt.figure(figsize=(fig_w, fig_h))
fig.patch.set_facecolor("white")

# One axes per node: tree panel
tree_axes = []
dot_axes = []

for idx, node in enumerate(sorted_nodes):
    row = idx // N_COLS
    col = idx % N_COLS

    # Normalised coordinates
    x0 = (col * CELL_W + 0.3) / fig_w
    y0 = 1.0 - (row + 1) * (CELL_H + DOT_ROW) / fig_h
    w = CELL_W * 0.88 / fig_w
    h = CELL_H * 0.82 / fig_h
    hd = DOT_ROW * 0.6 / fig_h

    ax_tree = fig.add_axes([x0, y0 + hd, w, h])
    ax_dot = fig.add_axes([x0, y0, w, hd])
    tree_axes.append(ax_tree)
    dot_axes.append(ax_dot)

# ── 6. Draw flow trees ────────────────────────────────────────────────────────
for idx, node in enumerate(sorted_nodes):
    ax_tree = tree_axes[idx]
    ax_dot = dot_axes[idx]
    defs = membership[node]
    n_defs = len(defs)

    # Coloured panel background depending on intersection depth
    if n_defs >= 3:
        bg = "#FFF5E0"  # warm amber tint
    elif n_defs == 2:
        bg = "#F0F4FC"  # cool blue tint
    else:
        bg = "white"

    ax_tree.set_facecolor(bg)
    ax_dot.set_facecolor(bg)

    # Draw flow tree
    trees = ft_dict.get(node, {}).get("human", [])
    ax_tree.axis("off")
    if trees:
        plot_flow_tree(trees[0], node, ax_tree, just_edges=True, no_title=True)
    else:
        ax_tree.text(
            0.5,
            0.5,
            "—",
            transform=ax_tree.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color="#bbb",
        )

    # Node label
    label = node.replace("_", " ")
    if len(label) > 18:
        # break at nearest space
        mid = len(label) // 2
        sp = label.rfind(" ", 0, mid + 6)
        label = label[:sp] + "\n" + label[sp + 1 :] if sp > 0 else label

    ax_tree.set_title(label, fontsize=5.5, pad=2, color="#333", linespacing=1.2)

    # Border: thick coloured edge if in multiple definitions, thin grey otherwise
    for spine in ax_tree.spines.values():
        if n_defs >= 2:
            spine.set_visible(True)
            spine.set_edgecolor(
                C_MULTI
                if n_defs >= 3
                else DEF_COLOURS[defs[0]]
                if n_defs == 1
                else C_MULTI
            )
            spine.set_linewidth(1.6 if n_defs >= 3 else 1.1)
        else:
            spine.set_visible(True)
            spine.set_edgecolor(DEF_COLOURS[defs[0]] if defs else "#DDDDDD")
            spine.set_linewidth(0.7)

    # ── Membership dots row ───────────────────────────────────────────────────
    ax_dot.axis("off")
    ax_dot.set_xlim(0, 4)
    ax_dot.set_ylim(0, 1)
    for d_i, (d_col, d_name) in enumerate(zip(DEF_COLOURS, DEF_NAMES)):
        filled = d_i in defs
        ax_dot.scatter(
            [0.35 + d_i * 0.82],
            [0.5],
            s=28 if filled else 18,
            color=d_col if filled else "#DDDDDD",
            linewidths=0.4 if filled else 0,
            edgecolors="white" if filled else "#CCCCCC",
            zorder=2,
        )

# ── 7. Legend ─────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(facecolor=DEF_COLOURS[i], label=DEF_NAMES[i], linewidth=0)
    for i in range(4)
]
legend_patches += [
    mpatches.Patch(
        facecolor="#F0F4FC", edgecolor=C_MULTI, linewidth=1.1, label="in 2 definitions"
    ),
    mpatches.Patch(
        facecolor="#FFF5E0",
        edgecolor=C_MULTI,
        linewidth=1.6,
        label="in 3–4 definitions",
    ),
]
# Small dot legend
dot_legend = [
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=DEF_COLOURS[i],
        markersize=5,
        label=DEF_NAMES[i],
    )
    for i in range(4)
]

fig.legend(
    handles=legend_patches + dot_legend,
    loc="lower center",
    ncol=5,
    frameon=False,
    fontsize=6.5,
    bbox_to_anchor=(0.5, 0.005),
    handlelength=1.2,
    columnspacing=0.9,
)

fig.text(
    0.5,
    0.995,
    f"Landmark Flow Trees — top {K} per definition",
    ha="center",
    va="top",
    fontsize=9,
    color="#333",
)

# ── 8. Save ────────────────────────────────────────────────────────────────────
# plt.savefig("figures/fig_landmark_gallery.pdf", bbox_inches="tight", facecolor="white")
plt.savefig(
    "figures/fig_landmark_gallery5.png", bbox_inches="tight", dpi=300, facecolor="white"
)
print("Saved figures/fig_landmark_gallery.pdf and .png")

# Print intersection summary
print(f"\nIntersection summary (top-{K} per definition):")
in_all_4 = struct_lm & player_lm & semantic_lm & ft_lm
in_3plus = [n for n in union_nodes if len(membership[n]) >= 3]
in_2plus = [n for n in union_nodes if len(membership[n]) >= 2]
in_1only = [n for n in union_nodes if len(membership[n]) == 1]
print(f"  All 4 definitions:  {len(in_all_4)} nodes  {sorted(in_all_4)}")
print(f"  3+ definitions:     {len(in_3plus)} nodes")
print(f"  2+ definitions:     {len(in_2plus)} nodes")
print(f"  Unique to 1 only:   {len(in_1only)} nodes")
print("\nFT-only (not in structural/player/semantic):")
ft_only = ft_lm - struct_lm - player_lm - semantic_lm
for n in sorted(ft_only, key=lambda x: -ft_scores.get(x, 0)):
    print(
        f"  {n:40s}  ft_score={ft_scores.get(n, 0):.2f}  "
        f"out_degree={node_outdegrees.get(n, 0):3d}"
    )
