"""
make_fig3.py
============
Generates Fig 3 panels C and D for the Wikispeedia Flow Trees paper.

Run from the mazedyn directory with the mazedyn-test conda environment:
    conda activate mazedyn-test
    python make_fig3.py

Outputs: figures/fig3C.pdf, figures/fig3D.pdf (and .png versions)
"""

import os
import pickle
import random

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from wordfreq import zipf_frequency

from flow_tree_utils import (
    generate_flow_tree_from_trajs,
    get_matched_random_walk,
    plot_flow_tree,
)

# ── Style ────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
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
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

HUMAN_COLOR = "#E05C5C"   # warm red
RANDOM_COLOR = "#5B8DB8"  # muted blue
LANDMARK_ALPHA = 1.0
NON_LANDMARK_ALPHA = 0.45

os.makedirs("figures", exist_ok=True)
CACHE_FILE = "figures/fig3_cache.pkl"

# ── 1. Load data ─────────────────────────────────────────────────────────────
print("Loading data...")
data_path = "data/wikispeedia_paths-and-graph"

article_names = []
with open(f"{data_path}/articles.tsv") as f:
    for line in f:
        line = line.strip()
        if not line or line[0] == "#":
            continue
        article_names.append(line)

edges_df = pd.read_csv(
    f"{data_path}/links.tsv",
    names=["linkSource", "linkTarget"],
    sep="\t",
    skiprows=13,
)
wikiG = nx.DiGraph()
wikiG.add_edges_from(zip(edges_df.linkSource, edges_df.linkTarget))
print(f"  Graph: {wikiG}")


def get_nth_article(s, n): return s.split(";")[n]
def process_path(path):
    assert ";<" * 23 not in path
    for i in range(21, 0, -1):
        seq = "<" + ";<" * i
        if seq in path:
            path = path.replace(seq, f"<{i+1}")
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
    sep="\t", skiprows=16,
)
finished_df["StartAt"] = finished_df["path"].apply(lambda x: get_nth_article(x, 0))
finished_df["EndAt"]   = finished_df["path"].apply(lambda x: get_nth_article(x, -1))
finished_df["finished"] = True

unfinished_df = pd.read_csv(
    f"{data_path}/paths_unfinished.tsv",
    names=["hashedIpAddress", "timestamp", "durationInSec", "path", "target", "type"],
    sep="\t", skiprows=17,
)
unfinished_df["StartAt"] = unfinished_df["path"].apply(lambda x: get_nth_article(x, 0))
unfinished_df["EndAt"]   = unfinished_df["target"]
unfinished_df["finished"] = False

trajs_df = pd.concat([finished_df, unfinished_df]).reset_index(drop=True)
trajs_df = trajs_df.drop_duplicates(subset=["hashedIpAddress", "StartAt", "target"])
trajs_df["traj"] = trajs_df["path"].apply(process_path)
print(f"  Trajectories: {len(trajs_df)}")

# ── 2. Node properties ───────────────────────────────────────────────────────
node_counts, node_indegrees, node_outdegrees, node_zipf = {}, {}, {}, {}
STOP_WORDS = {"a", "an", "the", "and", "for", "of"}
for traj in trajs_df.traj:
    for n in traj:
        node_counts[n] = node_counts.get(n, 0) + 1
for n in node_counts:
    if n in wikiG:
        node_indegrees[n]  = wikiG.in_degree(n)
        node_outdegrees[n] = wikiG.out_degree(n)
        words = [w for w in n.split("_") if w.lower() not in STOP_WORDS]
        node_zipf[n] = np.mean([zipf_frequency(w, "en") for w in words]) if words else 0

def top_k(d, k, restrict=None, ascending=False):
    keys = list(restrict or d.keys())
    return sorted(keys, key=lambda x: d.get(x, 0), reverse=not ascending)[:k]

BIG_K, SMALL_K = 500, 50
most_common_nodes = set(top_k(node_counts, BIG_K))
most_common_nodes.discard("Wikipedia_Text_of_the_GNU_Free_Documentation_License")

outdegree_landmarks = set(top_k(node_outdegrees, SMALL_K, restrict=most_common_nodes))
player_landmarks    = set(top_k(node_counts,    SMALL_K, restrict=most_common_nodes))
semantic_landmarks  = set(top_k(node_zipf,      SMALL_K, restrict=most_common_nodes))
print(f"  Landmark overlaps — outd∩player: {len(outdegree_landmarks & player_landmarks)}, "
      f"player∩semantic: {len(player_landmarks & semantic_landmarks)}, "
      f"outd∩semantic: {len(outdegree_landmarks & semantic_landmarks)}")

# ── 3. Flow tree helpers ─────────────────────────────────────────────────────
MIN_PATH = 6
TRAJ_PER_TREE = 13
K_FOLD = 5
rev_wikiG = wikiG.reverse()

def make_traj_dict(nodes):
    d = {n: [] for n in nodes}
    for traj in trajs_df.traj:
        for i in range(len(traj) - MIN_PATH - 1):
            if traj[i] in nodes:
                d[traj[i]].append(traj[i: i + MIN_PATH])
    return d

def make_rev_traj_dict(nodes):
    d = {n: [] for n in nodes}
    for traj in trajs_df.traj:
        for i in range(MIN_PATH, len(traj)):
            if traj[i] in nodes:
                d[traj[i]].append(list(reversed(traj[i - MIN_PATH: i])))
    return d

def sample_trees(traj_dict, graph, k_fold=K_FOLD, n=TRAJ_PER_TREE):
    """Returns dict: node -> {'human': [G, ...], 'random': [G, ...]}"""
    result = {}
    for node, trajs in traj_dict.items():
        human_trees, random_trees = [], []
        for _ in range(k_fold):
            sampled = trajs if len(trajs) <= n else [
                trajs[i] for i in np.random.choice(len(trajs), n, replace=False)]
            if not sampled:
                continue
            G = generate_flow_tree_from_trajs(sampled, END_NODE="1")
            G = nx.relabel.convert_node_labels_to_integers(G, ordering="default")
            human_trees.append(G)

            start = sampled[0][0]
            if start in graph:
                rand_trajs = [get_matched_random_walk(graph, start, len(s)) for s in sampled]
                rG = generate_flow_tree_from_trajs(rand_trajs, END_NODE="1")
                rG = nx.relabel.convert_node_labels_to_integers(rG, ordering="default")
                if (0, 0) in rG.edges:
                    rG.remove_edge(0, 0)
                random_trees.append(rG)
        result[node] = {"human": human_trees, "random": random_trees}
    return result

def extract_features(G, root=0):
    depths = nx.single_source_shortest_path_length(G, root)
    depth = max(depths.values())
    diameter = nx.diameter(G)
    num_nodes = G.number_of_nodes()
    num_leaves = sum(1 for n in G.nodes() if G.degree(n) == 1)
    bfs = [len(list(G.neighbors(n))) for n in G.nodes() if G.degree(n) != 1]
    avg_bf = np.mean(bfs) if bfs else 0
    return [num_nodes, num_leaves, depth, avg_bf, diameter]

FEAT_NAMES  = ["# nodes", "# leaves", "depth", "branching\nfactor", "diameter"]
FEAT_KEYS   = ["num_nodes", "num_leaves", "depth", "avg_bf", "diameter"]

# ── 4. Compute or load from cache ────────────────────────────────────────────
if os.path.exists(CACHE_FILE):
    print(f"Loading cached flow trees from {CACHE_FILE}...")
    with open(CACHE_FILE, "rb") as f:
        cache = pickle.load(f)
    ft_dict     = cache["ft_dict"]
    rev_ft_dict = cache["rev_ft_dict"]
else:
    print("Computing flow trees (this takes ~5-10 min)...")
    traj_dict     = make_traj_dict(most_common_nodes)
    rev_traj_dict = make_rev_traj_dict(most_common_nodes)
    ft_dict     = sample_trees(traj_dict,     wikiG)
    rev_ft_dict = sample_trees(rev_traj_dict, rev_wikiG)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump({"ft_dict": ft_dict, "rev_ft_dict": rev_ft_dict}, f)
    print(f"  Cached to {CACHE_FILE}")

# ── 5. Build feature arrays ──────────────────────────────────────────────────
def get_features_for_group(nodes, ft_d, person):
    trees = [t for n in nodes if n in ft_d for t in ft_d[n][person]]
    return np.array([extract_features(G) for G in trees if G.number_of_nodes() > 1])

DEFINITIONS = [
    ("Structural\n(out-degree)", outdegree_landmarks),
    ("Data-driven\n(player freq.)", player_landmarks),
    ("Semantic\n(word freq.)", semantic_landmarks),
]

feat_data = {}
for label, lm_set in DEFINITIONS:
    non_lm = most_common_nodes - lm_set
    feat_data[label] = {
        "human_L":   get_features_for_group(lm_set,  ft_dict, "human"),
        "human_NL":  get_features_for_group(non_lm,  ft_dict, "human"),
        "random_L":  get_features_for_group(lm_set,  ft_dict, "random"),
        "random_NL": get_features_for_group(non_lm,  ft_dict, "random"),
    }

# ── 6. Fig 3D: Violin / box plots ───────────────────────────────────────────
print("Making Fig 3D...")

N_METRICS = len(FEAT_NAMES)
N_DEFS    = len(DEFINITIONS)

fig, axes = plt.subplots(
    N_DEFS, N_METRICS,
    figsize=(7.5, 5.5),
    sharey="col",
)
fig.subplots_adjust(hspace=0.45, wspace=0.35, left=0.10, right=0.97, top=0.92, bottom=0.12)

# ── colour / hatch scheme
GROUP_STYLES = {
    "human_L":   dict(color=HUMAN_COLOR,  alpha=0.85, label="Human landmark"),
    "human_NL":  dict(color=HUMAN_COLOR,  alpha=0.35, label="Human non-landmark"),
    "random_L":  dict(color=RANDOM_COLOR, alpha=0.85, label="Random landmark"),
    "random_NL": dict(color=RANDOM_COLOR, alpha=0.35, label="Random non-landmark"),
}
GROUP_ORDER = ["human_L", "human_NL", "random_L", "random_NL"]

def sig_stars(p):
    if   p < 0.001: return "***"
    elif p < 0.01:  return "**"
    elif p < 0.05:  return "*"
    else:           return "ns"

for row_i, (def_label, lm_set) in enumerate(DEFINITIONS):
    fd = feat_data[def_label]
    for col_i, (feat_name, feat_key) in enumerate(zip(FEAT_NAMES, FEAT_KEYS)):
        ax = axes[row_i][col_i]
        fi = FEAT_KEYS.index(feat_key)

        vp_data   = [fd[g][:, fi] for g in GROUP_ORDER]
        positions = [1, 2, 3.5, 4.5]
        colors    = [GROUP_STYLES[g]["color"] for g in GROUP_ORDER]
        alphas    = [GROUP_STYLES[g]["alpha"] for g in GROUP_ORDER]

        # ── violin bodies
        vp = ax.violinplot(
            vp_data, positions=positions,
            widths=0.7, showmedians=False, showextrema=False,
        )
        for body, col, alp in zip(vp["bodies"], colors, alphas):
            body.set_facecolor(col)
            body.set_alpha(alp)
            body.set_edgecolor("none")

        # ── box plots on top
        bp = ax.boxplot(
            vp_data, positions=positions,
            widths=0.25, patch_artist=True,
            medianprops=dict(color="white", linewidth=1.5),
            boxprops=dict(linewidth=0),
            whiskerprops=dict(linewidth=0.7, color="#555"),
            capprops=dict(linewidth=0.7, color="#555"),
            flierprops=dict(marker=".", markersize=2, alpha=0.3, color="#555"),
        )
        for patch, col, alp in zip(bp["boxes"], colors, alphas):
            patch.set_facecolor(col)
            patch.set_alpha(alp)

        # ── significance bracket: human L vs human NL
        h_L_vals  = fd["human_L"][:, fi]
        h_NL_vals = fd["human_NL"][:, fi]
        _, p_h = stats.ttest_ind(h_L_vals, h_NL_vals)
        stars = sig_stars(p_h)
        y_max = max(np.percentile(h_L_vals, 95), np.percentile(h_NL_vals, 95))
        y_range = ax.get_ylim()
        bracket_y = y_max * 1.07
        ax.annotate(
            "", xy=(2, bracket_y), xytext=(1, bracket_y),
            arrowprops=dict(arrowstyle="-", color="#333", lw=0.8),
        )
        ax.text(1.5, bracket_y * 1.02, stars, ha="center", va="bottom",
                fontsize=6.5, color="#333")

        # ── x-axis ticks
        ax.set_xticks(positions)
        ax.set_xticklabels(["L", "NL", "L", "NL"], fontsize=6)
        ax.tick_params(axis="x", length=2)
        ax.set_xlim(0.3, 5.7)

        # ── column headers (metric names) on top row
        if row_i == 0:
            ax.set_title(feat_name, fontsize=8, pad=4)

        # ── row labels on leftmost column
        if col_i == 0:
            ax.set_ylabel(def_label.replace("\n", " "), fontsize=7.5, labelpad=4)

        ax.tick_params(axis="y", labelsize=6.5)

# ── shared x-label indicating human | random grouping
fig.text(0.30, 0.035, "Human", ha="center", fontsize=8, color=HUMAN_COLOR, fontweight="bold")
fig.text(0.57, 0.035, "Random", ha="center", fontsize=8, color=RANDOM_COLOR, fontweight="bold")

# ── legend
legend_handles = [
    mpatches.Patch(facecolor=HUMAN_COLOR,  alpha=0.85, label="Landmark"),
    mpatches.Patch(facecolor=HUMAN_COLOR,  alpha=0.35, label="Non-landmark"),
]
fig.legend(
    handles=legend_handles,
    loc="lower right", ncol=1, frameon=False,
    fontsize=7, bbox_to_anchor=(0.98, 0.02),
)

plt.savefig("figures/fig3D.pdf", bbox_inches="tight")
plt.savefig("figures/fig3D.png", bbox_inches="tight", dpi=300)
print("  Saved figures/fig3D.pdf and figures/fig3D.png")

# ── 7. Fig 3C: Representative Flow Trees (landmark vs non-landmark) ──────────
print("Making Fig 3C...")

# Pick 3 representative nodes per group (landmark, non-landmark)
def pick_representatives(ft_d, node_set, n=3, seed=42):
    rng = random.Random(seed)
    available = [n for n in node_set if n in ft_d and ft_d[n]["human"]]
    return rng.sample(sorted(available), min(n, len(available)))

lm_nodes_outd  = pick_representatives(ft_dict, outdegree_landmarks, n=3)
nlm_nodes_outd = pick_representatives(ft_dict, most_common_nodes - outdegree_landmarks, n=3)

fig_c, axes_c = plt.subplots(
    4, 3,
    figsize=(5.5, 7.0),
)
fig_c.subplots_adjust(hspace=0.5, wspace=0.1)

def _label(node): return node.replace("_", " ")

row_configs = [
    ("LANDMARK\n(human)",     lm_nodes_outd,  "human",  HUMAN_COLOR),
    ("NON-LANDMARK\n(human)", nlm_nodes_outd, "human",  HUMAN_COLOR),
    ("LANDMARK\n(random)",    lm_nodes_outd,  "random", RANDOM_COLOR),
    ("NON-LANDMARK\n(random)",nlm_nodes_outd, "random", RANDOM_COLOR),
]

for row_i, (row_label, nodes, person, col) in enumerate(row_configs):
    for col_i, node in enumerate(nodes):
        ax = axes_c[row_i][col_i]
        trees = ft_dict.get(node, {}).get(person, [])
        if trees:
            plot_flow_tree(trees[0], _label(node), ax, just_edges=True, no_title=False)
        else:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
        ax.set_title(_label(node), fontsize=6.5, pad=2, color="#333")
        ax.axis("off")
        if col_i == 0:
            ax.text(-0.15, 0.5, row_label, transform=ax.transAxes,
                    fontsize=6.5, va="center", ha="right",
                    color=col, fontweight="bold",
                    rotation=0, wrap=True)

plt.savefig("figures/fig3C.pdf", bbox_inches="tight")
plt.savefig("figures/fig3C.png", bbox_inches="tight", dpi=300)
print("  Saved figures/fig3C.pdf and figures/fig3C.png")

print("\nDone! Check the figures/ directory.")
