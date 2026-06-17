"""Utillity functions for generating, manipulating and visualising Flow Trees."""

import random

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import norm

OBJECTS = "OYNKIWLAP"


def load_explore_df(filename="data/MLINDIV_train_full.csv"):
    df = pd.read_csv(filename)
    return df[df["eprocs"].str.contains("Explore")]


def load_test_df(
    filename="data/MLINDIV_train_full.csv",
    remove_incompletes=True,
    include_subject_info=True,
    subject_filename="data/MLINDIV_subject_info.csv",
):

def load_test_df(
    filename="data/MLINDIV_train_full.csv",
    remove_incompletes=True,
    include_subject_info=True,
    subject_filename="data/MLINDIV_subject_info.csv",
):
    df = pd.read_csv(filename)

    # load test df
    test_df = df[df["eprocs"].str.contains("Test")]

    # remove incompletes
    if remove_incompletes:
        clean_test_df = test_df.copy()

        for subject, group in test_df.groupby("Subject"):
            if group.shape[0] < 48:
                clean_test_df = clean_test_df[
                    clean_test_df["Subject"] != subject
                ].copy()
                clean_test_df = clean_test_df[
                    clean_test_df["Subject"] != subject
                ].copy()

        test_df = clean_test_df

    if include_subject_info:
        subject_df = pd.read_csv(subject_filename)

        subject_df = subject_df[~subject_df["Spatial Neuro ID"].isna()]

        is_man_map = {}
        age_map = {}
        for _, row in subject_df.iterrows():
            subj_id = row["Spatial Neuro ID"]

            # Data handling bc the csv is interesting
            if type(subj_id) == str and subj_id != "Subject Counts":
                subj_id = int(subj_id.split("S")[0])

            is_man_map[subj_id] = row["Sex"] == "M"
            age_map[subj_id] = row["Age"]

        test_df["is_man"] = test_df["Subject"].apply(lambda x: is_man_map[x])
        test_df["age"] = test_df["Subject"].apply(lambda x: age_map[x])

    test_df["subj_mean_acc"] = test_df.groupby("Subject")["accuracy"].transform("mean")
    test_df["path_mean_acc"] = test_df.groupby(["StartAt", "EndAt"])[
        "accuracy"
    ].transform("mean")
    test_df["path_mean_acc"] = test_df.groupby(["StartAt", "EndAt"])[
        "accuracy"
    ].transform("mean")

    test_df["quartile"] = pd.qcut(test_df["subj_mean_acc"], 4, labels=[1, 2, 3, 4])
    test_df["quartile"] = pd.qcut(test_df["subj_mean_acc"], 4, labels=[1, 2, 3, 4])

    test_df["path_mean_acc_by_Q"] = test_df.groupby(["StartAt", "EndAt", "quartile"])[
        "accuracy"
    ].transform("mean")
    test_df["path_mean_acc_by_Q"] = test_df.groupby(["StartAt", "EndAt", "quartile"])[
        "accuracy"
    ].transform("mean")

    test_df["trajs"] = test_df["paths"].apply(generate_traj_from_path)
    test_df["accuracy"] = test_df["accuracy"].astype(bool)

    test_df["StartEnd"] = test_df["StartAt"] + "_" + test_df["EndAt"]

    return test_df


def all_traj_same(trajs, which_trajs, max_traj_len):
    """Determines whether any trajectory differs from the others.

    Helper function for generating Reeb tree.

    Parameters
    ----------
    trajs : list,
        List of all trajectories over which to compute the Reeb tree.
    which_trajs : list,
        Indices of the trajectories to consider.
    max_traj_len : integer
        Length of the longest path, for convenience.

    Returns
    -------
    boolean
        True if all trajectories are the same length, otherwise False.
    """
    if len(set([len(trajs[t]) for t in which_trajs])) != 1:
        return False

    for i in range(max_traj_len):
        if len(set([trajs[t][i] for t in which_trajs])) != 1:
            return False
    return True


def collapse_traj(traj):
    """
    Remove turns so that A A B B G G G becomes A B G.

    Parameters
    ----------
    traj: list,
        Trajectory through maze.

    Returns
    -------
    new_traj: list,
        Same trajectory through maze, no turns.
    """
    new_traj = [traj[0]]
    for t in traj:
        if t != new_traj[-1]:
            new_traj.append(t)
    return new_traj


def generate_traj_from_path(path):
    states = path.split()
    states[-1] = "." + states[-1]
    states = collapse_traj(states)
    return states


def generate_trajs_from_df(this_df, use_turns=False):
    trajs = []
    for _, row in this_df.iterrows():
        path = row["e_paths"] if use_turns else row["paths"]
        if path[-2:] == "NA":
            path = path[:-2]  # this happens if the test ends unsuccessfully

        states = path.split()
        states[-1] = "." + states[-1]
        if not use_turns:
            states = collapse_traj(states)
        trajs.append(states)

    return trajs


def generate_flow_tree_from_trajs(trajs, END_NODE="BLAH", DEBUG=False):
    start_node = trajs[0][0]

    G = nx.Graph()
    G.add_node(start_node, color="blue")

    which_trajs = [i for i in range(len(trajs))]
    max_traj_len = max([len(trajs[t]) for t in which_trajs])

    groups = [which_trajs[:]]
    parents = [start_node]

    for index in range(1, max_traj_len):
        if DEBUG:
            print("LEVEL", index)

        new_groups = []
        new_parents = []

        for group, parent in zip(groups, parents):
            # SEE IF CAN SPLIT GROUP INTO SUBGROUPS
            done = {}
            new_subgroup_split = {}
            for group_member in group:
                traj = trajs[group_member]
                if index + 1 >= len(traj):
                    # THIS GUY IS DONE
                    done["".join(traj)] = done.get("".join(traj), []).copy() + [
                        group_member
                    ]
                else:
                    new_node = traj[index]
                    new_subgroup_split[new_node] = new_subgroup_split.get(
                        new_node, []
                    ).copy() + [group_member]

            for done_node, done_which_trajs in done.items():
                if DEBUG:
                    print(f"BASE CASE ({done_node}): {group}")

                color = "green" if END_NODE in done_node else "yellow"
                G.add_node(done_node, color=color, end=True)
                if DEBUG:
                    print(f"ADDING NEW END NODE {done_node}")
                G.add_edge(parent, done_node, weight=len(done_which_trajs))
                if DEBUG:
                    print(
                        f"ADDING NEW END EDGE from {parent} to {done_node}, weight is {len(done_which_trajs)}"
                    )

            if len(new_subgroup_split.keys()) == 0:
                continue
            elif len(new_subgroup_split.keys()) == 1:
                if DEBUG:
                    print("NO NEW SPLITS FROM SUBGROUP")
                new_groups.append(group)
                new_parents.append(parent)
            else:
                if DEBUG:
                    print("YES NEW SPLITS FROM SUBGROUP --", new_subgroup_split.keys())
                for new_node, new_which_trajs in new_subgroup_split.items():
                    # CAN COLLAPSE NEW BOY
                    max_traj_len = max([len(trajs[t]) for t in new_which_trajs])
                    if len(group) == 1 or all_traj_same(
                        trajs, new_which_trajs, max_traj_len
                    ):
                        end_node = "".join(trajs[new_which_trajs[0]])  # [-1]

                        if DEBUG:
                            print(
                                f"BASE CASE - GROUP UNIFORM: {new_which_trajs}, last: {end_node}"
                            )

                        color = "green" if END_NODE in end_node else "yellow"
                        G.add_node(end_node, color=color, end=True)
                        G.add_edge(parent, end_node, weight=len(new_which_trajs))

                    else:
                        new_node = "".join(trajs[new_which_trajs[0]][: index + 1])

                        G.add_node(new_node, color="blue", end=False)
                        if DEBUG:
                            print(f"ADDING NEW BLUE NODE {new_node}")

                        G.add_edge(parent, new_node, weight=len(new_which_trajs))
                        if DEBUG:
                            print(
                                f"ADDING NEW EDGE from {parent} to {new_node}, weight is {len(new_which_trajs)}"
                            )

                        new_groups.append(new_which_trajs)
                        new_parents.append(new_node)

        groups = new_groups
        parents = new_parents

    return G


def plot_flow_tree(
    G,
    title,
    ax,
    no_title=False,
    just_edges=False,
    special_edges=None,
    edge_boldness=1.0,
):
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")  # graphviz_layout(G, prog="dot")

    # Draw nodes
    nodes = G.nodes()

    # Draw edges with weights
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    normalized_weights = [
        edge_boldness * 6 * w / max_weight for w in edge_weights
    ]  # Scale for visualization
    if special_edges is None:
        nx.draw_networkx_edges(G, pos, width=normalized_weights, ax=ax)
    else:
        edge_colors = {}
        for edge in G.edges():
            if edge in special_edges:
                edge_colors[edge] = "magenta"
            else:
                edge_colors[edge] = "black"
        nx.draw_networkx_edges(
            G, pos, width=normalized_weights, edge_color=edge_colors.values(), ax=ax
        )

    if not just_edges:
        n_colors = [G.nodes[n].get("color", "blue") for n in nodes]
        nx.draw_networkx_nodes(
            G, pos, node_color=n_colors, node_size=500, alpha=0.7, ax=ax
        )

        # Draw edge labels
        edge_labels = {(u, v): f"{G[u][v]['weight']}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

        # Draw labels
        labels = {n: n for n in G}
        nx.draw_networkx_labels(G, pos, labels, ax=ax)

    ax.axis("off")
    if not no_title:
        ax.set_title(title)


def plot_treeb_special_path(G, special_edges, with_nodes=False):
    fig = plt.figure(figsize=(10, 5))
    this_ax = plt.gca()

    pos = graphviz_layout(G, prog="dot")

    edges = G.edges()
    weights = [G[u][v]["weight"] * 0.4 for u, v in edges]

    edge_colors = {}
    for edge in G.edges():
        if edge in special_edges:
            edge_colors[edge] = "magenta"
        else:
            edge_colors[edge] = "black"

    nodes = G.nodes()
    n_labels = {n: n for n in G}
    n_colors = [G.nodes[n].get("color", "blue") for n in nodes]

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edges,
        edge_color=edge_colors.values(),
        width=weights,
        ax=this_ax,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edges,
        edge_color=edge_colors.values(),
        width=weights,
        ax=this_ax,
    )
    if with_nodes:
        nx.draw_networkx_nodes(
            G, pos, node_color=n_colors, node_size=500, alpha=0.7, ax=this_ax
        )
        nx.draw_networkx_labels(
            G, pos, labels=n_labels, font_size=13, ax=this_ax
        )  # font_weight="bold",
        nx.draw_networkx_nodes(
            G, pos, node_color=n_colors, node_size=500, alpha=0.7, ax=this_ax
        )
        nx.draw_networkx_labels(
            G, pos, labels=n_labels, font_size=13, ax=this_ax
        )  # font_weight="bold",

    this_ax.axis("off")
    plt.show()
    plt.show()


## For Explore



def compute_extended_explore_traj(e_path):
    # need to replace V-east (V2) with Y and F-south (F3) with N
    # because they are facing and can see the object from there
    path = e_path.replace("V2", "Y2")
    path = e_path.replace("F3", "N3")


    nodes = path.split()
    nodes.remove("NA")
    nodes = [n[0] for n in nodes]

    return "".join(nodes)




def compute_explore_traj(e_path):
    # need to replace V-east (V2) with Y and F-south (F3) with N
    # because they are facing and can see the object from there
    path = e_path.replace("V2", "Y2")
    path = e_path.replace("F3", "N3")


    nodes = path.split()
    nodes.remove("NA")
    nodes = [n[0] for n in nodes]

    return "".join(collapse_traj(nodes))


def steiger_z_test(r12, r13, r23, n, alternative="two-sided"):
    # Steiger's Z formula for comparing two dependent correlations
    # compares corr 1-2 vs. corr 1-3

    if n < 4:
        raise ValueError("Sample size must be at least 4")

    z_num = (r12 - r13) * np.sqrt((n - 3) * (1 + r23))
    z_denom = 2 * (1 - r12**2 - r13**2 - r23**2 + 2 * r12 * r13 * r23)
    z_denom = np.sqrt(z_denom)

    if z_denom == 0:
        raise ValueError("Denominator is zero — check your correlations.")

    z = z_num / z_denom

    # Choose p-value based on test direction
    if alternative == "two-sided":
        p = 2 * (1 - norm.cdf(abs(z)))
    elif alternative == "greater":  # r12 > r13
        p = 1 - norm.cdf(z)
    elif alternative == "less":  # r12 < r13
        p = norm.cdf(z)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    return z, p


def get_matched_random_walk(G, start_node, walk_length):
    """
    Simulates a random walk on a graph and returns nodes visited.

    Parameters
    ----------
        G : nx.DiGraph
            The networkx graph.
        start_node : node in G
        walk_length : int
            The number of steps the walker takes.

    Returns
    ----------
        trajectory : list[str]
            A list of nodes visited.
    """
    trajectory = [start_node]
    current_node = start_node

    for _ in range(walk_length):
        neighbors = list(G.neighbors(current_node))
        if len(neighbors) == 0:
            # Dead end — backtrack to the last node that had outgoing edges
            backtrack_idx = len(trajectory) - 2
            while backtrack_idx >= 0:
                if len(list(G.neighbors(trajectory[backtrack_idx]))) > 0:
                    current_node = trajectory[backtrack_idx]
                    break
                backtrack_idx -= 1
            else:
                # No valid node found (start itself is a dead end), stay put
                trajectory.append(current_node)
                continue
        else:
            current_node = random.choice(neighbors)
        trajectory.append(current_node)

    return trajectory
