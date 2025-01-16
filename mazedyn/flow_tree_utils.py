"""Utillity functions for generating, manipulating and visualising Flow Trees."""

import networkx as nx

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
    if len(set([len(trajs[t]) for t in which_trajs])) != 1: return False

    for i in range(max_traj_len):
        if len(set([trajs[t][i] for t in which_trajs])) != 1: return False
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
    states[-1] = "."+states[-1]
    states = collapse_traj(states)
    return states

def generate_trajs_from_df(this_df, use_turns=False): 
    trajs = []
    for _, row in this_df.iterrows():
        path = row["e_paths"] if use_turns else row["paths"]
        if path[-2:] == "NA": path = path[:-2] # this happens if the test ends unsuccessfully
        
        states = path.split()
        states[-1] = "."+states[-1]
        if not use_turns: states = collapse_traj(states)
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
        if DEBUG: print("LEVEL", index)
        
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
                    done["".join(traj)] = done.get("".join(traj), []).copy() + [group_member]
                else:
                    new_node = traj[index]     
                    new_subgroup_split[new_node] = new_subgroup_split.get(new_node, []).copy() + [group_member]

            for done_node, done_which_trajs in done.items():
                if DEBUG: print(f"BASE CASE ({done_node}): {group}")

                color = "green" if END_NODE in done_node else "yellow"
                G.add_node(done_node, color=color, end=True)
                if DEBUG: print(f"ADDING NEW END NODE {done_node}")
                G.add_edge(parent, done_node, weight=len(done_which_trajs))
                if DEBUG: print(f"ADDING NEW END EDGE from {parent} to {done_node}, weight is {len(done_which_trajs)}")

            if len(new_subgroup_split.keys()) == 0:
                continue
            elif len(new_subgroup_split.keys()) == 1:
                if DEBUG: print("NO NEW SPLITS FROM SUBGROUP")
                new_groups.append(group)
                new_parents.append(parent)
            else:
                if DEBUG: print("YES NEW SPLITS FROM SUBGROUP --", new_subgroup_split.keys())        
                for new_node, new_which_trajs in new_subgroup_split.items():
                    
                    # CAN COLLAPSE NEW BOY
                    max_traj_len = max([len(trajs[t]) for t in new_which_trajs])
                    if len(group) == 1 or all_traj_same(trajs, new_which_trajs, max_traj_len):
                        end_node = "".join(trajs[new_which_trajs[0]])#[-1]
                        
                        if DEBUG: print(f"BASE CASE - GROUP UNIFORM: {new_which_trajs}, last: {end_node}")

                        color = "green" if END_NODE in end_node else "yellow"
                        G.add_node(end_node, color=color, end=True)
                        G.add_edge(parent, end_node, weight=len(new_which_trajs))
            
                    else:
                        new_node = "".join(trajs[new_which_trajs[0]][:index+1])
                    
                        G.add_node(new_node, color="blue", end=False)
                        if DEBUG: print(f"ADDING NEW BLUE NODE {new_node}")
                    
                        G.add_edge(parent, new_node, weight=len(new_which_trajs))
                        if DEBUG: print(f"ADDING NEW EDGE from {parent} to {new_node}, weight is {len(new_which_trajs)}")

                        new_groups.append(new_which_trajs)
                        new_parents.append(new_node)

                
        groups = new_groups
        parents = new_parents

    return G

def plot_flow_tree(G, title, ax, just_edges=False, special_edges=None):
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")#graphviz_layout(G, prog="dot")

    # Draw nodes
    nodes = G.nodes()
        
    # Draw edges with weights
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    normalized_weights = [6 * w/max_weight for w in edge_weights]  # Scale for visualization
    if special_edges is None:
        nx.draw_networkx_edges(G, pos, width=normalized_weights, ax=ax)
    else:
        edge_colors = {}
        for edge in G.edges():
            if edge in special_edges:
                edge_colors[edge] = 'magenta'
            else:
                edge_colors[edge] = 'black'
        nx.draw_networkx_edges(G, pos, width=normalized_weights, edge_color=edge_colors.values(), ax=ax)

    if not just_edges:
        n_colors = [G.nodes[n].get("color", "blue") for n in nodes]
        nx.draw_networkx_nodes(G, pos, node_color=n_colors, node_size=500, alpha=0.7, ax=ax)

        # Draw edge labels
        edge_labels = {(u,v): f'{G[u][v]["weight"]}' for u,v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

        # Draw labels
        labels = {n: n for n in G}
        nx.draw_networkx_labels(G, pos, labels, ax=ax)
        
    ax.axis("off")
    ax.set_title(title)  


def plot_treeb_special_path(G, special_edges, with_nodes=False):
    fig = plt.figure(figsize=(10,5))
    this_ax = plt.gca()

    pos = graphviz_layout(G, prog="dot")
    
    edges = G.edges()
    weights = [G[u][v]['weight'] * .4 for u,v in edges]

    edge_colors = {}
    for edge in G.edges():
        if edge in special_edges:
            edge_colors[edge] = 'magenta'
        else:
            edge_colors[edge] = 'black'

    nodes = G.nodes()
    n_labels = {n: n for n in G}
    n_colors = [G.nodes[n].get("color", "blue") for n in nodes]

    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors.values(), width=weights, ax=this_ax)
    if with_nodes:
        nx.draw_networkx_nodes(G, pos, node_color=n_colors, node_size=500, alpha=0.7, ax=this_ax)
        nx.draw_networkx_labels(G, pos, labels=n_labels, font_size=13, ax=this_ax) # font_weight="bold",

    this_ax.axis("off")
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