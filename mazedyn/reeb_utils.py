"""Defines the the functions for generating Reeb trees."""

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import os
import pandas as pd
import pydot


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
        
def generate_reeb_tree(start_i, start_node, trajs, which_trajs, G, visited, SUCCESS="", DEBUG=False):
    """Generates Reeb tree.

    TODO talk about what this means

    Parameters
    ----------
    start_i : TODO,
        TODO
    start_node : TODO,
        TODO
    trajs : list,
        Shape = [n_trajs, various]
        List of all trajectories over which to compute the Reeb tree.
    G : nx.Graph,
        Reeb tree to mutate in place as graph is construted.
    visited : TODO,
        TODO
    DEBUG : boolean,
        If True print more things.

    Returns
    -------
    None
    """
    if DEBUG: print(f"DO REEB, {start_i}, {start_node}, {which_trajs}")
    times_before = visited.get(start_node, 0) 
    visited[start_node] = times_before
    to_add = start_node + "_" + str(times_before)

    max_traj_len = max([len(trajs[t]) for t in which_trajs])

    ## BASE CASE 1: SINGLETON ##
    ## BASE CASE 2: ALL THE SAME ##
    if len(which_trajs) == 1 or all_traj_same(trajs, which_trajs, max_traj_len):
        end_node = trajs[which_trajs[0]][-1]
        times_before = visited.get(end_node, -1) + 1
        visited[end_node] = times_before
        
        end_node +=  "_" + str(times_before)
        if DEBUG: print(f"BASE CASE: {which_trajs}, last: {end_node}")

        color = "green" if SUCCESS in end_node else "yellow"
        G.add_node(end_node, color=color, end=True)
        G.add_edge(to_add, end_node, weight=len(which_trajs))

        return
    

    ## RECURSIVE STEP ##
    if DEBUG: print(f"FOR LOOP, {start_i}, {which_trajs}")
    for step in range(start_i, max_traj_len):
        
        new_gn = {}
        for i in which_trajs:
            traj = trajs[i]

            ## STOPPING CONDITION FOR THIS ONE GUY ##
            if step >= len(traj): 
                new_gn["DONE"] = new_gn.get("DONE", []).copy() + [i]
            else:
                new_node = traj[step]     
                new_gn[new_node] = new_gn.get(new_node, []).copy() + [i]

        if DEBUG: print("NEW_GN", new_gn)
        # decide whether we need a reeb node
        
        # people went different ways!!
        if len(new_gn.keys()) > 1:
            times_before = visited.get(start_node, 0)
            sn = start_node + "_" + str(times_before)
            
            # the thing happened at the node before this! all the paths in this fn are at the same point there so it doesnt mattter
            old_one = trajs[which_trajs[0]][step-1]
            times_before = visited.get(old_one, -1) + 1
            visited[old_one] = times_before
            
            on = old_one + "_" + str(times_before)
            G.add_node(on, color="blue", end=False)
            if DEBUG: print(f"ADDING NEW BLUE NODE {on}")
            
            G.add_edge(sn, on, weight=len(which_trajs))
            if DEBUG: print(f"ADDING NEW EDGE from {sn} to {on}, weight is {len(which_trajs)}")

            
            for k, v in new_gn.items():
                if DEBUG: print("RECURSE")
                if DEBUG and k == "DONE":
                    print("*@#)$*$#()*(#@($")
                    print(new_gn)
                generate_reeb_tree(step+1, old_one, trajs,  v, G, visited, SUCCESS=SUCCESS)

            return

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


######################################
## Specific generation functions    ##
##      for test and explore.       ##
###################################### 
def do_reeb_for_start_end(df, start_node, end_node, Q, use_turns=True):
    this_df = df[(df["StartAt"] == start_node) & (df["EndAt"] == end_node) & (df["quartile"]  == Q)]
    
    trajs = []
    count = 0
    for _, row in this_df.iterrows():
        path = row["e_paths"] if use_turns else row["paths"]
        if path[-2:] == "NA": path = path[:-2] # this happens if the test ends unsuccessfully
        
        states = path.split()
        if not use_turns: states = collapse_traj(states)
        trajs.append(states)
        count += 1
        # if count == 15: break

    visited = {}

    G = nx.Graph()
    G.add_node(start_node + "_0", color="red", end=False)

    # print(start_node)
    # for t in trajs:
    #     print(t)
    generate_reeb_tree(0, start_node, trajs, [i for i in range(len(trajs))], G, visited, SUCCESS=end_node[0], DEBUG=False)
    return G

def do_reeb_for_explore_node(path, salient_node, use_turns=False):
    
    trajs = []
    if path[-2:] == "NA": path = path[:-2] # this happens if the test ends unsuccessfully
    
    states = path.split()
    if not use_turns: states = collapse_traj(states)
    for i, s in enumerate(states): 
        if salient_node in s:
            trajs.append(states[i: i+10])

    G = nx.Graph()
    G.add_node(salient_node + "_0", color="red", end=False)
    
    if len(trajs) == 0: return G, 0

    # for t in trajs:
    #     print(t)
    visited = {}
    generate_reeb_tree(0, salient_node, trajs, [i for i in range(len(trajs))], G, visited, SUCCESS="IMPOSSIBLE")
    return G, len(trajs)

######################################
## Plotting functions               ##
##      for test and explore.       ##
###################################### 
def plot_treeb(G):
    fig = plt.figure(figsize=(20,10))
    this_ax = plt.gca()

    pos = graphviz_layout(G, prog="dot")
    
    edges = G.edges()
    # colors = [G[u][v]['color'] for u,v in edges]
    weights = [G[u][v]['weight'] * .8 for u,v in edges]

    nodes = G.nodes()
    # for n in nodes:
    #     print(n, G.nodes[n])
    n_labels = {n: n[:-2] for n in G}
    n_colors = [G.nodes[n].get("color", "blue") for n in nodes]
    # print(n_colors)
    # print ((q-1) // 2, (q-1) % 2)
    nx.draw_networkx_nodes(G, pos, node_color=n_colors, node_size=1000, alpha=0.7, ax=this_ax)
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, ax=this_ax)
    nx.draw_networkx_labels(G, pos, labels=n_labels, font_size=26, font_weight="bold", ax=this_ax)

    this_ax.axis("off")

def plot_quartile_reebs_for_start_end(test_df, start_node, end_node, save=True, use_turns=True):
    fig, axs = plt.subplots(2, 2, figsize=(40,20))
    for q in (1, 2, 3, 4):
        this_ax = axs[(q-1) // 2][(q-1) % 2]
        
        G = do_reeb_for_start_end(test_df, start_node, end_node, q, use_turns=use_turns)
        # print(G.nodes)
        # print(G.edges)
        # pos = nx.spectral_layout(G)
        # pos = nx.shell_layout(G)
        # pos = nx.spring_layout(G)
        # pos = nx.planar_layout(G)
        pos = graphviz_layout(G)#, prog="dot")
    
        edges = G.edges()
        # colors = [G[u][v]['color'] for u,v in edges]
        weights = [G[u][v]['weight'] * .8 for u,v in edges]
    
        nodes = G.nodes()
        # for n in nodes:
        #     print(n, G.nodes[n])
        n_labels = {n: n[:-2] for n in G}
        n_colors = [G.nodes[n].get("color", "blue") for n in nodes]
        # print(n_colors)
        # print ((q-1) // 2, (q-1) % 2)
        nx.draw_networkx_nodes(G, pos, node_color=n_colors, node_size=1000, alpha=0.7, ax=this_ax)
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, ax=this_ax)
        nx.draw_networkx_labels(G, pos, labels=n_labels, font_size=26, font_weight="bold", ax=this_ax)

        
        this_ax.axis("off")
        this_ax.set_title(f"Q{q}", fontsize=30)
    
    fig.suptitle(f"{start_node} to {end_node}", fontsize=50)
    
    if save:
        fig.savefig(f'figures/testpath-reeb-by-quartile/{start_node+end_node}_{"YES" if use_turns else "NO"}TURN.png')
        plt.close()
    else:
        plt.show()


def plot_reeb_explore(df, person, salient_node, q="", save=True, use_turns=True):
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))
    # print(df[df["Subject"]==person])
    c = 0
    for i, person_info in df[df["Subject"]==person].iterrows():
        ax = axs[c]
        c += 1
        
        path = person_info["e_paths"] if use_turns else person_info["paths"]
    
        G, num_trajs = do_reeb_for_explore_node(path, salient_node, use_turns=use_turns)
        pos = graphviz_layout(G, prog="dot")
    
        edges = G.edges()
        # colors = [G[u][v]['color'] for u,v in edges]
        weights = [G[u][v]['weight'] * 2 for u,v in edges]
    
        nodes = G.nodes()
        # for n in nodes:
        #     print(n, G.nodes[n])
        n_labels = {n: n[:-2] for n in G}
        n_colors = ["blue" for n in nodes]
        # print(n_colors)
    
        nx.draw_networkx_nodes(G, pos, node_color=n_colors, node_size=1000, alpha=0.7, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, ax=ax)
        nx.draw_networkx_labels(G, pos, labels=n_labels, font_size=26, font_weight="bold", ax=ax)
    
        ax.axis("off")
        ax.set_title(f"{num_trajs} trajs")
        
    fig.suptitle(f"Explore paths starting at {salient_node} for person {person} (Q{q})", fontsize=30)
    
    if save:
        os.makedirs(f'figures/explore/starting_{salient_node}/', exist_ok=True)  
        fig.savefig(f'figures/explore/starting_{salient_node}/subj_{person}-{"YES" if use_turns else "NO"}TURN.png')
        plt.close()
    else:
        plt.show()