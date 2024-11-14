import networkx as nx
import numpy as np
import pandas as pd

from scipy import stats 
from scipy.spatial.distance import pdist

from reeb_utils import do_reeb_for_start_end

def load_test_df(filename="data/MLINDIV_train_full.csv"):
    # load data into dataframe
    df = pd.read_csv(filename)

    test_df = df[df["eprocs"].str.contains("Test")]

    test_df["subj_mean_acc"] = test_df.groupby("Subject")["accuracy"].transform("mean")
    test_df["path_mean_acc"] = test_df.groupby(["StartAt", "EndAt"])["accuracy"].transform("mean")

    test_df['quartile'] = (
        pd.qcut(test_df['subj_mean_acc'], 4, labels=[1, 2, 3, 4])
    )

    test_df["path_mean_acc_by_Q"] = test_df.groupby(["StartAt", "EndAt", "quartile"])["accuracy"].transform("mean")

    return test_df


def load_subject_df(filename="data/MLINDIV_subject_info.csv"):
    subject_df = pd.read_csv(filename)
    subject_df["Subject"] = pd.to_numeric(subject_df["Spatial Neuro ID"], errors="coerce")
    subject_df = subject_df[~subject_df["Subject"].isna()]

    subject_df["is_man"] = (subject_df["Sex"] == "M")

    return subject_df

def compute_flow_trees_by_q(test_df):
    flow_trees_by_q = {i+1: [] for i in range(4)}
    accs = []
    for (start_node, end_node), group in test_df.groupby(["StartAt", "EndAt"]):
        G = do_reeb_for_start_end(test_df, start_node, end_node, 0, use_turns=False) # 0 means all quartiles
        new_G = nx.relabel.convert_node_labels_to_integers(G, first_label=0, ordering='default') #first_label is the starting integer label, in this case zero
        accs.append(group["path_mean_acc"].iloc[0])

        for i in range(1,5):
            G = do_reeb_for_start_end(test_df, start_node, end_node, i, use_turns=False)
            new_G = nx.relabel.convert_node_labels_to_integers(G, first_label=0, ordering='default') #first_label is the starting integer label, in this case zero
            flow_trees_by_q[i].append(new_G)

    return flow_trees_by_q, accs


def scalar_statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

def vector_statistic(x, y, axis):
    mean_x = np.mean(x, axis=axis)
    mean_y = np.mean(y, axis=axis)
    
    # Calculate Euclidean distance
    return np.linalg.norm(mean_x - mean_y, axis=-1)


def run_hypothesis_tests(reps_by_q):
    q12_reps = np.concatenate(reps_by_q[0:2, :])
    q34_reps = np.concatenate(reps_by_q[2:4, :])

    between_group_dist = pdist(np.concatenate(reps_by_q))

    q12_dist = pdist(q12_reps)
    q34_dist = pdist(q34_reps)

    # Perform a t-test on distances between the two groups
    t_stat, p_value = stats.ttest_ind(between_group_dist, np.concatenate([q12_dist, q34_dist]))
    print(f"T-statistic: {t_stat}, P-value: {p_value}")

    # Permutation test
    res = stats.permutation_test((q12_reps, q34_reps), vector_statistic, n_resamples=9999,
                        vectorized=True, alternative='two-sided')
    print(f"Permutation test: statistic={res.statistic:0.2f}, p={res.pvalue:0.4f}")