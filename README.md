# Flow Trees

Code for Flow Trees, a model of how groups of navigators move through an environment.

Two papers:
* **Flow Tree: A model for navigator dynamics and environmental complexity** (virtual mazes; mice in a labyrinth)
* **Dynamic signatures of collective knowledge through navigation in a network with semantic structure**
  (Wikispeedia; Bertics, Chrastil & Carlson, submitted to PLOS One)

The Flow Tree implementation is in `mazedyn/flow_tree_utils.py`.

## Environment setup (Apple silicon Mac)

```bash
make conda-osx-arm.lock
conda create --name mazedyn-test --file conda-osx-arm64.lock
conda activate mazedyn-test
make poetry.lock
poetry add notebook
conda install pydot
```

The Wikispeedia notebook also needs:

```bash
conda install -c conda-forge graphviz pygraphviz
pip install networkx statsmodels transformers wordfreq
```

It was run with Python 3.11, networkx 3.4, scipy 1.15, statsmodels 0.14, scikit-learn 1.7, transformers 4.41,
torch 2.6 and wordfreq 3.1. The `bert-base-uncased` model is downloaded from Hugging Face on first run.

## Data

All data go in `mazedyn/data/` (not tracked by git).

**Maze-learning task.** From https://osf.io/dujhy/ (under Maze-Learning Task), download:

* `MLINDIV_behavioral_full.csv`
* `MLINDIV_subject_info.csv`
* `MLINDIV_train_full.csv`

**Wikispeedia.** From the Stanford Network Analysis Project (https://snap.stanford.edu/data/wikispeedia.html),
download `wikispeedia_paths-and-graph.tar.gz` and extract it to `mazedyn/data/wikispeedia_paths-and-graph/`. The
notebook uses `articles.tsv`, `links.tsv`, `paths_finished.tsv` and `paths_unfinished.tsv`.

## Code structure

### Maze paper

* `00_animations_and_traces.ipynb`: visualisation of maze and navigation traces (Figures 2, 4)
* `0_dataset_statistics.ipynb`: dataset analysis (Figure 3)
* `1_flow_trees_and_regression.ipynb`: feature computation and regression (Figures 4, 5)
* `2_hypothesis_testing.ipynb`: hypothesis tests (Table 1)
* `3_dynamic_prediction.ipynb`: predicting individual results from group data (Figure 8)
* `4_backbone.ipynb`: backbone-specific analysis (Figure 6)
* `5_ego_v_allocentric.ipynb`: additional exploration of discretisation
* `6_start_node_end_node_trees.ipynb`: node Flow Trees and analysis (Figure 7)
* `7_mice_maze_analysis_3C.ipynb`: analysis of the Rosenberg et al. mice dataset (Figure 10)
* `rosenbergmice/`: Flow Trees for the mice dataset

### Wikispeedia paper

`wikispeedia_analysis.ipynb` reproduces every figure, table and reported number in the manuscript from the raw
Wikispeedia data, in one top-to-bottom run (10–20 minutes on an M-series Mac). Run it from `mazedyn/` so the
relative paths resolve. Its sections follow the paper: dataset, article features, random-walker null models,
Results (1)–(6), S2 and S4 Appendices. Outputs are written to `mazedyn/paper_outputs/`.

AI use: Claude was used to consolidate human-written code from three notebooks into `wikispeedia_analysis.ipynb`
and to write its documentation. All cells were seen by human eyeballs before release.

## Data sources

* Wikispeedia: West R, Leskovec J. Human wayfinding in information networks. WWW 2012; and West R, Pineau J,
  Precup D. Wikispeedia: An online game for inferring semantic distances between concepts. IJCAI 2009.
* Mice: Rosenberg M, Zhang T, Perona P, Meister M. Mice in a labyrinth show rapid learning, sudden insight, and
  efficient exploration. eLife 2021.
