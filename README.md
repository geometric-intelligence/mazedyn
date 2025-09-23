# Flow Tree: 
## A model for navigator dynamics and environmental complexity


### Environment setup (mac m3):
* make conda-osx-arm.lock
* conda create --name mazedyn-test --fil conda-osx-arm64.lock
* conda activate mazedyn-test
* make poetry.lock
* poetry add notebook

* conda install pydot

### Data download:
* Data is located here: https://osf.io/zp5ge/
* Download the following files and put in mazedyn/data/
    * MLINDIV_behavioral_full.csv
    * MLINDIV_subject_info.csv
    * MLINDIV_train_full.csv

### Code structure:
The code is organised into seven main notebooks, each of which concern a main application of Flow Trees. Flow Tree implementation is in flow_tree_utils.py.
* 00_animations_and_traces.ipynb -- visualisation of maze and navigation traces (Figures 2, 4)       
* 0_dataset_statistics.ipynb -- dataset analysis (Figure 3)          
* 1_flow_trees_and_regression.ipynb -- feature computation and regression (Figures 4, 5)
* 2_hypothesis_testing.ipynb -- hypothesis tests (Table 1)
* 3_dynamic_prediction.ipynb -- predicting individual results from group data (Figure 8)
* 4_backbone.ipynb -- backbone-specific analysis (Figure 6)
* 5_ego_v_allocentric.ipynb -- additional exploration of discretisation        
* 6_start_node_end_node_trees.ipynb -- node Flow Trees and analysis (Figure 7)
* flow_tree_utils.py
