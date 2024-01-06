This folder contains the code for the paper _'Learning Persistent Community Structures in Dynamic Networks via Topological Data Analysis'_, submitted to The 38th Annual AAAI Conference on Artificial Intelligence (AAAI), 2024.

## Setup

### Enviroments
* Python (Jupyter notebook) 

### Python requirements
* python=3.8.715
* cudatoolkit=11.6
* pytorch=1.12.1+cu116
* numpy=1.23.4
* matplotlib=3.6.0
* scipy=1.9.3
* networkx=2.8.7
* gudhi=3.6.0

## Datasets
* Datasets with ground truth labels are all available from "Data" folder
* Synthetic data are generated in the Jupyter notebook script
* Datasets source and processing code:
    * Enron: https://doi.org/10.1371/journal.pone.0195993
    * Highschool:  https://doi.org/10.1371/journal.pone.0195993
    * DBLP: https://github.com/houchengbin/GloDyNE
    * Cora: https://github.com/houchengbin/GloDyNE
    * DBLPdyn: We edit data processing code of generating DBLP dataset. The code and raw data are given in "Data/dblp_dyn" folder. 

## Run
* Folder "Experiments": contains all code (Python / Jupyter Notebook) for producing the results in the experiments
* Training procedure was performed  with a NVIDIA 3090 GPU on PyTorch platfom.