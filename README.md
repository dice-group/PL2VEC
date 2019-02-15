### Physical Embedding Model for Knowledge Graphs ###

PL2VEC is a Python API for learning continous vector representation of entities and relations in RDF knowledge graphs. PL2VEC API is originated from master's theses of Caglar Demir in Paderborn University.

- [More About PL2VEC](#more-about-PL2VEC)
    - [Parser](#parser)
    - [PL2VEC](#PL2VEC)
    - [Data Analyser](#data-analyser)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Motivation of Author on working embeddings](#motivation-of-author-on-working-embeddings)


## More about PL2VEC
PL2VEC is in Python 3.6.4 with using [NumPy](http://www.numpy.org/), [SciPy](https://www.scipy.org/), [scikit-learn](http://scikit-learn.org) and [Pandas](https://pandas.pydata.org/).
#### Workflow of PL2VEC
PL2VEC process RDF Knowledge Graph  in its own pipeline as seen below.

![alt text](https://raw.githubusercontent.com/Demirrr/physical_embedding/master/other/pl2vec_uml.png "Pipeline of PL2VEC")


At a granular level, PL2VEC API consists of following components:

### Parser

| Component | Description |
| ---- | --- |
| **Parser** | API for extracting information from RDF Knowledge Graph|
| **Parser.construct_comatrix** | Constructs a positive pointwise mutual information co-occurrence matrix from RDF KG.|
| **Parser.get_attactive_repulsive_entities** | Extracts the indexes of entities such that yield highest K  PPMI values.|

### PL2VEC

| Component | Description |
| ---- | --- |
| **PL2VEC** |  API for learning continous vector representaion of entities and relations in RDF Knowledge Graph.|
| **PL2VEC.randomly_initialize_embedding_space** | self explanatory |
| **PL2VEC.initialize_with_SVD** | Initialize embedding space via appling Singular Value Decomposion on PPMI co-occurance matrix. |
| **PL2VEC.go_through_entities** | self explanatory  |
| **PL2VEC.apply_hooke_s_law**   | Calculate attractive forces for a given entitiy. |
| **PL2VEC.apply_coulomb_s_law** | Calculate respulsive forces for a given entitiy. |

### DataAnalyser

| Component | Description |
| ---- | --- |
| **DataAnalyser** |  API for evaluation process|
| **DataAnalyser.calculate_euclidean_distance** | Calculates euclidiean distance between given entities or list of entities |
| **DataAnalyser.topN_cosine** | Returns top N similar entities for a given entitiy.  |
| **DataAnalyser.pseudo_label_DBSCAN** | Pseudo labels embeddings via DBSCAN |
| **DataAnalyser.perform_sampling** | Samples N entities from each cluster's mean. |
| **DataAnalyser.execute_DL** | returns class expressions in Description logics Syntax and  F-scores from DL-Learner for each cluster|


## Installation
* Clone repository and preferably create a new virtual python evnroirment.

# Install basic dependencies
```
python --version
Python 3.6.4
git clone https://github.com/Demirrr/physical_embedding.git
pip install -r requirements.txt
```

## Getting Started
TODO

## Experiments
TODO