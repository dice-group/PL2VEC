### Physical Embedding Model for Knowledge Graphs ###

PL2VEC is a Python API for learning continous vector representation of entities and relations in RDF knowledge graphs.
- [More About PL2VEC](#more-about-PL2VEC)
    - [Parser](#parser)
    - [PL2VEC](#PL2VEC)
    - [Data Analyser](#data-analyser)
- [Installation](#installation)


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
| **Parser.construct_comatrix** | Constructs a co-occurrence matrix from RDF KG. |
| **Parser.get_attractive_repulsive_entities** | Each entity is assigned with two lists of entities.|

### PL2VEC

| Component | Description |
| ---- | --- |
| **PL2VEC** |  API for learning continuous vector representation of entities and relations in RDF Knowledge Graph.|
| **PL2VEC.randomly_initialize_embedding_space** | self explanatory |
| **PL2VEC.initialize_with_SVD** | Initialize embedding space via applying Singular Value Decomposition on the constructed co-occurrence matrix. |
| **PL2VEC.go_through_entities** | self explanatory  |
| **PL2VEC.apply_hooke_s_law**   | Calculate attractive forces for a given entity. |
| **PL2VEC.apply_coulomb_s_law** | Calculate repulsive forces for a given entity. |

### DataAnalyser

| Component | Description |
| ---- | --- |
| **DataAnalyser** |  API for evaluation process|
| **DataAnalyser.calculate_euclidean_distance** | Calculates euclidean distance between given entities or list of entities |
| **DataAnalyser.topN_cosine** | Returns top N similar entities for a given entitiy.  |
| **DataAnalyser.pseudo_label_DBSCAN** | Pseudo labels embeddings via DBSCAN |
| **DataAnalyser.perform_sampling** | Samples N entities from each cluster's mean. |
| **DataAnalyser.execute_DL** | returns class expressions in description logic syntax and  F-scores from DL-Learner for each cluster|


## Installation

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