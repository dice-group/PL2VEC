### Physical Embedding Model for Knowledge Graphs ###

PL2VEC is an open source project for knowledge graph embedding.

- [What is PL2VEC](#PL2VEC)
- [Implementation](#Implementation)
    - [Parser](#parser)
    - [PL2VEC](#PL2VEC)
    - [Data Analyser](#data-analyser)
- [Installation](#installation)
- [Interactive playground](#playground)



## PL2VEC
PL2VEC is a novel physical embedding model for RDF knowledge graphs. 
It is a physical model as its properties are inherited from disciplines:  
Hooke's Law and Coulomb's Law from ***Physics***,
an optimization technique inspired by Simulated Annealing and several similarity measurements pointwise mutual information and entropy weighted jaccard similarity.
. For more information please refer to URL of paper.


## Implementation
PL2VEC is in Python 3.6.4 with using [NumPy](http://www.numpy.org/), [SciPy](https://www.scipy.org/), [scikit-learn](http://scikit-learn.org) and [Pandas](https://pandas.pydata.org/).
#### Workflow of PL2VEC
PL2VEC process RDF Knowledge Graph  in its own pipeline as seen below.

![alt text](https://raw.githubusercontent.com/Demirrr/physical_embedding/master/other/pl2vec_uml.png "Pipeline of PL2VEC")


At a granular level, PL2VEC API consists of following components:

### Parser

| Component | Description |
| ---- | --- |
| **Parser.pipeline_of_preprocessing** | Implements the workflow of Parser. |

### PL2VEC

| Component | Description |
| ---- | --- |
| **PL2VEC.pipeline_of_learning_embeddings**| Implements the worfklow of PL2VEC. |
| **PL2VEC.apply_hooke_s_law**   | Calculate attractive forces for a given particle. |
| **PL2VEC.apply_coulomb_s_law** | Calculate repulsive forces for a given particle. |
| **PL2VEC.equilibrium** | Calculate to the distance to equilibrium of the embedding space. |


### DataAnalyser

| Component | Description |
| ---- | --- |
| **DataAnalyser.calculate_euclidean_distance** | Returns the ratio of sum of distances between attractive particles and and repulsive particles.|
| **DataAnalyser.topN_cosine** | Returns top N similar particles for a given particle.  |
| **DataAnalyser.pseudo_label_DBSCAN** | Pseudo labels embeddings via DBSCAN |
| **DataAnalyser.perform_sampling** | Samples N particles from each cluster's mean. |
| **DataAnalyser.execute_DL** | returns class expressions in description logic syntax and  F-scores from DL-Learner for each cluster|

### Util

| Component | Description |
| ---- | --- |
| **randomly_initialize_embedding_space** | self explanatory |
| **initialize_with_SVD** | Initialize embedding space via applying Singular Value Decomposition on the constructed co-occurrence matrix. |

## Installation

```
python --version
Python 3.6.4
git clone https://github.com/Demirrr/physical_embedding.git
pip install -r requirements.txt
```
## Interactive playground

We provide an interactive playground of PL2VEC via Google Colab. Users can easly reproduce our evaluations as well as 
observe easy usage of our framework.
[Interactive PL2VEC](https://colab.research.google.com/drive/1Rh37e8J_FoIk1rsQdCwNoN7_1Xy-kvEQ)