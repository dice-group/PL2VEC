from helper_classes import PL2VEC
from helper_classes import Parser
from helper_classes import DataAnalyser
from helper_classes import Saver
from helper_classes import PPMI
from helper_classes import WeightedJaccard

import pandas as pd
import util as ut
import numpy as np
import random

## set random number generator
random_state = 1
np.random.seed(random_state)
random.seed(random_state)


def save_all():
    Saver.settings.append("random_state:"+str(random_state))
    Saver.settings.append("K:"+ str(K))
    Saver.settings.append("num_of_dims:"+ str(num_of_dims))
    Saver.settings.append("bound_on_iter:"+ str(bound_on_iter))
    Saver.settings.append("negative_constant:"+ str(negative_constant))
    Saver.settings.append("e_release:"+ str(e_release))
    Saver.settings.append("system_energy:"+ str(system_energy))
    Saver.settings.append("num_sample_from_clusters:"+ str(num_sample_from_clusters))


# DEFINE MODEL PARAMS
K = 20
num_of_dims = 20
bound_on_iter = 10
negative_constant = -10
e_release = 0.01
num_sample_from_clusters = 3
system_energy = 1

# Define paths

#kg_root = 'KGs/AKSW'
#kg_path = kg_root + '/kb.nt'

#kg_root = 'KGs/DBpedia'
#kg_path = kg_root + '/'

#kg_root = 'KGs/DBpedia_2016_10_core'
#kg_path = kg_root + '/'

kg_root = 'KGs/Drugbank'
kg_path = kg_root + '/drugbank.nq'

#kg_root = 'KGs/Wikidata'
#kg_path = kg_root + '/wikidata-simple-statements.nt'

#kg_root = 'KGs/SWDF'
#kg_path = kg_root + '/SWDF.nt'

#kg_root = 'KGs/example'
#kg_path = kg_root + '/father.nt'

#kg_root = 'KGs/Mutag'
#kg_path = kg_root + '/'

#kg_root = 'KGs/Car'
#kg_path = kg_root + '/'


dl_learner_path = 'dllearner-1.3.0/bin/cli'

storage_path, experiment_folder = ut.create_experiment_folder()

parser = Parser(p_folder=storage_path,k=K)

parser.set_similarity_measure(PPMI)
#parser.set_similarity_measure(WeightedJaccard)

#parser.set_similarity_function(parser.apply_entropy_jaccard_new)
#parser.set_similarity_function(parser.apply_ppmi_on_entitiy_adj_matrix)


model = PL2VEC(system_energy=system_energy)


analyser = DataAnalyser(p_folder=storage_path, execute_DL_Learner=dl_learner_path)


holder=parser.pipeline_of_preprocessing(kg_path, bound=10_000_000)

exit(1)
vocab_size=len(holder)

save_all()
embeddings = ut.randomly_initialize_embedding_space(vocab_size, num_of_dims)

learned_embeddings = model.pipeline_of_learning_embeddings(e=embeddings,
                                                           max_iteration=bound_on_iter, energy_release_at_epoch=e_release,
                                                           holder=holder, negative_constant=negative_constant)
del embeddings
del holder




#df=analyser.pseudo_label_DBSCAN(pd.DataFrame(learned_embeddings),eps=0.01,min_samples=5)
df = analyser.pseudo_label_HDBSCAN(pd.DataFrame(learned_embeddings),min_cluster_size=15,min_samples=5)
#df = analyser.pseudo_label_Kmeans(pd.DataFrame(learned_embeddings),n_clusters=len(learned_embeddings)//10)

analyser.perform_clustering_quality(df)

ut.write_settings(parser.p_folder, Saver.settings)




"""
################### EVALUATION STEPS FOR ONTOLOGY LEARNING############################################################
representative_entities = analyser.pipeline_of_data_processing_single_run(learned_embeddings, num_sample_from_clusters)
analyser.pipeline_of_single_evaluation_dl_learner(representative_entities)
analyser.kg_path=analyser.p_folder
#run DL learner
dl = analyser.generated_responds(experiment_folder)
"""