from helper_classes import PL2VEC
from helper_classes import Parser
from helper_classes import DataAnalyser
from helper_classes import Saver

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
K = 10
num_of_dims = 2
bound_on_iter = 15
negative_constant = -5
e_release = 0.01
num_sample_from_clusters = 3
system_energy = 1

# Define paths

#kg_root = 'KGs/AKSW'
#kg_path = kg_root + '/kb.nt'

#kg_root = 'KGs/DBpedia'
#kg_path = kg_root + '/'

#kg_root = 'KGs/Bio2RDF'
#kg_path = kg_root + '/drugbank.nq'

#kg_root = 'KGs/Wikidata'
#kg_path = kg_root + '/wikidata-simple-statements.nt'


kg_root = 'KGs/SWDF'
kg_path = kg_root + '/SWDF.nt'

dl_learner_path = 'dllearner-1.3.0/bin/cli'

storage_path, experiment_folder = ut.create_experiment_folder()

parser = Parser(p_folder=storage_path,K=K)


parser.set_similarity_function(parser.apply_entropy_jaccard_on_entitiy_adj_matrix)
#parser.set_similarity_function(parser.apply_ppmi_on_entitiy_adj_matrix)
#parser.set_similarity_function(parser.apply_similarity_on_laplacian)
#parser.set_similarity_function(parser.apply_entropy_jaccard_with_networkx)


model = PL2VEC(system_energy=system_energy)


analyser = DataAnalyser(p_folder=storage_path, execute_DL_Learner=dl_learner_path)


#P, N= parser.construct_comatrix(kg_path, bound=5000)
holder=parser.pipeline_of_preprocessing(kg_path, bound=1000)



vocab_size=len(holder)

save_all()
embeddings = ut.randomly_initialize_embedding_space(vocab_size, num_of_dims)

learned_embeddings = model.pipeline_of_learning_embeddings(e=embeddings,
                                                           max_iteration=bound_on_iter, energy_release_at_epoch=e_release,
                                                           holder=holder, negative_constant=negative_constant)
del embeddings
del holder

representative_entities = analyser.pipeline_of_data_processing_single_run(learned_embeddings, num_sample_from_clusters)

analyser.pipeline_of_single_evaluation_dl_learner(representative_entities)

#run DL learner
dl = analyser.generated_responds(experiment_folder)
