from helper_classes import PL2VEC
from helper_classes import Parser
from helper_classes import DataAnalyser
import util as ut
import numpy as np
import random

## set random number generator
random_state = 1
np.random.seed(random_state)
random.seed(random_state)

# DEFINE MODEL PARAMS
K = 5
num_of_dims = 200
bound_on_iter = 15
negative_constant = -1
e_release = 0.01
num_sample_from_clusters = 1
system_energy = 1

# Define paths
#kg_root = 'KGs/DBpedia'
#kg_path = kg_root + '/skos_categories_en.ttl.bz2'
kg_root = 'KGs/Bio2RDF'
kg_path = kg_root + '/drugbank.nq'

dl_learner_path = 'dllearner-1.3.0/bin/cli'

storage_path, experiment_folder = ut.create_experiment_folder()

parser = Parser(p_folder=storage_path)

parser.set_similarity_function(parser.apply_entropy_jaccard_on_co_matrix)

#parser.set_similarity_function(parser.apply_ppmi_on_co_matrix)


model = PL2VEC(system_energy=system_energy)

analyser = DataAnalyser(p_folder=storage_path, execute_DL_Learner=dl_learner_path)

stats_corpus_info = parser.construct_comatrix(kg_path, bound=1000, bound_flag=True)

P, N = parser.get_attractive_repulsive_entities(stats_corpus_info, K)
vocab_size = len(stats_corpus_info)
del stats_corpus_info

ut.serializer(object_=N, path=parser.p_folder, serialized_name='Negative_URIs')
ut.serializer(object_=P, path=parser.p_folder, serialized_name='Positive_URIs')

holder = model.combine_information(P, N)
del P
del N

embeddings = model.randomly_initialize_embedding_space(vocab_size, num_of_dims)
# embeddings = model.initialize_with_svd(stats_corpus_info, num_of_dims)
# ut.do_scatter_plot(embeddings,folder_path)

#ut.visualize_2D(low_embeddings=embeddings, storage_path=storage_path, title='Randomly Initialized Embedding Space')

learned_embeddings = model.start(e=embeddings,
                                 max_iteration=bound_on_iter, energy_release_at_epoch=e_release,
                                 holder=holder, negative_constant=negative_constant)
del embeddings
del holder

#ut.visualize_2D(low_embeddings=learned_embeddings, storage_path=storage_path, title='Learned Embedding Space')


#model.plot_distance_function(title='asd',K=K,NC=negative_constant,delta_e=e_release)

representative_entities = analyser.pipeline_of_data_processing_single_run(learned_embeddings, num_sample_from_clusters)

analyser.pipeline_of_single_evaluation_dl_learner(representative_entities)

# run DL learner
dl = analyser.generated_responds(experiment_folder)
