import pandas as pd
from SALib.sample import saltelli
import numpy as np
from helper_classes import PL2VEC
from helper_classes import Parser
from helper_classes import DataAnalyser
from helper_classes import Saver
import util as ut

# Define Paths
kg_root = 'KGs/DBpedia'
kg_path = kg_root + '/skos_categories_en.ttl.bz2'
dl_learner_path = '/home/demir/Desktop/physical_embedding/dllearner-1.3.0/bin/cli'

# Define model parameters that will not be determined by SOBOL
num_of_dims = 50
bound_on_iter = 15
num_sample_from_clusters = 10
system_energy = 1

# Define the model inputs
problem = {
    'num_vars': 3,
    'names': ['K',
              'energy_release_at_epoch',
              'negative_constant'],
    'bounds': [[1, 30],
               [0.0001, 0.9],
               [-20, -0.5]]
}

_, experiment_folder = ut.create_experiment_folder()

# Generate samples
sobol_sampled_parameters = pd.DataFrame(saltelli.sample(problem, 13), columns=['K',
                                                                              'energy_release_at_epoch',
                                                                              'negative_constant']).astype(np.float32)

sobol_sampled_parameters['K'] = sobol_sampled_parameters.K.astype(np.uint32)
sobol_sampled_parameters['negative_constant'] = sobol_sampled_parameters.K.astype(np.int32)
sobol_sampled_parameters.to_csv(experiment_folder + '/SOBOL_Parameters.csv')

# Initialize modules
parser = Parser(p_folder=experiment_folder)
model = PL2VEC(system_energy)
analyser = DataAnalyser(execute_DL_Learner=dl_learner_path, kg_path=experiment_folder)

#parser.set_similarity_function(parser.apply_entropy_jaccard_on_co_matrix)
parser.set_similarity_function(parser.apply_ppmi_on_co_matrix)

# Read KG as we do not need to read KG for each individual sampled input parameter
stats_corpus_info = parser.construct_comatrix(kg_path, bound=50000, bound_flag=True)

ut.serializer(object_=stats_corpus_info, path=experiment_folder, serialized_name='stats_corpus_info')
del stats_corpus_info

for parameters in sobol_sampled_parameters.itertuples():
    # Sampled model parameters
    K = parameters.K
    energy_release_at_epoch = parameters.energy_release_at_epoch
    negative_constant = parameters.negative_constant

    # Define experiment folder
    storage_path, _ = ut.create_experiment_folder()

    # Set the respective experiment folder- day-time etc.
    parser.set_experiment_path(storage_path)
    analyser.set_experiment_path(storage_path)

    stats_corpus_info = ut.deserializer(path=experiment_folder, serialized_name='stats_corpus_info')

    P, N = parser.get_attractive_repulsive_entities(stats_corpus_info, K)
    vocab_size = len(stats_corpus_info)
    del stats_corpus_info

    ut.serializer(object_=N, path=parser.p_folder, serialized_name='Negative_URIs')
    ut.serializer(object_=P, path=parser.p_folder, serialized_name='Positive_URIs')

    print('K:', K)
    print('energy_release_at_epoch:', energy_release_at_epoch)
    print('NC:', negative_constant)

    Saver.settings.append('Size of vocabulary :' + str(vocab_size))
    #    Saver.settings.append('Num of RDF :' + str(kg_size))
    #    Saver.settings.append('Negative Constant :' + str(NC))
    Saver.settings.append('energy_release_at_epoch :' + str(energy_release_at_epoch))

    holder = model.combine_information(P, N)
    del P
    del N

    embeddings = model.randomly_initialize_embedding_space(vocab_size, num_of_dims)
    # embeddings = model.initialize_with_svd(stats_corpus_info, num_of_dims)
    # ut.do_scatter_plot(embeddings,folder_path)

    # ut.visualize_2D(low_embeddings=embeddings, storage_path=storage_path, title='Randomly Initialized Embedding Space')

    learned_embeddings = model.start(e=embeddings,
                                     max_iteration=bound_on_iter, energy_release_at_epoch=energy_release_at_epoch,
                                     holder=holder, negative_constant=negative_constant)
    del embeddings
    del holder
    # ut.visualize_2D(low_embeddings=learned_embeddings, storage_path=storage_path, title='Learned Embedding Space')

    representative_entities = analyser.pipeline_of_data_processing(experiment_folder, learned_embeddings,
                                                                   num_sample_from_clusters)

    analyser.pipeline_of_dl_learner(experiment_folder, representative_entities)

    # run DL learner
    # dl_ = analyser.generated_responds(experiment_folder)
