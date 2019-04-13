import pandas as pd
from SALib.sample import saltelli
import numpy as np
from helper_classes import PL2VEC
from helper_classes import Parser
from helper_classes import PPMI
from helper_classes import DataAnalyser
from helper_classes import Saver
import util as ut

# Define Paths
#kg_root = 'KGs/DBpedia_2016_10_core'
#kg_path = kg_root + '/'

kg_root = 'KGs/Drugbank'
kg_path = kg_root + '/drugbank.nq'

dl_learner_path = '/home/demir/Desktop/physical_embedding/dllearner-1.3.0/bin/cli'


# Define model parameters that will not be determined by SOBOL
num_of_dims = 50
bound_on_iter = 2
num_sample_from_clusters = 5

# Define the model inputs
problem = {
    'num_vars': 5,
    'names': ['K',
              'energy_release_at_epoch',
              'negative_constant',
              'HDBSCAN_min_cluster_size',
              'HDBSCAN_min_sample'],
    'bounds': [[1, 50],
               [0.0001, 0.9],
               [-10, -0.5],
               [2, 50],
               [2, 100]]
}


## set random number generator
random_state = 1
np.random.seed(random_state)

current_param_folder, experiment_folder = ut.create_experiment_folder()

# Generate samples
sobol_sampled_parameters = pd.DataFrame(saltelli.sample(problem, 20), columns=['K',
                                                                              'energy_release_at_epoch',
                                                                              'negative_constant',
                                                                              'HDBSCAN_min_cluster_size',
                                                                              'HDBSCAN_min_sample'
                                                                               ])


sobol_sampled_parameters['K'] = sobol_sampled_parameters.K.astype(np.uint32)
sobol_sampled_parameters['negative_constant'] = np.around(sobol_sampled_parameters.negative_constant,5)
sobol_sampled_parameters['energy_release_at_epoch'] = np.around(sobol_sampled_parameters.energy_release_at_epoch,5)
sobol_sampled_parameters['HDBSCAN_min_cluster_size'] = sobol_sampled_parameters.HDBSCAN_min_cluster_size.astype(np.uint32)
sobol_sampled_parameters['HDBSCAN_min_sample'] = sobol_sampled_parameters.HDBSCAN_min_sample.astype(np.uint32)


print(len(sobol_sampled_parameters))
exit(1)
sobol_sampled_parameters.sort_values(by=['K'],inplace=True)


sobol_sampled_parameters.to_csv(experiment_folder + '/SOBOL_Parameters.csv')

# Initialize modules
parser = Parser(p_folder=experiment_folder)

analyser = DataAnalyser(execute_DL_Learner=dl_learner_path, p_folder=experiment_folder,kg_path=experiment_folder)


parser.set_similarity_measure(PPMI)

# Read KG as we do not need to read KG for each individual sampled input parameter
num_of_rdf=parser.process_KB_w_Sobol(kg_path, bound=50_000)

old_K=None
for parameters in sobol_sampled_parameters.itertuples():

    # Sampled model parameters
    parser.set_k_entities(parameters.K)


    energy_release_at_epoch = parameters.energy_release_at_epoch
    negative_constant = parameters.negative_constant

    HDBSCAN_min_cluster_size = parameters.HDBSCAN_min_cluster_size
    HDBSCAN_min_sample = parameters.HDBSCAN_min_sample


    Saver.settings.append('K:' + str(parameters.K))
    Saver.settings.append('Negative Constant :' + str(negative_constant))
    Saver.settings.append('energy_release_at_epoch :' + str(energy_release_at_epoch))
    Saver.settings.append('HDBSCAN_eps:' + str(HDBSCAN_min_cluster_size))
    Saver.settings.append('HDBSCAN_min_sample :' + str(HDBSCAN_min_sample))


    print('K:', parameters.K)
    print('energy_release_at_epoch:', energy_release_at_epoch)
    print('NC:', negative_constant)
    print('HDBSCAN_min_cluster_size:', HDBSCAN_min_cluster_size)
    print('HDBSCAN_min_sample:', HDBSCAN_min_sample)

    # Define experiment folder
    storage_path, _ = ut.create_experiment_folder()

    if old_K!=parameters.K:
        inverted_index = ut.deserializer(path=experiment_folder, serialized_name='inverted_index')
        vocab_size = len(inverted_index)
        holder = parser.similarity_measurer().get_similarities(inverted_index, num_of_rdf, parser.K)


    embeddings = ut.randomly_initialize_embedding_space(vocab_size, num_of_dims)

    model = PL2VEC()
    learned_embeddings = model.pipeline_of_learning_embeddings(e=embeddings,
                                                               max_iteration=bound_on_iter,
                                                               energy_release_at_epoch=energy_release_at_epoch,
                                                               holder=holder, negative_constant=negative_constant)

    del embeddings

#    df = analyser.pseudo_label_DBSCAN(pd.DataFrame(learned_embeddings), eps=DBSCAN_eps, min_samples=DBSCAN_min_sample)
    df = analyser.pseudo_label_HDBSCAN(pd.DataFrame(learned_embeddings), min_cluster_size=HDBSCAN_min_cluster_size, min_samples=HDBSCAN_min_sample)
    #df = analyser.pseudo_label_Kmeans(pd.DataFrame(learned_embeddings), n_clusters=len(learned_embeddings) // 10)

    del learned_embeddings

    mean_of_scores=analyser.perform_clustering_quality(df)


    ut.write_settings(storage_path,Saver.settings)
    Saver.settings.clear()


    break