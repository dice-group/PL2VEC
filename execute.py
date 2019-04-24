from helper_classes import PL2VEC
from helper_classes import Parser
from helper_classes import DataAnalyser
from helper_classes import Saver
from helper_classes import PPMI


import util as ut
import numpy as np
import random

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
K = 45
num_of_dims = 50
bound_on_iter = 30
negative_constant = -1.45557
e_release = 0.01
num_sample_from_clusters = 3
system_energy = 1

#kg_root = 'KGs/DBpedia_2016_10_core_sub'
#kg_path = kg_root + '/'

kg_root = 'KGs/Drugbank'
kg_path = kg_root + '/'



storage_path, experiment_folder = ut.create_experiment_folder()

parser = Parser(p_folder=storage_path,k=K)

parser.set_similarity_measure(PPMI)

model = PL2VEC(system_energy=system_energy)


analyser = DataAnalyser(p_folder=storage_path)


holder=parser.pipeline_of_preprocessing(kg_path, bound=1000000000000)


vocab_size=len(holder)

save_all()
embeddings = ut.randomly_initialize_embedding_space(vocab_size, num_of_dims)

learned_embeddings = model.pipeline_of_learning_embeddings(e=embeddings,
                                                           max_iteration=bound_on_iter, energy_release_at_epoch=e_release,
                                                           holder=holder, negative_constant=negative_constant)
del embeddings
del holder




analyser.perform_type_prediction(learned_embeddings)


df = analyser.pseudo_label_HDBSCAN(learned_embeddings,min_cluster_size=26,min_samples=29)

analyser.perform_clustering_quality(df)


ut.write_settings(parser.p_folder, Saver.settings)
