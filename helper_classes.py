import os
import pickle
import re
import math
from bz2 import BZ2File
import bz2
from collections import Counter, defaultdict
import itertools
from scipy import spatial
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from scipy.spatial.distance import cosine
from sklearn.neighbors.kd_tree import KDTree

import util as ut
import os.path
import matplotlib.pyplot as plt
from numpy import linalg as LA
import numpy as np
import pandas as pd
from scipy.spatial import distance
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import typing
import warnings
import sortednp as snp
import sys
from abc import ABC, abstractmethod
import hdbscan
from itertools import chain

print('Do not forget warnigngs')
warnings.filterwarnings('ignore')
defined_min = np.finfo(np.float32).min
defined_max = np.finfo(np.float32).max


def performance_debugger(func_name):
    def function_name_decoratir(func):
        def debug(*args, **kwargs):
            long_string = ''
            starT = time.time()
            print('\n\n######', func_name, ' starts ######')
            r = func(*args, **kwargs)
            print(func_name, ' took ', time.time() - starT, ' seconds\n')
            long_string += str(func_name) + ' took:' + str(time.time() - starT) + ' seconds'
            Saver.settings.append(long_string)
            return r

        return debug

    return function_name_decoratir


class SimilarityCalculator(ABC):
    def __init__(self):
        self._inverted_index = None
        self._num_triples = None

    @abstractmethod
    def get_similarities(self, inverted_index, num_triples, top_K):
        pass


class PPMI(SimilarityCalculator):
    def __init__(self):
        """

        :param co_occurrences: term to list of terms
        :param num_triples:
        """
        super().__init__()
        self._marginal_probs = None

    def calculate_marginal_probabilities(self):
        marginal_probs = dict()
        for unq_ent, list_of_context_ent in enumerate(self.inverted_index):
            # N is multipled by 2 as list_of_context_ent contains other two element of an RDF triple
            probability = len(list_of_context_ent) / (self._num_triples * 2)

            marginal_probs[unq_ent] = probability
        self._marginal_probs = marginal_probs

    @performance_debugger('Calculation of PPMIs')
    def calculate_ppmi(self) -> np.array:

        holder = list()

        for unq_ent, list_of_context_ent in enumerate(self.inverted_index):
            top_k_sim = dict()

            marginal_prob_of_target = self._marginal_probs[unq_ent]

            statistical_info_of_cooccurrences = Counter(list_of_context_ent)

            top_k_sim.setdefault(unq_ent, dict())

            for context_ent, co_occuring_freq in statistical_info_of_cooccurrences.items():

                joint_prob = co_occuring_freq / self._num_triples

                marginal_prob_of_context = self._marginal_probs[context_ent]

                denominator = marginal_prob_of_target * marginal_prob_of_context

                PMI_val = np.log2(joint_prob) - np.log2(denominator)

                if PMI_val <= 0:
                    continue

                if len(top_k_sim[unq_ent]) <= self._topK:
                    top_k_sim[unq_ent][context_ent] = PMI_val.astype(np.float32)
                else:
                    for k, v in top_k_sim[unq_ent].items():
                        if v < PMI_val:
                            top_k_sim[unq_ent][context_ent] = PMI_val
                            del top_k_sim[unq_ent][k]
                            break

            context = np.array(list(top_k_sim[unq_ent].keys()), dtype=np.uint32)
            sims = np.array(list(top_k_sim[unq_ent].values()), dtype=np.float32)
            sims.shape = (sims.size, 1)

            # sampled may contain dublicated variables
            sampled = np.random.choice(len(self.inverted_index), self._topK)

            # negatives must be disjoint from context of k.th vocabulary term and k.term itsel
            negatives = np.setdiff1d(sampled, np.append(context, unq_ent), assume_unique=True)

            holder.append((context, sims, negatives))

        return holder

    def get_similarities(self, inverted_index, num_triples, top_K):
        """

        :param inverted_index:
        :param num_triples:
        :return: similarities data structure is a numpy array of dictionaries.

                i.th element of the numpy array corresponds to i.th element in the vocabulary.
                The dictionary stored in the i.th element:
                    Key: a vocabulary term
                    Val: PPMI value

        """
        self.inverted_index = inverted_index
        self._num_triples = num_triples
        self._topK = top_K
        self.calculate_marginal_probabilities()

        similarities = self.calculate_ppmi()

        return similarities


class WeightedJaccard(SimilarityCalculator):
    def __init__(self, similar_characteristics, p_folder):
        super().__init__()
        self._entropies = None
        self._similar_characteristics = similar_characteristics
        self._inverted_index = None
        self._num_triples = None
        self.p_folder = p_folder

        self._lowerbound = 0.09
        self._upperbound = 0.3  # 0.3

    def calculate_entropies(self):
        entropies = list()

        for index, postings_list in enumerate(self._inverted_index):

            self._inverted_index[index] = np.array(postings_list, dtype=np.uint32)

            # each vocabulary term contains both co-occurring terms per rdf triple
            marginal_prob = len(postings_list) / (self._num_triples * 2)

            with np.errstate(divide='raise'):
                try:
                    entropies.append(-marginal_prob * np.log2(marginal_prob))
                except FloatingPointError:
                    print('entropy of term-', index, ': is set 0')
                    print('P( term-', index, '- is', marginal_prob)
                    exit(1)

        self._entropies = np.array(entropies, dtype=np.float32)

        print('Min entropy ', self._entropies.min())
        print('Mean entropy', self._entropies.mean())
        print('Median entropy', np.median(self._entropies))
        print('Variance entropy', np.var(self._entropies))
        print('Max entropy', self._entropies.max())

    @performance_debugger('Calculate Weighted Jaccard Similarity')
    def calculate_weighted_jaccard(self, results):
        """
        Computes weighted jaccard similarity between subjects whose contexts contains sufficiently enough information.
        :param results:
        :return:
        """

        def calculate_sim(subjects):
            """
            Inverted_index contains each unique subject predicate and object.
            We select each subject from inverted index and compute similarities given their contex contains suffc.
            enough inforation.
            Thereafter each index reindex starting from 0 to N.
            :param subjects:
            :return:
            """

            #            l = [vocabulary[i] for i in subjects]
            #           print(l)

            for i in subjects:
                context_i = self._inverted_index[i]
                sum_ent_context_i = self._entropies[context_i].sum()

                re_indexer.setdefault(i, len(re_indexer))
                similarities.setdefault(re_indexer[i], dict())

                for j in subjects:
                    if i == j:
                        continue

                    re_indexer.setdefault(j, len(re_indexer))
                    context_j = self._inverted_index[j]
                    sum_ent_context_j = self._entropies[context_j].sum()

                    intersection = snp.intersect(context_i, context_j)

                    sum_ent_intersection = self._entropies[intersection].sum()

                    sim = 2 * sum_ent_intersection / (sum_ent_context_i + sum_ent_context_j)

                    # print(vocabulary[i],'',vocabulary[j],'=',sim)
                    if sim > 0:
                        similarities.setdefault(re_indexer[j], dict())

                        similarities[re_indexer[i]][re_indexer[j]] = sim
                        similarities[re_indexer[j]][re_indexer[i]] = sim

        similarities = dict()
        re_indexer = dict()

        # vocabulary = np.array(ut.deserializer(path=self.p_folder, serialized_name='vocabulary'))

        seen = set()
        for groups in results:

            if not groups:
                print('empty')
                continue

            for subjects in groups:

                str_ = np.array_str(subjects)

                if str_ in seen:
                    continue
                else:
                    seen.add(str_)
                calculate_sim(subjects)

        ut.serializer(object_=re_indexer, path=self.p_folder, serialized_name='re_indexer')
        return similarities

    @performance_debugger('Pruning less information p-o pairs')
    def prune_pairs_of_p_o(self):
        def gen():
            print('lower bound', self._lowerbound, ' - upper bound', self._upperbound)

            for p, l_p in self._similar_characteristics.items():

                objects = np.array(list(l_p.keys()))
                sum_of_ent_of_p_objects = self._entropies[objects] + self._entropies[p]

                valid_objects_to_be_paired = objects[
                    (sum_of_ent_of_p_objects >= self._lowerbound) & (sum_of_ent_of_p_objects <= self._upperbound)]
                if valid_objects_to_be_paired.size > 0:
                    yield p, valid_objects_to_be_paired

        def delete_empty_results(futures):
            """
            Futures is a list of future objects of similarity computations.
            Each item in the list contains similarity computations between a predicate p and a list of objects occurred with p


            :param futures: list of future objects
            :return:
            """
            for f in futures:
                result = f.result()
                if result:
                    yield result

        def get_intersections(context_p, context_l_o):

            subjects = list()
            for context_of_o in context_l_o:
                intersection = snp.intersect(context_p, context_of_o)
                if intersection.size > 1:
                    subjects.append(intersection)
            return subjects

        vocabulary = ut.deserializer(path=self.p_folder, serialized_name='vocabulary')

        pruned_p_o_pairs = gen()

        e = ThreadPoolExecutor(8)
        futures = []
        for p, list_o in pruned_p_o_pairs:
            print('Predicates: ', vocabulary[p])
            context_p = self._inverted_index[p]
            context_l_o = [self._inverted_index[_] for _ in list_o]
            futures.append(e.submit(get_intersections, context_p, context_l_o))

        return delete_empty_results(futures)

    def get_similarities(self, inverted_index, num_triples):

        self._inverted_index = inverted_index
        self._num_triples = num_triples

        self.calculate_entropies()

        similarities = self.calculate_weighted_jaccard(self.prune_pairs_of_p_o())

        similarities = np.array(list(similarities.values()))

        print('|subjects|', len(similarities))

        return similarities


class Parser:
    def __init__(self, logger=False, p_folder: str = 'not initialized', k=1):
        self.path = 'uninitialized'
        self.logger = logger
        self.p_folder = p_folder
        self.similarity_function = None
        self.similarity_measurer = None
        self.K = int(k)

    def set_similarity_function(self, f):
        self.similarity_function = f
        Saver.settings.append('similarity_function:' + repr(f))

    def set_similarity_measure(self, f):
        self.similarity_measurer = f
        Saver.settings.append('similarity_function:' + repr(f))

    def set_experiment_path(self, p):
        self.p_folder = p

    def set_k_entities(self, k):
        self.K = k

    """
    @staticmethod
    def calculate_marginal_probabilities(binary_co_matrix: Dict, number_of_rdfs: int):
        marginal_probs = dict()
        for unq_ent, list_of_context_ent in binary_co_matrix.items():
            # N is multipled by 2 as list_of_context_ent contains other two element of an RDF triple
            marginal_prob = len(list_of_context_ent) / (number_of_rdfs * 2)
            marginal_prob = round(marginal_prob, 5)
            marginal_probs[unq_ent] = marginal_prob
        return marginal_probs

    @performance_debugger('Calculating entropies')
    def calculate_entropies(self, freq_adj_matrix: Dict, number_of_rdfs: int) -> typing.Tuple[np.array, np.array]:
        co_occurrences = deque()
        entropies = deque()

        for unq_ent, contexts_w_co_freq in freq_adj_matrix.items():

            marginal_prob = sum(contexts_w_co_freq.values()) / number_of_rdfs
            co_occurrences.append(np.array(list(contexts_w_co_freq.keys()), dtype=np.uint32))

            with np.errstate(divide='raise'):
                try:
                    entropy = -marginal_prob * np.log2(marginal_prob)
                    entropies.append(entropy)
                except FloatingPointError:
                    print('entropy of term-', unq_ent, ': is set 0')
                    print('P( term-', unq_ent, '- is', marginal_prob)
                    entropies.append(0)

        entropies = np.array(entropies, dtype=np.float32)

        ut.serializer(object_=entropies, path=self.p_folder, serialized_name='entropies')
        # todo what is the type of co_pccurrences
        ut.serializer(object_=np.array(co_occurrences), path=self.p_folder, serialized_name='co_occurrences')

        return entropies, np.array(co_occurrences)

    def calculate_ppmi(self, binary_co_matrix: Dict, marginal_probs: Dict, number_of_rdfs) -> Dict:
        top_k_sim = dict()
        negatives = dict()
        for unq_ent, list_of_context_ent in binary_co_matrix.items():

            marginal_prob_of_target = marginal_probs[unq_ent]

            statistical_info_of_cooccurrences = Counter(list_of_context_ent)

            top_k_sim.setdefault(unq_ent, dict())

            for context_ent, co_occuring_freq in statistical_info_of_cooccurrences.items():

                joint_prob = round(co_occuring_freq / number_of_rdfs, 5)

                marginal_prob_of_context = marginal_probs[context_ent]

                denominator = marginal_prob_of_target * marginal_prob_of_context

                if denominator > 0.00 and joint_prob > 0.0000:
                    PMI_val = np.round(np.log2(joint_prob) - np.log2(denominator), 5)

                    if len(top_k_sim[unq_ent]) <= self.K:

                        top_k_sim[unq_ent][context_ent] = PMI_val
                    else:

                        for k, v in top_k_sim[unq_ent].items():
                            if v < PMI_val:
                                top_k_sim[unq_ent][context_ent] = PMI_val
                                del top_k_sim[unq_ent][k]
                                break

            n = np.random.choice(len(binary_co_matrix), self.K)

            negatives[unq_ent] = np.setdiff1d(n, list(list_of_context_ent))

        assert len(top_k_sim) == len(negatives)

        top_k_sim = np.array(list(top_k_sim.values()))

        return top_k_sim, negatives

    @performance_debugger('Applying apply_entropy_jaccard_new')
    def apply_entropy_jaccard_new(self, inverted_index: np.array, num_triples: int, inv_p_o: dict):

        def calculate_from_p_o():
            similarities = dict()
            for __, set_of_subjects in inv_p_o.items():

                #            p,o = __.split(' ')
                #            p,o=int(p),int(o)

                for s_i in set_of_subjects:

                    similarities.setdefault(s_i, dict())

                    context_i = inverted_index[s_i]

                    sum_ent_context_vocab_i = entropies[context_i].sum()

                    for s_j in set_of_subjects:
                        if s_i == s_j or s_j in similarities[s_i]:
                            continue

                        similarities.setdefault(s_j, dict())

                        context_j = inverted_index[s_j]
                        sum_ent_context_vocab_j = entropies[context_j].sum()

                        sim = entropies[snp.intersect(context_i, context_j)].sum() / (
                                sum_ent_context_vocab_j + sum_ent_context_vocab_i)

                        similarities[s_i][s_j] = sim
                        similarities[s_j][s_i] = sim

            return similarities

        entropies = deque()

        for index, postings_list in enumerate(inverted_index):

            inverted_index[index] = np.array(postings_list, dtype=np.uint32)

            # each vocabulary term contains both co-occurring terms per rdf triple
            marginal_prob = len(postings_list) / (num_triples * 2)

            with np.errstate(divide='raise'):
                try:
                    entropies.append(-marginal_prob * np.log2(marginal_prob))
                except FloatingPointError:
                    print('entropy of term-', index, ': is set 0')
                    print('P( term-', index, '- is', marginal_prob)
                    exit(1)

        entropies = np.array(entropies, dtype=np.float32)

        holder = self.choose_to_K_attract_repulsives(similarities)

        return holder

    @performance_debugger('Applying Entropy Jaccard ')
    def apply_entropy_jaccard_on_entitiy_adj_matrix(self, freq_adj_matrix: Dict, num_triples: int):
        try:
            entropies = ut.deserializer(path=self.p_folder, serialized_name='entropies')
            co_occurrences = ut.deserializer(path=self.p_folder, serialized_name='co_occurrences')
            print('Entropies and co_occurrences are deserialized.')
        except:
            entropies, co_occurrences = self.calculate_entropies(freq_adj_matrix, num_triples)

        print(entropies)
        print(co_occurrences)
        exit(1)
        holder = self.calculate_entropy_jaccard(entropies, co_occurrences)

        return holder

    def calculate_entropy_jaccard(self, entropies: np.array, domain: np.array):

        holder = list()

        for i, domain_i in enumerate(domain):
            top_k_sim = dict()
            negatives = dict()
            top_k_sim.setdefault(i, dict())
            negatives.setdefault(i, list())

            sum_ent_domain_i = entropies[domain_i].sum()

            for j in domain_i:

                domain_j = domain[j]

                intersection = snp.intersect(domain_i, domain_j)

                if len(intersection) > 0:
                    if len(top_k_sim[i]) <= self.K:

                        sum_ent_domain_j = entropies[domain_j].sum()
                        sim = entropies[intersection].sum() / (sum_ent_domain_i + sum_ent_domain_j)
                        top_k_sim[i][j] = sim
                    else:
                        for k, v in top_k_sim[i].items():
                            sim = entropies[intersection].sum() / (sum_ent_domain_i + entropies[domain_j].sum())

                            if v < sim:
                                top_k_sim[i][j] = sim
                                del top_k_sim[i][k]
                                break
                else:
                    if len(negatives[i]) <= self.K:
                        negatives[i].append(j)

            context = np.array(list(top_k_sim[i].keys()), dtype=np.uint32)
            sim = np.array(list(top_k_sim[i].values()))
            sim.shape = (sim.size, 1)

            repulsives = np.array(negatives[i], dtype=np.uint32)

            del top_k_sim
            del negatives
            holder.append((context, sim, repulsives))

        return holder
    """

    # TODO apply both seach and spectral custerin on tf-idf and and entropy co-occurrences.
    # TODO thereafter for each vocabulary term find most similar K terms

    """
    
    def apply_similarity_on_laplacian(self, freq_adj_matrix: typing.Dict[int, Counter], num_triples: int)
        row = list()
        col = list()
        data = list()

        num_of_unqiue_entities = len(freq_adj_matrix)
        for i, co_info in freq_adj_matrix.items():
            vals = np.array(list(co_info.values())).sum()
            row.extend(np.array([i]))
            col.extend(np.array([i]))
            data.extend(np.array([vals]))

            j = list(co_info.keys())
            vals = np.array(list(co_info.values()))

            row.extend(np.array([i] * len(j)))

            col.extend(j)

            data.extend(-1 * vals)

        laplacian_matrix = sparse.csc_matrix((data, (row, col)), shape=(num_of_unqiue_entities, num_of_unqiue_entities))

        print(laplacian_matrix.toarray())

        exit(1)

    def apply_entropy_jaccard_with_connected_component_analysis(self, freq_adj_matrix: typing.Dict[int, Counter],
                                                                num_triples: int):


        entropies, co_occurrences = self.calculate_entropies(freq_adj_matrix, num_triples)

        row = list()
        col = list()
        data = list()

        num_of_unqiue_entities = len(freq_adj_matrix)
        for i, co_info in freq_adj_matrix.items():
            j = list(co_info.keys())
            vals = np.array(list(co_info.values()))

            row.extend(np.array([i] * len(j)))

            col.extend(j)

            data.extend(entropies[vals])

        entropy_adj_matirx = sparse.csc_matrix((data, (row, col)),
                                               shape=(num_of_unqiue_entities, num_of_unqiue_entities))

        n_components, labels = connected_components(csgraph=entropy_adj_matirx, directed=False, return_labels=True)

        print('Found components')
        for i_component in range(n_components):
            itemindex = np.where(labels == i_component)[0]
            print(len(itemindex))

        exit(1)
        pass
    """
    """
    def apply_entropy_jaccard_with_networkx(self, freq_adj_matrix: typing.Dict[int, Counter], num_triples: int):
        entropies, co_occurrences = self.calculate_entropies(freq_adj_matrix, num_triples)

        for i, domain_i in enumerate(co_occurrences):

            # get top K highest entropy from domain:

            for j in domain_i:
                # get top K highest entriofy from each domain
                pass
            break

        #            compare i with K^2 terms

        exit(1)
        import networkx as nx
        from networkx.algorithms import approximation
        from networkx.algorithms.traversal import depth_first_search
        entropies, co_occurrences = self.calculate_entropies(freq_adj_matrix, num_triples)
        del co_occurrences

        row = list()
        col = list()
        data = list()

        num_of_unqiue_entities = len(freq_adj_matrix)
        for i, co_info in freq_adj_matrix.items():
            j = list(co_info.keys())
            vals = np.array(list(co_info.values()))

            row.extend(np.array([i] * len(j)))

            col.extend(j)

            data.extend(entropies[vals])

        entropy_adj_matirx = sparse.csc_matrix((data, (row, col)),
                                               shape=(num_of_unqiue_entities, num_of_unqiue_entities))

        G = nx.from_scipy_sparse_matrix(entropy_adj_matirx)
        # del entropy_adj_matirx
        #       k_components = approximation.min_maximal_matching(G)
        #        print(k_components)
        # print(list(nx.dfs_predecessors(G, source=3)))
        # https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.approximation.connectivity.local_node_connectivity.html#networkx.algorithms.approximation.connectivity.local_node_connectivity

        l = np.array(MiniBatchKMeans(n_clusters=10).fit(entropy_adj_matirx).labels_)
        print(l)
        exit(1)
        pass
    
    """

    def get_path_knowledge_graphs(self, path: str):
        """

        :param path: str represents path of a KB or path of folder containg KBs
        :return:
        """
        KGs = list()

        if os.path.isfile(path):
            KGs.append(path)
        else:
            for root, dir, files in os.walk(path):
                for file in files:
                    if '.nq' in file or '.nt' in file or 'ttl' in file:
                        KGs.append(path + '/' + file)
        if len(KGs) == 0:
            print(path + ' is not a path for a file or a folder containing any .nq or .nt formatted files')
            exit(1)
        return KGs

    def decompose_rdf(self, sentence):

        flag = 0

        components = re.findall('<(.+?)>', sentence)
        if len(components) == 2:
            s, p = components
            remaining_sentence = sentence[sentence.index(p) + len(p) + 2:]
            literal = remaining_sentence[:-1]
            o = literal
            flag = 2

        elif len(components) == 4:
            del components[-1]
            s, p, o = components

            flag = 4

        elif len(components) == 3:
            s, p, o = components
            flag = 3

        elif len(components) > 4:

            s = components[0]
            p = components[1]
            remaining_sentence = sentence[sentence.index(p) + len(p) + 2:]
            literal = remaining_sentence[:remaining_sentence.index(' <http://')]
            o = literal

        else:

            ## This means that literal contained in RDF triple contains < > symbol
            """ pass"""
            # print(sentence)
            raise ValueError()

        o = re.sub("\s+", "", o)

        s = re.sub("\s+", "", s)

        p = re.sub("\s+", "", p)

        return s, p, o, flag

    @performance_debugger('KG to PPMI Matrix')
    def construct_adj_sobol(self, f_name, bound=10):

        freq_adj_matrix, num_of_rdf = self.process_file(f_name, bound)
        # freq_adj_matrix, num_of_rdf = self.process_file_with_node(f_name, bound)

        ut.serializer(object_=freq_adj_matrix, path=self.p_folder, serialized_name='freq_adj_matrix')

        return num_of_rdf

    @performance_debugger('KG to PPMI Matrix')
    def construct_adj_matrix(self, f_name, bound=''):

        if isinstance(bound, int):
            freq_matrix, vocab, num_triples, subjects_to_indexes, predicates_to_indexes = self.create_dic_from_text(
                f_name, bound)
        else:
            freq_matrix, vocab, num_triples, only_resources = self.create_dic_from_text_all(f_name)

        print('Size of vocabulary', len(vocab))
        print('Number of RDF triples', num_triples)
        print('Number of subjects', len(subjects_to_indexes))

        indexes_of_subjects = np.array(list(subjects_to_indexes.values()), dtype=np.uint32)
        indexes_of_predicates = np.array(list(predicates_to_indexes.values()), dtype=np.uint32)

        # Prune those subject that occurred in predicate position
        valid_subjects = np.setdiff1d(indexes_of_subjects, indexes_of_predicates)
        print('Number of valid subjects for DL-Learner', len(valid_subjects))

        texec = ThreadPoolExecutor(4)
        future = texec.submit(self.similarity_function, freq_matrix, num_triples)

        ut.serializer(object_=dict(zip(list(vocab.values()), list(vocab.keys()))), path=self.p_folder,
                      serialized_name='i_vocab')

        ut.serializer(object_=vocab, path=self.p_folder, serialized_name='vocab')
        del vocab

        index_of_predicates = np.array(list(predicates_to_indexes.values()), dtype=np.uint32)
        ut.serializer(object_=index_of_predicates, path=self.p_folder, serialized_name='index_of_predicates')
        del predicates_to_indexes
        del index_of_predicates

        index_of_resources = np.array(list(subjects_to_indexes.values()), dtype=np.uint32)
        ut.serializer(object_=index_of_resources, path=self.p_folder, serialized_name='index_of_resources')
        del index_of_resources

        ut.serializer(object_=subjects_to_indexes, path=self.p_folder, serialized_name='subjects_to_indexes')
        del subjects_to_indexes

        similarities = future.result()

        return similarities

    @performance_debugger('KG to PPMI Matrix')
    def construct(self, f_name, bound=''):

        if isinstance(bound, int):
            freq_matrix, vocab, num_triples, subjects_to_indexes, predicates_to_indexes = self.create_dic_from_text(
                f_name, bound)
        else:
            freq_matrix, vocab, num_triples, only_resources = self.create_dic_from_text_all(f_name)

        print('Size of vocabulary', len(vocab))
        print('Number of RDF triples', num_triples)
        print('Number of subjects', len(subjects_to_indexes))

        indexes_of_subjects = np.array(list(subjects_to_indexes.values()), dtype=np.uint32)
        indexes_of_predicates = np.array(list(predicates_to_indexes.values()), dtype=np.uint32)

        # Prune those subject that occurred in predicate position
        valid_subjects = np.setdiff1d(indexes_of_subjects, indexes_of_predicates)
        print('Number of valid subjects for DL-Learner', len(valid_subjects))

        texec = ThreadPoolExecutor(4)
        future = texec.submit(self.similarity_function, freq_matrix, num_triples)

        ut.serializer(object_=dict(zip(list(vocab.values()), list(vocab.keys()))), path=self.p_folder,
                      serialized_name='i_vocab')

        ut.serializer(object_=vocab, path=self.p_folder, serialized_name='vocab')
        del vocab

        index_of_predicates = np.array(list(predicates_to_indexes.values()), dtype=np.uint32)
        ut.serializer(object_=index_of_predicates, path=self.p_folder, serialized_name='index_of_predicates')
        del predicates_to_indexes
        del index_of_predicates

        index_of_resources = np.array(list(subjects_to_indexes.values()), dtype=np.uint32)
        ut.serializer(object_=index_of_resources, path=self.p_folder, serialized_name='index_of_resources')
        del index_of_resources

        ut.serializer(object_=subjects_to_indexes, path=self.p_folder, serialized_name='subjects_to_indexes')
        del subjects_to_indexes

        similarities = future.result()

        return similarities

    @performance_debugger('Choose K attractives and repulsives')
    def choose_to_K_attract_repulsives(self, similarities):
        """

        :param co_similarity_matrix:
        :param size_of_iteracting_entities:
        :return:
        """

        holder = list()

        for k, v in similarities.items():

            if len(v) > self.K:
                _ = sorted(v.items(), key=lambda kv: kv[1], reverse=True)[0:self.K]

            else:
                _ = sorted(v.items(), key=lambda kv: kv[1], reverse=True)

            l = list(itertools.chain.from_iterable(_))
            context = np.array(l[::2], dtype=np.uint32)
            sims = np.array(l[1::2], dtype=np.float32)
            sims.shape = (sims.size, 1)

            holder.append((context, sims, np.random.choice(len(similarities), self.K, )))

        return holder

    def random_sample_repulsive_entities(self, ppmi_co_occurence_matrix, num_interacting_entities):

        return_val = list()
        for vocabulary_term, context_ppmi in ppmi_co_occurence_matrix.items():
            disjoint_entities = list(context_ppmi.keys())
            disjoint_entities.append(vocabulary_term)
            disjoint_entities = np.array(disjoint_entities)
            population = np.array(list(ppmi_co_occurence_matrix.keys()))
            sub_population = np.setdiff1d(population, disjoint_entities)

            sampled_negative_entities = np.random.choice(sub_population, num_interacting_entities)

            assert not np.isin(sampled_negative_entities, disjoint_entities).all()

            return_val.append(set(sampled_negative_entities))

        return return_val

    def apply_ppmi_on_entitiy_adj_matrix(self, freq_adj_matrix: typing.Dict[int, Counter], num_triples: int) -> \
            typing.Tuple[dict, dict]:
        """

        :param freq_adj_matrix: is mapping from the index of entities to Counter object.
                Counter object contains the index of co-occurring entities and the respectiveco-occurrences
        :param num_triples:
        :return:
        """

        marginal_probs = np.array([sum(context.values()) for i, context in freq_adj_matrix.items()],
                                  dtype=np.uint32) / num_triples

        holder = list()

        for unq_ent, contexts_w_co_freq in freq_adj_matrix.items():
            top_k_sim = dict()
            negatives = dict()

            marginal_prob_of_target = marginal_probs[unq_ent]

            top_k_sim.setdefault(unq_ent, dict())

            for context_ent, co_freq in contexts_w_co_freq.items():

                joint_prob = round(co_freq / num_triples, 5)

                marginal_prob_of_context = marginal_probs[context_ent]

                denominator = marginal_prob_of_target * marginal_prob_of_context

                try:
                    PMI_val = round(math.log2(joint_prob) - math.log2(denominator), 5)
                except ValueError:
                    continue

                if len(top_k_sim[unq_ent]) <= self.K:
                    top_k_sim[unq_ent][context_ent] = PMI_val
                else:
                    for k, v in top_k_sim[unq_ent].items():
                        if v < PMI_val:
                            top_k_sim[unq_ent][context_ent] = PMI_val
                            del top_k_sim[unq_ent][k]
                            break

            n = np.random.choice(len(freq_adj_matrix), self.K, replace=False)

            negatives[unq_ent] = np.setdiff1d(n, list(contexts_w_co_freq.keys()))

            context = np.array(list(top_k_sim[unq_ent].keys()), dtype=np.uint32)
            sim = np.array(list(top_k_sim[unq_ent].values()), dtype=np.uint32)
            sim.shape = (sim.size, 1)

            repulsives = np.array(negatives[unq_ent], dtype=np.uint32)

            del top_k_sim
            del negatives
            holder.append((context, sim, repulsives))

        return holder

    @performance_debugger('Assigning attractive and repulsive particles')
    def get_attractive_repulsive_entities(self, similarities: typing.Sequence[typing.Dict[int, np.float64]]):
        """

        :param similarities: typing.Sequence corresponds to a numpy array
        :return:
        """

        holder = list()

        for k, v in enumerate(similarities):

            if len(v) > self.K:
                _ = sorted(v.items(), key=lambda kv: kv[1], reverse=True)[0:self.K]

            else:
                _ = sorted(v.items(), key=lambda kv: kv[1], reverse=True)

            l = list(itertools.chain.from_iterable(_))

            context = np.array(l[::2], dtype=np.uint32)
            sims = np.array(l[1::2], dtype=np.float32)
            sims.shape = (sims.size, 1)

            # sampled may contain dublicated variables
            sampled = np.random.choice(len(similarities), self.K)

            # negatives must be disjoint from context of k.th vocabulary term and k.term itsel
            negatives = np.setdiff1d(sampled, np.append(context, k), assume_unique=True)

            holder.append((context, sims, negatives))

        # numpy array requires more memory, sys.getsizeof(holder)<sys.getsizeof(np.array(holder))
        return holder

    @performance_debugger('Preprocessing')
    def pipeline_of_preprocessing(self, f_name, bound=''):

        inverted_index, num_of_rdf, similar_characteristics = self.inverted_index(f_name, bound)
        exit(1)
        if 'helper_classes.WeightedJaccard' in repr(self.similarity_measurer):
            similarities = self.similarity_measurer(similar_characteristics, self.p_folder).get_similarities(
                inverted_index, num_of_rdf)
        else:
            holder = self.similarity_measurer().get_similarities(inverted_index, num_of_rdf, self.K)

        return holder

    @performance_debugger('Preprocessing')
    def process_KB_w_Sobol(self, f_name, bound=''):

        inverted_index, num_of_rdf, _ = self.inverted_index(f_name, bound)

        ut.serializer(object_=inverted_index, path=self.p_folder, serialized_name='inverted_index')

        return num_of_rdf

    def process_file(self, path, bound):

        freq_adj_matrix = {}
        vocabulary = {}
        num_of_rdf = 0
        index_to_prune_for_dl = dict()
        p_knowledge_graphs = self.get_path_knowledge_graphs(path)

        writer_kb = open(self.p_folder + '/' + 'KB.txt', 'w')

        for f_name in p_knowledge_graphs:

            if f_name[-4:] == '.bz2':
                reader = bz2.open(f_name, "rt")



            else:
                reader = open(f_name, "r")

            total_sentence = 0

            counter = 0
            for sentence in reader:
                if isinstance(sentence, bytes):
                    sentence = sentence.decode('utf-8')
                    print('asdad')
                    exit(1)

                # print(sentence)
                if total_sentence == bound: break

                counter += 1

                if '"' in sentence or "'" in sentence:
                    continue

                # if 'rdf-syntax-ns#type' in sentence:
                #     continue

                writer_kb.write(sentence)

                if counter % 100000 == 0:
                    print(counter)
                try:
                    s, p, o, flag = self.decompose_rdf(sentence)

                    # <..> <..> <..>
                    if flag != 3:
                        print(sentence)
                        print(flag)
                        continue

                except ValueError:
                    continue
                total_sentence += 1

                writer_kb.write(sentence)

                # mapping from string to vocabulary
                vocabulary.setdefault(s, len(vocabulary))
                vocabulary.setdefault(p, len(vocabulary))
                vocabulary.setdefault(o, len(vocabulary))

                index_to_prune_for_dl[vocabulary[p]] = p
                index_to_prune_for_dl[vocabulary[o]] = o

                freq_adj_matrix.setdefault(vocabulary[s], Counter())[vocabulary[o]] += 1
                freq_adj_matrix.setdefault(vocabulary[s], Counter())[vocabulary[p]] += 1

                freq_adj_matrix.setdefault(vocabulary[o], Counter())[vocabulary[s]] += 1
                freq_adj_matrix.setdefault(vocabulary[o], Counter())[vocabulary[p]] += 1

                freq_adj_matrix.setdefault(vocabulary[p], Counter())[vocabulary[s]] += 1
                freq_adj_matrix.setdefault(vocabulary[p], Counter())[vocabulary[p]] += 1

            print(f_name, ' - ', counter)
            reader.close()
            num_of_rdf += total_sentence

        writer_kb.close()

        print('Number of RDF triples in the input KG:', num_of_rdf)
        print('Number of entities:', len(vocabulary) - len(index_to_prune_for_dl))

        if len(vocabulary) == 0:
            print('exitting')
            exit(1)

        print(freq_adj_matrix[0])

        exit(1)

        ut.serializer(object_=index_to_prune_for_dl, path=self.p_folder, serialized_name='index_to_prune_for_dl')
        del index_to_prune_for_dl

        ut.serializer(object_=vocabulary, path=self.p_folder, serialized_name='vocabulary')
        del vocabulary

        return freq_adj_matrix, num_of_rdf

    def process_file_with_node(self, path, bound):

        freq_adj_matrix = {}
        vocabulary = {}
        num_of_rdf = 0
        index_to_prune_for_dl = dict()
        p_knowledge_graphs = self.get_path_knowledge_graphs(path)

        writer_kb = open(self.p_folder + '/' + 'KB.txt', 'w')

        for f_name in p_knowledge_graphs:

            if f_name[-4:] == '.bz2':
                reader = BZ2File(f_name, "r")
            else:
                reader = open(f_name, "r")

            total_sentence = 0

            for sentence in reader:
                if isinstance(sentence, bytes):
                    sentence = sentence.decode('utf-8')

                if total_sentence == bound: break

                if '"' in sentence:
                    continue

                try:

                    s, p, o, _ = sentence.split(' ')
                except ValueError:
                    print('val error')
                    continue
                total_sentence += 1

                writer_kb.write(sentence)

                # mapping from string to vocabulary
                vocabulary.setdefault(s, len(vocabulary))
                vocabulary.setdefault(p, len(vocabulary))
                vocabulary.setdefault(o, len(vocabulary))

                index_to_prune_for_dl[vocabulary[p]] = p

                if 'owl#' in o:
                    index_to_prune_for_dl[vocabulary[o]] = o
                    index_to_prune_for_dl[vocabulary[s]] = s

                if 'ontology' in o:
                    index_to_prune_for_dl[vocabulary[o]] = o
                    index_to_prune_for_dl[vocabulary[s]] = s

                if 'http://www.w3.org/2000/01/rdf-schema#subClassOf' in p:
                    index_to_prune_for_dl[vocabulary[o]] = o
                    index_to_prune_for_dl[vocabulary[s]] = s

                freq_adj_matrix.setdefault(vocabulary[s], Counter())[vocabulary[o]] += 1
                freq_adj_matrix.setdefault(vocabulary[s], Counter())[vocabulary[p]] += 1

                freq_adj_matrix.setdefault(vocabulary[o], Counter())[vocabulary[s]] += 1
                freq_adj_matrix.setdefault(vocabulary[o], Counter())[vocabulary[p]] += 1

                freq_adj_matrix.setdefault(vocabulary[p], Counter())[vocabulary[s]] += 1
                freq_adj_matrix.setdefault(vocabulary[p], Counter())[vocabulary[p]] += 1

            reader.close()
            num_of_rdf += total_sentence

        writer_kb.close()

        print('Number of RDF triples in the input KG:', num_of_rdf)
        print('Number of entities:', len(vocabulary))

        if len(vocabulary) == 0:
            print('exitting')
            exit(1)

        ut.serializer(object_=index_to_prune_for_dl, path=self.p_folder, serialized_name='index_to_prune_for_dl')
        del index_to_prune_for_dl

        ut.serializer(object_=vocabulary, path=self.p_folder, serialized_name='vocabulary')
        del vocabulary

        return freq_adj_matrix, num_of_rdf

    def read_kb(self, path, bound):

        co_occurrences = {}
        vocabulary = {}
        num_of_rdf = 0
        index_to_prune_for_dl = dict()
        p_knowledge_graphs = self.get_path_knowledge_graphs(path)

        writer_kb = open(self.p_folder + '/' + 'KB.txt', 'w')

        for f_name in p_knowledge_graphs:

            if f_name[-4:] == '.bz2':
                reader = bz2.open(f_name, "rt")

            else:
                reader = open(f_name, "r")

            total_sentence = 0

            counter = 0
            for sentence in reader:

                # print(sentence)
                if total_sentence == bound: break

                counter += 1

                if '"' in sentence or "'" in sentence:
                    continue

                writer_kb.write(sentence)

                if counter % 100000 == 0:
                    print(counter)
                try:
                    s, p, o, flag = self.decompose_rdf(sentence)

                    # <..> <..> <..>
                    if flag != 3:
                        print(sentence)
                        print(flag)
                        continue

                except ValueError:
                    continue
                total_sentence += 1

                # mapping from string to vocabulary
                vocabulary.setdefault(s, len(vocabulary))
                vocabulary.setdefault(p, len(vocabulary))
                vocabulary.setdefault(o, len(vocabulary))

                index_to_prune_for_dl[vocabulary[p]] = p
                index_to_prune_for_dl[vocabulary[o]] = o

                co_occurrences.setdefault(vocabulary[s], []).append(vocabulary[p])
                co_occurrences[vocabulary[s]].append(vocabulary[o])

                co_occurrences.setdefault(vocabulary[p], []).append(vocabulary[s])
                co_occurrences[vocabulary[p]].append(vocabulary[o])

                co_occurrences.setdefault(vocabulary[o], []).append(vocabulary[s])
                # co_occurrences[vocabulary[o]].append(vocabulary[p])

            print(f_name, ' - ', counter)
            reader.close()
            num_of_rdf += total_sentence

        writer_kb.close()

        print('Number of RDF triples in the input KG:', num_of_rdf)
        print('Number of entities:', len(vocabulary) - len(index_to_prune_for_dl))

        if len(vocabulary) == 0:
            print('exitting')
            exit(1)

        print(co_occurrences)
        exit(1)
        ut.serializer(object_=index_to_prune_for_dl, path=self.p_folder, serialized_name='index_to_prune_for_dl')
        del index_to_prune_for_dl

        ut.serializer(object_=vocabulary, path=self.p_folder, serialized_name='vocabulary')
        del vocabulary

        return co_occurrences, num_of_rdf

    @performance_debugger('Constructing Inverted Index')
    def inverted_index(self, path, bound):

        inverted_index = {}
        vocabulary = {}
        similar_characteristics = defaultdict(lambda: defaultdict(list))

        num_of_rdf = 0
        writer_kb = open(self.p_folder + '/' + 'KB.txt', 'w')

        type_info = defaultdict(set)

        sentences = ut.generator_of_reader(bound, self.get_path_knowledge_graphs(path), self.decompose_rdf)

        for s, p, o in sentences:

            num_of_rdf += 1
            #            writer_kb.write(s + ' ' + p + ' ' + o + ' .\n')
            writer_kb.write('<'+s + '> <' + p + '> <' + o + '> .\n')

            # mapping from string to vocabulary
            vocabulary.setdefault(s, len(vocabulary))
            vocabulary.setdefault(p, len(vocabulary))
            vocabulary.setdefault(o, len(vocabulary))

            inverted_index.setdefault(vocabulary[s], []).extend([vocabulary[o], vocabulary[p]])
            inverted_index.setdefault(vocabulary[p], []).extend([vocabulary[s], vocabulary[o]])
            inverted_index.setdefault(vocabulary[o], []).extend([vocabulary[s], vocabulary[p]])

            if 'rdf-syntax-ns#type' in p:
                type_info[vocabulary[s]].add(vocabulary[o])

        print('Number of RDF triples:', num_of_rdf)
        print('Number of vocabulary terms: ', len(vocabulary))
        print('Number of subjects: ', len(type_info))

        Saver.settings.append('Number of RDF triples:' + str(num_of_rdf))
        Saver.settings.append('Number of subjects:' + str(len(type_info)))
        Saver.settings.append('Number of vocabulary terms: ' + str(len(vocabulary)))

        # writer_kb.close()
        # This command ensures that inverted index starts from 0 to the size of vocabulary.
        # If this always true, we do not need to store key values.
        assert list(inverted_index.keys()) == list(range(0, len(vocabulary)))

        vocabulary = list(vocabulary.keys())
        ut.serializer(object_=vocabulary, path=self.p_folder, serialized_name='vocabulary')
        del vocabulary

        #        inverted_index = np.array(list(inverted_index.values()))
        inverted_index = list(inverted_index.values())

        ut.serializer(object_=inverted_index, path=self.p_folder, serialized_name='inverted_index')

        ut.serializer(object_=type_info, path=self.p_folder, serialized_name='type_info')
        del type_info

        return inverted_index, num_of_rdf, similar_characteristics


class PL2VEC(object):
    def __init__(self, system_energy=1):

        self.epsilon = 0.1
        self.texec = ThreadPoolExecutor(8)

        self.total_distance_from_attractives = list()
        self.total_distance_from_repulsives = list()
        self.ratio = list()
        self.system_energy = system_energy

    @staticmethod
    def apply_hooke_s_law(embedding_space, target_index, context_indexes, PMS):

        dist = embedding_space[context_indexes] - embedding_space[target_index]
        # replace all zeros to 1
        dist[dist == 0] = 1
        # replace all
        pull = dist * PMS
        total_pull = np.sum(pull, axis=0)

        norm_of_dist = np.nan_to_num(np.linalg.norm(dist))

        return total_pull, norm_of_dist

    @staticmethod
    def apply_coulomb_s_law(embedding_space, target_index, repulsive_indexes, negative_constant):
        # calculate distance from target to repulsive entities
        dist = embedding_space[repulsive_indexes] - embedding_space[target_index]

        # replace all zeros to 1
        dist[dist == 0] = 0.1
        with warnings.catch_warnings():
            try:
                #                r_square = dist ** 2
                #               total_push = np.sum((negative_constant / r_square), axis=0)

                total_push = negative_constant * np.reciprocal(dist).sum(axis=0)
                # replace all zeros to 1
                total_push[total_push == 0] = 0.1


            except RuntimeWarning as r:
                print(r)
                print("Unexpected error:", sys.exc_info()[0])

                exit(1)

        norm_of_dist = np.nan_to_num(np.linalg.norm(dist))

        return total_push, norm_of_dist

    def go_through_entities(self, e, holder, negative_constant):

        agg_att_d = 0
        agg_rep_d = 0

        for target_index in range(len(e)):
            indexes_of_attractive, pms_of_contest, indexes_of_repulsive = holder[target_index]

            pull, abs_att_dost = self.apply_hooke_s_law(e, target_index, indexes_of_attractive, pms_of_contest)

            push, abs_rep_dist = self.apply_coulomb_s_law(e, target_index, indexes_of_repulsive, negative_constant)
            """
            futures.append(self.texec.submit(self.apply_hooke_s_law, e, target_index, indexes_of_attractive, pms_of_contest))
            futures.append(self.texec.submit(self.apply_coulomb_s_law, e, target_index, indexes_of_repulsive, negative_constant))
            results = [f.result() for f in futures]
            pull, abs_att_dost = results[0]
            push, abs_rep_dist = results[1]
            """

            total_effect = (pull + push) * self.system_energy

            e[target_index] = e[target_index] + total_effect

            agg_att_d += abs_att_dost
            agg_rep_d += abs_rep_dist

        ratio = agg_att_d / agg_rep_d

        return e, ratio

    @performance_debugger('Generating Embeddings:')
    def pipeline_of_learning_embeddings(self, *, e, max_iteration, energy_release_at_epoch, holder, negative_constant):

        for epoch in range(max_iteration):
            print('EPOCH: ', epoch)

            previous_f_norm = LA.norm(e, 'fro')

            e, d_ratio = self.go_through_entities(e, holder, negative_constant)

            self.system_energy = self.system_energy - energy_release_at_epoch

            # replace nan with zero and inf with finite numbers
            # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.nan_to_num.html
            # check later whether it is the chase
            e = np.nan_to_num(e)

            with warnings.catch_warnings():
                try:
                    e = (e - e.min(axis=0)) / (e.max(axis=0) - e.min(axis=0))
                except RuntimeWarning as r:
                    print(r)
                    print(e.mean())
                    print(np.isnan(e).any())
                    print(np.isinf(e).any())
                    exit(1)

            new_f_norm = LA.norm(e, 'fro')

            if self.equilibrium(epoch, previous_f_norm, new_f_norm, d_ratio):
                break

        return pd.DataFrame(e)

    def equilibrium(self, epoch, p_n, n_n, d_ratio):

        val = np.abs(p_n - n_n)
        print('The norm diff in E', val)
        print('d(Similars)/d(Non Similars) ', d_ratio)
        # or d_ratio < 0.1
        if val < self.epsilon or self.system_energy <= 0:  # or np.isnan(val) or system_energy < 0.001:
            print("\n Epoch: ", epoch)
            print('Previous norm', p_n)
            print('New norm', n_n)
            print('The differences in matrix norm ', val)
            print('d(Semantically Similar)/d(Not Semantically Similar) ', d_ratio)
            print('System energy:', self.system_energy)

            Saver.settings.append('Epoch: ' + str(epoch))
            Saver.settings.append('The differences in matrix norm: ' + str(val))
            Saver.settings.append('Ratio of total Attractive / repulsives: ' + str(d_ratio))
            Saver.settings.append('self.system_energy: ' + str(self.system_energy))
            # print('The state of equilibrium is reached.')
            return True
        return False

    def plot_distance_function(self, title, K, NC, delta_e, path=''):

        X = np.array(self.total_distance_from_attractives)
        y = np.array(self.total_distance_from_repulsives)
        print(X)
        plt.plot(X / y, linewidth=4)

        plt.title('Ratio of attractive and repulsive distances on ' + title)
        plt.xlabel('Epoch')
        plt.ylabel('Ratio')
        plt.ylim([0, 1])
        plt.xlim([0, len(X) + 5])
        plt.text(len(X), 0.9, "K: " + str(K))
        plt.text(len(X), 0.8, "NC: " + str(NC))
        plt.text(len(X), 0.7, "$\Delta$ e: " + str(delta_e))
        plt.savefig(path + 'Ratio.png')
        plt.show()


class DataAnalyser(object):
    def __init__(self, p_folder: str = 'not initialized',
                 execute_DL_Learner="Not initialized", kg_path='asd'):

        self.execute_DL_Learner = execute_DL_Learner  # "/Users/demir/Desktop/Thesis/Project/dllearner-1.3.0/bin/cli"
        self.p_folder = p_folder
        self.kg_path = self.p_folder

    def set_experiment_path(self, p):
        self.p_folder = p

    def collect_configs(self, folder_path):
        """
        This function requires path in which many separet folders located
        it returns a list of tuples. T[0] current folders path
        T[1]config files name like 1.conf or 2.conf

        :param folder_path:
        :return:
        """

        dl_req = list()
        for root, dir, files in os.walk(folder_path):
            for i in dir:
                new_path = root + '/' + i
                for nroot, _, nfiles in os.walk(new_path):
                    # if 'DL_OUTPUT.txt' not in nfiles:
                    configs = [c for c in nfiles if 'conf' in c]
                    dl_req.append((nroot, configs))

        print(dl_req)
        print(len(dl_req))
        return dl_req

    def write_config(self, t):

        path, l = t

        assert os.path.isfile(self.execute_DL_Learner)

        output_of_dl = list()
        for confs in l:
            n_path = path + '/' + confs
            output_of_dl.append('\n\n')
            output_of_dl.append('### ' + confs + ' starts ###')
            print(n_path)

            result = subprocess.run([self.execute_DL_Learner, n_path], stdout=subprocess.PIPE, universal_newlines=True)

            lines = result.stdout.splitlines()
            output_of_dl.extend(lines)

            output_of_dl.append('### ' + confs + ' ends ###')

        f_name = path + '/' + 'DL_OUTPUT.txt'
        with open(f_name, 'w') as handle:
            for sentence in output_of_dl:
                handle.write(sentence + '\n')
        handle.close()

        return output_of_dl

    def generated_responds(self, folder_path):
        data = self.collect_configs(folder_path)

        assert os.path.isfile(self.execute_DL_Learner)

        # dl_outputs=list()
        # for setting in data:
        #   dl_outputs.append(self.write_config(setting))

        # If memory allows
        e = ProcessPoolExecutor()
        dl_outputs = list(e.map(self.write_config, data))

        return dl_outputs

    def extract_info_from_settings(self, path_settings):

        regex_num = re.compile('[-+]?[0-9]*\.?[0-9]+')
        k1 = 'Size of vocabulary :'
        size_of_vocab = list()
        k2 = 'Num of RDF :'
        num_of_rdfs = list()
        k3 = 'energy_release_at_epoch :'
        energe_relases = list()
        k4 = 'Negative Constant :'
        NC = list()
        k5 = 'Num of dimension in Embedding Space :'
        embed_space = list()
        k6 = 'Ratio of total Attractive / repulsives:'
        ratios = list()
        k7 = 'Generating Embeddings: took:'
        time_emb = list()
        k8 = 'Num of generated clusters:'
        num_of_clusters = list()
        k9 = '### cluster distribution##'
        distributon_of_cluster = list()

        with open(path_settings, 'r') as reader:

            sentences = reader.readlines()
            print(sentences)

            for index, item in enumerate(sentences):
                if k1 in item:
                    val = ''.join(regex_num.findall(item))
                    size_of_vocab.append(val)
                    continue

                if k2 in item:
                    val = ''.join(regex_num.findall(item))
                    num_of_rdfs.append(val)
                    continue

                if k3 in item:
                    val = ''.join(regex_num.findall(item))
                    energe_relases.append(val)
                    continue

                if k4 in item:
                    val = ''.join(regex_num.findall(item))
                    NC.append(val)
                    continue

                if k5 in item:
                    val = ''.join(regex_num.findall(item))
                    embed_space.append(val)
                    continue

                if k6 in item:
                    val = ''.join(regex_num.findall(item))
                    ratios.append(val)
                    continue

                if k7 in item:
                    val = ''.join(regex_num.findall(item))
                    time_emb.append(val)
                    continue

                if k8 in item:
                    val = ''.join(regex_num.findall(item))
                    num_of_clusters.append(val)
                    continue

                if k9 in item:
                    cluster_infos = sentences[index + 1:sentences.index('#####\n')]
                    val = ''.join(regex_num.findall(cluster_infos))
                    distributon_of_cluster.append(val)
                    continue
        reader.close()

        assert len(energe_relases) == len(embed_space)

        df = pd.DataFrame(
            {k1: size_of_vocab,
             k2: num_of_rdfs,
             k3: energe_relases,
             k4: NC,
             k5: embed_space,
             k6: ratios,
             k7: time_emb,
             k8: num_of_clusters,
             })

    def collect_data(self, folder_path):
        """
        Process all Settings.txt
        Process all d_ratios.txt
        Process all DL_OUTPUT
        Combine all infos into csv
        :param folder_path:
        :return:
        """
        # Top 3 values and class expressions regarded
        self.collect_dl_output(folder_path)

        import re
        import os
        import pandas as pd
        import numpy as np

        regex_num = re.compile('[-+]?[0-9]*\.?[0-9]+')

        K = list()
        NC = list()
        energe_relases = list()
        number_of_samples = list()
        Epoch = list()
        matrix_norm_diff = list()
        ratio_of_attractive_repulsive = list()
        runtime_of_learning_embeddings = list()
        cluster_dist = list()

        folder_names = list()
        for root, dir, files in os.walk(folder_path):
            for i in dir:
                new_path = root + '/' + i
                for nroot, _, nfiles in os.walk(new_path):
                    folder_names.append(i)

                    individual_path = nroot + '/Settings'

                    with open(individual_path, 'r') as reader:

                        sentences = reader.readlines()

                        for index, item in enumerate(sentences):

                            if 'K:' in item:
                                val = ''.join(regex_num.findall(item))
                                K.append(val)
                                continue

                            if 'Negative Constant :' in item:
                                val = ''.join(regex_num.findall(item))
                                NC.append(val)
                                continue

                            if 'energy_release_at_epoch :' in item:
                                val = ''.join(regex_num.findall(item))
                                energe_relases.append(val)
                                continue

                            if 'num_sample_from_clusters :' in item:
                                val = ''.join(regex_num.findall(item))
                                number_of_samples.append(val)
                                continue

                            if 'Epoch: ' in item:
                                val = ''.join(regex_num.findall(item))
                                Epoch.append(val)
                                continue

                            if 'The differences in matrix norm:' in item:
                                val = ''.join(regex_num.findall(item))
                                matrix_norm_diff.append(val)
                                continue

                            if 'Ratio of total Attractive / repulsives:' in item:
                                val = ''.join(regex_num.findall(item))
                                ratio_of_attractive_repulsive.append(val)
                                continue

                            if 'Generating Embeddings: took:' in item:
                                val = ''.join(regex_num.findall(item))
                                runtime_of_learning_embeddings.append(val)
                                continue

                            if '### cluster distribution##' in item:
                                values = sentences[index + 1:index + 3]
                                item = ' '.join(values)
                                val = '-'.join(regex_num.findall(item))

                                cluster_dist.append(val)

                    reader.close()

        df = pd.DataFrame(
            {
                'names': folder_names,
                'Energy_Relase': np.array(energe_relases),  # , dtype=np.float32),
                'negative_constant': np.array(NC, dtype=np.float32),
                'K': np.array(K, dtype=int),

                'ratios': np.array(ratio_of_attractive_repulsive),  # dtype=np.float32
                'Runtime_of_E': np.array(runtime_of_learning_embeddings),

                'Epoch': np.array(Epoch),
                'matrix_norm_diff': np.array(matrix_norm_diff),
                'num_of_clusters': np.array(cluster_dist)})

        df.to_csv(folder_path + '/generated_data.csv')

        """
        
        regex_num = re.compile('[-+]?[0-9]*\.?[0-9]+')

        K = list()
        NC = list()
        energe_relases = list()
        number_of_samples = list()
        Epoch = list()
        matrix_norm_diff = list()
        ratio_of_attractive_repulsive = list()
        runtime_of_learning_embeddings = list()
        cluster_dist = list()

        folder_names = list()
        for root, dir, files in os.walk(folder_path):
            for i in dir:
                new_path = root + '/' + i
                for nroot, _, nfiles in os.walk(new_path):
                    folder_names.append(i)

                    individual_path = nroot + '/Settings'

                    with open(individual_path, 'r') as reader:

                        sentences = reader.readlines()

                        for index, item in enumerate(sentences):
                            print(item)

                            if 'K:' in item:
                                val = ''.join(regex_num.findall(item))
                                K.append(val)
                                continue

                            if 'Negative Constant :' in item:
                                val = ''.join(regex_num.findall(item))
                                NC.append(val)
                                continue

                            if 'energy_release_at_epoch :' in item:
                                val = ''.join(regex_num.findall(item))
                                energe_relases.append(val)
                                continue

                            if 'num_sample_from_clusters :' in item:
                                val = ''.join(regex_num.findall(item))
                                number_of_samples.append(val)
                                continue

                            if 'Epoch: ' in item:
                                val = ''.join(regex_num.findall(item))
                                Epoch.append(val)
                                continue

                            if 'The differences in matrix norm:' in item:
                                val = ''.join(regex_num.findall(item))
                                matrix_norm_diff.append(val)
                                continue

                            if 'Ratio of total Attractive / repulsives:' in item:
                                val = ''.join(regex_num.findall(item))
                                ratio_of_attractive_repulsive.append(val)
                                continue

                            if 'Generating Embeddings: took:' in item:
                                val = ''.join(regex_num.findall(item))
                                runtime_of_learning_embeddings.append(val)
                                continue

                            if '### cluster distribution##' in item:
                                values = sentences[index + 1:index + 3]
                                cluster_dist.append('---'.join(values))

                    reader.close()
                    exit(1)

        df = pd.DataFrame(
            {
                'names': folder_names,
                'Energy_Relase': np.array(energe_relases),  # , dtype=np.float32),
                'negative_constant': np.array(NC, dtype=np.float32),
                'K': np.array(K, dtype=int),

                'ratios': np.array(ratio_of_attractive_repulsive),  # dtype=np.float32
                'Runtime_of_E': np.array(runtime_of_learning_embeddings),

                'Epoch': np.array(Epoch),
                'matrix_norm_diff': np.array(matrix_norm_diff),
                'num_of_clusters': np.array(cluster_dist)})

        df.to_csv(folder_path + '/generated_data')
        return folder_path + '/generated_data'
        """

    def combine_all_data(self, folder_path):

        assert os.path.isfile(folder_path + '/DL_responds.csv')
        assert os.path.isfile(folder_path + '/SOBOL_Parameters.csv')
        assert os.path.isfile(folder_path + '/generated_data.csv')

        dl_df = pd.read_csv(folder_path + '/DL_responds.csv', index_col=0)
        sobol_df = pd.read_csv(folder_path + '/SOBOL_Parameters.csv', index_col=0)
        gen_df = pd.read_csv(folder_path + '/generated_data.csv', index_col=0)

        assert np.all(dl_df.names == gen_df.names)

        dl_df.drop(['names'], axis=1, inplace=True)
        merged_df = pd.concat([gen_df, dl_df], axis=1, join='inner')
        print(merged_df.columns.values)
        del gen_df
        del dl_df

        assert len(merged_df) == len(sobol_df)
        merged_df['names'] = pd.to_datetime(merged_df.names)
        merged_df.sort_values(by=['names'], inplace=True, ascending=True)

        sobol_df.drop(['energy_release_at_epoch', 'negative_constant'], axis=1, inplace=True)
        merged_df = pd.concat([merged_df, sobol_df], axis=1, join='inner')

        merged_df.to_csv(folder_path + '/alldata.csv')

    def collect_dl_output(self, folder_path):
        """
        Collects all DL_OUTPUT FILEs
        :param folder_path:
        :return:
        """
        """
        def formatter(text):
            v = re.findall('[-+]?[0-9]*\.?[0-9]+', text)
            # rank, prediction accuracy, f mesaure
            if len(v) == 3:
                return float(v[-1])
            elif len(v) == 4 or len(v) == 5:
                # print(text)
                # print(v)
                # print('val to be returned',float(v[-1]))
                return float(v[-1])
            else:
                # print(text)
                # print(v)
                # print(len(v))
                print('unexpected val')
                exit(1)

        holder = list()
        for root, dir, files in os.walk(folder_path):
            for i in dir:
                new_path = root + '/' + i
                for nroot, _, nfiles in os.walk(new_path):
                    # folder_names.append(i)
                    individual_path = nroot + '/DL_OUTPUT.txt'

                    with open(individual_path, 'r') as reader:
                        f_scores = [formatter(item) for item in reader if '1:' in item or '2:' in item or '3:' in item]

                        if f_scores:
                            # This f scores is obtained by   only regarding top3
                            # print(individual_path)
                            # print(f_scores)
                            # firsts.append(f_scores[0])
                            # seconds.append(f_scores[1])
                            # thirds.append(f_scores[2])
                            holder.append((individual_path, f_scores[0], f_scores[1], f_scores[2]))
                        else:
                            print('EMPTY')
                            print(individual_path)
                            print(f_scores)

        folder_names = list()
        firsts = list()
        seconds = list()
        thirds = list()

        for i in holder:
            folder_names.append(i[0])
            firsts.append(i[1])
            seconds.append(i[2])
            thirds.append(i[3])

        df = pd.DataFrame(
            {'names': folder_names,
             'Rank1': np.array(firsts),
             'Rank2': np.array(seconds),
             'Rank3': np.array(thirds),
             })
        df.to_csv(folder_path + '/DL_responds.csv')
        
        """

        def formatter(text):

            v = re.findall('[-+]?[0-9]*\.?[0-9]+', text)
            # rank, prediction accuracy, f mesaure
            return v[-1]
            """
            # get first and last two collumn

            if len(v) == 3:
                return float(v[-1])
            elif len(v) == 4 or len(v) == 5:
                # print(text)
                # print(v)
                # print('val to be returned',float(v[-1]))
                return float(v[-1])
            else:
                print(text)
                print(v)
                # print(text)
                # print(v)
                # print(len(v))
                print('unexpected val')
                exit(1)
            """

        holder = list()
        for root, dir, files in os.walk(folder_path):

            for i in dir:
                new_path = root + '/' + i

                for nroot, _, nfiles in os.walk(new_path):

                    # folder_names.append(i)
                    individual_path = nroot + '/DL_OUTPUT.txt'

                    with open(individual_path, 'r') as reader:

                        f_scores = [formatter(item) for item in reader if '1:' in item or '2:' in item or '3:' in item]

                        if f_scores:
                            # This f scores is obtained by   only regarding top3
                            # print(individual_path)
                            # print(f_scores)
                            # firsts.append(f_scores[0])
                            # seconds.append(f_scores[1])
                            # thirds.append(f_scores[2])
                            holder.append((individual_path, f_scores[0], f_scores[1], f_scores[2]))
                        else:
                            print('EMPTY')
                            print(individual_path)
                            print(f_scores)

                        # means.append(np.mean(np.array(f_scores)))
                        # std.append(np.std(np.array(f_scores)))
                        # medians.append(np.median(np.array(f_scores)))
                        # mins.append(np.min(np.array(f_scores)))
                        # maxs.append(np.max(np.array(f_scores)))
        folder_names = list()
        firsts = list()
        seconds = list()
        thirds = list()
        for i in holder:
            folder_names.append(i[0])
            firsts.append(i[1])
            seconds.append(i[2])
            thirds.append(i[3])
        df = pd.DataFrame(
            {'names': folder_names,
             'Rank1': np.array(firsts),
             'Rank2': np.array(seconds),
             'Rank3': np.array(thirds),
             })
        df.to_csv(folder_path + '/DL_responds.csv')

    def topN_cosine(self, learned_embeddings, topN, times):

        vocabulary_of_entity_names = list(learned_embeddings.index)

        sampled_index_of_entities = np.random.randint(len(learned_embeddings), size=times)

        learned_embeddings = learned_embeddings.values

        list_of_clusters = []

        for i in sampled_index_of_entities:
            d = dict()
            sampled_vector = learned_embeddings[i]
            sampled_entitiy_name = vocabulary_of_entity_names[i]

            Saver.settings.append('\n Target: ' + sampled_entitiy_name)
            print('\n Target:', sampled_entitiy_name)

            distances = list()

            for index, ith_vec_of_embeddings in enumerate(learned_embeddings):
                try:
                    distances.append(spatial.distance.cosine(sampled_vector, ith_vec_of_embeddings))
                except RuntimeWarning as r:
                    print(r)
                    continue

            distances = np.array(distances)

            most_similar = distances.argsort()[1:topN]

            Saver.settings.append('\n Top similars\n')
            similars = list()
            for index, j in enumerate(most_similar):
                print(str(index) + '. similar: ' + vocabulary_of_entity_names[j])
                similars.append(vocabulary_of_entity_names[j])
                Saver.settings.append(str(index) + '. similar: ' + vocabulary_of_entity_names[j])

            d[sampled_entitiy_name] = similars
            list_of_clusters.append(d)

        return list_of_clusters

    def new_cosine(self, learned_embeddings):
        from sklearn.metrics.pairwise import cosine_similarity

        sims = cosine_similarity(learned_embeddings)

        vocabulary_of_entity_names = list(learned_embeddings.index)

        top_sims = np.argmax(sims, axis=0)

        for index, name in enumerate(vocabulary_of_entity_names):
            print('Target ', name)
            print('Most sim', vocabulary_of_entity_names[top_sims[index]])

            if index == 10:
                break

    @staticmethod
    def calculate_euclidean_distance(*, embeddings, entitiy_to_P_URI, entitiy_to_N_URI):
        """
        Calculate the difference
        Target entitiy
        Attractive entitiy entitiy_to_P_URI list of dictionaries
        Repulsive entitiy
        """

        total_distance_from_attractives = 0
        total_distance_from_repulsives = 0

        for index in range(len(entitiy_to_P_URI)):
            index_of_attractive_entitites = np.array(list(entitiy_to_P_URI[index].keys()), dtype=np.int32)
            index_of_repulsive_entitites = np.array(list(entitiy_to_N_URI[index]), dtype=np.int32)

            total_distance_from_attractives += np.linalg.norm(
                embeddings[index_of_attractive_entitites] - embeddings[index])

            total_distance_from_repulsives += np.linalg.norm(
                embeddings[index_of_repulsive_entitites] - embeddings[index])

        print('Distance comparision d(A)/d(R) ', total_distance_from_attractives / total_distance_from_repulsives)

    @performance_debugger('Sample from mean of clusters')
    def sample_from_mean_of_clusters(self, labeled_embeddings, num_sample_from_clusters):
        def calculate_difference_to_mean(data_point):
            euclidiean_dist = distance.euclidean(data_point, mean_of_cluster)
            return euclidiean_dist

        labels = labeled_embeddings.labels.unique().tolist()

        sampled_total_entities = pd.DataFrame()

        for label in labels:
            cluster = labeled_embeddings.loc[labeled_embeddings.labels == label].copy()
            mean_of_cluster = cluster.mean()

            cluster['d_to_mean'] = cluster.apply(calculate_difference_to_mean, axis=1)
            # cluster.loc[:, 'description']
            cluster.sort_values(by=['d_to_mean'], ascending=False)

            sampled_entities_from_cluster = cluster.head(num_sample_from_clusters)
            sampled_total_entities = pd.concat([sampled_total_entities, sampled_entities_from_cluster])

        sampled_total_entities.drop(['d_to_mean'], axis=1, inplace=True)

        representative_entities = dict()
        for key, val in sampled_total_entities.labels.to_dict().items():
            representative_entities.setdefault(val, []).append(key)

        return representative_entities

    @performance_debugger('Sample from mean of clusters')
    def sample_from_clusters(self, labeled_embeddings, num_sample_from_clusters):

        labels = labeled_embeddings.labels.unique().tolist()

        sampled_total_entities = pd.DataFrame()

        for label in labels:
            cluster = labeled_embeddings.loc[labeled_embeddings.labels == label]
            if len(cluster) > num_sample_from_clusters:
                sampled_entities_from_cluster = cluster.sample(n=num_sample_from_clusters)
            else:
                sampled_entities_from_cluster = cluster

            sampled_total_entities = pd.concat([sampled_total_entities, sampled_entities_from_cluster])

        representative_entities = dict()
        for key, val in sampled_total_entities.labels.to_dict().items():
            representative_entities.setdefault(val, set()).add(key)

        return representative_entities

    @performance_debugger('Pseudo labeling via HDBSCAN')
    def pseudo_label_HDBSCAN(self, df, min_cluster_size=None, min_samples=None):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit(df)
        df['labels'] = clusterer.labels_
        return df

    @performance_debugger('Pseudo labeling via DBSCAN')
    def pseudo_label_DBSCAN(self, df, eps=None, min_samples=None):
        df['labels'] = DBSCAN(eps=eps, min_samples=min_samples).fit(df).labels_
        return df

    @performance_debugger('Pseudo labeling via Kmeans')
    def pseudo_label_Kmeans(self, df, n_clusters=2):
        df['labels'] = MiniBatchKMeans(n_clusters).fit(df).labels_
        return df

    def perform_clustering_quality(self, df, type_info=None):
        """

        :param df:
        :param re_indexer:
        :return:
        """

        if type_info == None:
            type_info = ut.deserializer(path=self.p_folder, serialized_name='type_info')

        df_only_subjects = df.loc[list(type_info.keys())]

        print(len(df_only_subjects))
        clusters = pd.unique(df_only_subjects.labels)

        score = 0
        for c in clusters:
            if c == -1:
                'indicates noise'
                continue

            indexes_in_c = df_only_subjects[df_only_subjects.labels == c].index.values

            sum_of_cosines = 0
            valid_indexes_in_c = indexes_in_c

            if len(valid_indexes_in_c) == 1:
                continue
            print('##### CLUSTER', c, ' #####')

            if len(valid_indexes_in_c) > 1000:
                sampled_points = self.sample_from_mean_of_clusters(df_only_subjects[df_only_subjects.labels == c], 1000)
                index_of_sampled_points = list(*sampled_points.values())

                poss = [pos for pos, i in enumerate(index_of_sampled_points) if len(type_info[i]) > 0]
                valid_indexes_in_c = indexes_in_c[poss]

            for i in valid_indexes_in_c:

                # returns a set of indexes
                types_i = type_info[i]
                len_types_i = len(types_i)

                for j in valid_indexes_in_c:
                    nom = len(types_i & type_info[j])
                    denom = math.sqrt(len_types_i) * math.sqrt(len(type_info[j]))

                    sum_of_cosines += nom / denom

            aritiy = sum_of_cosines / (len(valid_indexes_in_c) ** 2)

            score += aritiy
            output = 'Size of the cluster(' + str(c) + ')=' + str(
                len(valid_indexes_in_c)) + '\t Cluster Quality=' + str(aritiy)
            print(output)

            Saver.settings.append(output)

            """
            type_historgram=vocabulary[np.array(type_historgram)]
            type_historgram=list(map(lambda x:x.rsplit('/')[-1],type_historgram))
#            type_historgram=list(map(lambda x: x[x.index('#'):],type_historgram))
            plt.title('Historgram of types in cluster_'+str(c)+'_score:'+str(aritiy))
            plt.xticks(rotation=90)
            plt.hist(type_historgram,bins=100)

            plt.show()
            """

        mean_of_scores = score / len(clusters)
        print('\nMean of cluster quality', mean_of_scores)
        Saver.settings.append('Mean of cluster quality: ' + str(mean_of_scores))

        return mean_of_scores

    def perform_type_prediction(self, df):

        def get_similarities(e:pd.DataFrame):
            """

            :param e:
            :return:
            """
            kdt = KDTree(e, metric='euclidean')
            _ = kdt.query(e, k=101, return_distance=False)
            s=pd.DataFrame(_)
            
            s.to_csv(self.p_folder + '/Similarities_pyke_50.csv')

            #vocabulary = ut.deserializer(path=self.p_folder, serialized_name='vocabulary')
            #l = s.applymap(lambda x: vocabulary[x])
            #l.to_csv(self.p_folder + '/mapped_most_sims.csv')

            # reindex the similarity results
            mapper = dict(zip(list(range(len(s))), e.index.values))
            s = s.applymap(lambda x: mapper[x])

            return s


        # get the types. Mapping from the index of subject to the index of object
        type_info = ut.deserializer(path=self.p_folder, serialized_name='type_info')

        # get the index of objects / get type information =>>> s #type o
        all_types = sorted(set.union(*list(type_info.values())))

        # Consider only points with type infos.
        e_w_types = df.loc[list(type_info.keys())]

        # get most similar 100 points
        df_most_similars=get_similarities(e_w_types)

        k_values = [1, 3, 5, 10, 15, 30, 50, 100]


        e = ThreadPoolExecutor(20)

        def create_binary_type_vector(t_types,a_types):
            vector = np.zeros(len(all_types))
            i = [a_types.index(_) for _ in t_types]
            vector[i] = 1
            return vector

        def create_binary_type_prediction_vector(t_types,a_types):
            vector = np.zeros(len(all_types))
            i = [a_types.index(_) for _ in chain.from_iterable(t_types)]
            vector[i] += 1
            return vector




        for k in k_values:
            print('#####', k, '####')
            similarities = list()
            for _, S in df_most_similars.iterrows():

                true_types = type_info[S.values[0]]
                type_predictions = [type_info[_] for _ in S.values[1:k + 1]]


                f_target = e.submit(create_binary_type_vector, true_types, all_types)
                f_pre = e.submit(create_binary_type_prediction_vector, type_predictions, all_types)

                vector_true=f_target.result()
                vector_prediction=f_pre.result()


                sim = cosine(vector_true, vector_prediction)
                similarities.append(1 - sim)


            report = pd.DataFrame(similarities)
            print('Mean type prediction', report.mean().values)
            #            print(report.describe())
            report.to_csv(self.p_folder + '/TypePrediction_' + str(k) + '_PYKE_50__cosine.csv')

    @performance_debugger('Prune non resources')
    def prune_non_subject_entities(self, embeddings, upper_folder='not init'):

        if upper_folder != 'not init':
            index_to_predicates = ut.deserializer(path=upper_folder, serialized_name="predicates_to_indexes")
        else:
            #           index_to_predicates = ut.deserializer(path='/home/demir/Desktop/work/DICE_pl2vec/Experiments/',
            #                                                  serialized_name="index_to_prune_for_dl")
            #          vocabulary = ut.deserializer(path='/home/demir/Desktop/work/DICE_pl2vec/Experiments/',
            #                                      serialized_name="vocabulary")
            index_to_predicates = ut.deserializer(path=self.p_folder, serialized_name="index_to_prune_for_dl")
            vocabulary = ut.deserializer(path=self.p_folder, serialized_name="vocabulary")

        raw_predicates = list(index_to_predicates.values())
        for i in raw_predicates:
            if i in vocabulary:
                del vocabulary[i]

        # PRUNE non subject entities
        embeddings = embeddings[list(vocabulary.values())]

        #       names=[re.findall('<(.+?)>', name)[0] for name in list(vocabulary.keys())]

        embeddings = pd.DataFrame(embeddings, index=list(vocabulary.keys()))

        del vocabulary

        # subembeddings = embeddings[~embeddings.index.str.match('http://dl-learner.org/carcinogenesis')]

        return embeddings

    @performance_debugger('Embedding space partitioning')
    def apply_partitioning(self, X, partitions, samp_part):

        if len(X) < partitions:
            partitions = len(X) // 5

        # print('Partition embedding space into: ', partitions)
        Saver.settings.append('KMEANS partitions space into: ' + str(partitions))
        df = pd.DataFrame(X)
        df["labels"] = np.array(MiniBatchKMeans(n_clusters=partitions).fit(X).labels_)

        sampled_total_entities = pd.DataFrame()

        for item in df.labels.unique():
            cluster = df.loc[df.labels == item]
            sampled_total_entities = pd.concat([sampled_total_entities, cluster.head(samp_part * 2)])
        del df

        sampled_total_entities.drop(columns=['labels'], inplace=True)
        return sampled_total_entities

    def perform_sampling(self, embeddings, num_of_sample):
        """
        Partion embeddings space into 100 cluster and sample num_sample entities from
        each cluster mean
        :param embeddings:
        :param num_sample:
        :return:
        """
        embeddings = self.sample_from_mean_of_clusters(embeddings, num_of_sample)
        embeddings.drop(columns=['labels'], inplace=True)
        return embeddings

    @performance_debugger('Pipeline of DP')
    def pipeline_of_data_processing(self, path, E, num_sample_from_clusters):

        # prune predicates and literalse
        embeddings_of_resources = self.prune_non_subject_entities(embeddings=E, upper_folder=path)

        # Partition total embeddings into number of 10
        E = self.apply_partitioning(embeddings_of_resources, partitions=20, samp_part=20)

        pseudo_labelled_embeddings = self.pseudo_label_DBSCAN(embeddings_of_resources)

        ut.serializer(object_=self.p_folder, path=path,
                      serialized_name='pseudo_labelled_resources')

        Saver.settings.append("### cluster distribution##")
        Saver.settings.append(pseudo_labelled_embeddings.labels.value_counts().to_string())
        Saver.settings.append("#####")

        sampled_embeddings = self.sample_from_mean_of_clusters(pseudo_labelled_embeddings, num_sample_from_clusters)

        representative_entities = dict()

        for key, val in sampled_embeddings.labels.to_dict().items():
            representative_entities.setdefault(val, []).append(key)

        Saver.settings.append(
            'Num of generated clusters: ' + str(len(list(representative_entities.keys()))))

        return representative_entities

    @performance_debugger('Pipeline of DP')
    def pipeline_of_data_processing_single_run(self, E, num_sample_from_clusters):
        # prune predicates and literals
        embeddings_of_resources = self.prune_non_subject_entities(E)

        # Partition total embeddings into number of 10
        # embeddings_of_resources = self.apply_partitioning(embeddings_of_resources, partitions=50, samp_part=50)

        pseudo_labelled_embeddings = self.pseudo_label_DBSCAN(embeddings_of_resources)

        # pseudo_labelled_embeddings = self.pseudo_label_Kmeans(embeddings_of_resources)

        self.topN_cosine(pseudo_labelled_embeddings, 10, 10)

        pseudo_labelled_embeddings.to_csv(self.p_folder + '/pseudo_labelled_embeddings.csv')

        # ut.serializer(object_=pseudo_labelled_embeddings, path=self.p_folder,
        #               serialized_name='pseudo_labelled_resources')

        Saver.settings.append("### cluster distribution##")
        Saver.settings.append(pseudo_labelled_embeddings.labels.value_counts().to_string())
        Saver.settings.append("##" * 20)

        # representative_entities = self.sample_from_clusters(pseudo_labelled_embeddings, num_sample_from_clusters)

        representative_entities = self.sample_from_mean_of_clusters(pseudo_labelled_embeddings,
                                                                    num_sample_from_clusters)

        Saver.settings.append('Num of generated clusters: ' + str(len(list(representative_entities.keys()))))

        return representative_entities

    def apply_tsne_and_plot_only_subjects(self, e, prune=False):
        """
        e should be V x 2 dataframe index must be names
        :param e:
        :param prune:
        :return:
        """
        import dash
        import dash_core_components as dcc
        import dash_html_components as html
        import plotly.graph_objs as go

        if prune:
            e = self.prune_non_subject_entities(e)

        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

        names = list(e.index.values)

        x = e.values[:, 0]
        y = e.values[:, 1]

        app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
        app.layout = html.Div([
            dcc.Graph(
                id='life-exp-vs-gdp',
                figure={
                    'data': [
                        go.Scatter(
                            x=x,
                            y=y,
                            text=names,
                            mode='markers',
                            opacity=0.7,
                            marker={
                                'size': 15,
                                'line': {'width': 0.5, 'color': 'white'}
                            },
                            name=i
                        ) for i in names
                    ],
                    'layout': go.Layout(
                        xaxis={'title': 'X'},
                        yaxis={'title': 'Y'},
                        hovermode='closest'
                    )
                }
            )
        ])

        app.run_server(debug=True)

    def create_config(self, config_path, pos_examples, sampled_neg):

        Text = list()
        pos_string = "{ "
        neg_string = "{ "
        for i in pos_examples:
            pos_string += "\"" + i + "\","

        for j in sampled_neg:
            neg_string += "\"" + str(j) + "\","

        pos_string = pos_string[:-1]
        pos_string += "}"

        neg_string = neg_string[:-1]
        neg_string += "}"

        Text.append("rendering = \"dlsyntax\"")
        Text.append("// knowledge source definition")
        Text.append("ks.type = \"OWL File\"")
        Text.append("\n")

        Text.append("// knowledge source definition")

        Text.append(
            "ks.fileName = \"" + self.kg_path + '/KB.txt\"')  #

        Text.append("\n")

        Text.append("reasoner.type = \"closed world reasoner\"")
        Text.append("reasoner.sources = { ks }")
        Text.append("fmeasure.type = \"fmeasure\"")
        Text.append("\n")

        # Text.append("lp.type = \"posonlylp\"")#celoe

        Text.append("lp.type = \"PosNegLPStandard\"")  # ocel
        # Text.append("lp.accuracyMethod = \"pred_acc\"")
        Text.append("\n")

        Text.append("lp.positiveExamples =" + pos_string)
        Text.append("\n")

        Text.append("lp.negativeExamples =" + neg_string)
        Text.append("\n")

        Text.append('lp.accuracyMethod = fmeasure')

        Text.append("alg.type = \"celoe\"")
        Text.append("op.type = \"rho\"")
        Text.append("op.useHasValueConstructor = false")
        Text.append("\n")
        Text.append("// create learning algorithm to run")
        # Text.append("alg.writeSearchTree = true")
        Text.append("alg.replaceSearchTree = true")
        Text.append("\n")
        Text.append("alg.maxExecutionTimeInSeconds = 30")
        Text.append("alg.expandAccuracy100Nodes = true")
        # Text.append("alg.noisePercentage = 100")

        pathToConfig = config_path + '.conf'

        file = open(pathToConfig, "wb")

        for i in Text:
            file.write(i.encode("utf-8"))
            file.write("\n".encode("utf-8"))
        file.close()

    def create_config_pos_only(self, pathToconfig, pos_examples):

        Text = list()
        pos_string = "{ "
        for i in pos_examples:
            pos_string += "\"" + i + "\","

        pos_string = pos_string[:-1]
        pos_string += "}"

        Text.append("rendering = \"dlsyntax\"")
        Text.append("// knowledge source definition")
        Text.append("ks.type = \"OWL File\"")
        Text.append("\n")

        Text.append("// knowledge source definition")
        Text.append(
            "ks.fileName = \"/Users/demir/Desktop/Thesis/illustration_of_thesis/DBpedia/KB.txt\"")  # DBpedia/KB.txt.txt

        Text.append("\n")

        Text.append("reasoner.type = \"closed world reasoner\"")
        Text.append("reasoner.sources = { ks }")

        Text.append("lp.type = \"posonlylp\"")
        Text.append("\n")

        Text.append("lp.positiveExamples =" + pos_string)
        Text.append("\n")

        Text.append("alg.type = \"celoe\"")
        Text.append("alg.maxExecutionTimeInSeconds = 30")
        Text.append("alg.expandAccuracy100Nodes = true")
        Text.append("alg.noisePercentage = 100")

        pathToConfig = pathToconfig + '.conf'

        file = open(pathToConfig, "wb")

        for i in Text:
            file.write(i.encode("utf-8"))
            file.write("\n".encode("utf-8"))
        file.close()

    def execute_DL(self, resources, dict_of_cluster):

        clusters = np.array(list(dict_of_cluster.keys()))

        print('Total Num of clusters ', clusters)

        for cluster_label, uris_in_pos in dict_of_cluster.items():
            uris_in_pos = set(uris_in_pos)

            other_clusters = clusters[clusters != cluster_label]

            sampled_negatives = list()

            if len(clusters) > 1:
                _ = [list(dict_of_cluster[item]) for item in other_clusters]
                sampled_negatives = list(itertools.chain.from_iterable(_));
                del _
            # sampled_negatives = set(np.random.choice(sampled_negatives, len(sampled_negatives), replace=False))
            else:
                sampled_negatives.extend((np.random.choice(resources, 2, replace=False)))

            self.create_config(self.p_folder + '/' + str(cluster_label), uris_in_pos, sampled_negatives)

    def execute_dl(self, sampled_representatives):

        if len(sampled_representatives) == 1:
            print('Only one cluster found in the embedding space')
            print('Exitting.')
            return None
        try:

            sum_lists = list(sampled_representatives.values())

            all_samples = set()

            for i in sum_lists:

                for j in i:
                    all_samples.add(j)

        #           all_samples = set.union(*set(list(sampled_representatives.values())))
        except TypeError as t:
            print(t)
            print(sampled_representatives)
            print(sampled_representatives.values())
            print(type(sampled_representatives.values()))
            exit(1)

        for cluster_label, positive_subjects in sampled_representatives.items():
            sampled_negatives = all_samples.difference(positive_subjects)

            """
            
            try:
                sampled_negatives = np.random.choice(candidates_of_negative_subjects, len(positive_subjects),
                                                     replace=False)
            except ValueError as t:
                print(t)
                sampled_negatives = candidates_of_negative_subjects
            """
            self.create_config(self.p_folder + '/' + str(cluster_label), positive_subjects, sampled_negatives)

    @performance_debugger('DL-Learner')
    def pipeline_of_dl_learner(self, path, dict_of_cluster_with_original_term_names):

        print(dict_of_cluster_with_original_term_names)

        exit(1)
        with open(path + "/subjects_to_indexes.p", "rb") as f:
            str_subjects = list(pickle.load(f).keys())

        f.close()

        for key, val in dict_of_cluster_with_original_term_names.items():
            dict_of_cluster_with_original_term_names[key] = [str_subjects[item] for item in val]

        self.execute_DL(resources=str_subjects, dict_of_cluster=dict_of_cluster_with_original_term_names)

    @performance_debugger('DL-Learner')
    def pipeline_of_single_evaluation_dl_learner(self, dict_of_cluster_with_original_term_names):

        self.execute_dl(dict_of_cluster_with_original_term_names)


class Saver:
    settings = []
    similarities = []
