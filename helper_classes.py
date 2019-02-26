import math
import os
import pickle
import random
import re
from bz2 import BZ2File
from collections import Counter
import itertools
from scipy import spatial
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import util as ut
import os.path
import matplotlib.pyplot as plt
from numpy import linalg as LA
import numpy as np
import pandas as pd
from scipy.spatial import distance
import subprocess
import time
from scipy import sparse
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

import warnings

warnings.filterwarnings('error')


def performance_debugger(func_name):
    def function_name_decoratir(func):
        def debug(*args, **kwargs):
            long_string = ''
            starT = time.time()
            #            long_string += '\n'
            #            long_string += '#####' * 10 + '\n'
            print('\n######', func_name, ' starts ######')
            r = func(*args, **kwargs)
            print(func_name, ' took ', time.time() - starT, ' seconds\n')
            long_string += str(func_name) + ' took:' + str(time.time() - starT) + ' seconds'
            Saver.settings.append(long_string)

            return r

        return debug

    return function_name_decoratir


class Parser:
    def __init__(self, logger=False, p_folder: str = 'not initialized'):
        self.path = 'uninitialized'
        self.logger = logger
        self.p_folder = p_folder
        self.similarity_function = None

    def set_similarity_function(self, f):
        self.similarity_function = f

    def set_experiment_path(self, p):
        self.p_folder = p

    @staticmethod
    def calculate_marginal_probabilities(binary_co_matrix: Dict, number_of_rdfs: int):
        marginal_probs = dict()
        for unq_ent, list_of_context_ent in binary_co_matrix.items():
            # N is multipled by 2 as list_of_context_ent contains other two element of an RDF triple
            marginal_prob = len(list_of_context_ent) / (number_of_rdfs * 2)
            marginal_prob = round(marginal_prob, 5)
            marginal_probs[unq_ent] = marginal_prob
        return marginal_probs

    @staticmethod
    @performance_debugger('Calculating entropies')
    def calculate_entropies(binary_co_matrix: Dict, number_of_rdfs: int):
        entropies = dict()
        for unq_ent, list_of_context_ent in binary_co_matrix.items():
            # N is multiplied by 2 as list_of_context_ent contains other two element of an RDF triple
            marginal_prob = len(list_of_context_ent) / (number_of_rdfs * 2)

            with np.errstate(divide='raise'):
                try:
                    entropy = -marginal_prob * np.log2(marginal_prob)
                    entropies[unq_ent] = entropy
                except FloatingPointError:
                    print('entropy of term-', unq_ent, ': is set 0')
                    print('P( term-', unq_ent, '- is', marginal_prob)
                    entropies[unq_ent] = 0

        return entropies

    @staticmethod
    def calculate_ppmi(binary_co_matrix: Dict, marginal_probs: Dict, number_of_rdfs) -> Dict:
        pmi_val_target_to_context_positive = dict()

        for unq_ent, list_of_context_ent in binary_co_matrix.items():

            marginal_prob_of_target = marginal_probs[unq_ent]

            statistical_info_of_cooccurrences = Counter(list_of_context_ent)

            pmi_val_target_to_context_positive.setdefault(unq_ent, dict())

            for context_ent, co_occuring_freq in statistical_info_of_cooccurrences.items():

                joint_prob = round(co_occuring_freq / number_of_rdfs, 5)

                marginal_prob_of_context = marginal_probs[context_ent]

                denominator = marginal_prob_of_target * marginal_prob_of_context

                if denominator > 0.00 and joint_prob > 0.0000:
                    PMI_val = round(math.log2(joint_prob) - math.log2(denominator), 5)

                    if PMI_val > 0.00000000:
                        pmi_val_target_to_context_positive[unq_ent][context_ent] = PMI_val
                        continue

        return pmi_val_target_to_context_positive

    @staticmethod
    @performance_debugger('Calculating entropy weighted Jaccard index')
    def calculate_entropy_jaccard(freq_matrix: Dict, entropies: Dict) -> Dict:
        entropy_jaccard = {}

        for target, list_of_context_ent in freq_matrix.items():

            entropy_jaccard.setdefault(target, dict())
            co_occurring_points_w_target = set(list_of_context_ent)

            # Sum of entropies of all points occurred with target (x).
            sum_of_entropy_points_target = sum(list(map(lambda i: entropies[i], co_occurring_points_w_target)))

            for co_point in co_occurring_points_w_target:
                co_occurring_points_w_co_point = set(freq_matrix[co_point])

                overlapped_elements = co_occurring_points_w_target.intersection(co_occurring_points_w_co_point)

                if len(overlapped_elements) == 0:
                    print('No overlapping')
                    continue

                # Sum of entropies of all overlapping points
                sum_of_entropy_overlapping_points = sum(list(map(lambda i: entropies[i], overlapped_elements)))

                sum_of_entropy_points_co_point = sum(list(map(lambda i: entropies[i], co_occurring_points_w_co_point)))

                sim = sum_of_entropy_overlapping_points / (
                        sum_of_entropy_points_target + sum_of_entropy_points_co_point)


                exit(1)
                entropy_jaccard[target][co_point] = np.round(sim, 6)

        assert len(freq_matrix) == len(entropy_jaccard)
        return entropy_jaccard

    @staticmethod
    @performance_debugger('Calculating entropy weighted Jaccard index')
    def efficient_calculate_entropy_jaccard(freq_matrix: Dict, entropies: Dict) -> Dict:
        """

        :param freq_matrix:
        :param entropies: is a dictionary that maps index of point to its entropy.
        :return:
        """
        assert np.all(np.array(list(entropies.keys())) == np.arange(len(entropies)))
        # we utilize the following assumption. As entropies dictionary start from 0 to rows of freg_matrix
        # so that we compress the required memory for entropies even further.
        num_array_entropies = np.array(list(entropies.values()))
        assert len(num_array_entropies) == len(freq_matrix)
        del entropies

        texec = ThreadPoolExecutor(8)

        def f(l):
            return np.unique(np.array(l))


        # as we count each occurrences of points
        # Contexes is a list of list from index to the indexes of co-occurred points.
        contexts = np.array(np.array(list(map(f, list(freq_matrix.values())))))
        entropy_jaccard = dict()

        for component_a, context_of_component_a in enumerate(contexts):

            #entropy_jaccard.setdefault(component_a, dict())

            all_context_of_component_b = contexts[context_of_component_a]

            futures = []
            for item in all_context_of_component_b:
                futures.append(
                    texec.submit(ut.calculate_similarities, context_of_component_a, item, num_array_entropies))

            similarities = [ff.result() for ff in futures]

            d=dict(zip(context_of_component_a,similarities))
            entropy_jaccard[component_a] = d

        assert len(freq_matrix) == len(entropy_jaccard)

        return entropy_jaccard

    def apply_ppmi_on_co_matrix(self, binary_co_matrix: Dict, num_triples: int) -> Dict:
        marginal_probabilities = self.calculate_marginal_probabilities(binary_co_matrix, num_triples)
        return self.calculate_ppmi(binary_co_matrix, marginal_probabilities, num_triples)

    def apply_entropy_jaccard_on_co_matrix(self, freq_matrix: Dict, num_triples: int) -> Dict:
        """
        Calculate Shannon entropy weighted jaccard from freq_matrix
                A=  Sum of entropies of overlapping elements
                B=  Sum of entropies of all elements that occurred with x
                C=  Sum of entropies of all elements that occurred with y

       Sim(x,y) = A / (B+C)


        :param freq_matrix: freq_matrix is mapping from a vertex or an edge to co-occurring vertices or edges.
        :param num_triples:
        :return: Mapping from points to a mapping containing a point and positive jaccard sim.
        """

        entropies = self.calculate_entropies(freq_matrix, num_triples)

        entropy_jaccard = self.efficient_calculate_entropy_jaccard(freq_matrix, entropies)

       #entropy_jaccard = self.calculate_entropy_jaccard(freq_matrix, entropies)

        return entropy_jaccard

    def create_dictionary(self, f_name, bound):

        total_rdf_tripples = 0
        binary_co_occurence_matrix = {}

        DBpedia_files = list()
        vocabulary = dict()
        subjects_to_indexes = dict()

        for root, dir, files in os.walk(f_name):
            for file in files:
                if '.bz' in file:
                    DBpedia_files.append(f_name + '/' + file)

        assert len(DBpedia_files) > 0

        handle = open(root + '/' + 'KB.txt', 'w')

        for f_name in DBpedia_files:
            counter = 0
            with BZ2File(f_name, "r") as reader:

                for bytes_of_sentence in reader:
                    sentence = bytes_of_sentence.decode('utf-8')

                    components = re.findall('<(.+?)>', sentence)
                    if len(components) == 3:
                        s, p, o = components

                    elif len(components) == 2:
                        s, p = components
                        __ = len(s) + len(p)
                        # To obtain literal
                        o = (sentence[__ + 6:-3])
                    else:

                        if self.logger:
                            print('Wrong formatted data: ', re.sub("\s+", " ", sentence), ' in', f_name)

                        continue
                    counter += 1

                    if counter == bound:
                        break

                    # Write into file to be used by DL-Learner
                    handle.write(sentence)
                    total_rdf_tripples += 1

                    # mapping from string to vocabulary
                    vocabulary.setdefault(s, len(vocabulary))
                    subjects_to_indexes.setdefault(s, len(vocabulary) - 1)

                    vocabulary.setdefault(p, len(vocabulary))

                    o = re.sub("\s+", " ", o)
                    vocabulary.setdefault(o, len(vocabulary))

                    # if 'resource' in o:
                    #    subjects_to_indexes.setdefault(o, len(vocabulary) - 1)

                    binary_co_occurence_matrix.setdefault(vocabulary[s], []).append(vocabulary[p])
                    binary_co_occurence_matrix[vocabulary[s]].append(vocabulary[o])

                    binary_co_occurence_matrix.setdefault(vocabulary[p], []).append(vocabulary[s])
                    binary_co_occurence_matrix[vocabulary[p]].append(vocabulary[o])

                    binary_co_occurence_matrix.setdefault(vocabulary[o], []).append(vocabulary[s])
                    binary_co_occurence_matrix[vocabulary[o]].append(vocabulary[p])

        reader.close()
        """
        #        folder = f_name[:8]
        f_name = root + '/' + 'KB.txt.txt'
        total_rdf_tripples = len(kb)
        with open(f_name, 'w') as handle:
            for sentence in kb:
                handle.write(sentence)
        handle.close()
        """
        handle.close()

        print('Number of total RDF triples ', total_rdf_tripples)

        return binary_co_occurence_matrix, list(vocabulary.keys()), total_rdf_tripples, subjects_to_indexes

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
                    if '.nq' in file or '.nt' in file:
                        KGs.append(path + '/' + file)
        if len(KGs) == 0:
            print(path + ' is not a path for a file or a folder containing any .nq or .nt formatted files')
            exit(1)
        return KGs

    def is_literal_in_rdf(self, sentence):
        """
        Apply heuristic to find whether RDF contains Literals.

        Look at " symbol occurs after second occurrence of  >
        :param sentence:
        :return:
        """
        if '> "' in sentence:
            return True
        return False

    def decompose_rdf(self, sentence):

        # TODO include literals later
        # if self.is_literal_in_rdf(sentence):

        # return matching patter
        # return <what written here>
        # so "kinase activity"^^ <...> ignored
        components = re.findall('<(.+?)>', sentence)
        if len(components) == 2:
            s, p = components
            remaining_sentence = sentence[sentence.index(p) + len(p) + 2:]
            literal = remaining_sentence[:-1]
            o = literal
        elif len(components) == 4:
            del components[-1]
            s, p, o = components
        elif len(components) == 3:
            s, p, o = components
        elif len(components) > 4:

            s = components[0]
            p = components[1]
            remaining_sentence = sentence[sentence.index(p) + len(p) + 2:]
            literal = remaining_sentence[:remaining_sentence.index(' <http://')]
            o = literal
        else:

            ## This means that literal contained in RDF triple contains < > symbol
            """ pass"""
            # print('WRONG FORMAT RDF found and will be ignored')
            # print(sentence)
            raise ValueError()

        o = re.sub("\s+", " ", o)

        return s, p, o

    def create_dic_from_text(self, f_name: str, bound: int):
        """

        :param f_name: path of a KG or of a folder containg KGs
        :param bound: number of RDF triples for a KG.
        :return:
        """

        binary_co_occurence_matrix = {}
        vocabulary = {}
        subjects_to_indexes = {}
        predicates_to_indexes = {}

        num_of_rdf = 0

        p_knowledge_graphs = self.get_path_knowledge_graphs(f_name)
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

                total_sentence += 1
                try:
                    s, p, o = self.decompose_rdf(sentence)
                except ValueError:
                    continue

                # Replace each next line character with space.
                writer_kb.write(re.sub("\s+", " ", sentence) + '\n')

                # mapping from string to vocabulary
                vocabulary.setdefault(s, len(vocabulary))
                subjects_to_indexes[s] = vocabulary[s]

                vocabulary.setdefault(p, len(vocabulary))
                predicates_to_indexes[p] = vocabulary[p]

                vocabulary.setdefault(o, len(vocabulary))

                binary_co_occurence_matrix.setdefault(vocabulary[s], []).append(vocabulary[p])
                binary_co_occurence_matrix[vocabulary[s]].append(vocabulary[o])

                binary_co_occurence_matrix.setdefault(vocabulary[p], []).append(vocabulary[s])
                binary_co_occurence_matrix[vocabulary[p]].append(vocabulary[o])

                binary_co_occurence_matrix.setdefault(vocabulary[o], []).append(vocabulary[s])
                binary_co_occurence_matrix[vocabulary[o]].append(vocabulary[p])

            reader.close()
            num_of_rdf += total_sentence

        writer_kb.close()
        assert len(vocabulary) == len(binary_co_occurence_matrix)

        print('Size of vocabulary', len(vocabulary))
        print('Number of RDF triples', num_of_rdf)
        print('Number of subjects', len(subjects_to_indexes))
        print('Number of predicates', len(predicates_to_indexes))

        return binary_co_occurence_matrix, vocabulary, num_of_rdf, subjects_to_indexes, predicates_to_indexes

    def create_dic_from_text_all(self, f_name):

        background_knowledge = list()
        binary_co_occurence_matrix = {}

        KGs = list()
        vocabulary = dict()
        subjects_to_indexes = dict()

        if os.path.isfile(f_name):
            KGs.append(f_name)
        else:
            for root, dir, files in os.walk(f_name):
                for file in files:
                    if '.nq' in file or '.nt' in file:
                        KGs.append(f_name + '/' + file)

        try:
            assert len(KGs) > 0
        except (FileNotFoundError, IOError):
            print(f_name + '  does not contain any .nq or .nt file file containing')

        for f_name in KGs:
            # print('File name ',f_name)
            with open(f_name, "r") as reader:

                for sentence in reader:

                    components = re.findall('<(.+?)>', sentence)

                    if len(components) == 4:
                        del components[-1]
                        s, p, o = components
                    elif len(components) < 4:
                        s, p, o = components
                    else:
                        """ pass"""
                        print(sentence)
                        del sentence

                    processed_rdf = '<' + s + '> ' + '<' + p + '> ' + '<' + p + '> .'
                    background_knowledge.append(processed_rdf)

                    """
                    background_knowledge.append(sentence)

                    sentence = self.rreplace(sentence,
                                             '<http://bio2rdf.org/drugbank_resource:bio2rdf.dataset.drugbank.R3>', '',
                                             1)
                    sentence = self.rreplace(sentence,
                                             '<http://bio2rdf.org/drugbank_resource:bio2rdf.dataset.sider.R3>', '', 1)
                    sentence = self.rreplace(sentence, '<http://bio2rdf.org/sider_resource:bio2rdf.dataset.sider.R4>',
                                             '', 1)
                    sentence = self.rreplace(sentence,
                                             '<http://bio2rdf.org/wormbase_resource:bio2rdf.dataset.wormbase.R4>', '',
                                             1)

                    sentence = self.rreplace(sentence,
                                             '<http://bio2rdf.org/pubmed_resource:bio2rdf.dataset.pubmed.R3.statistic>',
                                             '',
                                             1)

                    s = sentence[:sentence.find('>') + 1]
                    sentence = sentence[len(s) + 1:]
                    p = sentence[:sentence.find('>') + 1]
                    o = sentence[len(p) + 1:]

                    s = self.modifier(s)
                    p = self.modifier(p)
                    o = self.modifier(o)
                    
                    """

                    # mapping from string to vocabulary
                    vocabulary.setdefault(s, len(vocabulary))
                    subjects_to_indexes[s] = vocabulary[s]

                    vocabulary.setdefault(p, len(vocabulary))
                    vocabulary.setdefault(o, len(vocabulary))
                    # subjects_to_indexes.setdefault(o, len(vocabulary) - 1)

                    binary_co_occurence_matrix.setdefault(vocabulary[s], []).append(vocabulary[p])
                    binary_co_occurence_matrix[vocabulary[s]].append(vocabulary[o])

                    binary_co_occurence_matrix.setdefault(vocabulary[p], []).append(vocabulary[s])
                    binary_co_occurence_matrix[vocabulary[p]].append(vocabulary[o])

                    binary_co_occurence_matrix.setdefault(vocabulary[o], []).append(vocabulary[s])
                    binary_co_occurence_matrix[vocabulary[o]].append(vocabulary[p])

            reader.close()

        f_name = 'KG' + '/' + 'KB.txt'

        with open(f_name, 'w') as handle:
            for sentence in background_knowledge:
                handle.write(sentence)
        handle.close()

        num_of_rdf = len(background_knowledge)
        del background_knowledge

        assert len(binary_co_occurence_matrix) > 0

        return binary_co_occurence_matrix, vocabulary, num_of_rdf, subjects_to_indexes

    @performance_debugger('KG to PPMI Matrix')
    def construct_comatrix(self, f_name, bound=10, bound_flag=False):

        if bound_flag:
            freq_matrix, vocab, num_triples, subjects_to_indexes, predicates_to_indexes = self.create_dic_from_text(
                f_name, bound)
        else:
            freq_matrix, vocab, num_triples, only_resources = self.create_dic_from_text_all(f_name)

        similarities = self.similarity_function(freq_matrix, num_triples)


        ut.serializer(object_=dict(zip(list(vocab.values()), list(vocab.keys()))), path=self.p_folder,
                      serialized_name='i_vocab')

        ut.serializer(object_=vocab, path=self.p_folder, serialized_name='vocab')
        del vocab

        index_of_resources = np.array(list(subjects_to_indexes.values()), dtype=np.uint32)
        ut.serializer(object_=index_of_resources, path=self.p_folder, serialized_name='index_of_resources')
        del index_of_resources

        index_of_predicates = np.array(list(predicates_to_indexes.values()), dtype=np.uint32)
        ut.serializer(object_=index_of_predicates, path=self.p_folder, serialized_name='index_of_predicates')
        del predicates_to_indexes
        del index_of_predicates

        ut.serializer(object_=subjects_to_indexes, path=self.p_folder, serialized_name='subjects_to_indexes')
        del subjects_to_indexes

        ut.serializer(object_=num_triples, path=self.p_folder, serialized_name='num_triples')
        del num_triples

        print('Exitting')
        exit(1)
        return similarities

    @performance_debugger('KG to PPMI Matrix')
    def process_knowledge_graph_to_construct_PPMI_co_matrix(self, f_name, bound):

        binary_co_occurrences, vocab, num_triples, only_resources = self.create_dictionary(f_name, bound)

        ppmi_stats = self.binary_to_ppmi_matrix(binary_co_occurrences, num_triples)

        return ppmi_stats, vocab, num_triples, only_resources

    def choose_to_K_attractive_entities(self, ppmi_co_occurence_matrix, size_of_iteracting_entities):
        # PPMIs are sorted and disregarded some entities
        for k, v in ppmi_co_occurence_matrix.items():
            if len(v) > size_of_iteracting_entities:
                ppmi_co_occurence_matrix[k] = dict(
                    sorted(v.items(), key=lambda kv: kv[1], reverse=True)[0:size_of_iteracting_entities])
            else:
                ppmi_co_occurence_matrix[k] = dict(sorted(v.items(), key=lambda kv: kv[1], reverse=True))
        return ppmi_co_occurence_matrix

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

    @performance_debugger('Assigning attractive and repulsive particles')
    def get_attractive_repulsive_entities(self, stats_corpus_info, K):
        pruned_stats_corpus_info = self.choose_to_K_attractive_entities(stats_corpus_info, K)
        context_entitiy_pms = list(pruned_stats_corpus_info.values())
        del pruned_stats_corpus_info

        repulsitve_entities = self.random_sample_repulsive_entities(stats_corpus_info, K)

        return context_entitiy_pms, repulsitve_entities


class PL2VEC(object):
    def __init__(self, system_energy=1):

        self.epsilon = 0.001
        self.texec = ThreadPoolExecutor(8)

        self.total_distance_from_attractives = list()
        self.total_distance_from_repulsives = list()
        self.ratio = list()
        self.system_energy = system_energy

    @staticmethod
    def initialize_with_svd(stats_corpus_info: Dict, embeddings_dim):
        """
        Ini
        :param stats_corpus_info:
        :param embeddings_dim:
        :return:
        """
        row = list()
        col = list()
        data = list()

        num_of_unqiue_entities = len(stats_corpus_info)
        for target_entitiy, co_info in stats_corpus_info.items():
            row.extend([int(target_entitiy)] * len(co_info))
            col.extend(list(co_info.keys()))
            data.extend(list(co_info.values()))

        sparse_ppmi = sparse.csc_matrix((data, (row, col)), shape=(num_of_unqiue_entities, num_of_unqiue_entities))

        svd = TruncatedSVD(n_components=embeddings_dim, n_iter=500)  # , random_state=self.random_state)

        return StandardScaler().fit_transform(svd.fit_transform(sparse_ppmi))

    @staticmethod
    def randomly_initialize_embedding_space(num_vocab, embeddings_dim):
        return np.random.rand(num_vocab, embeddings_dim).astype(np.float64) + 1

    @staticmethod
    def get_qualifiy_repulsive_entitiy(distances, give_threshold):
        absolute_distance = np.abs(distances)
        mask = absolute_distance < give_threshold
        index_of_qualifiy_entitiy = np.all(mask, axis=1)
        return distances[index_of_qualifiy_entitiy]

    @performance_debugger('Compressing Information of Attractives and Repulsives')
    def combine_information(self, context_entitiy_pms, repulsitve_entities):
        assert len(context_entitiy_pms) == len(repulsitve_entities)

        holder = list()

        for index, item in enumerate(context_entitiy_pms):
            context = np.array(list(item.keys()), dtype=np.int32)

            pms = np.around(list(item.values()), 3).astype(np.float32)
            pms.shape = (pms.size, 1)

            repulsive = np.array(list(repulsitve_entities[index]), dtype=np.int32)

            holder.append((context, pms, repulsive))
        del repulsitve_entities
        # del ppmi_co_occurence_matrix
        del context_entitiy_pms

        return holder

    @staticmethod
    def apply_hooke_s_law(embedding_space, target_index, context_indexes, PMS):

        dist = embedding_space[context_indexes] - embedding_space[target_index]
        pull = dist * PMS
        return np.sum(pull, axis=0), np.linalg.norm(dist)

    @staticmethod
    def apply_coulomb_s_law(embedding_space, target_index, repulsive_indexes, negative_constant):
        # calculate distance from target to repulsive entities
        dist = np.abs(embedding_space[repulsive_indexes] - embedding_space[target_index])
        #        dist = np.ma.array(dist, mask=np.isnan(dist))  # Use a mask to mark the NaNs
        #        dist=(dist ** 2).filled(np.maxi)

        with warnings.catch_warnings():
            try:
                r_square = (dist ** 2) + 1
            except RuntimeWarning:
                print(dist)
                print('Overflow')
                #                r_square = dist + 0.5
                # print(r_square)
                exit(1)

        with warnings.catch_warnings():
            try:
                total_push = np.sum((negative_constant / (r_square)), axis=0)
            except RuntimeWarning:
                print(r_square)
                exit(1)

        return total_push, np.linalg.norm(dist)

    def go_through_entities(self, e, holder, negative_constant, system_energy):

        agg_att_d = 0
        agg_rep_d = 0
        # futures = []
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

            total_effect = (pull + push) * system_energy

            e[target_index] = e[target_index] + total_effect

            agg_att_d += abs_att_dost
            agg_rep_d += abs_rep_dist

        return e, agg_att_d / agg_rep_d

    @performance_debugger('Generating Embeddings:')
    def start(self, *, e, max_iteration, energy_release_at_epoch, holder, negative_constant):
        scaler = MinMaxScaler()

        for epoch in range(max_iteration):
            print('EPOCH: ', epoch)

            previous_f_norm = LA.norm(e, 'fro')

            e, d_ratio = self.go_through_entities(e, holder, negative_constant, self.system_energy)

            self.system_energy = self.system_energy - energy_release_at_epoch
            self.ratio.append(d_ratio)

            # e[np.isnan(np.inf)] = e.max()

            # Z-score
            e = (e - e.mean()) / e.std()

            #            e = scaler.fit_transform(e)

            new_f_norm = LA.norm(e, 'fro')

            if self.equilibrium(epoch, previous_f_norm, new_f_norm, d_ratio):
                break

        return e

    def equilibrium(self, epoch, p_n, n_n, d_ratio):
        val = np.abs(p_n - n_n)
        # or d_ratio < 0.1
        if val < self.epsilon:  # or np.isnan(val) or system_energy < 0.001:
            print("\n Epoch: ", epoch)
            print('Previous norm', p_n)
            print('New norm', n_n)
            print('The differences in matrix norm ', val)
            print('d(Semantically Similar)/d(Not Semantically Similar) ', d_ratio)
            Saver.settings.append('Epoch: ' + str(epoch))
            Saver.settings.append('The differences in matrix norm: ' + str(val))
            Saver.settings.append('Ratio of total Attractive / repulsives: ' + str(d_ratio))
            print('The state of equilibrium is reached.')
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
        self.kg_path = kg_path

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
                    configs = [c for c in nfiles if 'conf' in c]
                    dl_req.append((nroot, configs))
        print(dl_req)
        return dl_req

    def write_config(self, t):

        path, l = t
        print(path)
        print(l)

        assert os.path.isfile(self.execute_DL_Learner)

        output_of_dl = list()
        for confs in l:
            n_path = path + '/' + confs
            output_of_dl.append('\n\n')
            output_of_dl.append('### ' + confs + ' starts ###')

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

        folder_names = list()
        for root, dir, files in os.walk(folder_path):
            for i in dir:
                new_path = root + '/' + i
                for nroot, _, nfiles in os.walk(new_path):
                    folder_names.append(i)

                    individual_path = nroot + '/Settings.txt'

                    with open(individual_path, 'r') as reader:

                        sentences = reader.readlines()

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

                                distributon_of_cluster.append(''.join(cluster_infos))
                                continue
                    reader.close()

                    assert len(energe_relases) == len(embed_space)

        df = pd.DataFrame(
            {
                'names': folder_names,
                'size of vocab': np.array(size_of_vocab, dtype=np.uint32),
                'num of rdfs': np.array(num_of_rdfs, dtype=np.uint32),
                'ER': np.array(energe_relases),  # , dtype=np.float32),
                'negative_constant': np.array(NC, dtype=np.float32),
                'dim': np.array(embed_space, dtype=np.float32),
                'ratios': np.array(ratios),  # dtype=np.float32
                'time': np.array(time_emb, dtype=np.float32),
                'num_of_clusters': np.array(num_of_clusters)})

        df.to_csv(folder_path + '/generated_data')
        return folder_path + '/generated_data'

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
                        """
                        means.append(np.mean(np.array(f_scores)))
                        std.append(np.std(np.array(f_scores)))
                        medians.append(np.median(np.array(f_scores)))
                        mins.append(np.min(np.array(f_scores)))
                        maxs.append(np.max(np.array(f_scores)))
                        """

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

            Saver.similarities.append('\n Target: ' + sampled_entitiy_name)
            # print('\n Target:', sampled_entitiy_name)

            distances = list()

            for index, ith_vec_of_embeddings in enumerate(learned_embeddings):
                distances.append(spatial.distance.cosine(sampled_vector, ith_vec_of_embeddings))

            distances = np.array(distances)

            most_similar = distances.argsort()[1:topN]

            Saver.similarities.append('\n Top similars\n')
            similars = list()
            for index, j in enumerate(most_similar):
                similars.append(vocabulary_of_entity_names[j])
                Saver.similarities.append(str(index) + '. similar: ' + vocabulary_of_entity_names[j])

            d[sampled_entitiy_name] = similars
            list_of_clusters.append(d)

        return list_of_clusters

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
    def sample_from_clusters(self, labeled_embeddings, num_sample_from_clusters):

        labels = labeled_embeddings.labels.unique().tolist()

        # Saver.settings.append('Number of clusters created:' + str(len(labels)))
        # Saver.settings.append('Number of entities will be sampled from clusters:' + str(num_sample))
        # str_predicates = list(ut.deserializer(path=self.p_folder, serialized_name='predicates_to_indexes').keys())
        sampled_total_entities = pd.DataFrame()

        def calculate_difference_to_mean(data_point):
            euclidiean_dist = distance.euclidean(data_point, mean_of_cluster)
            return euclidiean_dist

        for label in labels:
            cluster = labeled_embeddings.loc[labeled_embeddings.labels == label].copy()
            mean_of_cluster = cluster.mean()

            cluster['euclidean_distance_to_mean'] = cluster.apply(calculate_difference_to_mean, axis=1)
            # cluster.loc[:, 'description']
            cluster.sort_values(by=['euclidean_distance_to_mean'], ascending=False)

            sampled_entities_from_cluster = cluster.head(num_sample_from_clusters)
            sampled_total_entities = pd.concat([sampled_total_entities, sampled_entities_from_cluster])

        sampled_total_entities.drop(['euclidean_distance_to_mean'], axis=1, inplace=True)

        return sampled_total_entities

    @performance_debugger('Pseudo labeling via DBSCAN')
    def pseudo_label_DBSCAN(self, df):
        df['labels'] = DBSCAN().fit(df).labels_
        return df

    @performance_debugger('Prune non resources')
    def prune_non_subject_entities(self, embeddings, upper_folder='not init'):

        if upper_folder != 'not init':
            index_of_resources = ut.deserializer(path=upper_folder, serialized_name="index_of_resources")
        else:
            index_of_resources = ut.deserializer(path=self.p_folder, serialized_name="index_of_resources")

        index_of_predicates = ut.deserializer(path=self.p_folder, serialized_name="index_of_predicates")

        # Prune those subject that occurred in predicate position
        index_of_only_subjects = np.setdiff1d(index_of_resources, index_of_predicates)

        i_vocab = ut.deserializer(path=self.p_folder, serialized_name="i_vocab")

        names = np.array(list(i_vocab.values()))[index_of_only_subjects]

        # PRUNE non subject entities
        embeddings = embeddings[index_of_only_subjects]

        embeddings = pd.DataFrame(embeddings, index=names)

        return embeddings

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
        embeddings = self.sample_from_clusters(embeddings, num_of_sample)
        embeddings.drop(columns=['labels'], inplace=True)
        return embeddings

    @performance_debugger('Pipeline of DP')
    def pipeline_of_data_processing(self, path, E, num_sample_from_clusters):

        # prune predicates and literalse
        embeddings_of_resources = self.prune_non_subject_entities(embeddings=E, upper_folder=path)

        # Partition total embeddings into number of 10
        # E = self.apply_partitioning(embeddings_of_resources, partitions=20, samp_part=10)

        pseudo_labelled_embeddings = self.pseudo_label_DBSCAN(embeddings_of_resources)

        print(pseudo_labelled_embeddings)

        exit(1)
        ut.serializer(object_=self.p_folder, path=path,
                      serialized_name='pseudo_labelled_resources')

        Saver.settings.append("### cluster distribution##")
        Saver.settings.append(pseudo_labelled_embeddings.labels.value_counts().to_string())
        Saver.settings.append("#####")

        sampled_embeddings = self.sample_from_clusters(pseudo_labelled_embeddings, num_sample_from_clusters)

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
        # E = self.apply_partitioning(embeddings_of_resources, partitions=20, samp_part=10)

        pseudo_labelled_embeddings = self.pseudo_label_DBSCAN(embeddings_of_resources)

        ut.serializer(object_=pseudo_labelled_embeddings, path=self.p_folder,
                      serialized_name='pseudo_labelled_resources')

        Saver.settings.append("### cluster distribution##")
        Saver.settings.append(pseudo_labelled_embeddings.labels.value_counts().to_string())
        Saver.settings.append("#####")

        sampled_embeddings = self.sample_from_clusters(pseudo_labelled_embeddings, num_sample_from_clusters)

        representative_entities = dict()

        for key, val in sampled_embeddings.labels.to_dict().items():
            representative_entities.setdefault(val, []).append(key)

        Saver.settings.append(
            'Num of generated clusters: ' + str(len(list(representative_entities.keys()))))

        return representative_entities

    def apply_tsne_and_plot_only_subjects(self, e):
        import dash
        import dash_core_components as dcc
        import dash_html_components as html
        import plotly.graph_objs as go

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

    """
    def modifier(self, item):
        item = item.replace('>', '')
        item = item.replace('<', '')
        item = item.replace(' .', '')
        item = item.replace('\n', '')
        return item

    def converter(self, item):
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        return item
    """

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

        all_samples = np.array(list(sampled_representatives.values()))

        if len(sampled_representatives) == 1:
            print('Only one cluster found in the embedding space')
            print('Exitting.')
            exit(1)

        for cluster_label, val in sampled_representatives.items():
            positive_subjects = list(val)

            candidates_of_negative_subjects = np.setdiff1d(all_samples, list(positive_subjects))

            sampled_negatives = np.random.choice(candidates_of_negative_subjects, len(positive_subjects), replace=False)

            self.create_config(self.p_folder + '/' + str(cluster_label), positive_subjects, sampled_negatives)

    @performance_debugger('DL-Learner')
    def pipeline_of_dl_learner(self, path, dict_of_cluster_with_original_term_names):

        with open(path + "/subjects_to_indexes.p", "rb") as f:
            str_subjects = list(pickle.load(f).keys())

        f.close()

        for key, val in dict_of_cluster_with_original_term_names.items():
            dict_of_cluster_with_original_term_names[key] = [str_subjects[item] for item in val]

        self.execute_DL(resources=str_subjects, dict_of_cluster=dict_of_cluster_with_original_term_names)

    @performance_debugger('DL-Learner')
    def pipeline_of_single_evaluation_dl_learner(self, dict_of_cluster_with_original_term_names):

        # vocab.p is mapping from string to pos
        # str_subjects = list(pickle.load(open(path + "/subjects_to_indexes.p", "rb")).keys())

        str_subjects = list(ut.deserializer(path=self.p_folder, serialized_name='subjects_to_indexes').keys())

        # for key, val in dict_of_cluster_with_original_term_names.items():
        #    dict_of_cluster_with_original_term_names[key] = [str_subjects[item] for item in val]

        self.kg_path = self.p_folder

        self.execute_dl(dict_of_cluster_with_original_term_names)
        # self.execute_DL(resources=str_subjects, dict_of_cluster=dict_of_cluster_with_original_term_names)


class Saver:
    settings = []
    similarities = []
