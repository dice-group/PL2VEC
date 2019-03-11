import os
import pickle
import re
import math
from bz2 import BZ2File
from collections import Counter, deque
import itertools
from scipy import spatial
from sklearn.cluster import DBSCAN, MiniBatchKMeans
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
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

from typing import Tuple
import typing
from scipy import sparse
import sys

import warnings
import sortednp as snp
from scipy.sparse.csgraph import connected_components

#warnings.filterwarnings('error')


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
    def __init__(self, logger=False, p_folder: str = 'not initialized', K='not init'):
        self.path = 'uninitialized'
        self.logger = logger
        self.p_folder = p_folder
        self.similarity_function = None
        self.K = int(K)

    def set_similarity_function(self, f):
        self.similarity_function = f
        Saver.settings.append('similarity_function:' + repr(f))

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

    @performance_debugger('Calculating entropies')
    def calculate_entropies(self, freq_adj_matrix: Dict, number_of_rdfs: int) -> typing.Tuple[np.array,np.array]:
        """
         Calculate shannon entropy for each vocabulary term.

         ###### Parse RDF triples to co occurrence matrix  starts ######
        :param freq_adj_matrix:
        :param number_of_rdfs:
        :return:
        """

        co_occurrings = deque()
        entropies = deque()

        for unq_ent, contexts_w_co_freq in freq_adj_matrix.items():


            marginal_prob = sum(contexts_w_co_freq.values()) / number_of_rdfs


            co_occurrings.append(np.array(list(contexts_w_co_freq.keys()),dtype=np.uint32))

            with np.errstate(divide='raise'):
                try:
                    entropy = -marginal_prob * np.log2(marginal_prob)
                    entropies.append(entropy)
                except FloatingPointError:
                    print('entropy of term-', unq_ent, ': is set 0')
                    print('P( term-', unq_ent, '- is', marginal_prob)
                    entropies.append(0)

        entropies = np.array(entropies, dtype=np.float32)

        return entropies, np.array(co_occurrings)

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

    def apply_ppmi_on_co_matrix(self, binary_co_matrix: Dict, num_triples: int) -> Dict:
        marginal_probabilities = self.calculate_marginal_probabilities(binary_co_matrix, num_triples)
        top_similarities, negatives = self.calculate_ppmi(binary_co_matrix, marginal_probabilities, num_triples)
        return top_similarities, negatives

    @performance_debugger('Jaccards of subjects')
    def calculatted_e_j_s(self, co_occurrences, entropies, index_of_only_subjects, index_of_only_predicates):
        top_k_sim = dict()
        negatives = dict()

        most_info = 10

        for subject_a in index_of_only_subjects:

            top_k_sim.setdefault(subject_a, dict())
            negatives.setdefault(subject_a, list())

            # context of subject contains any co-occurence of a subject predicate or an object
            context_of_subject_a = co_occurrences[subject_a]

            # domain of subject consists of predicates and objects

            domain_of_subject_a = np.setdiff1d(context_of_subject_a, index_of_only_subjects, assume_unique=True)

            #            domain_of_subject_a = snp.intersect(context_of_subject_a, index_of_only_predicates)

            contextes_of_domain_of_subject_a = co_occurrences[domain_of_subject_a]

            select_domain_wi = np.array([entropies[i].sum() for i in contextes_of_domain_of_subject_a])

            most_informative_domain_indexes = select_domain_wi.argsort()[-most_info:][::-1]

            most_informative__pre_ob_of_subject_a = domain_of_subject_a[most_informative_domain_indexes]

            sum_of_ent_context_of_a = entropies[subject_a].sum()

            for subject_b in most_informative__pre_ob_of_subject_a:

                intersection = snp.intersect(context_of_subject_a, co_occurrences[subject_b])

                sum_of_ent_context_of_b = entropies[subject_b].sum()

                sum_of_ent_intersections = entropies[intersection].sum()

                sim = sum_of_ent_intersections / (sum_of_ent_context_of_a + sum_of_ent_context_of_b)

                if sim > 0:
                    top_k_sim[subject_a][subject_b] = round(sim, 4)

        return top_k_sim

        """
        old_subject_index_to_new = dict(zip(indexes_of_subjects, np.arange(len(indexes_of_subjects))))

        for a, context_of_subject_a in enumerate(domain_of_subjects):
            top_k_sim.setdefault(a, dict())
            negatives.setdefault(a, list())

            sum_of_ent_context_of_a = entropies[context_of_subject_a].sum()

            for component in context_of_subject_a:

                context_of_b = co_occurrences[component]

                intersection = snp.intersect(context_of_subject_a, context_of_b)

                if len(intersection) < 2:
                    continue

                sum_of_ent_intersections = entropies[intersection].sum()

                sum_of_ent_context_of_b = entropies[context_of_b].sum()

                sim = sum_of_ent_intersections / (sum_of_ent_context_of_a + sum_of_ent_context_of_b)

                if sim > 0 and (component in indexes_of_subjects):
                    component = old_subject_index_to_new[component]
                    if len(top_k_sim[a]) <= self.K:
                        top_k_sim[a][component] = round(sim, 4)
                    else:
                        for k, v in top_k_sim[a].items():

                            if v < sim:
                                top_k_sim[a][component] = sim
                                del top_k_sim[a][k]
                                break
                else:
                    if len(negatives[a]) <= self.K:
                        negatives[a].append(component)

            assert len(top_k_sim) == len(negatives)
        return top_k_sim, negatives
    
        """

    def apply_entropy_jaccard_subjects(self, freq_matrix: Dict, index_of_only_subjects, indexes_of_predicates,
                                       num_triples: int):
        """

        :param index_of_only_subjects:
        :param freq_matrix:
        :param subjects_to_indexes:
        :param num_triples:
        :return:
        """

        holder = list()

        entropies, co_occurrences = self.calculate_entropies(freq_matrix, num_triples)

        top_k_sim = self.calculatted_e_j_s(co_occurrences, entropies, index_of_only_subjects, indexes_of_predicates)

        new_index = dict(zip(list(top_k_sim.keys()), list(range(len(top_k_sim)))))

        for i, index_of_sub in enumerate(index_of_only_subjects):
            d = top_k_sim[index_of_sub]

            # remove the most similar term as it is dupplicatep
            attractives = np.array(list(d.keys()))

            attractives = [new_index[i] for i in attractives]

            sim = np.array(list(d.values()), dtype=np.float32).reshape(len(attractives), 1)

            negatives = np.random.choice(len(index_of_only_subjects), len(attractives) * 2)

            T = (attractives, sim, np.setdiff1d(negatives, attractives))
            holder.append(T)

        return holder

    def apply_entropy_jaccard_on_entitiy_adj_matrix(self, freq_adj_matrix: Dict, num_triples: int):
        """
        Calculate Shannon entropy weighted jaccard from freq_matrix
                A=  Sum of entropies of overlapping elements
                B=  Sum of entropies of all elements that occurred with x
                C=  Sum of entropies of all elements that occurred with y

       Sim(x,y) = A / (B+C)


        :param subjects_to_indexes:
        :param freq_matrix: freq_matrix is mapping from a vertex or an edge to co-occurring vertices or edges.
        :param num_triples:
        :return: Mapping from points to a mapping containing a point and positive jaccard sim.
        """

        entropies, co_occurrences = self.calculate_entropies(freq_adj_matrix, num_triples)

        holder = self.calculate_entropy_jaccard(entropies, co_occurrences)

        return holder

    @performance_debugger('Calculating entropy weighted Jaccard index')
    def efficient_calculate_entropy_jaccard(self, entropies, contexts):
        """

        :param contexts: ith item  corresponds a numpy array whose each element corresponds with ith vocabulary term.
        :param entropies:  ith item in entropies indicate the entropy of ith vocabulary term.
        :return:

        """
        top_k_sim = dict()
        negatives = dict()
        for a, context_of_a in enumerate(contexts):
            top_k_sim.setdefault(a, dict())
            negatives.setdefault(a, deque())

            sum_of_ent_context_of_a = sum([entropies[_] for _ in context_of_a])

            for b, context_of_b in enumerate(contexts):

                intersection = context_of_a.intersection(context_of_b)

                sum_of_ent_intersections = sum([entropies[_] for _ in intersection])

                sum_of_ent_context_of_b = sum([entropies[_] for _ in context_of_b])

                sim = sum_of_ent_intersections / (sum_of_ent_context_of_a + sum_of_ent_context_of_b)

                if sim > 0:

                    if len(top_k_sim[a]) <= self.K:
                        top_k_sim[a][b] = round(sim, 4)
                    else:
                        for k, v in top_k_sim[a].items():

                            if v < sim:
                                top_k_sim[a][b] = sim
                                del top_k_sim[a][k]
                                break
                else:
                    if len(negatives[a]) <= self.K:
                        negatives[a].append(b)

        return top_k_sim, negatives

        """
        texec = ThreadPoolExecutor(4)
        holder = list()  # .append((context, sim, repulsive))
        for a, context_of_a in enumerate(contexts):

            similarities = list()

            futures = []
            for item in all_context_of_component_b:
                futures.append(
                    texec.submit(ut.calculate_similarities, context_of_component_a, item, num_array_entropies))

            similarities = [ff.result() for ff in futures]

            print(similarities)
            exit(1)
            index_of_top_K = (-similarity_evals_a).argsort()[:self.K]
            top_K_similarity_val = np.around(similarity_evals_a[index_of_top_K], 4)
            top_K_similarity_val.shape = (top_K_similarity_val.size, 1)

            possible_negatives = np.argwhere(np.isnan(similarity_evals_a)).flatten()

            if len(possible_negatives) == 0:

                repulsives = np.array([np.argmin(similarity_evals_a)])

            #            N.append(np.argwhere(np.min(similarity_evals_a))[0])
            elif len(possible_negatives) <= self.K:
                repulsives = possible_negatives
            else:
                repulsives = np.random.choice(possible_negatives, self.K, replace=False)

            holder.append((index_of_top_K, top_K_similarity_val, repulsives))
        return holder
        """
        """
        for context_of_a in contexts:
            p_sim_evals_a = deque()

            for context_of_b in contexts:
                # 2 times more time efficient then setdiff1d of numpy
                # 1.5 times more time efficient then intersect1d of numpy
                #intersection = snp.intersect(context_of_a, context_of_b)

                intersection = context_of_a.intersection(context_of_b)

                print(intersection)
                exit(1)
                #sum_of_ent_context_of_a = entropies[context_of_a].sum()

                #sum_of_ent_context_of_b = entropies[context_of_b].sum()

                #sum_of_ent_intersections = entropies[intersection].sum()

                #sim = sum_of_ent_intersections / (sum_of_ent_context_of_a + sum_of_ent_context_of_b)

                #p_sim_evals_a.append(sim)

            #jaccard.append(p_sim_evals_a)

        """

    def calculate_entropy_jaccard(self, entropies:np.array, domain:np.array):

        print(entropies)
        print(domain)
#        exit(1)
        holder=list()

        for i, domain_i in enumerate(domain):
            top_k_sim = dict()
            negatives = dict()
            top_k_sim.setdefault(i, dict())
            negatives.setdefault(i, list())

            sum_ent_domain_i = entropies[domain_i].sum()

            for j in domain_i:

                domain_j = domain[j]




                intersection = snp.intersect(domain_i, domain_j)

                if len(intersection) > 1:
                    if len(top_k_sim[i]) <= self.K:

                        sum_ent_domain_j = entropies[domain_j].sum()
                        sim = entropies[intersection].sum() / (sum_ent_domain_i + sum_ent_domain_j)
                        top_k_sim[i][j] = sim
                    else:
                        for k, v in top_k_sim[i].items():
                            sim = entropies[intersection].sum() / (sum_ent_domain_i + sum_ent_domain_j)

                            if v < sim:
                                top_k_sim[i][j] = np.around(sim, 6)
                                del top_k_sim[i][k]
                                break
                else:
                    if len(negatives[i]) <= self.K:
                        negatives[i].append(j)

            context=np.array(list(top_k_sim[i].keys()),dtype=np.uint32)
            sim=np.array(list(top_k_sim[i].values()),dtype=np.uint32)
            sim.shape = (sim.size, 1)

            repulsives=np.array(negatives[i],dtype=np.uint32)

            del top_k_sim
            del negatives
            holder.append((context,sim,repulsives))

        return holder

    # TODO apply both seach and spectral custerin on tf-idf and and entropy co-occurrences.
    # TODO thereafter for each vocabulary term find most similar K terms
    def apply_similarity_on_laplacian(self, freq_adj_matrix: typing.Dict[int, Counter], num_triples: int) -> typing.Tuple[dict,dict]:
        """
        Partition the graph into two pieces such the resultsing pieces have low conductance

        Spectral Graph Partitioning

        Graph Laplacian
        :param freq_adj_matrix:
        :param num_triples:
        :return:
        """


        row = list()
        col = list()
        data = list()

        num_of_unqiue_entities = len(freq_adj_matrix)
        for i, co_info in freq_adj_matrix.items():
            vals = np.array(list(co_info.values())).sum()
            row.extend(np.array([i]))
            col.extend(np.array([i]))
            data.extend(np.array([vals]))


            j=list(co_info.keys())
            vals=np.array(list(co_info.values()))

            row.extend(np.array([i] * len(j)))

            col.extend(j)

            data.extend(-1*vals)

        laplacian_matrix = sparse.csc_matrix((data, (row, col)), shape=(num_of_unqiue_entities, num_of_unqiue_entities))

        print(laplacian_matrix.toarray())


        exit(1)

    def apply_entropy_jaccard_with_connected_component_analysis(self,freq_adj_matrix: typing.Dict[int, Counter], num_triples: int):
        """
        Construc entropy adjeseny matrix Thereafter apply analysis of connected components of a sparse graph.

        Calculate jaccard similarty measure on connected components
        :param freq_adj_matrix:
        :param num_triples:
        :return:
        """


        entropies, co_occurrences = self.calculate_entropies(freq_adj_matrix, num_triples)

        row = list()
        col = list()
        data = list()

        num_of_unqiue_entities = len(freq_adj_matrix)
        for i, co_info in freq_adj_matrix.items():

            j=list(co_info.keys())
            vals=np.array(list(co_info.values()))

            row.extend(np.array([i] * len(j)))

            col.extend(j)

            data.extend(entropies[vals])

        entropy_adj_matirx = sparse.csc_matrix((data, (row, col)), shape=(num_of_unqiue_entities, num_of_unqiue_entities))

        n_components, labels = connected_components(csgraph=entropy_adj_matirx, directed=False, return_labels=True)

        print('Found components')
        for i_component in range(n_components):
            itemindex = np.where(labels == i_component)[0]
            print(len(itemindex))


        exit(1)
        pass

    def apply_entropy_jaccard_with_networkx(self, freq_adj_matrix: typing.Dict[int, Counter], num_triples: int):
        """
        Construc entropy adjeseny matrix Thereafter apply analysis of connected components of a sparse graph.

        Calculate jaccard similarty measure on connected components
        :param freq_adj_matrix:
        :param num_triples:
        :return:
        """
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

            j=list(co_info.keys())
            vals=np.array(list(co_info.values()))

            row.extend(np.array([i] * len(j)))

            col.extend(j)

            data.extend(entropies[vals])

        entropy_adj_matirx = sparse.csc_matrix((data, (row, col)), shape=(num_of_unqiue_entities, num_of_unqiue_entities))


        G = nx.from_scipy_sparse_matrix(entropy_adj_matirx)
        #del entropy_adj_matirx
 #       k_components = approximation.min_maximal_matching(G)
#        print(k_components)
        #print(list(nx.dfs_predecessors(G, source=3)))
        #https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.approximation.connectivity.local_node_connectivity.html#networkx.algorithms.approximation.connectivity.local_node_connectivity


        l=np.array(MiniBatchKMeans(n_clusters=10).fit(entropy_adj_matirx).labels_)
        print(l)
        exit(1)
        pass

    def construct_entitiy_adj_matrix_from_list(self,kb:typing.List[Tuple[str,str,str]]):

        num_of_rdf=len(kb)
        binary_adj_matrix = {}
        vocabulary = {}
        entities_to_indexes = {}

        for s, p, o in kb:
            # mapping from string to vocabulary
            vocabulary.setdefault(s, len(vocabulary))
            entities_to_indexes[s] = vocabulary[s]


            vocabulary.setdefault(o, len(vocabulary))
            entities_to_indexes[o] = vocabulary[o]

            binary_adj_matrix.setdefault(vocabulary[s], Counter())[vocabulary[o]] += 1
            binary_adj_matrix.setdefault(vocabulary[o], Counter())[vocabulary[s]] += 1


        holder = self.similarity_function(binary_adj_matrix, num_of_rdf)


        return holder


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
                    print(file)
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

            # if '"' in sentence:
            #    remaining = sentence[len(s) + len(p) + 5:]
            #    literal = remaining[1:remaining.index(' <http://')]
            #    o = literal

        elif len(components) > 4:

            s = components[0]
            p = components[1]
            remaining_sentence = sentence[sentence.index(p) + len(p) + 2:]
            literal = remaining_sentence[:remaining_sentence.index(' <http://')]
            o = literal

        else:

            ## This means that literal contained in RDF triple contains < > symbol
            """ pass"""
            flag = 0
            # print(sentence)
            raise ValueError()

        # o = re.sub("\s+", "", o)

        # s = re.sub("\s+", "", s)

        # p = re.sub("\s+", "", p)

        return s, p, o, flag

    @performance_debugger('Parse RDF triples to co occurrence matrix')
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
                #                    sentence = sentence.decode('ascii')

                if total_sentence == bound: break

                #             if ('rdf-schema#' in sentence) or ('description' in sentence):
                #                  pass
                #               else:
                #                    continue

                #                if '"' in sentence:
                #                   continue

                try:
                    s, p, o, flag = self.decompose_rdf(sentence)

                    if not ((flag == 4) or (flag == 3) or (flag == 2)):
                        print(sentence)
                    #    print('exitting')
                    #   continue

                except ValueError:
                    continue

                total_sentence += 1

                #                processed_triples = '<' + s + '> ' + '<' + p + '> ' + '<' + o + '> .\n'
                #               writer_kb.write(processed_triples)
                writer_kb.write(sentence)
                # Replace each next line character with space.
                # writer_kb.write(re.sub("<http://bio2rdf.org/drugbank_resource:bio2rdf.dataset.drugbank.R3> .", ".", sentence))

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
    def construct_comatrix(self, f_name, bound=''):

        if isinstance(bound, int):
            freq_matrix, vocab, num_triples, subjects_to_indexes, predicates_to_indexes = self.create_dic_from_text(
                f_name, bound)
        else:
            freq_matrix, vocab, num_triples, only_resources = self.create_dic_from_text_all(f_name)

        print(vocab)
        print(sys.getsizeof(freq_matrix) / 1000000)
        exit(1)
        string_vocab = np.array(list(vocab.keys()))
        indexes_of_subjects = np.array(list(subjects_to_indexes.values()), dtype=np.uint32)
        indexes_of_predicates = np.array(list(predicates_to_indexes.values()), dtype=np.uint32)

        # Prune those subject that occurred in predicate position
        indexes_of_valid_subjects = np.setdiff1d(indexes_of_subjects, indexes_of_predicates)
        del indexes_of_subjects
        del indexes_of_predicates

        valid_subjects = string_vocab[indexes_of_valid_subjects]
        ut.serializer(object_=valid_subjects, path=self.p_folder, serialized_name='subjects')

        num_subjects = len(subjects_to_indexes)
        del subjects_to_indexes

        print('Number of valid subjects for DL-Learner', len(indexes_of_valid_subjects))
        print('Size of vocabulary', len(vocab))
        print('Number of RDF triples', num_triples)
        print('Number of subjects', num_subjects)
        print('Number of predicates', len(predicates_to_indexes))

        Saver.settings.append("Size of vocabulary:" + str(len(vocab)))
        Saver.settings.append("Number of RDF triples:" + str(num_triples))
        Saver.settings.append("Number of subjects:" + str(num_subjects))
        Saver.settings.append("Number of predicates:" + str(len(predicates_to_indexes)))
        Saver.settings.append("Number of valid subjects for DL-Learner:" + str(indexes_of_valid_subjects))
        del vocab

        ut.serializer(object_=indexes_of_valid_subjects, path=self.p_folder,
                      serialized_name='indexes_of_valid_subjects')
        del indexes_of_valid_subjects

        texec = ThreadPoolExecutor(4)
        future = texec.submit(self.similarity_function, freq_matrix, num_triples)

        del freq_matrix

        f = future.result()

        top_k_sim = f[0]
        negatives = f[1]

        return top_k_sim, negatives

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

    @performance_debugger('Choose K attractives')
    def choose_to_K_attractive_entities(self, co_similarity_matrix, size_of_iteracting_entities):
        """

        :param co_similarity_matrix:
        :param size_of_iteracting_entities:
        :return:
        """

        for k, v in co_similarity_matrix.items():
            if len(v) > size_of_iteracting_entities:
                co_similarity_matrix[k] = dict(
                    sorted(v.items(), key=lambda kv: kv[1], reverse=True)[0:size_of_iteracting_entities])
            else:
                co_similarity_matrix[k] = dict(sorted(v.items(), key=lambda kv: kv[1], reverse=True))
        return co_similarity_matrix

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
        attractives = list(pruned_stats_corpus_info.values())
        del pruned_stats_corpus_info

        repulsive_entities = self.random_sample_repulsive_entities(stats_corpus_info, K)

        ut.serializer(object_=repulsive_entities, path=self.p_folder, serialized_name='Negative_URIs')
        ut.serializer(object_=attractives, path=self.p_folder, serialized_name='Positive_URIs')

        return attractives, repulsive_entities

    def apply_ppmi_on_entitiy_adj_matrix(self, freq_adj_matrix: typing.Dict[int, Counter], num_triples: int) -> typing.Tuple[dict,dict]:
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

    @performance_debugger('Constructing entitiy adj matrix')
    def construct_entity_adj_matrix(self, f_name, bound=''):
        vocabulary = None
        freq_adj_matrix = None
        num_of_rdf = None
        index_to_valid_sub=None

        # Todo IDEA, do not save valid indexes but while reading write into file

        if isinstance(bound, int):
            freq_adj_matrix, vocabulary, num_of_rdf, index_to_valid_sub = self.process_file(f_name, bound)

        else:
            print('not yet implemented')
            exit(1)


        ut.serializer(object_=np.array(list(index_to_valid_sub.keys()),dtype=np.uint32), path=self.p_folder, serialized_name='indexes_of_valid_subjects')
        ut.serializer(object_=list(index_to_valid_sub.values()), path=self.p_folder, serialized_name='subjects')

        del index_to_valid_sub

        ut.serializer(object_=vocabulary, path=self.p_folder, serialized_name='vocabulary')
        del vocabulary

        holder = self.similarity_function(freq_adj_matrix, num_of_rdf)


        return holder
    def process_file(self, path, bound):

        freq_adj_matrix = {}
        vocabulary = {}
        num_of_rdf = 0
        i_vocab=dict()
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

                try:
                    s, p, o, flag = self.decompose_rdf(sentence)

                    if flag != 3:
                        # print(sentence)
                        continue
                except ValueError:
                    continue
                total_sentence += 1

                writer_kb.write(sentence)

                # mapping from string to vocabulary
                vocabulary.setdefault(s, len(vocabulary))
                # entities_to_indexes[s] = vocabulary[s]
                i_vocab.setdefault(len(vocabulary),s)

                # vocabulary.setdefault(p, len(vocabulary))

                vocabulary.setdefault(o, len(vocabulary))
                # entities_to_indexes[o] = vocabulary[o]

                freq_adj_matrix.setdefault(vocabulary[s], Counter())[vocabulary[o]] += 1
                freq_adj_matrix.setdefault(vocabulary[o], Counter())[vocabulary[s]] += 1

            reader.close()
            num_of_rdf += total_sentence

        writer_kb.close()
        # TODO think later bout predicates
        print('Number of RDF triples in the input KG:', num_of_rdf)
        print('Number of entities:', len(vocabulary))

        return freq_adj_matrix, vocabulary, num_of_rdf,i_vocab


class PL2VEC(object):
    def __init__(self, system_energy=1):

        self.epsilon = 0.001
        self.texec = ThreadPoolExecutor(8)

        self.total_distance_from_attractives = list()
        self.total_distance_from_repulsives = list()
        self.ratio = list()
        self.system_energy = system_energy

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
        for index_of_i, top_k_index in context_entitiy_pms.items():

            context = np.array(list(top_k_index.keys()), dtype=np.int32)


            pms = np.around(list(top_k_index.values()), 3).astype(np.float32)
            pms.shape = (pms.size, 1)

            repulsive = np.array(list(repulsitve_entities[index_of_i]), dtype=np.int32)

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

        return e, 0  # agg_att_d / agg_rep_d

    @performance_debugger('Generating Embeddings:')
    def start(self, *, e, max_iteration, energy_release_at_epoch, holder, negative_constant):
        scaler = MinMaxScaler()

        for epoch in range(max_iteration):
            print('EPOCH: ', epoch)

            previous_f_norm = LA.norm(e, 'fro')

            e, d_ratio = self.go_through_entities(e, holder, negative_constant, self.system_energy)

            self.system_energy = self.system_energy - energy_release_at_epoch
            # self.ratio.append(d_ratio)

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

            Saver.settings.append('\n Target: ' + sampled_entitiy_name)
            print('\n Target:', sampled_entitiy_name)

            distances = list()

            for index, ith_vec_of_embeddings in enumerate(learned_embeddings):
                distances.append(spatial.distance.cosine(sampled_vector, ith_vec_of_embeddings))

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
        def calculate_difference_to_mean(data_point):
            euclidiean_dist = distance.euclidean(data_point, mean_of_cluster)
            return euclidiean_dist

        labels = labeled_embeddings.labels.unique().tolist()

        sampled_total_entities = pd.DataFrame()

        for label in labels:
            cluster = labeled_embeddings.loc[labeled_embeddings.labels == label].copy()
            mean_of_cluster = cluster.mean()

            cluster['euclidean_distance_to_mean'] = cluster.apply(calculate_difference_to_mean, axis=1)
            # cluster.loc[:, 'description']
            cluster.sort_values(by=['euclidean_distance_to_mean'], ascending=False)

            sampled_entities_from_cluster = cluster.head(num_sample_from_clusters)
            sampled_total_entities = pd.concat([sampled_total_entities, sampled_entities_from_cluster])

        sampled_total_entities.drop(['euclidean_distance_to_mean'], axis=1, inplace=True)

        representative_entities = dict()
        for key, val in sampled_total_entities.labels.to_dict().items():
            representative_entities.setdefault(val, []).append(key)

        return representative_entities

    @performance_debugger('Pseudo labeling via DBSCAN')
    def pseudo_label_DBSCAN(self, df):
        df['labels'] = DBSCAN().fit(df).labels_
        return df

    @performance_debugger('Prune non resources')
    def prune_non_subject_entities(self, embeddings, upper_folder='not init'):

        if upper_folder != 'not init':
            index_of_only_subjects = ut.deserializer(path=upper_folder, serialized_name="indexes_of_valid_subjects")
        else:
            index_of_only_subjects = ut.deserializer(path=self.p_folder, serialized_name="indexes_of_valid_subjects")

        # index_of_predicates = ut.deserializer(path=self.p_folder, serialized_name="index_of_predicates")

        # Prune those subject that occurred in predicate position
        # index_of_only_subjects = np.setdiff1d(index_of_resources, index_of_predicates)

        # i_vocab = ut.deserializer(path=self.p_folder, serialized_name="i_vocab")

        # names = np.array(list(i_vocab.values()))[index_of_only_subjects]

        # PRUNE non subject entities
        embeddings = embeddings[index_of_only_subjects]
        del index_of_only_subjects

        names = ut.deserializer(path=self.p_folder, serialized_name="subjects")

        embeddings = pd.DataFrame(embeddings, index=names)
        del names

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
        # embeddings_of_resources = self.apply_partitioning(embeddings_of_resources, partitions=50, samp_part=10)

        pseudo_labelled_embeddings = self.pseudo_label_DBSCAN(embeddings_of_resources)

        self.topN_cosine(pseudo_labelled_embeddings, 5, 20)

        ut.serializer(object_=pseudo_labelled_embeddings, path=self.p_folder,
                      serialized_name='pseudo_labelled_resources')

        Saver.settings.append("### cluster distribution##")
        Saver.settings.append(pseudo_labelled_embeddings.labels.value_counts().to_string())
        Saver.settings.append("##" * 20)

        representative_entities = self.sample_from_clusters(pseudo_labelled_embeddings, num_sample_from_clusters)

        Saver.settings.append('Num of generated clusters: ' + str(len(list(representative_entities.keys()))))

        f = open(self.p_folder + '/Settings', 'w')

        for text in Saver.settings:
            f.write(text)
            f.write('\n')
        f.close()

        print(representative_entities)

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

            candidates_of_negative_subjects = np.setdiff1d(all_samples, positive_subjects)

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

        #        str_subjects = list(ut.deserializer(path=self.p_folder, serialized_name='subjects').keys())

        # for key, val in dict_of_cluster_with_original_term_names.items():
        #    dict_of_cluster_with_original_term_names[key] = [str_subjects[item] for item in val]

        self.kg_path = self.p_folder

        self.execute_dl(dict_of_cluster_with_original_term_names)
        # self.execute_DL(resources=str_subjects, dict_of_cluster=dict_of_cluster_with_original_term_names)


class Saver:
    settings = []
    similarities = []
