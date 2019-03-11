import helper_classes
import numpy as np


def test_binary_to_ppmi_matrix():
    """Create dummy binary_co_matrix and construct PPMI values"""
    parser = helper_classes.Parser()

    # 0 1 2         freq(0)     = 2 , freq(1)     = 3 , freq(2)     = 3 , freq(3)     = 1 , freq(4)     = 3
    # 0 3 4
    # 2 1 4
    # 2 1 4
    #
    #               co_occ(0,0) = 2 , co_occ(0,1) = 1 , co_occ(0,2) = 1 , co_occ(0,3) = 1 , co_occ(0,4) = 1
    #               co_occ(1,0) = 1 , co_occ(1,1) = 3 , co_occ(1,2) = 3 , co_occ(1,3) = 0 , co_occ(1,4) = 2
    #               co_occ(2,0) = 1 , co_occ(2,1) = 3 , co_occ(2,2) = 3 , co_occ(3,2) = 0 , co_occ(2,4) = 2
    #               co_occ(3,0) = 1 , co_occ(3,1) = 0 , co_occ(3,2) = 0 , co_occ(3,3) = 1 , co_occ(3,4) = 1
    #               co_occ(4,0) = 1 , co_occ(4,1) = 2 , co_occ(4,2) = 2 , co_occ(4,3) = 1 , co_occ(4,4) = 3

    co_occurrences = {0: [1, 2, 3, 4], 1: [0, 2, 4, 2, 4, 2], 2: [0, 1, 1, 4, 1, 4], 3: [0, 4],
                      4: [0, 3, 1, 2, 2, 1]}

    number_of_rdf = 4
    # PPMI of a,b = > log_2 ( joint prob of a and b/ marginal prob of a times b
    ppmi_co_occurences = parser.apply_ppmi_on_co_matrix(co_occurrences, number_of_rdf)

    print(ppmi_co_occurences)
    # Test the symetciy of co-occurences
    for event_a, v in ppmi_co_occurences.items():
        for event_b, ppmi_val in v.items():
            assert ppmi_val == (ppmi_co_occurences[event_b])[event_a]

    # Justification => math.log2(0.25/(0.5*0.25))
    assert (ppmi_co_occurences[0])[3] == 1
    assert (ppmi_co_occurences[4])[3] == 0.41504
    assert (ppmi_co_occurences[2])[1] == 0.41504


def test_calculate_entropies():
    parser = helper_classes.Parser()

    # 0 1 2
    # 0 3 4

    co_occurrences = {0: [1, 2, 3, 4], 1: [0, 2], 2: [0, 1], 3: [0, 4],
                      4: [0, 1]}
    number_of_rdf = 2

    entropies = parser.calculate_entropies(co_occurrences, number_of_rdf)

    # Calculation of Entropy for 1
    # N is multiplied by 2 as list_of_context_ent contains other two element of an RDF triple

    p_one = len(co_occurrences[1]) / (number_of_rdf * 2)
    e_one = - p_one * np.log2(p_one)
    assert entropies[1] == round(e_one, 5)



def test_frequency_to_entropy_jaccard_index():
    parser = helper_classes.Parser()

    # 0 1 2
    # 0 3 4

    co_occurrences = {0: [1, 2, 3, 4], 1: [0, 2], 2: [0, 1], 3: [0, 4],
                      4: [0, 3]}
    number_of_rdf = 2

    entropy_jaccard_sim = parser.apply_entropy_jaccard_on_entitiy_adj_matrix(co_occurrences, num_triples=number_of_rdf)
    print(entropy_jaccard_sim)
    # Is matrix symmetric ?
    for event_a, v in entropy_jaccard_sim.items():
        for event_b, sim in v.items():
            assert sim == (entropy_jaccard_sim[event_b])[event_a]

    # Sim(0,2)= (sum of entropies of 1, 4) /  ( sum of entropies of 1,2,3,4 + sum of entropies of 0,1,2,3,4,5

    entropies = parser.calculate_entropies(co_occurrences, number_of_rdf)

    sum_of_entropies_of_all_point_of_zero = entropies[1] + entropies[2] + entropies[3] + entropies[4]
    sum_of_entropies_of_all_point_of_two = entropies[0]+entropies[1]
    sum_of_entropies_of_overlapped_points = entropies[1]

    var = sum_of_entropies_of_overlapped_points / (
                sum_of_entropies_of_all_point_of_zero + sum_of_entropies_of_all_point_of_two)

    var=round(var,6)
    assert entropy_jaccard_sim[0][2] == var
