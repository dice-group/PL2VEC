import helper_classes


def test_parser_create_dictionary():
    parser = helper_classes.Parser()

    # path of knowledge graphs
    # KG must be under DBpedia folder
    f_name = 'DBpedia'
    # Read until bound
    bound = 100
    co_occurrences, m_term_to_index, num_triples, subjects_to_indexes = parser.create_dictionary(f_name, bound)

    assert isinstance(co_occurrences, dict)
    # keys of dict must be int
    assert isinstance(list(co_occurrences.keys())[0], int)
    # values of dictionary must contain int
    assert isinstance(list(co_occurrences.values())[0][0], int)
    # Mapping from raw entities to indexes of vocabulary
    assert isinstance(m_term_to_index, list)
    assert isinstance(num_triples, int)

    # Number of subjects must be less than all entities
    assert len(subjects_to_indexes) < len(co_occurrences)


def test_parser_create_dictionary_text():
    parser = helper_classes.Parser()

    # path of knowledge graphs
    # KG must be under Bio2RDF folder
    f_name = 'Bio2RDF'
    bound = 100
    co_occurrences, m_term_to_index, num_triples, subjects_to_indexes = parser.create_dic_from_text(f_name, bound)

    assert isinstance(co_occurrences, dict)
    # keys of dict must be int
    assert isinstance(list(co_occurrences.keys())[0], int)
    # values of dictionary must contain int
    assert isinstance(list(co_occurrences.values())[0][0], int)
    # Mapping from raw entities to indexes of vocabulary
    assert isinstance(m_term_to_index, dict)
    assert isinstance(list(m_term_to_index.values())[0], int)
    assert isinstance(num_triples, int)
    # Number of subjects must be less than all entities
    assert len(subjects_to_indexes) < len(co_occurrences)

    ppmi_co_occurences = parser.binary_to_ppmi_matrix(co_occurrences, num_triples)

    assert isinstance(ppmi_co_occurences, dict)

    assert len(ppmi_co_occurences) == len(co_occurrences)

    assert isinstance(list(ppmi_co_occurences.keys())[0], int)


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
    ppmi_co_occurences = parser.binary_to_ppmi_matrix(co_occurrences, number_of_rdf)

    print(ppmi_co_occurences)
    # Test the symetciy of co-occurences
    for event_a, v in ppmi_co_occurences.items():
        for event_b, ppmi_val in v.items():
            assert ppmi_val == (ppmi_co_occurences[event_b])[event_a]

    # Justification => math.log2(0.25/(0.5*0.25))
    assert (ppmi_co_occurences[0])[3] == 1
    assert (ppmi_co_occurences[4])[3] == 0.41504
    assert (ppmi_co_occurences[2])[1] == 0.41504


def test_retrieve_interactting_entities():
    parser=helper_classes.Parser()
    co_occurrences = {0: [1, 2, 3, 4], 1: [0, 2, 4, 2, 4, 2], 2: [0, 1, 1, 4, 1, 4], 3: [0, 4],
                      4: [0, 3, 1, 2, 2, 1]}

    number_of_rdf = 4
    K=3
    # PPMI of a,b = > log_2 ( joint prob of a and b/ marginal prob of a times b
    ppmi_of_entities = parser.binary_to_ppmi_matrix(co_occurrences, number_of_rdf)

    P, N = parser.get_attractive_repulsive_entities(ppmi_of_entities, K)
    assert P[0] [3]==1