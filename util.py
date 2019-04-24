import datetime
import os
import pickle
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt

import numpy as np
#import sortednp as snp
import re
from typing import Dict
from sklearn.decomposition import TruncatedSVD

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from bz2 import BZ2File
import bz2



def get_path_knowledge_graphs(path: str):
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


def file_type(f_name):
    if f_name[-4:] == '.bz2':
        reader = bz2.open(f_name, "rt")
        return reader
    return open(f_name, "r")



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

def create_experiment_folder():
    directory = os.getcwd() + '/Experiments/'
    folder_name = str(datetime.datetime.now())
    path_of_folder = directory + folder_name  # 'Spring'+str(folder_name)
    os.makedirs(path_of_folder)
    return path_of_folder, path_of_folder[:path_of_folder.rfind('/')]


def serializer(*, object_: object, path: str, serialized_name: str):
    with open(path + '/' + serialized_name + ".p", "wb") as f:
        pickle.dump(object_, f)
    f.close()


def deserializer(*, path: str, serialized_name: str):
    with open(path + "/" + serialized_name + ".p", "rb") as f:
        obj_ = pickle.load(f)
    f.close()
    return obj_
#    return pickle.load(open(path + "/" + serialized_name + ".p", "rb"))



def do_scatter_plot(X, path):
    import dash
    import dash_core_components as dcc
    import dash_html_components as html
    import plotly.graph_objs as go


    print(X)
    exit(1)
    names = list(pickle.load(open(path + "/vocab.p", "rb")).keys())

    x = X[:, 0]
    y = X[:, 1]
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

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
                    legend={'x': 2, 'y': 0},
                    hovermode='closest'
                )
            }
        )
    ])

    app.run_server(debug=True)


def visualize_2D(low_embeddings, storage_path, title='default name'):
    """
    Visiualize only first two columns
    :param low_embeddings:
    :param storage_path:
    :param title:
    :return:
    """
    d = deserializer(path=storage_path, serialized_name='subjects_to_indexes')

    counter = 0
    for k, v in d.items():
        counter += 1
        if counter > 10:
            break
        x = low_embeddings[v][1]
        y = low_embeddings[v][0]

        annotation = k.replace('http://example.com/', '')
        annotation = annotation.replace('http://dbpedia.org/resource/', '')
        annotation = annotation.replace('Category:', '')
        plt.annotate(annotation, (x, y))
        plt.title(title)
        plt.scatter(x, y)
        # plt.legend()

    plt.show()



def apply_pca(X):
    pca = PCA(n_components=2)
    low_embeddings=pca.fit_transform(X)

    return low_embeddings



def decompose_rdf(sentence):

    flag=0
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
        flag=2

    elif len(components) == 4:
        del components[-1]
        s, p, o = components

        flag=4

    elif len(components) == 3:
        s, p, o = components
        flag = 3

        if '"' in sentence:
            remaining=sentence[len(s)+len(p)+5:]
            literal=remaining[1:remaining.index(' <http://')]
            o=literal

    elif len(components) > 4:

        s = components[0]
        p = components[1]
        remaining_sentence = sentence[sentence.index(p) + len(p) + 2:]
        literal = remaining_sentence[:remaining_sentence.index(' <http://')]
        o = literal

    else:

        ## This means that literal contained in RDF triple contains < > symbol
        """ pass"""
        flag=0
        # print(sentence)
        raise ValueError()


    return s, p, o,flag

def randomly_initialize_embedding_space(num_vocab, embeddings_dim):
    return np.random.rand(num_vocab, embeddings_dim)


def write_settings(path,settings):
    f = open(path + '/Settings', 'w')
    for text in settings:
        f.write(text)
        f.write('\n')
    f.close()



def read_kb_for_w2v():
    path='/home/demir/Desktop/complex/datasets/drugbank/train.txt'
    required_formated=list()

    with open(path, "r") as reader:
        for rdf_triple in reader:
            s,p,o,_=rdf_triple.split()
            required_formated.extend((s,p,o))

    return required_formated


def generator_of_reader(bound,knowledge_graphs,rdf_decomposer,vocab=None):
    for f_name in knowledge_graphs:
        reader = file_type(f_name)
        total_sentence = 0
        for sentence in reader:

            #Ignore Literals
            if '"' in sentence or "'" in sentence:
                continue

            if total_sentence == bound: break
            total_sentence += 1

            try:
                s, p, o, flag = rdf_decomposer(sentence)

                # <..> <..> <..>
                if flag !=4:
                    print(sentence, '+', flag)
                    continue

            except ValueError as v:
                ## started 2017-03-31T16:36:17Z
#                print(sentence)
                continue

            yield s, p, o


        reader.close()