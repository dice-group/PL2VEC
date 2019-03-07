import datetime
import os
import pickle
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
#import sortednp as snp
import re
from typing import Dict
from sklearn.decomposition import TruncatedSVD

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

    #pickle.dump(object_, open(path + '/' + serialized_name + ".p", "wb"))


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


def calculate_similarities(context_of_a, context_of_b, entropies):

    intersection = snp.intersect(context_of_a, context_of_b)

    if len(intersection) == 0:
        return 0

    sum_of_ent_context_of_a= entropies[context_of_a].sum()

    sum_of_ent_context_of_b = entropies[context_of_b].sum()

    sum_of_ent_intersections = entropies[intersection].sum()

    sim = sum_of_ent_intersections / (sum_of_ent_context_of_a + sum_of_ent_context_of_b)

    return sim


def cal_demir(context_of_a, context_of_b, entropies):
    intersection = snp.intersect(context_of_a, context_of_b)

    sum_of_ent_context_of_a = entropies[context_of_a].sum()

    sum_of_ent_context_of_b = entropies[context_of_b].sum()

    sum_of_ent_intersections = entropies[intersection].sum()

    sim = sum_of_ent_intersections / (sum_of_ent_context_of_a + sum_of_ent_context_of_b)

    return sim




def apply_pca(X):
    pca = PCA(n_components=2)
    low_embeddings=pca.fit_transform(X)

    return low_embeddings