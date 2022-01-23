from pdb import set_trace
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import neighbors, svm
from sklearn.model_selection import GridSearchCV, train_test_split
import sys
import networkx as nx
import stellargraph as sg
from stellargraph.data import UnsupervisedSampler
from stellargraph.mapper import Attri2VecLinkGenerator, Attri2VecNodeGenerator
from stellargraph.layer import Attri2Vec, link_classification

from tensorflow import keras

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def convert_raw_data_to_weights(raw_feat, raw_data):

    # get station numbers
    station_numbers = raw_feat.shape[0]

    # generate weight matrix
    weight = np.zeros((station_numbers, station_numbers))

    # calculate weight matrix by aggregating raw data
    for i in range(raw_data.shape[0]):
        
        # find the index of flow in feat
        index_start = np.where(raw_feat[:, 0]==raw_data[i][0])
        index_end = np.where(raw_feat[: ,0]==raw_data[i][1])
        start = index_start[0][0]
        end = index_end[0][0]
        # aggregation
        weight[start][end] = weight[start][end] + 1
    
    weight = weight.astype(int)

    return weight

def creat_nx_graph(raw_node_feat, adj_matrix):

    # graph initialize
    G = nx.DiGraph()

    # add nodes
    G.add_nodes_from([i for i in range(raw_node_feat.shape[0])])
    for i in range(raw_node_feat.shape[0]):
        G.nodes[i]['id'] = raw_node_feat[i][0]

    # add edges
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i][j] == 1:
                G.add_edge(i, j)
    
    return G


# read in data
df_raw_feat_2016 = pd.read_csv ('Divvy_Station_2016.csv', usecols= ['id','latitude','longitude','dpcapacity'])
df_feat_2016 = pd.read_csv ('Divvy_Station_2016.csv', usecols= ['latitude','longitude','dpcapacity'])
df_raw_label_2016 = pd.read_csv ('July_trip_raw_data_2016.csv')

df_raw_feat_2017 = pd.read_csv ('Divvy_Station_2017.csv', usecols= ['id','latitude','longitude','dpcapacity'])
df_feat_2017 = pd.read_csv ('Divvy_Station_2017.csv', usecols= ['latitude','longitude','dpcapacity'])
df_raw_label_2017 = pd.read_csv ('July_trip_raw_data_2017.csv')

# processing and sort the feat accoring to id
raw_feat_2016 = df_raw_feat_2016.to_numpy(dtype=float)
raw_feat_2016.sort(axis=0)

raw_label_2016 = df_raw_label_2016.to_numpy()

raw_feat_2017 = df_raw_feat_2017.to_numpy()
raw_feat_2017.sort(axis=0)

raw_label_2017 = df_raw_label_2017.to_numpy()

# label processing
weight_2016 = convert_raw_data_to_weights(raw_feat_2016, raw_label_2016)
weight_2017 = convert_raw_data_to_weights(raw_feat_2017, raw_label_2017)

# thresholding (to be developed) 
weight_2016[weight_2016 < 5] = 0
weight_2017[weight_2017 < 5] = 0
weight_2016[weight_2016 >= 5] = 1
weight_2017[weight_2017 >= 5] = 1

# create graph
station_net_2016_nx = creat_nx_graph(raw_feat_2016, weight_2016)
station_net_2016_stellar = sg.StellarGraph.from_networkx(station_net_2016_nx, node_features=df_feat_2016)

station_net_2017_nx = creat_nx_graph(raw_feat_2017, weight_2017)
station_net_2017_stellar = sg.StellarGraph.from_networkx(station_net_2017_nx, node_features=df_feat_2017)

# specify optional parameter values
nodes = list(station_net_2016_stellar.nodes())
number_of_walks = 2
length = 5

# create the UnsupervisedSampler instance
unsupervised_samples = UnsupervisedSampler(station_net_2016_stellar, nodes=nodes, length=length, number_of_walks=number_of_walks)

# set the batch size and the number of epochs.
batch_size = 50
epochs = 100

# define an attri2vec training generator,
generator = Attri2VecLinkGenerator(station_net_2016_stellar, batch_size)

layer_sizes = [128]
attri2vec = Attri2Vec(
    layer_sizes=layer_sizes, generator=generator, bias=False, normalize=None
)

# build the model and expose input and output sockets of attri2vec, for node pair inputs:
x_inp, x_out = attri2vec.in_out_tensors()

prediction = link_classification(output_dim=1, output_act="sigmoid", edge_embedding_method="ip")(x_out)

# stack the attri2vec encoder and prediction layer into a Keras model, and specify the loss.
model = keras.Model(inputs=x_inp, outputs=prediction)

model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-2),
    loss=keras.losses.binary_crossentropy,
    metrics=[keras.metrics.binary_accuracy],
)

# train the model
print("Start training.")
history = model.fit(
    generator.flow(unsupervised_samples),
    epochs=epochs,
    verbose=2,
    use_multiprocessing=False,
    workers=1,
    shuffle=True,
)


