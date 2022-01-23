from pdb import set_trace
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import neighbors, svm
from sklearn.model_selection import GridSearchCV, train_test_split
import sys

np.set_printoptions(threshold=sys.maxsize)


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

def convert_raw_feat_to_classifiers_feat(raw_feat):

    # get station numbers
    station_numbers = raw_feat.shape[0]

    # get station connection numbers
    edge_numbers = raw_feat.shape[0] * raw_feat.shape[0]

    # generate feature matrix
    feat = np.zeros((edge_numbers, 6))

    # calculate feat matrix by assigning raw feat
    for i in range(feat.shape[0]):

        # find the index of flow in feat
        index_start = int(i / station_numbers)
        index_end = int(i % station_numbers)
        
        feat[i, 0:3] = raw_feat[index_start, 1:]
        feat[i, 3:] = raw_feat[index_end, 1:]

    return feat

def convert_adj_to_feat(raw_adj_matrix, feat):

    # initialize
    pruned_adj_matrix = np.zeros((feat.shape[0], feat.shape[0]))

    # delta in index
    delta = feat[-1][0] - raw_adj_matrix.shape[0] + 1

    # prune raw_adj_matrix according to matrix
    for i in range(feat.shape[0]):
        for j in range(feat.shape[0]):
            start_node = feat[i][0]
            end_node = feat[j][0]
            pruned_adj_matrix[i][j] = raw_adj_matrix[int(start_node - delta)][int(end_node - delta)]
            
    return pruned_adj_matrix

def rbf_SVC(X, y):

    rbf_SVC = svm.SVC(kernel='rbf', class_weight="balanced")
    Cs = [1]
    clf = GridSearchCV(estimator=rbf_SVC, param_grid=dict(C=Cs), scoring='accuracy', n_jobs=-1, cv=5)
    clf.fit(X, y)

    return clf

def predict(X, y, model):

    Z = model.predict(X)
    acc = 1 - 1/len(X) * ((Z != y).sum())

    return acc

def precision(prediction, target, class_index):
    idx_egde_in_prediction = np.argwhere(prediction==class_index)
    idx_egde_in_target = np.argwhere(target==class_index)
    correct_edge_in_prediction = 0
    prec = 0
    for i in range(idx_egde_in_target.shape[0]):
        correct_edge_in_prediction = (idx_egde_in_target[i] in idx_egde_in_prediction) + correct_edge_in_prediction
    if idx_egde_in_prediction.shape[0] != 0:
        prec = correct_edge_in_prediction/idx_egde_in_prediction.shape[0]
    return prec

def recall(prediction, target, class_index):
    idx_egde_in_prediction = np.argwhere(prediction==class_index)
    idx_egde_in_target = np.argwhere(target==class_index)
    correct_edge_in_prediction = 0
    rec = 0
    for i in range(idx_egde_in_target.shape[0]):
      correct_edge_in_prediction = (idx_egde_in_target[i] in idx_egde_in_prediction) + correct_edge_in_prediction
    if idx_egde_in_target.shape[0] != 0:
        rec = correct_edge_in_prediction/idx_egde_in_target.shape[0]
    return rec




# read in data
df_raw_feat_2016 = pd.read_csv ('Divvy_Station_2016.csv', usecols= ['id','latitude','longitude','dpcapacity'])
df_raw_label_2016 = pd.read_csv ('July_trip_raw_data_2016.csv')

df_raw_feat_2017 = pd.read_csv ('Divvy_Station_2017.csv', usecols= ['id','latitude','longitude','dpcapacity'])
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

threshold_method = True

# thresholding (to be developed)
if threshold_method is True:
    # read in data from matlab outputed csv
    df_matrix_2016 = pd.read_csv('2016.csv', header=None)
    df_matrix_2017 = pd.read_csv('2017.csv', header=None)
    matrix_2016 = df_matrix_2016.to_numpy(dtype=int)
    matrix_2017 = df_matrix_2017.to_numpy(dtype=int)

    # prune no use nodes and their corresponding edges
    weight_2016 = convert_adj_to_feat(matrix_2016, raw_feat_2016)
    weight_2017 = convert_adj_to_feat(matrix_2017, raw_feat_2017)
else:
    weight_2016[weight_2016 < 5] = 0
    weight_2017[weight_2017 < 5] = 0
    weight_2016[weight_2016 >= 5] = 1
    weight_2017[weight_2017 >= 5] = 1

# preprocessing for classifier
X_train = convert_raw_feat_to_classifiers_feat(raw_feat_2016)
X_test = convert_raw_feat_to_classifiers_feat(raw_feat_2017)
y_train = weight_2016.flatten()
y_test = weight_2017.flatten()

# classifier
print("Start training")

rbf_SVC_model = rbf_SVC(X_train, y_train)
acc_rbf = predict(X_test, y_test, rbf_SVC_model)

predict_test = rbf_SVC_model.predict(X_test)
target = y_test
prec_zero = precision(predict_test, target, 0)
reca_zero = recall(predict_test, target, 0)
print("zero_class Accuracy:{}, Precision:{}, Recall:{}".format(acc, prec_zero, reca_zero))
prec_one = precision(predict_test, target, 1)
reca_one = recall(predict_test, target, 1)
print("one_class Accuracy:{}, Precision:{}, Recall:{}".format(acc, prec_one, reca_one))

print(acc_rbf)