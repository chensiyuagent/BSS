from pdb import set_trace
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import neighbors, svm
from sklearn.model_selection import GridSearchCV, train_test_split
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter


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


def predict(X, y, model):
    Z = model.predict(X)
    acc = 1 - 1/len(X) * ((Z != y).sum())

    return acc

def post_process(prediction):
    threshold = 0.8
    prediction[prediction > threshold] = 1
    prediction[prediction <= threshold] = 0
    return prediction

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

#defining dataset class
class dataset(Dataset):
  def __init__(self,x,y):
    self.x = torch.tensor(x,dtype=torch.float32)
    self.y = torch.tensor(y,dtype=torch.float32)
    self.length = self.x.shape[0]
 
  def __getitem__(self,idx):
    return self.x[idx],self.y[idx]

  def __len__(self):
    return self.length


# read in data
df_raw_feat_2016 = pd.read_csv ('https://raw.githubusercontent.com/Li-ai-cell/Machine-Learning-Trial/main/Divvy_Station_2016.csv', usecols= ['id','latitude','longitude','dpcapacity'])
df_raw_label_2016 = pd.read_csv ('https://raw.githubusercontent.com/Li-ai-cell/Machine-Learning-Trial/main/July_trip_raw_data_2016.csv')

df_raw_feat_2017 = pd.read_csv ('https://raw.githubusercontent.com/Li-ai-cell/Machine-Learning-Trial/main/Divvy_Station_2017.csv', usecols= ['id','latitude','longitude','dpcapacity'])
df_raw_label_2017 = pd.read_csv ('https://raw.githubusercontent.com/Li-ai-cell/Machine-Learning-Trial/main/July_trip_raw_data_2017.csv')

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

# one hot encoding
y_train = weight_2016.flatten()
y_test = weight_2017.flatten()

# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

trainset = dataset(X_train, y_train)
testset = dataset(X_test, y_test)
 
weight = [1/9, 1]

samples_weight = np.array([weight[int(t)] for t in y_train])
samples_weight = torch.from_numpy(samples_weight)

sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

# DataLoader
trainloader = DataLoader(trainset, batch_size=100, sampler=sampler)
testloader = DataLoader(testset, batch_size=100, shuffle=False)

# classifier
class Net(nn.Module):
  def __init__(self,input_shape):
    super(Net,self).__init__()
    self.fc1 = nn.Linear(input_shape,32)
    self.fc2 = nn.Linear(15,32)
    self.fc3 = nn.Linear(32,16)
    self.fc4 = nn.Linear(16,1)

  def forward(self,x):
    x = torch.relu(self.fc1(x))
    #x = torch.relu(self.fc2(x))
    x = torch.relu(self.fc3(x))
    x = torch.sigmoid(self.fc4(x))
    return x

#hyper parameters
learning_rate = 0.01
epochs = 10
# Model , Optimizer, Loss
model = Net(input_shape=X_train.shape[1])
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
weight_classes = torch.tensor([0.9, 0.1])
# loss_fn = nn.BCELoss(weight=weight_classes)
loss_fn = nn.BCELoss()

def binary_cross_entropy(input, y, weight): 

    if y[0][0] == 1:
        loss = -(input.log()*y + (1-y)*(1-input).log()).mean() * weight
    else:
        loss = -(input.log()*y + (1-y)*(1-input).log()).mean() * (1 - weight)

    return loss

# forward loop

print("Start training.")
losses = []
accur = []
for i in range(epochs):
  for j,(x_train,y_train) in enumerate(trainloader):
    
    # calculate output
    output = model(x_train)

    # calculate loss
    # loss = F.cross_entropy(output, y_train, weight_classes)
    loss = loss_fn(output, y_train.reshape(-1 ,1))
    # loss = binary_cross_entropy(output, y_train, 0.95)
    
    # accuracy
    predicted = model(torch.tensor(x_train, dtype=torch.float32))
    x = predicted.reshape(-1).detach().numpy().round() == y_train.reshape(-1).detach().numpy()
    acc = (predicted.reshape(-1).detach().numpy().round() == y_train.reshape(-1).detach().numpy()).mean()

    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  if i%2 == 0:
    losses.append(loss)
    accur.append(acc)
    print("epoch {}\tloss : {}\t accuracy : {}".format(i,loss,acc))

plt.plot(losses)
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('loss')

#printing the accuracy
plt.plot(accur)
plt.title('Accuracy vs Epochs')
plt.xlabel('Accuracy')
plt.ylabel('loss')

y_test = weight_2017.flatten()
predicted = model(torch.tensor(X_test, dtype=torch.float32))
acc = (predicted.reshape(-1).detach().numpy().round() == y_test.reshape(-1)).mean()
predict_test = post_process(predicted[:, 0]).detach().numpy()
target = y_test
prec_zero = precision(predict_test, target, 0)
reca_zero = recall(predict_test, target, 0)
print("zero_class Accuracy:{}, Precision:{}, Recall:{}".format(acc, prec_zero, reca_zero))
prec_one = precision(predict_test, target, 1)
reca_one = recall(predict_test, target, 1)
print("one_class Accuracy:{}, Precision:{}, Recall:{}".format(acc, prec_one, reca_one))

output_matrix = np.reshape(predicted[:, 0].detach().numpy(), weight_2017.shape) 

np.savetxt('2017_prediction.csv', output_matrix, fmt='%.1e')