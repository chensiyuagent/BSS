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
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from torch.autograd import Variable

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
set_trace()
# one hot encoding
y_train = np.vstack((weight_2016.flatten(), -1 * (weight_2016.flatten()-1))).transpose()
y_test = np.vstack((weight_2017.flatten(), -1 * (weight_2017.flatten()-1))).transpose()

# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

trainset = dataset(X_train, y_train)
testset = dataset(X_test, y_test)

# DataLoader
trainloader = DataLoader(trainset,batch_size=16,shuffle=True)
testloader = DataLoader(testset,batch_size=16,shuffle=True)

# classifier
class Net(nn.Module):

  def __init__(self,input_shape):
    super(Net,self).__init__()
    self.fc1 = nn.Linear(input_shape,32)
    self.fc2 = nn.Linear(32,64)
    self.fc3 = nn.Linear(64,2)

  def forward(self,x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = torch.sigmoid(self.fc3(x))
    return x

# criterion
class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

  	
#hyper parameters
learning_rate = 0.01
epochs = 100
# Model , Optimizer, Loss
model = Net(input_shape=X_train.shape[1])
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
weight_classes = torch.tensor([0.9, 0.1])
loss_fn = nn.BCELoss(weight=weight_classes)
a = torch.tensor(0.75)
criterion = FocalLoss(2, alpha=a)

#forward loop

print("Start training.")
losses = []
accur = []

for i in range(epochs):
  for j,(x_train,y_train) in enumerate(trainloader):
    
    #calculate output
    output = model(x_train)
 
    #calculate loss
    #loss = loss_fn(output,y_train)
    loss = F.cross_entropy(output, y_train, weight_classes)
 
    #accuracy
    predicted = model(torch.tensor(x_train, dtype=torch.float32))
    x = predicted.reshape(-1).detach().numpy().round() == y_train.reshape(-1).detach().numpy()
    acc = (predicted.reshape(-1).detach().numpy().round() == y_train.reshape(-1).detach().numpy()).mean()
    #backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  if i%50 == 0:
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

predicted = model(torch.tensor(X_test, dtype=torch.float32))
acc = (predicted.reshape(-1).detach().numpy().round() == y_test).mean()

print(acc)