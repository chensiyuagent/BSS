from ast import Str
from pdb import set_trace
from tokenize import String
import numpy as np
import pandas as pd


df_raw_feat_2016 = pd.read_csv ('Divvy_Station_2016.csv', usecols= ['id', 'name', 'latitude','longitude','dpcapacity'])

df_raw_feat_2017 = pd.read_csv ('Divvy_Station_2017.csv', usecols= ['id', 'name', 'latitude','longitude','dpcapacity'])

raw_feat_2016 = df_raw_feat_2016.to_numpy()
raw_feat_2016.sort(axis=0)

raw_feat_2017 = df_raw_feat_2017.to_numpy()
raw_feat_2017.sort(axis=0)


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

def int_to_en(num):
    d = { 0 : 'zero', 1 : 'one', 2 : 'two', 3 : 'three', 4 : 'four', 5 : 'five',
          6 : 'six', 7 : 'seven', 8 : 'eight', 9 : 'nine', 10 : 'ten',
          11 : 'eleven', 12 : 'twelve', 13 : 'thirteen', 14 : 'fourteen',
          15 : 'fifteen', 16 : 'sixteen', 17 : 'seventeen', 18 : 'eighteen',
          19 : 'nineteen', 20 : 'twenty',
          30 : 'thirty', 40 : 'forty', 50 : 'fifty', 60 : 'sixty',
          70 : 'seventy', 80 : 'eighty', 90 : 'ninety' }
    k = 1000
    m = k * 1000
    b = m * 1000
    t = b * 1000

    assert(0 <= num)

    if (num < 20):
        return d[num]

    if (num < 100):
        if num % 10 == 0: return d[num]
        else: return d[num // 10 * 10] + '-' + d[num % 10]

    if (num < k):
        if num % 100 == 0: return d[num // 100] + ' hundred'
        else: return d[num // 100] + ' hundred and ' + int_to_en(num % 100)

    if (num < m):
        if num % k == 0: return int_to_en(num // k) + ' thousand'
        else: return int_to_en(num // k) + ' thousand, ' + int_to_en(num % k)

    if (num < b):
        if (num % m) == 0: return int_to_en(num // m) + ' million'
        else: return int_to_en(num // m) + ' million, ' + int_to_en(num % m)

    if (num < t):
        if (num % b) == 0: return int_to_en(num // b) + ' billion'
        else: return int_to_en(num // b) + ' billion, ' + int_to_en(num % b)

    if (num % t == 0): return int_to_en(num // t) + ' trillion'
    else: return int_to_en(num // t) + ' trillion, ' + int_to_en(num % t)

    raise AssertionError('num is too large: %s' % str(num))

df_matrix_2016 = pd.read_csv('2016.csv', header=None)
df_matrix_2017 = pd.read_csv('2017.csv', header=None)
matrix_2016 = df_matrix_2016.to_numpy(dtype=int)
matrix_2017 = df_matrix_2017.to_numpy(dtype=int)

# prune no use nodes and their corresponding edges
weight_2016 = convert_adj_to_feat(matrix_2016, raw_feat_2016)
weight_2017 = convert_adj_to_feat(matrix_2017, raw_feat_2017)

myText = open(r'2016_creat_file.txt','w')

myText.write('CREATE' + '\n')

for i in range(raw_feat_2016.shape[0]):
    station_name=""
    for m in raw_feat_2016[i][1]:
        if m.isdigit():
            m = int_to_en(int(m))
        if m == "&":
            m = "and"
        if m == "(" or m == ")":
            m = ""
        if m == "*":
            m = "star"
        if m == ".":
            m = "dot"
        if m == "-":
            m = "to"
        station_name = station_name + m
    raw_feat_2016[i][1] = station_name
    myText.write('  (' + str(raw_feat_2016[i][1].replace(" ", "_")) + ':Station {latitude: ' + str(raw_feat_2016[i][2]) + ', longitude: ' + str(raw_feat_2016[i][3]) + ', dpcapacity: ' + str(raw_feat_2016[i][4]) + '}), \n')

myText.write('\n')

for i in range(weight_2016.shape[0]):
    for j in range(weight_2016.shape[1]):
        if weight_2016[i][j] == 1:
            start_station = raw_feat_2016[i][1].replace(" ", "_")
            end_station = raw_feat_2016[j][1].replace(" ", "_")
            myText.write('  (' + str(start_station) + ')-[:transport]->(' + str(end_station) + '),\n')

myText.close()

#CREATE
#  (alice:Person {name: 'Alice', numberOfPosts: 38}),
#  (michael:Person {name: 'Michael', numberOfPosts: 67}),

#  (karin)-[:KNOWS]->(veselin),
#  (chris)-[:KNOWS]->(karin);