# -*- coding:utf-8 -*-
import math
import os
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch_geometric
from minepy import MINE
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import scipy.sparse as sp
from tqdm import tqdm
from scipy.stats import kde

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def calc_corr(a, b):
    s1 = Series(a)
    s2 = Series(b)
    return s1.corr(s2)



def calculate_MIC(x, y):
    m = MINE(alpha=0.99, c=64)
    m.compute_score(x, y)
    return m.mic()

def MI(x, y, l):
    d = (4 / (2 + 2)) ** (1 / (4 + 2)) * (l ** (-1 / (2 + 4)))  # #####silverman 方法

    px = kde.gaussian_kde(x, bw_method=d)
    pxy = kde.gaussian_kde((x, y), bw_method=d)
    py = kde.gaussian_kde(y, bw_method=d)

    a = px(x)
    b = py(y)
    c = pxy((x, y))

    mi = 0
    for i in range(l):
        mi = mi + math.log(c[i] / a[i] / b[i]) / math.log(2)
    mi = mi / l
    return mi


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def create_graph(num_nodes, data):
    features = torch.randn((num_nodes, 256))
    edge_index = [[], []]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            x, y = data[:, i], data[:, j]
            corr = MI(x, y, l=len(x))

            if corr >= 0.4:
                edge_index[0].append(i)
                edge_index[1].append(j)

    edge_index = torch.LongTensor(edge_index)
    graph = Data(x=features, edge_index=edge_index)
    graph.edge_index = to_undirected(graph.edge_index, num_nodes=num_nodes)

    return graph


def nn_seq_GraphSAGE(num_nodes, seq_len, B, pred_step_size):

    path = os.path.dirname(os.path.realpath(__file__)) + '/data/Debutanizer_Data.csv'
    data = pd.read_csv(path)
    print('data.shape = ', data.shape, '\n')

    train = data[:1300]
    val = data[1301:1500]
    test = data[1500 - seq_len + 1:len(data)]



    # path = os.path.dirname(os.path.realpath(__file__)) + '/data/SRU_data.csv'
    # data = pd.read_csv(path)
    # print('data.shape = ', data.shape, '\n')
    #
    # train = data[:8000]
    # val = data[8001:9000]
    # # val = data[8001:9000]
    # test = data[9000 - seq_len + 1:len(data)]

    scaler = MinMaxScaler()
    train = scaler.fit_transform(train.values)
    val = scaler.transform(val.values)
    test = scaler.transform(test.values)

    graph = create_graph(num_nodes, train)

    def process(dataset, batch_size, step_size, shuffle):
        dataset = dataset.tolist()
        graphs = []
        for i in tqdm(range(0, len(dataset) - seq_len - pred_step_size, step_size)):
            train_seq = []
            for j in range(i, i + seq_len):
                x = []
                for c in range(len(dataset[0])):
                    x.append(dataset[j][c])
                train_seq.append(x)
            train_labels = []
            for j in range(len(dataset[0])):
                train_label = []
                for k in range(i + seq_len, i + seq_len + pred_step_size):
                    train_label.append(dataset[k][j])
                train_labels.append(train_label)
            train_seq = torch.FloatTensor(train_seq)
            train_labels = torch.FloatTensor(train_labels)
            temp = Data(x=train_seq.T, edge_index=graph.edge_index, y=train_labels)
            graphs.append(temp)

        loader = torch_geometric.loader.DataLoader(graphs, batch_size=batch_size,
                                                   shuffle=shuffle, drop_last=False)

        return loader


    Dtr = process(train, B, step_size=1, shuffle=True)
    Val = process(val, B, step_size=1, shuffle=True)
    Dte = process(test, B, step_size=pred_step_size, shuffle=False)

    return graph, Dtr, Val, Dte, scaler


def save_pickle(dataset, file_name):
    f = open(file_name, "wb")
    pickle.dump(dataset, f)
    f.close()


def load_pickle(file_name):
    f = open(file_name, "rb+")
    dataset = pickle.load(f)
    f.close()
    return dataset
