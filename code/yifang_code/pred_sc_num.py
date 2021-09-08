import math
import os
import pandas as pd
import sys
import argparse

sys.path.append('../')
from utils import *
from model import *
import dgl
import torch
import numpy as np
from tqdm import tqdm
import joblib


def get_fileNames(inPath):
    # 获取当前数据集中所有文件名
    fileNames = []
    for fileName in os.listdir(inPath):
        if fileName.endswith('.csv'):
            fileNames.append(os.path.join(inPath, fileName))
    return fileNames


def build_dgl_retweet_graph(row, col, node_idx):
    g = dgl.DGLGraph()
    g.add_nodes(len(node_idx))
    g.add_edges(row, col)
    # add self loop
    g.add_edges(g.nodes(), g.nodes())

    node_idx = list(map(int, node_idx))
    node_idx = np.array(node_idx)
    g.ndata['id'] = torch.from_numpy(node_idx).long().view(-1, 1)
    g.edata['w'] = torch.from_numpy(np.array([1 for n in np.arange(0, g.num_edges())])).view(-1, 1)
    # print(g)
    return g


def build_retweet_graph(retweet_data, time_slice):
    slice_data = []
    # 把dataframe按时间切片划分
    for time_item in time_slice:
        from_time = pd.to_datetime(time_item[0])
        to_time = pd.to_datetime(time_item[1])
        s_data = retweet_data[(pd.to_datetime(retweet_data[createAt]) > from_time)
                              & (pd.to_datetime(retweet_data[createAt]) < to_time)]
        slice_data.append(s_data)
    slice_data = change_df_to_acc(slice_data)
    sc_num = [len(item) for item in slice_data]
    # 对切片数据构建图
    slice_graph = []
    origin_tweet_set = set()
    for s_data in slice_data:
        node_pairs = []
        nodes = set()
        for index, row in s_data.iterrows():
            s_n, t_n = row[retweetId], row[contentId]  # source_node and target_node
            origin_tweet_set.add(s_n)
            nodes.add(s_n)
            nodes.add(t_n)
            node_pairs.append([s_n, t_n])
        nodes_idx = list(nodes)
        row = [nodes_idx.index(pair[0]) for pair in node_pairs]
        col = [nodes_idx.index(pair[1]) for pair in node_pairs]
        slice_graph.append(build_dgl_retweet_graph(row, col, nodes_idx))
    for idx, graph in enumerate(slice_graph):
        if graph.num_nodes() == 0:
            slice_graph[idx] = build_dgl_retweet_graph([], [], list(origin_tweet_set))
    return slice_graph, sc_num


def getFileData(datas):
    start_t = datas.loc[0][createAt]
    end_t = datas.loc[len(datas) - 1][createAt]
    time_slice = split_time_ranges_avg(start_t, end_t, slice_num)
    origin_tweet = datas[datas[stateType] == 0][contentId].tolist()
    retweet_data = datas[datas[retweetId].isin(origin_tweet)]
    slice_graph, sc_num = build_retweet_graph(retweet_data, time_slice)
    return slice_graph, sc_num


slice_num = 8
contentId = 'contentId'
createAt = 'createdAt'
stateType = 'stateType'
retweetId = 'retweetId'
dataPath = "../../data/yifang/"
files = get_fileNames(dataPath)

parser = argparse.ArgumentParser(description='')
parser.add_argument("--dp", type=str, default="../data/yifang/", help="data path")
parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
parser.add_argument("--n-hidden-1", type=int, default=10, help="number of hidden units")
parser.add_argument("--n-hidden-2", type=int, default=5, help="number of hidden units")
parser.add_argument("--gpu", type=int, default=0, help="gpu")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight_decay")
parser.add_argument("-d", "--dataset", type=str, default='weibo/9.6', help="dataset to use")
parser.add_argument("--grad-norm", type=float, default=1.0, help="norm to clip gradient to")
parser.add_argument("--max-epochs", type=int, default=30, help="maximum epochs")
parser.add_argument("--seq-len", type=int, default=8, help="reference days of training")
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--rnn-layers", type=int, default=1)
parser.add_argument("--maxpool", type=int, default=1)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--use-lstm", type=int, default=1, help='1 use lstm 0 gru')
parser.add_argument("--attn", type=str, default='', help='dot/add/genera; default general')
parser.add_argument("--seed", type=int, default=42, help='random seed')
args = parser.parse_args()
num_nodes = 1
model = grsce(h_dim_1=args.n_hidden_1,
              h_dim_2=args.n_hidden_2,
              num_nodes=num_nodes,
              seq_len=args.seq_len,
              dropout=args.dropout,
              use_lstm=args.use_lstm)
model_name = model.__class__.__name__
print('Model:', model_name)
token = 'based_num_pred_grsce_sl8_max1_list1_attn'
print('Token:', token, args.dataset)

model_state_file = '../models/{}/{}.pth'.format(args.dataset, token)
checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['state_dict'])
model.cuda()

@torch.no_grad()
def evaluate(rel_g, sc_num):
    rel_g, sc_num = [rel_g], [sc_num]
    model.eval()
    pred, _ = model.predict(rel_g, sc_num)
    pred = float(pred.squeeze())
    pred_sc = math.ceil(sc_num[0][-1] * (1 + math.fabs(pred)))
    print(pred_sc)


for file in tqdm(files):
    try:
        datas = pd.read_csv(file, sep=',', engine='python')
        datas = datas.drop_duplicates()
        datas.sort_values(createAt, inplace=True)
        datas.dropna(subset=[createAt], inplace=True)
        datas = datas.reset_index(drop=True)
        result = getFileData(datas)
        evaluate(result[0], result[1])
        print(0)
    except:
        print(file)

