import os
import pandas as pd
import sys

sys.path.append('../')
from utils import *
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
final_x_list = []
final_y_list = []

for file in tqdm(files):
    try:
        datas = pd.read_csv(file, sep=',', engine='python')
        datas = datas.drop_duplicates()
        datas.sort_values(createAt, inplace=True)
        datas.dropna(subset=[createAt], inplace=True)
        datas = datas.reset_index(drop=True)
        result = getFileData(datas)
        final_x_list.append(result[0])
        final_y_list.append(result[1])
        print(0)
    except:
        print(file)
joblib.dump(final_x_list, dataPath+'final_x_list')
joblib.dump(final_y_list, dataPath+'final_y_list')
