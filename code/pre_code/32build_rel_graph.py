#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/25 9:58
# @Author  : glq
# @Email   : wycmglq@outlook.com
# @File    : 32build_rel_graph.py
# @Do      : something
import dgl
import joblib
import numpy as np
import torch
from tqdm import tqdm

# slice_retweet_num_sample = joblib.load("slice_retweet_num_sample.txt")
# slice_retweet_data_sample = joblib.load("slice_retweet_data_sample.txt")
# slice_retweet_content_sample = joblib.load("slice_retweet_content_sample.txt")
# event_time_slice_sample = joblib.load("event_time_slice_sample.txt")
path = "../../data/middle_data/"
slice_retweet_data_all = joblib.load(path + "slice_retweet_data_all")
origin_tweet2user_dict = joblib.load(path + "origin_tweet2user_dict")
print(0)


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


def build_retweet_graph(slice_retweet_data):
    slice_graph = []
    origin_user_id_set = set()
    for slice in slice_retweet_data:
        nodes_pairs = []
        for retweet_data in slice:
            origin_content_id = retweet_data[0]
            origin_user_id = origin_tweet2user_dict[origin_content_id]
            origin_user_id_set.add(origin_user_id)
            nodes_pairs.append([origin_user_id, retweet_data[1]])
        nodes = [pair[1] for pair in nodes_pairs] + list({pair[0] for pair in nodes_pairs})
        nodes_idx = list(set(nodes))
        all_nodes.extend(nodes_idx)
        row = [nodes_idx.index(pair[0]) for pair in nodes_pairs]
        col = [nodes_idx.index(pair[1]) for pair in nodes_pairs]
        slice_graph.append(build_dgl_retweet_graph(row, col, nodes_idx))
        # print(0)
    for idx, graph in enumerate(slice_graph):
        if graph.num_nodes() == 0:
            slice_graph[idx] = build_dgl_retweet_graph([], [], list(origin_user_id_set))
    return slice_graph


# def build_follow_graph():
#     print(0)

all_nodes = []
rel_graph_dict_full = []
for event_idx, slice_retweet_data in enumerate(tqdm(slice_retweet_data_all)):
    rel_graph_dict_full.append(build_retweet_graph(slice_retweet_data))
joblib.dump(rel_graph_dict_full, path + "rel_graph_dict_full")
all_nodes_idx = list(set(all_nodes))
outf = open(path + "all_nodes_idx.txt", 'w')
for idx, node_id in enumerate(all_nodes_idx):
    line = str(idx) + " " + str(node_id)
    outf.write("{}\n".format(line))
outf.close()
joblib.dump(all_nodes_idx, path + "all_nodes_idx")
