#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/25 2:40
# @Author  : glq
# @Email   : wycmglq@outlook.com
# @File    : 31build_word_graph.py
# @Do      : something
import sys
sys.path.append('../')
import utils
import joblib
import dgl
import torch
import numpy as np
from tqdm import tqdm


# undirected graph
def get_big_word_graph(row, col, weight, node_idx, vocab_s=None):
    g = dgl.DGLGraph()
    g.add_nodes(len(node_idx))
    g.add_edges(row, col)
    g.edata['w'] = torch.from_numpy(np.array(weight)).view(-1, 1)
    # add self loop
    g.add_edges(g.nodes(), g.nodes())
    # print(g.edges())
    # degs = g.in_degrees().float()
    # norm = torch.pow(degs, -0.5)
    # norm[torch.isinf(norm)] = 0
    # g.ndata['norm'] = norm.unsqueeze(1)
    node_idx = list(map(int, node_idx))
    node_idx = np.array(node_idx)
    g.ndata['id'] = torch.from_numpy(node_idx).long().view(-1, 1)

    # print(g)
    return g


path = "../../data/middle_data/"
# slice_retweet_num_sample = joblib.load("slice_retweet_num_sample.txt")
# slice_retweet_data_sample = joblib.load("slice_retweet_data_sample.txt")
# slice_retweet_content_sample = joblib.load("slice_retweet_content_sample.txt")
# event_time_slice_sample = joblib.load("event_time_slice_sample.txt")
slice_retweet_content = joblib.load(path + "slice_retweet_content_all")
word_graph_dict_full = []
for event_idx, event_text in enumerate(tqdm(slice_retweet_content)):
    # 每个时间段的词存在nodes中
    text_l_all_time = event_text
    text_l_all_time = [sent.split(' ') for text_l in text_l_all_time for sent in text_l]
    nodes = [item for sublist in text_l_all_time for item in sublist]
    nodes = list(set(nodes))
    nodes.remove('')
    # nodes = [x for x in nodes if x in vocab_l]  # filter empty strings
    # node_idx = get_idx(vocab_l, nodes)
    row, col, weight = utils.get_pmi(text_l_all_time, nodes)  # words here

    word_graph_dict_full.append(get_big_word_graph(row, col, weight, nodes))

print(0)
joblib.dump(word_graph_dict_full, path + "word_graph_dict_full")
