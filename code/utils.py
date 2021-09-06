#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/23 3:04
# @Author  : glq
# @Email   : wycmglq@outlook.com
# @File    : utils.py
# @Do      : something
import joblib
import pandas as pd
import numpy as np
import datetime
import scipy.sparse as sp
from math import log
import copy

import torch


def split_time_ranges_avg(from_time, to_time, n):
    """
    切分时间区间为n份
    """

    def get_time_frequency(from_time, to_time, n):
        from_time, to_time = pd.to_datetime(from_time), pd.to_datetime(to_time)
        delta = to_time - from_time
        interval = int(((delta.days * 86400) + delta.seconds) / n)
        return interval

    frequency = get_time_frequency(from_time, to_time, n)
    from_time, to_time = pd.to_datetime(from_time), pd.to_datetime(to_time)
    time_range = list(pd.date_range(from_time, to_time, freq='%sS' % frequency))
    if to_time not in time_range:
        time_range.append(to_time)
    time_range = [item.strftime("%Y-%m-%d %H:%M:%S") for item in time_range]
    time_ranges = []
    for item in time_range:
        f_time = item
        t_time = (datetime.datetime.strptime(item, "%Y-%m-%d %H:%M:%S") + datetime.timedelta(seconds=frequency))
        if t_time >= to_time:
            t_time = to_time.strftime("%Y-%m-%d %H:%M:%S")
            time_ranges.append([f_time, t_time])
            break
        time_ranges.append([f_time, t_time.strftime("%Y-%m-%d %H:%M:%S")])

        # n = 10

        # print(frequency)
        # res = split_time_ranges('2012-07-08-14:24:24', '2012-07-09-11:43:48', frequency)
        # print(res[0:n])
    return time_ranges[0:n]


# tlist = split_time_ranges_avg('2021-11-17-08:47:41', '2021-11-20-12:38:23', 8)
# print(0)

# use words
def get_pmi(words_l, nodes):
    # 每个时间段内词汇共现率计算pmi
    vocab = nodes
    vocab_size = len(vocab)
    word_id_map = {}
    for i in range(vocab_size):
        word_id_map[vocab[i]] = i
    # # filter words not in vocab
    new_words_l = []
    for l in words_l:
        l = set(l)  # unique
        new_words_l.append([w for w in l if w in nodes])
    windows = new_words_l
    # calculate term freq in all docs
    word_window_freq = {}
    for window in windows:
        window = list(set(window))
        for i in range(len(window)):
            if word_window_freq.get(window[i]):
                word_window_freq[window[i]] += 1
            else:
                word_window_freq[window[i]] = 1

    word_pair_count = {}
    for window in windows:
        for i in range(1, len(window)):
            for j in range(0, i):

                word_i = window[i]
                word_j = window[j]

                if word_id_map.get(word_i) != None and word_id_map.get(word_j) != None:

                    word_i_id = word_id_map[word_i]
                    word_j_id = word_id_map[word_j]
                    if word_i_id == word_j_id:
                        continue
                    word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                    if word_pair_str in word_pair_count:
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1
                    # two orders
                    word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                    if word_pair_str in word_pair_count:
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1

    row = []
    col = []
    weight = []
    num_window = len(windows)
    for key in word_pair_count:
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        count = word_pair_count[key]
        word_freq_i = word_window_freq[vocab[i]]
        word_freq_j = word_window_freq[vocab[j]]
        pmi = log((1.0 * count * num_window) / (1.0 * word_freq_i * word_freq_j))
        # print(pmi,'before round pmi')
        pmi = round(pmi, 3)
        # print(pmi,'round pmi')
        if pmi <= 0:
            continue
        row.append(i)
        col.append(j)
        weight.append(pmi)
    adj = sp.csr_matrix((weight, (row, col)), shape=(vocab_size, vocab_size))
    adj_new = normalize_adj(adj)
    _row = adj_new.row
    _col = adj_new.col
    _data = adj_new.data
    _data = [round(elem, 3) for elem in _data]
    __row, __col, __data = [], [], []
    for k in range(len(_data)):
        if _data[k] > 0:
            __row.append(_row[k])
            __col.append(_col[k])
            __data.append(_data[k])
        # else:
        # print('empty')
    return __row, __col, __data


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj += sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def collate_3(batch):
    rel_g = [item[0] for item in batch]
    word_g = [item[1] for item in batch]
    sc_num = [item[2] for item in batch]
    return [rel_g, word_g, sc_num]


def collate_2(batch):
    rel_g = [item[0] for item in batch]
    sc_num = [item[1] for item in batch]
    return [rel_g, sc_num]


'''
Loss function
'''


# Pick-all-labels normalised (PAL-N)
def soft_cross_entropy(pred, soft_targets):
    logsoftmax = torch.nn.LogSoftmax(dim=-1)  # pred (batch, #node/#rel)
    pred = pred.type('torch.DoubleTensor')
    if torch.cuda.is_available():
        pred = pred.cuda()
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))


def mean_square_error(pred, real):
    pred = pred.type('torch.DoubleTensor')
    if torch.cuda.is_available():
        pred = pred.cuda()

    return np.sum(np.power((real.reshape(-1, 1) - pred), 2)) / len(real)


def tensor_id_map_func(tensor_id, id_map):
    new_id_list = tensor_id.numpy().tolist()
    new_id_list = [[id_map.index(item[0])] for item in new_id_list]
    return torch.Tensor(new_id_list).type(torch.int64)


#  change graph to accumulated
#  input:time_slice type:list size:7
def change_graph_to_acc(slice_data):
    new_slice = []
    acc_data = []
    for one_slice in slice_data:
        acc_data.extend(one_slice)
        append_data = copy.deepcopy(acc_data)
        new_slice.append(append_data)
    return new_slice

def change_df_to_acc(slice_data):
    new_slice = []
    for idx, one_slice in enumerate(slice_data):
        if idx == 0:
            acc_data = one_slice
        else:
            acc_data = acc_data.append(one_slice)
        append_data = copy.deepcopy(acc_data)
        new_slice.append(append_data)
    return new_slice