#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 7:38
# @Author  : glq
# @Email   : wycmglq@outlook.com
# @File    : data.py
# @Do      : something
import torch
from torch.utils import data


# empirical distribution/counts s/r/o in one day, predict r
class DistData(data.Dataset):
    def __init__(self, data_set, set_name):
        rel_g, sc_num = data_set
        # rel_g, word_g, sc_num = data_set
        self.len = len(sc_num)
        # if torch.cuda.is_available():
        #     rel_g = rel_g.cuda()
        #     word_g = word_g.cuda()
        #     sc_num = sc_num.cuda()

        # self.times = times
        self.rel_g = rel_g
        # self.word_g = word_g
        self.sc_num = sc_num

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # return self.rel_g[index], self.word_g[index], self.sc_num[index]
        return self.rel_g[index], self.sc_num[index]
