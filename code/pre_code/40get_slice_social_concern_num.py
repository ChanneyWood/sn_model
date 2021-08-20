#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 7:44
# @Author  : glq
# @Email   : wycmglq@outlook.com
# @File    : 40get_slice_social_concern_num.py
# @Do      : something
import math

import pandas as pd
import joblib

path = '../../data/middle_data/'
slice_retweet_num_all = joblib.load(path + "slice_retweet_num_all")
origin_data_dict = joblib.load(path + "origin_data_dict")
sc_num_list = joblib.load(path + "sc_num_list")
event_originId_list = joblib.load(path + "event_originId_list")
slice_sc_num_all = []
for idx, slice_retweet_num in enumerate(slice_retweet_num_all):
    n = sum(slice_retweet_num)
    sc_list = []
    sc_acc = 0  # accumulated social concern num
    for re_num in slice_retweet_num:
        sc_acc += math.ceil(re_num * sc_num_list[idx] / n)
        sc_list.append(sc_acc)
    slice_sc_num_all.append(sc_list)
print(0)
joblib.dump(slice_sc_num_all, path + "slice_sc_num_all")