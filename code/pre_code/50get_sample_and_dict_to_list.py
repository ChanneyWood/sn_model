#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/27 8:07
# @Author  : glq
# @Email   : wycmglq@outlook.com
# @File    : 50get_sample_and_dict_to_list.py
# @Do      : something

import joblib
path = "../../data/middle_data/"
rel_g = joblib.load(path + "rel_graph_dict_full")
word_g = joblib.load(path + "word_graph_dict_full")
sc_num = joblib.load(path + "slice_sc_num_all")

sample_num = 50
rel_g_sample = rel_g[0:sample_num]
word_g_sample = word_g[0:sample_num]
sc_num_sample = sc_num[0:sample_num]
joblib.dump(rel_g_sample, path + "rel_graph_dict_sample")
joblib.dump(word_g_sample, path + "word_graph_dict_sample")
joblib.dump(sc_num_sample, path + "slice_sc_num_sample")