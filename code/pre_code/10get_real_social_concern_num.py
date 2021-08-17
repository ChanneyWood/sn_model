#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 12:27
# @Author  : glq
# @Email   : wycmglq@outlook.com
# @File    : 10get_real_social_concern_num.py
# @Do      : something
import pandas as pd
import joblib

path = "../../data/middle_data/"
origin_data = pd.read_csv('../../data/origin_content_join_all.csv')
hashtag_data = pd.read_csv('../../data/event_dict_with_time_satisfy_add_index_drop_without_userId.csv')
origin_tid_to_retnum_dict = {item['original_twitter_id']: item['retweet_real_num'] for _, item in origin_data.iterrows()}

sc_num_list = []
for _, item in hashtag_data.iterrows():
    sc_num = 0
    id_list = str(item['id_list']).split(" ")
    for t_id in id_list:
        sc_num += origin_tid_to_retnum_dict[int(t_id)]
    sc_num_list.append(sc_num)
print(0)
joblib.dump(sc_num_list, path + "sc_num_list")
