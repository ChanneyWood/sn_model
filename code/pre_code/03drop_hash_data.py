#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 8:52
# @Author  : glq
# @Email   : wycmglq@outlook.com
# @File    : 03drop_hash_data.py
# @Do      : something
import joblib
import pandas as pd

path = "../../data/middle_data/"
hashtag_data = pd.read_csv('../../data/event_dict_with_time_satisfy_add_index.csv')
# 清洗不存在对应用户id的hashtag
follow_user_all = joblib.load(path + "follow_user_all")
origin_tweet2user_dict = joblib.load(path + "origin_tweet2user_dict")
id_list = [str(il).split(" ") for il in hashtag_data['id_list'].to_list()]
tweet_id_all = list(origin_tweet2user_dict.keys())
idx_to_remove = []
new_id_list = []
for key, item in enumerate(id_list):
    new_item = []
    for tweet_id in item:
        if int(tweet_id) in tweet_id_all and origin_tweet2user_dict[int(tweet_id)] in follow_user_all:
            new_item.append(tweet_id)
    if len(new_item) == 0:
        idx_to_remove.append(key)
        new_id_list.append('')
    elif len(new_item) == 1:
        new_id_list.append(new_item[0])
    else:
        id_list_str = ''
        for idx in range(len(new_item)):
            id_list_str = id_list_str + new_item[idx] + ' '
        id_list_str += new_item[-1]
        new_id_list.append(id_list_str)

hashtag_data['id_list'] = new_id_list
hashtag_data = hashtag_data[~hashtag_data['index'].isin(idx_to_remove)].reset_index(drop=True)
hashtag_data.to_csv("../../data/event_dict_with_time_satisfy_add_index_drop_without_userId.csv")
print(0)
