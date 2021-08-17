#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/25 11:20
# @Author  : glq
# @Email   : wycmglq@outlook.com
# @File    : 02get_origin_tweet2user_dict.py
# @Do      : something
import pandas as pd
import joblib

path = "../../data/middle_data/"
origin_data = pd.read_csv('../../data/origin_content_join_all.csv')
origin_tweet2user = {item['original_twitter_id']: item["original_user_id"] for _, item in origin_data.iterrows()}
print(0)
joblib.dump(origin_tweet2user, path + "origin_tweet2user_dict")
