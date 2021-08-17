#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 2:37
# @Author  : glq
# @Email   : wycmglq@outlook.com
# @File    : 01get_follow_rel.py
# @Do      : something
import pandas as pd
import joblib

path = "../../data/middle_data/"
user2user = pd.read_csv("../../data/user2user.csv")
outf = open(path + "user2user.txt", 'w')
follow_user_all = set()
for _, item in user2user.iterrows():
    item = item.values.item().split(" ")
    f, t = item[0][1:], item[1][1:]
    follow_user_all.add(int(f))
    follow_user_all.add(int(t))
    line = f + " " + t
    outf.write("{}\n".format(line))
outf.close()
follow_user_all = list(follow_user_all)
joblib.dump(follow_user_all, path + "follow_user_all")
