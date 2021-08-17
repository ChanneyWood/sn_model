#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/23 2:35
# @Author  : liliang
# @Email   : wycmglq@outlook.com
# @File    : 20process.py
import sys
sys.path.append('../')
import math
import pandas as pd
import utils
import datetime
from tqdm import tqdm
import joblib

path = "../../data/middle_data/"
# 首先获得hashtag的信息 {idx,hashtag,oringinIdList,retweet_num,strat_time,end_time,subtract_time_hour}
# 为每个事件进行时间切片
slice_num = 8
print("loading event_dict")
hashtag_data = pd.read_csv('../../data/event_dict_with_time_satisfy_add_index_drop_without_userId.csv')

event_time_slice = {event_idx: utils.split_time_ranges_avg(item['strat_time'], item['end_time'], slice_num) for
                    event_idx, item in hashtag_data.iterrows()}

joblib.dump(event_time_slice, path + 'event_time_slice')
print("load event_dict completed")

# 获得每个原创博文的信息{oringinId，original_user_id，original_time，retweet_real_num，context}
print("loading origin data")
origin_data = pd.read_csv('../../data/origin_content_join_all.csv')
origin_data_dict = {item['original_twitter_id']: item.to_list()[1:5] for _, item in tqdm(origin_data.iterrows())}
joblib.dump(origin_data_dict, path + 'origin_data_dict')
# origin_data_dict = joblib.load(path + 'origin_data_dict')
print("load origin data completed")
# 每个事件切片内构建子图，统计每个子图的转发数，按比例记录网络真实的转发数
print("loading retweet_data")
retweet_data = pd.read_csv('../../data/retweet_content_sheet_1.csv')
print("load retweet_data completed")


# get idx of retweet content with origin id list
def find_retweet_list(id_list):
    index_of_t = retweet_data[retweet_data.loc[:, 'original_twitter_id'].isin(id_list)].index
    return index_of_t


# retweet num of each time slice of each event
slice_retweet_num_all = []
# relation data of each time slice of each event
slice_retweet_data_all = []
# semantic data of each time slice of each event
slice_retweet_content_all = []
event_originId_list = []
# slice retweet data for each origin tweeter
for key, id_list in enumerate(tqdm(hashtag_data['id_list'])):
    id_list = str(id_list).split(' ')
    event_originId_list.append(id_list)

    retweet_of_content = retweet_data.loc[find_retweet_list(id_list)]
    retweet_of_content = retweet_of_content.sort_values(by='retweet_time')
    slice_retweet_num = [0 for n in range(0, slice_num)]
    slice_retweet_data = [[] for n in range(0, slice_num)]
    slice_retweet_content = [set() for n in range(0, slice_num)]
    if len(retweet_of_content) > 0:
        time_slice = event_time_slice[key]
        slice_idx = 0
        from_time = pd.to_datetime(time_slice[slice_idx][0])
        to_time = pd.to_datetime(time_slice[slice_idx][1])
        for _, item in retweet_of_content.iterrows():
            retweet_time = pd.to_datetime(item['retweet_time'])
            origin_tweet_id = item['original_twitter_id']
            retweet_text = item['text_data']
            if retweet_time > to_time:
                if slice_idx >= slice_num - 1:
                    # time slice is over, break this process
                    break
                else:
                    # enter next time slice
                    slice_idx += 1
                    # find next time slice where retweet data not none
                    for n in range(slice_idx, slice_num):
                        from_time = pd.to_datetime(time_slice[slice_idx][0])
                        to_time = pd.to_datetime(time_slice[slice_idx][1])
                        if retweet_time > to_time:
                            slice_idx += 1
                            continue
                        else:
                            break
            if slice_idx >= 8:
                break
            if not isinstance(retweet_text, float) or not math.isnan(retweet_text):
                slice_retweet_content[slice_idx].add(origin_data_dict[origin_tweet_id][3])
                slice_retweet_content[slice_idx].add(retweet_text)
            slice_retweet_data[slice_idx].append(item.to_list()[:4])
            slice_retweet_num[slice_idx] += 1
    slice_retweet_num_all.append(slice_retweet_num)
    slice_retweet_data_all.append(slice_retweet_data)
    slice_retweet_content_all.append(slice_retweet_content)

joblib.dump(slice_retweet_num_all, path + 'slice_retweet_num_all')
joblib.dump(slice_retweet_data_all, path + 'slice_retweet_data_all')
joblib.dump(slice_retweet_content_all, path + 'slice_retweet_content_all')
joblib.dump(event_originId_list, path + "event_originId_list")
print(0)
