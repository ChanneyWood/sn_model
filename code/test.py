import math

import torch
import pandas as pd

# def get_time_frequency(from_time, to_time, n):
#     from_time, to_time = pd.to_datetime(from_time), pd.to_datetime(to_time)
#     delta = to_time - from_time
#     interval = int(((delta.days * 86400) + delta.seconds) / n)
#     return interval
#
# get_time_frequency('2012-10-04-12:35:54','2012-10-05-12:35:56',8)


# retweet_data = pd.read_csv('../data/retweet_content_sheet_1.csv', nrows=1)
# for _,item in retweet_data.iterrows():
#     item = item.to_list()[:5]
#     print(0)
# # retweet_data = retweet_data.head(100000)
# slice_retweet_num_all = []
# # get idx of retweet content with origin id list
# def find_retweet_list(id_list):
#     index_of_t = retweet_data[retweet_data.loc[:, 'original_twitter_id'].isin(id_list)].index
#     return index_of_t
# find_retweet_list(['3464198275618249'])

# num_with_fault=[1,2,4]
# df = pd.DataFrame(columns = ["num"])
# df['num'] =[3,4,2,55,6]
# new_df = df[~df['num'].isin(num_with_fault)].reset_index(drop=True)
# index_with_fault = df[~df['num'].isin(num_with_fault)].index
# df = df.loc[index_with_fault]
# len = len(df)
# print(0)

import joblib
import pickle
# slice_retweet_num_all = joblib.load('slice_retweet_num_all')
# slice_retweet_data_all = joblib.load('slice_retweet_data_all')
# slice_retweet_content_all = joblib.load('slice_retweet_content_all')
# event_time_slice = joblib.load('event_time_slice')
#
# slice_retweet_num_sample = slice_retweet_num_all[0:100]
# slice_retweet_data_sample = slice_retweet_data_all[0:100]
# slice_retweet_content_sample = slice_retweet_content_all[0:100]
# event_time_slice_sample = event_time_slice[0:100]
#
# joblib.dump(slice_retweet_num_sample, 'slice_retweet_num_sample.txt')
# joblib.dump(slice_retweet_data_sample, 'slice_retweet_data_sample.txt')
# joblib.dump(slice_retweet_content_sample, 'slice_retweet_content_sample.txt')
# joblib.dump(event_time_slice_sample, 'event_time_slice_sample.txt')
# print(0)

# user2user = pd.read_csv("../data/user2user.csv")
# outf = open("user2user.txt", 'w')
# for _, item in user2user.iterrows():
#     item = item.values.item().split(" ")
#     line = item[0][1:] + " " + item[1][1:]
#     outf.write("{}\n".format(line))
# outf.close()

import dgl

# node_idx = [0, 1, 2]
# g = dgl.DGLGraph()
# g.add_nodes(len(node_idx))
# g.add_edges([0, 0, 1, 2], [1, 1, 0, 0])
# nodes = g.num_nodes()
# print(0)
from tqdm import tqdm
path = "../data/middle_data/"
# tweet2id = joblib.load(path + "origin_tweet2user_dict")
# hashtag_data = pd.read_csv('../data/event_dict_with_time_satisfy_add_index.csv')
# event_originId_list = []
# event_originUserId_list = []
# for key, id_list in enumerate(tqdm(hashtag_data['id_list'])):
#     id_list = str(id_list).split(' ')
#     user_id_list = [tweet2id[int(t_id)] for t_id in id_list]
#     event_originId_list.append(id_list)
#     event_originUserId_list.append(user_id_list)
# print(0)
# joblib.dump(event_originId_list, path + "event_originUserId_list")
# joblib.dump(event_originUserId_list, path + "event_originUserId_list")

# event_time_slice = joblib.load(path + 'event_time_slice')
# print("load event_dict completed")
# l = [1,0,3,4,5]
# # n = sum(l)
# # l = [math.ceil(c*90/n+1) for c in l]
# idx = l.index(0)
# print(0)

# rel_g_list = joblib.load(path + "rel_graph_dict_sample")
# event_list = rel_g_list[:5]
# g_list = [item for slice in event_list for item in slice]
# batched_g = dgl.batch(g_list)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# batched_g = batched_g.to(device)
# id_list = batched_g.ndata['id']
# nodes_map = joblib.load(path + "all_nodes_idx")
# new_ilist = id_list.cpu().numpy().tolist()
# new_ilist = [[nodes_map.index(item[0])] for item in new_ilist]
# print(0)

pred = [[[1,2,4,5]]]
pred = torch.Tensor(pred)
pred = pred.squeeze()
print(0)