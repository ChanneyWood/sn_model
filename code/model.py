import torch.nn as nn
import dgl
import torch
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from utils import *
from modules import *
import math


class grsce(nn.Module):
    def __init__(self, h_dim_1, h_dim_2, num_nodes, seq_len, dropout, use_lstm=1, attn=''):
        super().__init__()
        self.h_dim_1 = h_dim_1
        self.h_dim_2 = h_dim_2
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        # self.maxpool = maxpool
        self.dropout = nn.Dropout(dropout)
        self.use_lstm = use_lstm
        # self.nodes_embeds = nn.Parameter(torch.Tensor(num_nodes, h_dim))

        self.node_map = None

        # out_feat = int(h_dim // 2)
        # self.g_aggr = GCN(1, out_feat, h_dim, 1, F.relu, dropout)
        self.g_aggr = GraphConv(h_dim_1, h_dim_2, norm='both', weight=True, bias=True)

        if self.use_lstm:
            self.encoder = nn.LSTM(h_dim_2, h_dim_2, batch_first=True, bidirectional=False)
        else:
            self.encoder = nn.GRU(h_dim_2, h_dim_2, batch_first=True)

        self.linear_r = nn.Linear(h_dim_2, 1, bias=True)

        self.threshold = 0.5
        self.out_func = torch.sigmoid
        self.criterion = nn.MSELoss(reduce=True, size_average=True)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, event_list, sc_num_list):
        pred, _ = self.__get_pred(event_list)
        loss = self.get_loss(pred, sc_num_list)
        return loss

    def get_loss(self, pred, sc_num_list):
        pred = pred.reshape(-1, 1)
        epsilon = 1e-5
        # 手动去前面数据为0的话题 如：[0,0,0,0,0,0,0,17300]
        # sc_num_list = [item[-1] / item[-2] - 1 + epsilon for item in sc_num_list]
        # sc_num = [item[-1] / item[-2] - 1 + epsilon for item in sc_num_list]
        sc_num = [1 - item[-2] / item[-1] + epsilon for item in sc_num_list]
        sc_num = torch.Tensor(sc_num).cuda().reshape(-1, 1)
        loss = self.criterion(pred, sc_num)
        return loss

    def __get_pred(self, event_list):
        embed_seq_tensor, len_non_zero = self.aggregator(event_list)
        # packed_input = torch.nn.utils.rnn.pack_padded_sequence(embed_seq_tensor,
        #                                                        len_non_zero,
        #                                                        batch_first=True)
        packed_input = embed_seq_tensor.reshape((len(len_non_zero), len_non_zero[0], 5))
        if self.use_lstm:
            # lstm返回值为out, (h, c)
            output, (feature, _) = self.encoder(packed_input)
        else:
            # rnn,gru返回值为out, h
            output, feature = self.encoder(packed_input)
        output = output.data
        feature = feature.squeeze(0)

        pred = self.linear_r(feature)
        # pred = self.linear_r(output)

        return pred, feature

    def aggregator(self, event_list):

        len_non_zero = []

        for item in event_list:
            length = len(item)
            # 预测t+1时刻的值，len_non_zero存储该时刻前的参考的时间区间长度
            len_non_zero.append(length)

        # entity graph
        g_list = [item for slice in event_list for item in slice]
        batched_g = dgl.batch(g_list)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 给图节点加入隐藏属性
        # map_id_tensor = tensor_id_map_func(batched_g.ndata['id'], self.node_map)
        batched_g.ndata['h'] = torch.ones((batched_g.num_nodes(), 1))
        batched_g = batched_g.to(device)  # torch.device('cuda:0')
        feat = torch.ones(batched_g.num_nodes(), self.h_dim_1).cuda()
        batched_g.ndata['h'] = self.g_aggr(batched_g, feat)
        bfnh = batched_g.ndata['h']

        # 用每个batch中最大的节点嵌入表示图嵌入
        global_node_info = dgl.mean_nodes(batched_g, 'h')

        embed_seq_tensor = torch.zeros(len(len_non_zero), self.seq_len, 1 * self.h_dim_2)
        if torch.cuda.is_available():
            embed_seq_tensor = embed_seq_tensor.cuda()

        move_idx = 0
        for i, ent_slice in enumerate(event_list):
            for j, t in enumerate(ent_slice):
                embed_seq_tensor[i, j, :] = global_node_info[move_idx]
                move_idx += 1

        embed_seq_tensor = self.dropout(embed_seq_tensor)
        return embed_seq_tensor, len_non_zero

    def predict(self, ent_list, sc_num_list):
        pred, feature = self.__get_pred(ent_list)
        loss = self.get_loss(pred, sc_num_list)
        return loss, pred, feature

    def evaluate(self, ent_list, sc_num):
        loss, pred, _ = self.predict(ent_list, sc_num)
        return loss