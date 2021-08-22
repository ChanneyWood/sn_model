import torch.nn as nn
import dgl
import torch
import torch.nn.functional as F
from utils import *
from modules import *
import math


class grsce(nn.Module):
    def __init__(self, h_dim, num_nodes, seq_len, maxpool, dropout, use_lstm=1, attn=''):
        super().__init__()
        self.h_dim = h_dim
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.maxpool = maxpool
        self.dropout = nn.Dropout(dropout)
        self.use_lstm = use_lstm
        self.nodes_embeds = nn.Parameter(torch.Tensor(num_nodes, h_dim))

        self.node_map = None
        # self.word_embeds = None
        # self.global_emb = None
        # self.word_graph_dict = None
        # self.graph_dict = None

        out_feat = int(h_dim // 2)
        self.g_aggr = GCN(1, out_feat, h_dim, 1, F.relu, dropout)
        # self.wg_aggr = GCN(100, h_dim, h_dim, 2, F.relu, dropout)
        # if attn == 'add':
        #     self.attn = Attention(h_dim, 'add')
        # elif attn == 'dot':
        #     self.attn = Attention(h_dim, 'dot')
        # else:
        #     self.attn = Attention(h_dim, 'general')
        if self.use_lstm:
            self.encoder = nn.LSTM(h_dim, h_dim, batch_first=True, bidirectional=False)
        else:
            self.encoder = nn.GRU(h_dim, h_dim, batch_first=True)

        self.linear_r = nn.Linear(h_dim, 1, bias=True)

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
        pred = pred.reshape(-1, 1)
        # sc_num = [item[-1] for item in sc_num_list]
        sc_num = torch.Tensor(sc_num_list).cuda().reshape(-1, 1)
        loss = self.criterion(pred, sc_num)
        return loss

    def __get_pred(self, event_list):
        # sorted_t, idx = t_list.sort(0, descending=True)
        embed_seq_tensor, len_non_zero = self.aggregator(event_list)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(embed_seq_tensor,
                                                               len_non_zero,
                                                               batch_first=True)
        if self.use_lstm:
            # lstm返回值为out, (h, c)
            output, (feature, _) = self.encoder(packed_input)
        else:
            # rnn,gru返回值为out, h
            output, feature = self.encoder(packed_input)
        output = output.data
        feature = feature.squeeze(0)


        # t_list_len = sum(len_non_zero)
        # if torch.cuda.is_available():
        #     feature = torch.cat((feature, torch.zeros(len(len_non_zero) - len(feature), feature.size(-1)).cuda()), dim=0)
        # else:
        #     feature = torch.cat((feature, torch.zeros(len(len_non_zero) - len(feature), feature.size(-1))), dim=0)

        # pred = self.linear_r(feature)
        pred = self.linear_r(output)

        return pred, feature

    def aggregator(self, event_list):
        # times = list(self.graph_dict.keys())
        # times.sort(reverse=False)
        # time_list = []

        len_non_zero = []
        # nonzero_idx = torch.nonzero(t_list, as_tuple=False).view(-1)
        # t_list = t_list[nonzero_idx]  # usually no duplicates

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
        # batched_g.ndata['h'] = self.nodes_embeds[map_id_tensor].view(-1, self.nodes_embeds.shape[1])
        # # word graph
        # wg_list = [self.word_graph_dict[tim.item()] for tim in unique_t]
        # batched_wg = dgl.batch(wg_list)
        # batched_wg = batched_wg.to(device)
        # with torch.no_grad():
        #     batched_wg.ndata['h'] = self.word_embeds[batched_wg.ndata['id']].view(-1, self.word_embeds.shape[1])
        batched_g.ndata['h'] = self.g_aggr(batched_g)
        bfnh = batched_g.ndata['h']
        # batched_wg.ndata['h'] = self.wg_aggr(batched_wg)
        # word_ids_wg = batched_wg.ndata['id'].view(-1).cpu().tolist()
        # id_dict = dict(zip(word_ids_wg, list(range(len(word_ids_wg)))))

        # # cpu operation for nodes
        # g_node_embs = batched_g.ndata.pop('h').data.cpu()
        # g_node_ids = batched_g.ndata['id'].view(-1)
        # max_query_ent = 0
        # num_nodes = len(g_node_ids)
        # c_g_node_ids = g_node_ids.data.cpu().numpy()
        # c_unique_ent_id = list(set(c_g_node_ids))
        # ent_gidx_dict = {}  # entid: [[gidx],[word_idx]]
        #
        # for ent_id in c_unique_ent_id:
        #     word_ids = self.ent_map[ent_id]
        #     word_idx = []
        #     for w in word_ids:
        #         try:
        #             word_idx.append(id_dict[w])
        #         except:
        #             continue
        #     if len(word_idx) > 1:
        #         gidx = (c_g_node_ids == ent_id).nonzero()[0]
        #         word_idx = torch.LongTensor(word_idx)
        #         ent_gidx_dict[ent_id] = [gidx, word_idx]
        #         max_query_ent = max(max_query_ent, len(word_idx))
        # # initialize a batch
        # wg_node_embs = batched_wg.ndata['h'].data.cpu()
        # Q_mx = g_node_embs.view(num_nodes, 1, self.h_dim)
        # H_mx = torch.zeros((num_nodes, max_query_ent, self.h_dim))
        #
        # for ent in ent_gidx_dict:
        #     [gidx, word_idx] = ent_gidx_dict[ent]
        #     if len(gidx) > 1:
        #         embeds = wg_node_embs.index_select(0, word_idx)
        #         for i in gidx:
        #             H_mx[i, range(len(word_idx)), :] = embeds
        #     else:
        #         H_mx[gidx, range(len(word_idx)), :] = wg_node_embs.index_select(0, word_idx)
        #
        # if torch.cuda.is_available():
        #     H_mx = H_mx.cuda()
        #     Q_mx = Q_mx.cuda()
        # output, weights = self.attn(Q_mx, H_mx)  # output (batch,1,h_dim)
        # batched_g.ndata['h'] = output[:num_nodes].view(-1, self.h_dim)

        # 用每个batch中最大的节点嵌入表示图嵌入
        global_node_info = dgl.mean_nodes(batched_g, 'h')
        # if self.maxpool == 1:
        #     global_node_info = dgl.max_nodes(batched_g, 'h')
        #     # global_word_info = dgl.max_nodes(batched_wg, 'h')
        # else:
        #     global_node_info = dgl.mean_nodes(batched_g, 'h')
        #     # global_word_info = dgl.mean_nodes(batched_wg, 'h')
        # global_node_info = torch.cat((global_node_info, global_word_info), -1)

        embed_seq_tensor = torch.zeros(len(len_non_zero), self.seq_len, 1 * self.h_dim)
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
        pred = pred.reshape(-1, 1)
        # sc_num = [item[-1] for item in sc_num_list]
        sc_num = torch.Tensor(sc_num_list).cuda().reshape(-1, 1)
        if sc_num is not None:
            loss = self.criterion(pred, sc_num)
        else:
            loss = None
        return loss, pred, feature

    def evaluate(self, ent_list, sc_num):
        loss, pred, _ = self.predict(ent_list, sc_num)
        # prob_rel = self.out_func(pred.view(-1))
        # sorted_prob_rel, prob_rel_idx = prob_rel.sort(0, descending=True)  # 得到模型预测的排序好的事件概率和它们对应的事件index
        # if torch.cuda.is_available():
        #     sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()).cuda())
        # else:
        #     sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()))
        # nonzero_prob_idx = torch.nonzero(sorted_prob_rel,as_tuple=False).view(-1)  # 去零后的长度计量
        # nonzero_prob_rel_idx = prob_rel_idx[:len(nonzero_prob_idx)]  # 去零后的事件index
        #
        # # target
        # true_prob_r = true_prob_r.view(-1)
        # nonzero_rel_idx = torch.nonzero(true_prob_r,as_tuple=False) # (x,1)->(x)
        # sorted_true_rel, true_rel_idx = true_prob_r.sort(0, descending=True)
        # nonzero_true_rel_idx = true_rel_idx[:len(nonzero_rel_idx)]
        # return nonzero_true_rel_idx, nonzero_prob_rel_idx, loss
        return loss