import argparse
import numpy as np
import time
import utils
import os
from sklearn.utils import shuffle
from model import *
from data import *
import joblib
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='')
parser.add_argument("--dp", type=str, default="../data/yifang/", help="data path")
parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
parser.add_argument("--n-hidden-1", type=int, default=10, help="number of hidden units")
parser.add_argument("--n-hidden-2", type=int, default=5, help="number of hidden units")
parser.add_argument("--gpu", type=int, default=0, help="gpu")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight_decay")
parser.add_argument("-d", "--dataset", type=str, default='weibo', help="dataset to use")
parser.add_argument("--grad-norm", type=float, default=1.0, help="norm to clip gradient to")
parser.add_argument("--max-epochs", type=int, default=30, help="maximum epochs")
parser.add_argument("--seq-len", type=int, default=8, help="reference days of training")
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--rnn-layers", type=int, default=1)
parser.add_argument("--maxpool", type=int, default=1)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--use-lstm", type=int, default=1, help='1 use lstm 0 gru')
parser.add_argument("--attn", type=str, default='', help='dot/add/genera; default general')
parser.add_argument("--seed", type=int, default=42, help='random seed')
args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
use_cuda = args.gpu >= 0 and torch.cuda.is_available()
print("cuda", use_cuda)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

print("loading data...")
rel_g_list = joblib.load(args.dp + "final_x_list")
sc_num_list = joblib.load(args.dp + "final_y_list")
data_len = len(sc_num_list)
s_idx = np.arange(data_len)
s_idx = s_idx.tolist()
np.random.shuffle(s_idx)
s_rel_g_list = [rel_g_list[idx] for idx in s_idx]
s_sc_num_list = [sc_num_list[idx] for idx in s_idx]

test_set = (s_rel_g_list, s_sc_num_list)
test_dataset_loader = DistData(
    test_set, set_name='test')
test_loader = DataLoader(test_dataset_loader, batch_size=1,
                         shuffle=False, collate_fn=collate_2)
num_nodes = 1

model = grsce(h_dim_1=args.n_hidden_1,
              h_dim_2=args.n_hidden_2,
              num_nodes=num_nodes,
              seq_len=args.seq_len,
              dropout=args.dropout,
              use_lstm=args.use_lstm,
              attn=args.attn)


@torch.no_grad()
def evaluate(data_loader, dataset_loader, set_name='valid'):
    model.eval()
    for i, batch in enumerate(tqdm(data_loader)):
        rel_g, sc_num = batch
        pred, _ = model.predict(rel_g)
        pred = pred.tolist()[0][0]
        pred_sc = sc_num[0][-1]*(1+math.fabs(pred))
        print(pred_sc)


model_name = model.__class__.__name__
print('Model:', model_name)
token = 'acc_graph_{}_sl{}_max{}_list{}_attn{}'.format(model_name, args.seq_len, int(args.maxpool),
                                                       int(args.use_lstm),
                                                       str(args.attn))
print('Token:', token, args.dataset)

model_state_file = 'models/{}/{}.pth'.format(args.dataset, token)
checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['state_dict'])
if use_cuda:
    model.cuda()
evaluate(test_loader, test_dataset_loader, set_name='Test')
