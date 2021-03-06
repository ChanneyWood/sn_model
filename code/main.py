def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

import argparse
import time
import os
from model import *
from data import *
import joblib
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='')
parser.add_argument("--dp", type=str, default="../data/middle_data/", help="data path")
parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
parser.add_argument("--n-hidden-1", type=int, default=10, help="number of hidden units")
parser.add_argument("--n-hidden-2", type=int, default=5, help="number of hidden units")
parser.add_argument("--gpu", type=int, default=0, help="gpu")
parser.add_argument("--lr", type=float, default=5e-3, help="learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-3, help="weight_decay")
parser.add_argument("--dataset", type=str, default='weibo', help="dataset to use")
parser.add_argument("--date", type=str, default='9.15', help="dataset to use")
parser.add_argument("--grad-norm", type=float, default=1.0, help="norm to clip gradient to")
parser.add_argument("--max-epochs", type=int, default=5, help="maximum epochs")
parser.add_argument("--seq-len", type=int, default=8, help="reference days of training")
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--rnn-layers", type=int, default=1)
parser.add_argument("--maxpool", type=int, default=1)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--use-lstm", type=int, default=1, help='1 use lstm 0 gru')
parser.add_argument("--attn", type=str, default='', help='dot/add/genera; default general')
parser.add_argument("--seed", type=int, default=42, help='random seed')
parser.add_argument("--runs", type=int, default=1, help='number of runs')

args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
use_cuda = args.gpu >= 0 and torch.cuda.is_available()
print("cuda", use_cuda)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# eval metrics
recall_list = []
f1_list = []
f2_list = []
hloss_list = []

iterations = 0
while iterations < args.runs:
    iterations += 1
    print('****************** iterations ', iterations, )
    file_prefix = 'loss_graph_num_gap_'
    if iterations == 1:
        print("loading data...")
        rel_g_list = joblib.load(args.dp + "rel_graph_dict_full")
        sc_num_list = joblib.load(args.dp + "slice_sc_num_all")
        # rel_g_list = joblib.load(args.dp + "rel_graph_dict_sample")
        # sc_num_list = joblib.load(args.dp + "slice_sc_num_sample")
        nodes_map = joblib.load(args.dp + "all_nodes_idx")
        num_nodes = len(nodes_map)
        data_len = len(sc_num_list)
        s_idx = np.arange(data_len)
        s_idx = s_idx.tolist()
        np.random.shuffle(s_idx)
        s_rel_g_list = [rel_g_list[idx] for idx in s_idx]
        s_sc_num_list = [sc_num_list[idx] for idx in s_idx]
        # ?????????????????????????????????????????????????????????
        sc_tensor = torch.Tensor(s_sc_num_list)
        line_tensor = sc_tensor.reshape(-1)
        max_value = max(line_tensor)
        min_value = min(line_tensor)
        scalar = max_value - min_value
        sc_tensor = sc_tensor / scalar
        s_sc_num_list = sc_tensor.tolist()

        train_set = (s_rel_g_list[0:3000], s_sc_num_list[0:3000])
        valid_set = (s_rel_g_list[3000:4000], s_sc_num_list[3000:4000])
        test_set = (s_rel_g_list[4000:], s_sc_num_list[4000:])

        # train_set = (s_rel_g_list[0:30], s_sc_num_list[0:30])
        # valid_set = (s_rel_g_list[30:40], s_sc_num_list[30:40])
        # test_set = (s_rel_g_list[40:], s_sc_num_list[40:])

        train_dataset_loader = DistData(
            train_set, set_name='train')
        valid_dataset_loader = DistData(
            valid_set, set_name='valid')
        test_dataset_loader = DistData(
            test_set, set_name='test')

        train_loader = DataLoader(train_dataset_loader, batch_size=args.batch_size,
                                  shuffle=True, collate_fn=collate_2)
        valid_loader = DataLoader(valid_dataset_loader, batch_size=1,
                                  shuffle=False, collate_fn=collate_2)
        test_loader = DataLoader(test_dataset_loader, batch_size=1,
                                 shuffle=False, collate_fn=collate_2)

    model = grsce(h_dim_1=args.n_hidden_1,
                  h_dim_2=args.n_hidden_2,
                  seq_len=args.seq_len,
                  dropout=args.dropout,
                  use_lstm=args.use_lstm,
                  scalar=scalar)

    model_name = model.__class__.__name__
    print('Model:', model_name)
    token = file_prefix + '{}_sl{}'.format(model_name, args.seq_len)
    print('Token:', token, args.dataset)

    # optimizer = torch.optim.SGD(
    #     model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('#params:', total_params)

    os.makedirs('models', exist_ok=True)
    os.makedirs('models/' + args.dataset, exist_ok=True)
    os.makedirs('models/{}/{}'.format(args.dataset, args.date), exist_ok=True)
    model_state_file = 'models/{}/{}/{}.pth'.format(args.dataset, args.date, token)
    model_result_file = 'models/{}/{}/{}_result.jpg'.format(args.dataset, args.date, token)
    # outf = 'models/{}/{}.result'.format(args.dataset, token)
    if use_cuda:
        model.cuda()

    model.node_map = nodes_map


    @torch.no_grad()
    def evaluate(data_loader, dataset_loader, set_name='valid'):
        model.eval()
        total_loss = 0
        for i, batch in enumerate(tqdm(data_loader)):
            rel_g, sc_num = batch
            loss = model.evaluate(rel_g, sc_num)
            total_loss += loss.item()

        print('{} results'.format(set_name))
        reduced_loss = total_loss / (dataset_loader.len / 1.0)
        print("{} Loss: {:.6f}".format(set_name, reduced_loss))
        return reduced_loss


    def train(data_loader, dataset_loader):
        model.train()
        total_loss = 0
        t0 = time.time()
        for i, batch in enumerate(tqdm(data_loader)):
            rel_g, sc_num = batch
            loss = model(rel_g, sc_num)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_norm)  # clip gradients
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        t2 = time.time()
        reduced_loss = total_loss / (dataset_loader.len / args.batch_size)
        print("Epoch {:04d} | Loss {:.6f} | time {:.2f} {}".format(
            epoch, reduced_loss, t2 - t0, time.ctime()))
        return reduced_loss


    bad_counter = 0
    loss_small = float("inf")
    train_loss_list, valid_loss_list, test_loss_list = [], [], []
    try:
        print("start training...")
        for epoch in range(1, args.max_epochs + 1):
            train_loss = train(train_loader, train_dataset_loader)

            valid_loss = evaluate(valid_loader, valid_dataset_loader, set_name='Valid')  # eval on train set

            if valid_loss < loss_small:
                loss_small = valid_loss
                bad_counter = 0
                print('save better model, loss={}'.format(str(loss_small)))
                torch.save({'state_dict': model.state_dict()},
                           model_state_file)
                test_loss = evaluate(test_loader, test_dataset_loader, set_name='Test')
            else:
                bad_counter += 1
            train_loss_list.append(train_loss)
            valid_loss_list.append(valid_loss)
            test_loss_list.append(test_loss)

        print("training done")
        import matplotlib.pyplot as plt

        x = range(1, args.max_epochs + 1)
        plt.plot(x, train_loss_list, label='train loss')
        plt.plot(x, valid_loss_list, label='valid loss')
        plt.plot(x, test_loss_list, label='test loss')
        plt.savefig(model_result_file, dip=1200)
        plt.show()

    except KeyboardInterrupt:
        print('-' * 80)
        print('Exiting from training early, epoch', epoch)

    # checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage)
    # model.load_state_dict(checkpoint['state_dict'])
    # if use_cuda:
    #     model.cuda()
    # evaluate(train_loader, train_dataset_loader, set_name='Valid')

    # # Load the best saved model.
    # print("\nstart testing...")
    # checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage)
    # model.load_state_dict(checkpoint['state_dict'])
    # print("Using best epoch: {}".format(checkpoint['epoch']))
    # hloss, recall, f1, f2 = evaluate(test_loader, test_dataset_loader, set_name='Test')
    # print(args)
    # print(token, args.dataset)
    # recall_list.append(recall)
    # f1_list.append(f1)
    # f2_list.append(f2)
    # hloss_list.append(hloss)
