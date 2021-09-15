import datetime
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import config


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=2, output_size=1):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)  # 输入的特征维度，隐藏层维度，LSTM叠加层数
        self.out = nn.Linear(hidden_size, output_size)  # 线性回归

    def forward(self, x):
        x, _ = self.rnn(x)  # (seq, batch, hidden)
        s, b, h = x.shape
        x = x.view(s * b, h)  # 转换成线性层的输入格式（二维）
        x = self.out(x)
        x = x.view(s, b, -1)  # 将二维转换为三维，传入下一个LSTM
        return x


def get_time_line(start_time, end_time, gap=24):
    """
    按照gap（小时）分割时间线
    """
    time_lines = []
    s_time = datetime.datetime.strptime(start_time, '%Y-%m-%d')
    e_time = datetime.datetime.strptime(end_time, '%Y-%m-%d') + datetime.timedelta(days=1)
    while s_time < e_time:
        time_lines.append(s_time.strftime('%Y-%m-%d %H:%M:%S'))
        s_time += datetime.timedelta(hours=gap)
    time_lines.append(e_time.strftime('%Y-%m-%d %H:%M:%S'))
    return time_lines


def preprocess_raw(data_file, target_file):
    """
    :param data_file: 原始datafile路径
    :param target_file: 中间文件目标路径
    :return gap
    """
    data = pd.read_csv(data_file, encoding='utf-8', dtype=str)
    # 剔除时间为nan的值
    data.dropna(axis=0, inplace=True, subset=['createdAt'])
    # 获取到不同的天数
    diff_days = list(set(map(lambda x: x.split()[0], data['createdAt'].tolist())))
    diff_days.sort()
    # 天数低于3天，按照6h为时间线分割；低于6天，按照12h；否则按照24h
    if len(diff_days) < 3:
        gap = 6
    elif len(diff_days) < 6:
        gap = 12
    else:
        gap = 24
    time_lines = get_time_line(start_time=diff_days[0], end_time=diff_days[-1], gap=gap)

    line_counts = []
    for i in range(len(time_lines) - 1):
        t1, t2 = time_lines[i], time_lines[i + 1]
        count = data[(data['createdAt'] >= t1) & (data['createdAt'] < t2)].shape[0]
        line_counts.append({
            'date': t1,
            'real': count
        })
    # 转换为DataFrame并保存csv
    df = pd.DataFrame.from_dict(line_counts, orient='columns')
    df.to_csv(target_file, index=False, header=True, encoding='utf-8')
    return gap, diff_days


def creat_dataset(dataset, look_back=2):
    """
    数据集和目标值赋值，dataset为数据，look_back为以几行数据为特征维度数量
    """
    data_x, data_y = [], []
    for i in range(len(dataset) - look_back):
        data_x.append(dataset[i:i + look_back])
        data_y.append(dataset[i + look_back])
    return np.asarray(data_x), np.asarray(data_y)  # 转为ndarray数据


def train_model(model_path, csv_file='hk_data.csv', use_cuda=True, k=3, hidden_layers=100):
    # csv_file with headers (date, real_count)
    df = pd.read_csv(csv_file, delimiter=',', encoding='utf-8-sig', dtype=str)
    dataset = df['real'].values.astype(float)
    # 归一化处理，不然后面训练数据误差会很大
    max_value = np.max(dataset)
    min_value = np.min(dataset)
    scalar = max_value - min_value
    dataset = list(map(lambda x: x / scalar, dataset))

    # 以k为特征维度，得到数据集
    data_x, data_y = creat_dataset(dataset, k)
    train_size = int(len(data_x) * 0.7)
    x_train = data_x[:train_size]  # 训练数据
    y_train = data_y[:train_size]  # 训练数据目标值
    x_train = x_train.reshape(-1, 1, k)  # 将训练数据调整成pytorch中lstm算法的输入维度
    y_train = y_train.reshape(-1, 1, 1)  # 将目标值调整成pytorch中lstm算法的输出维度
    # 将ndarray数据转换为张量
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    model = LSTM(input_size=k, hidden_size=hidden_layers)

    if use_cuda:
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        model.cuda()

    # 参数寻优，计算损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    loss_func = nn.MSELoss()

    # 训练模型
    for epoch in range(1000):
        var_x = Variable(x_train).type(torch.FloatTensor)
        var_y = Variable(y_train).type(torch.FloatTensor)
        if use_cuda:
            var_x = var_x.cuda()
            var_y = var_y.cuda()
        # 前向传播
        out = model(var_x)
        loss = loss_func(out, var_y)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print('Epoch:{}, Loss:{:.5f}'.format(epoch + 1, loss.item()))

    # 转换为测试模式
    model = model.eval()
    test_x = data_x.reshape(-1, 1, k)
    test_x = torch.from_numpy(test_x)

    if use_cuda:
        test_x = test_x.cuda()
        var_test = Variable(test_x).type(torch.FloatTensor).cuda()
    else:
        var_test = Variable(test_x).type(torch.FloatTensor)

    predictions = model(var_test)  # 测试集的预测结果
    predictions = predictions.cpu().view(-1).data.numpy()  # 转换成一维的ndarray数据，这是预测值
    # model_name = topicId+"_model.pt"
    # torch.save(model, model_path)

    # 将归一化后的值转换回去
    df['predict'] = df['real'].tolist()[:k] + list(map(lambda p: str(int(p * scalar)), predictions))
    return df


# def predict(model_path,csv_file='hk_data.csv', use_cuda=True, k=3, hidden_layers=50):
#     # csv_file with headers (date, real_count)
#     df = pd.read_csv(csv_file, delimiter=',', encoding='utf-8-sig', dtype=str)
#     dataset = df['real'].values.astype(float)
#
#     # 归一化处理，不然后面训练数据误差会很大
#     max_value = np.max(dataset)
#     min_value = np.min(dataset)
#     scalar = max_value - min_value
#     dataset = list(map(lambda x: x / scalar, dataset))
#
#     # 以k为特征维度，得到数据集
#     data_x, data_y = creat_dataset(dataset, k)
#
#     x_train = data_x # 训练数据
#     y_train = data_y  # 训练数据目标值
#
#     x_train = x_train.reshape(-1, 1, k)  # 将训练数据调整成pytorch中lstm算法的输入维度
#     y_train = y_train.reshape(-1, 1, 1)  # 将目标值调整成pytorch中lstm算法的输出维度
#
#     # 将ndarray数据转换为张量
#     x_train = torch.from_numpy(x_train)
#     y_train = torch.from_numpy(y_train)
#
#     model = LSTM(input_size=k, hidden_size=hidden_layers)
#
#     if use_cuda:
#         x_train = x_train.cuda()
#         y_train = y_train.cuda()
#         model.cuda()
#
#     # 参数寻优，计算损失函数
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
#     loss_func = nn.MSELoss()
#     # model_name = topicId + "_model.pt"
#     model = torch.load(model_path)
#     var_x = Variable(x_train).type(torch.FloatTensor)
#     var_y = Variable(y_train).type(torch.FloatTensor)
#     out = model(var_x).tolist()
#     return {'real': df['real'].values.tolist(), 'predict': [int(i[0][0] * scalar) for i in out]}


def main(topicId):
    data_file = config.file_dir + topicId + "-status.csv"
    target_file = config.attention_base_path + "" + topicId + ".csv"
    # gap, diff_days用于将预测结果按照day合并
    gap, diff_days = preprocess_raw(data_file, target_file)
    model_path = config.model_path + topicId + "_model.pt"
    res_df = train_model(model_path, target_file, use_cuda=False)
    # gap不是按照24h则要还原每天的预测结果
    if gap != 24:
        res_df['index'] = pd.to_datetime(res_df['date'])
        res_df.set_index('index', inplace=True)
        res = list(map(lambda day: {
            'date': day,
            'real': sum(res_df[day]['real'].astype(int).tolist()),
            'predict': sum(res_df[day]['predict'].astype(int).tolist())
        }, diff_days))
    else:
        res_df['date'] = res_df['date'].apply(lambda x: x.split()[0])
        res = res_df.to_dict(orient='records')
    # res = predict(model_path,target_file,use_cuda=False)
    return res


# if __name__ == '__main__':
#     # 原始数据预处理
#     preprocess_raw(data_file, target_file)
#     # 模型训练及预测
#     for i in range(6):
#         result = train_model(csv_file=str(i + 1), use_cuda=False, k=2, hidden_layers=100)
#         print(result)
#     print(predict(use_cuda=False))
# res = main("5ebbf329-d6b6-4f53-a088-953881808864")
# print(res)
# print("Fi")
