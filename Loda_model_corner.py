import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import data_process
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

T = 1
# Check gpu

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


# get data
def read_excel(path):
    workbook = pd.read_excel(path, engine='openpyxl')
    output = workbook.values
    return output


class LSTMPredictor(nn.Module):
    # def __init__(self, n_hidden1=16, n_hidden2=72, n_hidden3=32):
    #     super(LSTMPredictor, self).__init__()
    #     self.n_hidden1 = n_hidden1
    #     self.n_hidden2 = n_hidden2
    #     self.n_hidden3 = n_hidden3
    #     # lstm1, lstm2, linear1， linear2, linear3
    #     self.lstm1 = nn.LSTMCell(2, self.n_hidden1)
    #     self.lstm2 = nn.LSTMCell(self.n_hidden2, self.n_hidden2)
    #     self.lstm3 = nn.LSTMCell(self.n_hidden1 + self.n_hidden2, self.n_hidden3)
    #
    #     self.linear1 = nn.Linear(100, 64)
    #     self.linear2 = nn.Linear(64, self.n_hidden2)
    #     self.linear3 = nn.Linear(self.n_hidden3, 16)
    #     self.linear4 = nn.Linear(16, 2)
    #
    #     self.relu = nn.ReLU(inplace=True)
    # def __init__(self, n_hidden1=32, n_hidden2=256, n_hidden3=256, n_hidden_map=256):
    #     super(LSTMPredictor, self).__init__()
    #     self.n_hidden1 = n_hidden1
    #     self.n_hidden2 = n_hidden2
    #     self.n_hidden3 = n_hidden3
    #     # self.n_hidden_map = n_hidden_map
    #     # lstm1, lstm2, linear1， linear2, linear3
    #     self.lstm1 = nn.LSTMCell(2, self.n_hidden1)
    #     self.lstm2 = nn.LSTMCell(self.n_hidden2, self.n_hidden2)
    #     self.lstm3 = nn.LSTMCell(self.n_hidden1 + self.n_hidden2, self.n_hidden3)
    #     # self.lstm_map = nn.LSTMCell(256, self.n_hidden_map)
    #
    #     self.linear1 = nn.Linear(36, 128)
    #     self.linear2 = nn.Linear(128, self.n_hidden2)
    #     self.linear3 = nn.Linear(self.n_hidden3, 128)
    #     self.linear4 = nn.Linear(128, 2)
    #
    #     # self.linear_map = nn.Linear(100, 256)
    #
    #     self.relu = nn.ReLU(inplace=True)
    def __init__(self, n_hidden1=32, n_hidden2=128, n_hidden3=128, n_hidden_map=256):
        super(LSTMPredictor, self).__init__()
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_hidden3 = n_hidden3
        # self.n_hidden_map = n_hidden_map
        # lstm1, lstm2, linear1， linear2, linear3
        self.lstm1 = nn.LSTMCell(2, self.n_hidden1)
        self.lstm2 = nn.LSTMCell(self.n_hidden2, self.n_hidden2)
        self.lstm3 = nn.LSTMCell(self.n_hidden1 + self.n_hidden2, self.n_hidden3)
        # self.lstm_map = nn.LSTMCell(256, self.n_hidden_map)

        self.linear1 = nn.Linear(36, 72)
        self.linear2 = nn.Linear(72, self.n_hidden2)
        self.linear3 = nn.Linear(self.n_hidden3, 36)
        self.linear4 = nn.Linear(36, 2)

    def forward(self, x, y):
        outputs = []
        n_sample = x.size(0)
        seq_length = x.size(1)

        h_t = torch.zeros(n_sample, self.n_hidden1, dtype=torch.float32).to(device)
        c_t = torch.zeros(n_sample, self.n_hidden1, dtype=torch.float32).to(device)
        h_t2 = torch.zeros(n_sample, self.n_hidden2, dtype=torch.float32).to(device)
        c_t2 = torch.zeros(n_sample, self.n_hidden2, dtype=torch.float32).to(device)
        h_t3 = torch.zeros(n_sample, self.n_hidden3, dtype=torch.float32).to(device)
        c_t3 = torch.zeros(n_sample, self.n_hidden3, dtype=torch.float32).to(device)

        for t in range(seq_length):
            h_t, c_t = self.lstm1(x[:, t, :], (h_t, c_t))
            l1 = self.linear1(y[:, t, :])
            l2 = self.linear2(self.relu(l1))
            h_t2, c_t2 = self.lstm2(l2, (h_t2, c_t2))
            h = torch.cat((h_t, h_t2), 1)
            h_t3, c_t3 = self.lstm3(h, (h_t3, c_t3))
            l3 = self.linear3(h_t3)
            output = self.linear4(self.relu(l3))
            outputs.append(output)
        outputs = torch.cat(outputs, dim=1)
        return outputs

    def pred(self, x, future=0):
        outputs = []
        agent_num = x.size(0)
        len_seq = T
        h_t = torch.zeros(agent_num, self.n_hidden1, dtype=torch.float32).to(device)
        c_t = torch.zeros(agent_num, self.n_hidden1, dtype=torch.float32).to(device)
        h_t2 = torch.zeros(agent_num, self.n_hidden2, dtype=torch.float32).to(device)
        c_t2 = torch.zeros(agent_num, self.n_hidden2, dtype=torch.float32).to(device)
        h_t3 = torch.zeros(agent_num, self.n_hidden3, dtype=torch.float32).to(device)
        c_t3 = torch.zeros(agent_num, self.n_hidden3, dtype=torch.float32).to(device)

        for t in range(len_seq):
            h_t, c_t = self.lstm1(x[:, t, :], (h_t, c_t))
            neighbor_list = []
            for j in range(agent_num):
                neighbor = data_process.neighbor_grid(0.5, 3, j, x[:, t, :])
                neighbor_list.append(neighbor)
            y = torch.from_numpy(np.array(neighbor_list).reshape(agent_num, 1, 36)).float().to(device)
            l1 = self.linear1(y[:, 0, :])
            l2 = self.linear2(self.relu(l1))
            h_t2, c_t2 = self.lstm2(l2, (h_t2, c_t2))
            h = torch.cat((h_t, h_t2), 1)
            h_t3, c_t3 = self.lstm3(h, (h_t3, c_t3))
            l3 = self.linear3(h_t3)
            output = self.linear4(self.relu(l3))

        outputs.append(output)

        for k in range(future):
            for j in range(len(start_end)):
                if k + 1 >= start_end[j][0] - start_end[0][0] and k + 1 < start_end[j][0] - start_end[0][0] + 5:
                    output[j, :] = test[j, k + 1 - (start_end[j][0] - start_end[0][0]), :]
                if k + 1 == start_end[j][0] - start_end[0][0]:
                    h_t[j, :], c_t[j, :], h_t2[j, :], c_t2[j, :], h_t3[j, :], c_t3[j, :] = 0, 0, 0, 0, 0, 0
                # if k + 1 == start_end[j][0] - start_end[0][0]:
                #     output[j, :] = test[j, 0, :]
                #     h_t[j, :], c_t[j, :], h_t2[j, :], c_t2[j, :], h_t3[j, :], c_t3[j, :] = 0, 0, 0, 0, 0, 0



            h_t, c_t = self.lstm1(output, (h_t, c_t))
            neighbor_list = []
            for j in range(agent_num):
                neighbor = data_process.neighbor_grid(0.5, 3, j, output)
                neighbor_list.append(neighbor)
            y = torch.from_numpy(np.array(neighbor_list).reshape(agent_num, 1, 36)).float().to(device)
            l1 = self.linear1(y[:, 0, :])
            l2 = self.linear2(self.relu(l1))
            h_t2, c_t2 = self.lstm2(l2, (h_t2, c_t2))
            h = torch.cat((h_t, h_t2), 1)
            h_t3, c_t3 = self.lstm3(h, (h_t3, c_t3))
            l3 = self.linear3(h_t3)
            output = self.linear4(self.relu(l3))
            outputs.append(output)

            for i in range(len(output)):
                if output[i][1] > 350:
                    output[i] = np.nan

        outputs = torch.cat(outputs, dim=1)
        return outputs



load_model = torch.load("model_240_050.pth")

path_test = 'D:\CuiProject\PythonProject\ModelFree_corner_One2one\_240-050-240\_240-050-240.xlsx'
path_startend = 'D:\CuiProject\PythonProject\ModelFree_corner_One2one\_240-050-240\_240-050-240_startend.xlsx'
test = np.array(read_excel(path_test).reshape(64, 119, 2))
test = torch.from_numpy(test).float()

start_end = np.array(read_excel(path_startend).reshape(64, 2))

x = np.empty((test.shape[0], 1000, 2))
x[:] = numpy.nan

for i in range(len(start_end)):
    for j in range(x.shape[1]):
        if j + 1 == start_end[i][0]:
            x[i, j + 1:j + 6, :] = test[i, 0:5, :]
            break

# for i in range(len(start_end)):
#     for j in range(x.shape[1]):
#         if j + 1 == start_end[i][0]:
#             x[i, j + 1, :] = test[i, 0, :]
#             break

x = torch.from_numpy(x[:, 61:, :]).float().to(device)





with torch.no_grad():
    future = 1000
    pred = load_model.pred(x, future=future)
    data_process.write_excel(pred.tolist(), 'results')
