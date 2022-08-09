import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import data_process
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

N = 10000
T = 51


# Check gpu

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# get data
def read_excel(path):
    workbook = pd.read_excel(path, engine='openpyxl')
    output = workbook.values
    return output

data_collection = ["eo-240-050-240_combined_MB"]

for i in data_collection:
    path = 'D:\CuiProject\PythonProject\Corner\eo\eo-240-050-240\_' + i + '.txt'
    data_exp, start_end = data_process.get_data(path)
    agent, neighbor = data_process.get_training(data_exp, start_end)
    neighbor = neighbor.reshape(N, 52, 36)
    # map = map.reshape(N, 127, 100)
    if data_collection.index(i) == 0:
        train_agent = agent
        train_neighbor = neighbor
        # train_map = map
    else:
        train_agent = np.concatenate((train_agent, agent), axis=0)
        train_neighbor = np.concatenate((train_neighbor, neighbor), axis=0)
        # train_map = np.concatenate((train_map, map), axis=0)




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
    #     self.linear1 = nn.Linear(36, 64)
    #     self.linear2 = nn.Linear(64, self.n_hidden2)
    #     self.linear3 = nn.Linear(self.n_hidden3, 16)
    #     self.linear4 = nn.Linear(16, 2)

        # self.relu = nn.ReLU(inplace=True)
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

        # self.linear_map = nn.Linear(100, 256)

        self.relu = nn.ReLU(inplace=True)

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

        outputs = torch.cat(outputs, dim=1)
        return outputs







if __name__ == "__main__":

    batch_size = 200


    train_agent = torch.from_numpy(train_agent).float().to(device)
    train_neighbor = torch.from_numpy(train_neighbor).float().to(device)
    # train_map = torch.from_numpy(train_map).float().to(device)

    train = TensorDataset(train_agent, train_neighbor)
    train_ds, valid_ds = torch.utils.data.random_split(train, [int(0.8 * N * len(data_collection)), int(0.2 * N * len(data_collection))])
    train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=batch_size, shuffle=True)





    model = LSTMPredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    n_epochs = 50
    for epoch in range(n_epochs):
        losses_train = []
        losses_valid = []
        # train
        for mini_batch in train_dl:
            optimizer.zero_grad()
            model.zero_grad()
            agent, neighbor = mini_batch
            agent_input = agent[:, :-1, :]
            agent_target = agent[:, 1:, :]
            neighbor = neighbor[:, :-1, :]
            # map = map[:, :-1, :]
            out = model.forward(agent_input, neighbor)
            loss = criterion(out, agent_target.reshape(batch_size, 102))
            loss.backward()
            optimizer.step()
            losses_train.append(loss.tolist())
        # valid
        for mini_batch in valid_dl:
            agent, neighbor = mini_batch
            agent_input = agent[:, :-1, :]
            agent_target = agent[:, 1:, :]
            neighbor = neighbor[:, :-1, :]
            # map = map[:, :-1, :]
            valid_out = model.forward(agent_input, neighbor)
            valid_loss = criterion(valid_out, agent_target.reshape(batch_size, 102))
            losses_valid.append(valid_loss.tolist())
        print('EPOCH: {}, Train Loss: {:.5f}, Valid Loss: {:.5f}'.format(
            epoch,
            np.mean(losses_train),
            np.mean(losses_valid), average='macro'))


    torch.save(model, "model_240_050.pth")