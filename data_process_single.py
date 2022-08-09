import codecs
import numpy as np
import random
import math
import xlsxwriter

straight_length = 2.5
radius = 1.85
width = 0.8
sample_num = 5000
import Map

def get_data(path):
    f = codecs.open(path, mode='r',
                    encoding='utf=8')
    line = f.readline()
    L = []
    while line:
        d = line.split()
        L.append(d)
        line = f.readline()
    f.close()
    del L[0:7]

    data_all = []
    data = []
    N = 1
    for i in range(len(L)):
        d = L[i]
        if int(L[i][0]) != N:
            N += 1
            data_all.append(data)
            data = []
        # data.append(Map.map([float(L[i][3]) / 100 + (straight_length + radius * 2 + width) / 2,
        #              (width + radius * 2) / 2 - float(L[i][2]) / 100]))
        data.append([float(L[i][3]) / 100 + (straight_length + radius * 2 + width) / 2,
                     (width + radius * 2) / 2 - float(L[i][2]) / 100])
    data_all.append(data)

    new_data = []
    new_data_all = []
    ini_pos = []
    egress_time = []
    for i in range(len(data_all)):

        t = 0
        for j in range(len(data_all[i])):
            if j % 1 == 0:
                new_data.append(data_all[i][j])
                t += 1

            if j == 0:
                ini_pos.append(data_all[i][j])
        new_data = new_data[250:2000]
        new_data_all.append(new_data)
        new_data = []
        egress_time.append(t)
    # return new_data_all, ini_pos, egress_time
    return new_data_all


# get k-nearest neighbours
# def k_nearest(k, index, list):
#     list = list.tolist()
#     agent = list[index]
#     del list[index]
#     D = []
#     output = []
#     for i in list:
#         d = np.linalg.norm(np.array(i) - np.array(agent))
#         D.append(d)
#     min_index = []
#     for _ in range(k):
#         number = min(D)
#         index = D.index(number)
#         D[index] = math.inf
#         min_index.append(index)
#         output.append(list[index])
#     return output

# get neighbor grid
def neighbor_grid(unit, k, index, list):
    interval_list = []
    for i in range(2 * k):
        interval = [(i-k) * unit, (i-k+1) * unit]
        interval_list.append(interval)
    list = list.tolist()
    agent = list[index]
    del list[index]
    output = np.zeros([2*k, 2*k])
    for i in list:
        dis = np.array(i) - np.array(agent)
        if dis[0] < -k * unit or dis[0] > k * unit or dis[1] < -k * unit or dis[1] > k * unit:
            continue
        for j in interval_list:
            if dis[0] >= j[0] and dis[0] <= j[1]:
                index_x = interval_list.index(j)
                break
        for j in interval_list:
            if dis[1] >= j[0] and dis[1] <= j[1]:
                index_y = interval_list.index(j)
                break
        # bug Feb. 22
        output[index_x, index_y] += 1
        # output[index_y, index_x] += 1
    output = output.reshape((2 * k) ** 2, 1)
    return output


def write_excel(raw_data, indicator):
    # Create an new Excel file and add a worksheet.
    workname = str(indicator) + '.xlsx'
    workbook = xlsxwriter.Workbook('D:\CuiProject\PythonProject\ModelFree_One2one\_ModelFree_' + workname)
    worksheet = workbook.add_worksheet()
    for i in range(len(raw_data)):
        if isinstance(raw_data[i], list):
            for j in range(len(raw_data[i])):
                worksheet.write(i, j, raw_data[i][j])
        else:
            worksheet.write(i, 0, raw_data[i])
    workbook.close()


# def get_training(data):
#     train_agent = []
#     train_neighbor = []
#     for i in range(sample_num):
#         print(i)
#         pwall1 = random.randint(0, data.shape[0] - 1)
#         pwall2 = random.randint(0, data.shape[1] - 127 - 1)
#         train_agent.append(data[pwall1,pwall2:pwall2+127,:])
#         neighbor_list = []
#         for j in range(127):
#             neighbor = k_nearest(2, pwall1, data[:,pwall2+j,:])
#             neighbor_list.append(neighbor)
#         train_neighbor.append(neighbor_list)
#     return np.array(train_agent), np.array(train_neighbor)

def get_training(data):
    train_agent = []
    train_neighbor = []
    for i in range(sample_num):
        print(i)
        # Short term: 127 Long term: 252
        pwall1 = random.randint(0, data.shape[0] - 1)
        pwall2 = random.randint(0, data.shape[1] - 127 - 1)
        train_agent.append(data[pwall1, pwall2:pwall2 + 127, :])
        neighbor_list = []
        for j in range(127):
            neighbor = neighbor_grid(0.5, 3, pwall1, data[:,pwall2+j,:])
            neighbor_list.append(neighbor)
        train_neighbor.append(neighbor_list)
        # pwall1 = random.randint(0, data.shape[0] - 1)
        # pwall2 = random.randint(0, data.shape[1] - 252 - 1)
        # train_agent.append(data[pwall1, pwall2:pwall2 + 252, :])
        # neighbor_list = []
        # for j in range(252):
        #     neighbor = neighbor_grid(0.5, 3, pwall1, data[:, pwall2 + j, :])
        #     neighbor_list.append(neighbor)
        # train_neighbor.append(neighbor_list)
    return np.array(train_agent), np.array(train_neighbor)


# # get data
# path = 'D:\CuiProject\PythonProject\Single-file\_trajectories\_1_1_05.txt'
# data_exp = get_data(path)
# train_input = np.array(data_exp)
#
# # # get train data
# # train_data, neighbor_data = get_training(train_input)
# # write_excel(train_data.reshape(sample_num, 254).tolist(), 'agent')
# # write_excel(neighbor_data.reshape(sample_num, 12700).tolist(), 'neighbor')
#
#
# # get test data
# data_exp = np.array(data_exp)
# write_excel(data_exp.reshape(data_exp.shape[0], 1600).tolist(), 'test_1_1_05')
