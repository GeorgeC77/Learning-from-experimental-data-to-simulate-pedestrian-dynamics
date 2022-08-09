import codecs
import numpy as np
import random
import math
import xlsxwriter
import Map

sample_num = 10000

# # generate map
# grid_map = np.ones((38, 50))
#
# for i in range(grid_map.shape[0]):
#     for j in range(grid_map.shape[1]):
#         if (i + 1) * 0.2 > (room_size / 2 + 1 - 0.25) and (i + 1) * 0.2 < (room_size / 2 + 1 + 0.25):
#             grid_map[i][j] = 0
#         elif (i + 1) * 0.2 > (3.8 - room_size / 2) and (i + 1) * 0.2 < (3.8 + room_size / 2) and (j+1) * 0.2 > 2:
#             grid_map[i][j] = 0




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
    del L[0:0]

    data_all = []
    data = []
    start_end = []
    start = int(L[0][1])
    N = 1
    for i in range(len(L)):
        d = L[i]
        if int(L[i][0]) != N:
            N += 1
            data_all.append(data)
            data = []
            # record the start point and end point
            end = int(L[i - 1][1])
            start_end.append([start, end])
            start = int(L[i][1])
        # data.append([float(L[i][2])+room_size/2, room_length-float(L[i][3])])
        data.append([float(L[i][2]), float(L[i][3])])
    end = int(L[len(L) - 1][1])
    start_end.append([start, end])

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
        # new_data = new_data[250:1750]
        new_data_all.append(new_data)
        new_data = []
        egress_time.append(t)
    # return new_data_all, ini_pos, egress_time
    return new_data_all, start_end


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
        if dis[0] < -k * unit or dis[0] > k * unit or dis[1] < -k * unit or dis[1] > k * unit or np.isnan(np.sum(dis)):
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
    output = output.reshape((2 * k) ** 2, 1)
    return output


# x = []
# y = []
# z = []
#
# for i in range(10):
#     x = np.array([random.uniform(-2.8, 2.8), random.uniform(0, 7)])
#     y.append(x)
#     print(x)
#     z.append(occupancy_map(x, room_size))

def write_excel(raw_data, indicator):
    # Create a new Excel file and add a worksheet.
    workname = str(indicator) + '.xlsx'
    workbook = xlsxwriter.Workbook('D:\CuiProject\PythonProject\ModelFree_corner_One2one\_' + workname)
    worksheet = workbook.add_worksheet()
    for i in range(len(raw_data)):
        if isinstance(raw_data[i], list):
            for j in range(len(raw_data[i])):
                if not np.isnan(raw_data[i][j]):
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

def get_training(data, start_end):
    train_agent = []
    train_neighbor = []
    train_map = []
    for i in range(sample_num):
        print(i)
        # Short term: 127 Long term: 252
        pwall1 = random.randint(0, len(data) - 1)
        pwall2 = random.randint(start_end[pwall1][0], start_end[pwall1][1] - 52 - 1)
        train_agent.append(data[pwall1][pwall2 - start_end[pwall1][0]:pwall2 - start_end[pwall1][0] + 52])
        neighbor_list = []
        map_list = []
        for j in range(52):
            agents = []
            for k in range(len(start_end)):
                if start_end[k][0] <= pwall2 + j and start_end[k][1] >= pwall2 + j:
                    agents.append(data[k][pwall2 - start_end[k][0] + j])
            neighbor = neighbor_grid(0.5, 3, agents.index(data[pwall1][pwall2 - start_end[pwall1][0] + j]), np.array(agents))
            neighbor_list.append(neighbor)
            # map = occupancy_map(data[pwall1,pwall2+j,:], room_size)
            # map_list.append(map)
        train_neighbor.append(neighbor_list)
        # train_map.append(map_list)
        # pwall1 = random.randint(0, data.shape[0] - 1)
        # pwall2 = random.randint(0, data.shape[1] - 252 - 1)
        # train_agent.append(data[pwall1, pwall2:pwall2 + 252, :])
        # neighbor_list = []
        # for j in range(252):
        #     neighbor = neighbor_grid(0.5, 3, pwall1, data[:, pwall2 + j, :])
        #     neighbor_list.append(neighbor)
        # train_neighbor.append(neighbor_list)
    return np.array(train_agent), np.array(train_neighbor)


# def get_training(data, start_end):
#     train_agent = []
#     train_neighbor = []
#     train_map = []
#     # for i in range(len(data)):
#     for i in range(100):
#         print(i)
#         for j in range(10):
#             pwall2 = random.randint(start_end[i][0], start_end[i][1] - 52 - 1)
#             train_agent.append(data[i][pwall2 - start_end[i][0]:pwall2 - start_end[i][0] + 52])
#             neighbor_list = []
#             map_list = []
#             for j in range(52):
#                 agents = []
#                 for k in range(len(start_end)):
#                     if start_end[k][0] <= pwall2 + j and start_end[k][1] >= pwall2 + j:
#                         agents.append(data[k][pwall2 - start_end[k][0] + j])
#                 neighbor = neighbor_grid(0.5, 3, agents.index(data[i][pwall2 - start_end[i][0] + j]),
#                                          np.array(agents))
#                 neighbor_list.append(neighbor)
#
#             train_neighbor.append(neighbor_list)
#
#
#     # for i in range(sample_num):
#     #     print(i)
#     #     # Short term: 127 Long term: 252
#     #     pwall1 = random.randint(0, len(data) - 1)
#     #     pwall2 = random.randint(start_end[pwall1][0], start_end[pwall1][1] - 52 - 1)
#     #     train_agent.append(data[pwall1][pwall2 - start_end[pwall1][0]:pwall2 - start_end[pwall1][0] + 52])
#     #     neighbor_list = []
#     #     map_list = []
#     #     for j in range(52):
#     #         agents = []
#     #         for k in range(len(start_end)):
#     #             if start_end[k][0] <= pwall2 + j and start_end[k][1] >= pwall2 + j:
#     #                 agents.append(data[k][pwall2 - start_end[k][0] + j])
#     #         neighbor = neighbor_grid(0.5, 3, agents.index(data[pwall1][pwall2 - start_end[pwall1][0] + j]), np.array(agents))
#     #         neighbor_list.append(neighbor)
#     #         # map = occupancy_map(data[pwall1,pwall2+j,:], room_size)
#     #         # map_list.append(map)
#     #     train_neighbor.append(neighbor_list)
#     #     # train_map.append(map_list)
#     #     # pwall1 = random.randint(0, data.shape[0] - 1)
#     #     # pwall2 = random.randint(0, data.shape[1] - 252 - 1)
#     #     # train_agent.append(data[pwall1, pwall2:pwall2 + 252, :])
#     #     # neighbor_list = []
#     #     # for j in range(252):
#     #     #     neighbor = neighbor_grid(0.5, 3, pwall1, data[:, pwall2 + j, :])
#     #     #     neighbor_list.append(neighbor)
#     #     # train_neighbor.append(neighbor_list)
#     return np.array(train_agent), np.array(train_neighbor)

# def get_training(data):
#     train_agent = []
#     train_neighbor = []
#     for i in range(sample_num):
#         print(i)
#         # Short term: 127 Long term: 252
#         pwall1 = random.randint(0, len(data) - 1)
#
#         while len(data[pwall1]) < 127:
#             pwall1 = random.randint(0, len(data) - 1)
#         pwall2 = random.randint(0, len(data[pwall1]) - 127)
#         train_agent.append(np.array(data[pwall1])[pwall2:pwall2 + 127, :])
#         neighbor_list = []
#         for j in range(127):
#
#             neighbor = neighbor_grid(0.5, 3, pwall1, data[:,pwall2+j,:])
#             neighbor_list.append(neighbor)
#         train_neighbor.append(neighbor_list)
#         # pwall1 = random.randint(0, data.shape[0] - 1)
#         # pwall2 = random.randint(0, data.shape[1] - 252 - 1)
#         # train_agent.append(data[pwall1, pwall2:pwall2 + 252, :])
#         # neighbor_list = []
#         # for j in range(252):
#         #     neighbor = neighbor_grid(0.5, 3, pwall1, data[:, pwall2 + j, :])
#         #     neighbor_list.append(neighbor)
#         # train_neighbor.append(neighbor_list)
#     return np.array(train_agent), np.array(train_neighbor)

# # get data
# path = 'D:\CuiProject\PythonProject\Corner\eo\eo-300-150-300\_eo-300-150-300_combined_MB.txt'
# data_exp, start_end = get_data(path)
#
# # get test data
# # data_exp = np.array(data_exp)
# data = []
# for element in data_exp:
#     data.append(np.array(element).reshape(len(element) * 2).tolist())
#
#
# write_excel(data, '300-150-300')
# write_excel(start_end, '300-150-300_startend')
