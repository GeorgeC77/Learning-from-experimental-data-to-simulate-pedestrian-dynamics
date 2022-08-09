import random
import math

import matplotlib.pyplot as plt
import numpy as np
from pylab import *
from numpy import *
import matplotlib.animation as animation
import pandas as pd

# Floor plan, unit: meter
room_length = 7.0
room_size = 5.6


# path_test = 'D:\CuiProject\PythonProject\ModelFree_bottleneck\_40.xlsx'
path_test = 'D:\CuiProject\PythonProject\ModelFree_corner_One2one\_300-150-300\_300-150-300.xlsx'
path_startend = 'D:\CuiProject\PythonProject\ModelFree_corner_One2one\_300-150-300\_300-150-300_startend.xlsx'
# get data
def read_excel(path):
    workbook = pd.read_excel(path, engine='openpyxl')
    output = workbook.values
    return output

# test = np.array(read_excel(path_test).reshape(42, 1014, 2))
# test = np.array(read_excel(path_test).reshape(42, 508, 2))
test = np.array(read_excel(path_test).reshape(137, 179, 2))
start_end = np.array(read_excel(path_startend).reshape(137, 2))
# Matplotlib based visualization

def update(n):
    agents = []
    for j in range(len(start_end)):
        if start_end[j][0] <= n and start_end[j][1] >= n:
            agents.append(test[j, n - start_end[j][0], :])

    if agents:
        agents = np.array(agents)
        x = agents[:, 0] / 100
        y = agents[:, 1] / 100
        points.set_data(x, y)
    # ax.scatter(x,y, c='r')

    return points,
fig, ax = plt.subplots()
font = {'family' : 'Times New Roman',
  'weight' : 'normal',
  'size'   : 15,
  }
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
plt.plot([-3,4], [-3,-3], color='k', linewidth=3)
plt.plot([-3,-3], [-3,4], color='k', linewidth=3)
ax.set_xlabel('x', font)
ax.set_ylabel('y', font)
ax.set_aspect('equal')
plt.grid()
# plt.plot([-room_size/2,-room_size/2], [room_length,0], color='k', linewidth=3)
# plt.plot([room_size/2,room_size/2], [room_length,0], color='k', linewidth=3)
# plt.plot([-room_size/2,-0.25], [0,0], color='k', linewidth=3)
# plt.plot([room_size/2,0.25], [0,0], color='k', linewidth=3)
plt.plot([0,0], [0,4], color='k', linewidth=3)
plt.plot([0,4], [0,0], color='k', linewidth=3)
points, = plt.plot(test[:, 0, 0], test[:, 0, 1], "ro")



plt.yticks(fontproperties = 'Times New Roman', size=15)
plt.xticks(fontproperties = 'Times New Roman', size=15)
plt.title('Ground truth\n$b_{Korr}=3.0$ m, $b_{Zu}=1.5$ m', font)

ani = animation.FuncAnimation(fig, update, np.arange(53, 1000), interval=100, blit=True)
ani.save('gt_3_150.gif', writer='imagemagick', fps=10)
# ani.save('scatter.mp4')
plt.show()