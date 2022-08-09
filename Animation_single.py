import random
import math

import matplotlib.pyplot as plt
import numpy as np
from pylab import *
from numpy import *
import matplotlib.animation as animation
import pandas as pd

# Floor plan, unit: meter
straight_length = 2.5
radius = 1.85
width = 0.8
left_center = np.array([width / 2 + radius, width / 2 + radius])
right_center = np.array([width / 2 + radius + straight_length, width / 2 + radius])

# # gt
# path_test = 'D:\CuiProject\PythonProject\ModelFree_One2one\_1_1_04\_ModelFree_test_1_1_04.xlsx'
# lstm
path_test = 'D:\CuiProject\PythonProject\ModelFree_One2one\_ModelFree_results.xlsx'
# get data
def read_excel(path):
    workbook = pd.read_excel(path, engine='openpyxl')
    output = workbook.values
    return output

# test = np.array(read_excel(path_test).reshape(11, 876, 2))
# test = test[:, 876:1377, :]

test = np.array(read_excel(path_test).reshape(33, 751, 2))
test = test[:, :-1, :]

# test = np.array(read_excel(path_test).reshape(33, 800, 2))
# test = test[:, 25:775, :]
# Matplotlib based visualization

def update(n):
    x = test[:,n,0]
    y = test[:,n,1]
    # x = x[0:16:5]
    # y = y[0:16:5]
    # ax.scatter(x,y, c='r')
    points.set_data(x, y)
    return points,
fig, ax = plt.subplots()
font = {'family' : 'Times New Roman',
  'weight' : 'normal',
  'size'   : 15,
  }
ax.set_xlim(-0.25, 2 * radius + width + straight_length + 0.25)
ax.set_ylim(-0.25, 2 * radius + width + 0.25)
ax.set_xlabel('x', font)
ax.set_ylabel('y', font)
plt.grid()
plt.plot([radius + width / 2,radius + width / 2 + straight_length], [0,0], color='k', linewidth=3)
plt.plot([radius + width / 2,radius + width / 2 + straight_length], [width,width], color='k', linewidth=3)
plt.plot([radius + width / 2,radius + width / 2 + straight_length], [2 * radius,2 * radius], color='k', linewidth=3)
plt.plot([radius + width / 2,radius + width / 2 + straight_length], [2 * radius + width,2 * radius + width], color='k', linewidth=3)
theta = np.arange(np.pi / 2, 3/2*np.pi, 0.01)
x = left_center[0] + (radius - width / 2) * np.cos(theta)
y = left_center[1] + (radius - width / 2) * np.sin(theta)
plt.plot(x,y, color='k', linewidth=3)
x = left_center[0] + (radius + width / 2) * np.cos(theta)
y = left_center[1] + (radius + width / 2) * np.sin(theta)
plt.plot(x,y, color='k', linewidth=3)
theta = np.arange(-np.pi / 2, np.pi / 2, 0.01)
x = right_center[0] + (radius - width / 2) * np.cos(theta)
y = right_center[1] + (radius - width / 2) * np.sin(theta)
plt.plot(x,y, color='k', linewidth=3)
x = right_center[0] + (radius + width / 2) * np.cos(theta)
y = right_center[1] + (radius + width / 2) * np.sin(theta)
plt.plot(x,y, color='k', linewidth=3)
points, = plt.plot(test[:,0,0], test[:,0,1], "ro")

plt.yticks(fontproperties = 'Times New Roman', size=15)
plt.xticks(fontproperties = 'Times New Roman', size=15)
# plt.title('$N=17$', font)
# plt.title('Ground truth\n$N=17$', font)
plt.title('LSTM\n$N=33$', font)
ani = animation.FuncAnimation(fig, update, np.arange(0, 750), interval=100, blit=True)
# ani.save('gt_1104.gif', writer='imagemagick', fps=10)
ani.save('lstm_1104.gif', writer='imagemagick', fps=10)
# ani.save('scatter.mp4')
plt.show()