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
path_test = 'D:\CuiProject\PythonProject\ModelFree_corner_One2one\_240-050-240\_results.xlsx'
path_startend = 'D:\CuiProject\PythonProject\ModelFree_corner_One2one\_240-050-240\_240-050-240_startend.xlsx'
# get data
def read_excel(path):
    workbook = pd.read_excel(path, engine='openpyxl')
    output = workbook.values
    return output

# test = np.array(read_excel(path_test).reshape(42, 1014, 2))
# test = np.array(read_excel(path_test).reshape(42, 508, 2))
test = np.array(read_excel(path_test).reshape(64, 910, 2))
# Matplotlib based visualization

def update(n):
    x = test[:,n,0] / 100
    y = test[:,n,1] / 100
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
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
plt.plot([-3,4], [-3,-3], color='k', linewidth=3)
plt.plot([-3,-3], [-3,4], color='k', linewidth=3)
ax.set_aspect('equal')
ax.set_xlabel('x', font)
ax.set_ylabel('y', font)
plt.grid()
plt.plot([0,0], [0,4], color='k', linewidth=3)
plt.plot([0,4], [0,0], color='k', linewidth=3)
points, = plt.plot(test[:,0,0] / 100, test[:,0,1] / 100, "ro")

plt.yticks(fontproperties = 'Times New Roman', size=15)
plt.xticks(fontproperties = 'Times New Roman', size=15)
plt.title('LSTM\n$b_{Korr}=3.0$ m, $b_{Zu}=1.5$ m', font)

ani = animation.FuncAnimation(fig, update, np.arange(0, 910), interval=100, blit=True)
# ani.save('lstm_30_150.gif', writer='imagemagick', fps=10)
# ani.save('scatter.mp4')
plt.show()