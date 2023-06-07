import csv
 
filename='/home/hongyi/gpgpu-sim_hasp/figure/Results.csv'
data = []
with open(filename) as csvfile:
    csv_reader = csv.reader(csvfile)
    header = next(csv_reader)
    for row in csv_reader:
        data.append(row)

print(header)
print(data)

import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.default'] = 'regular'
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

COLOR = ["blue", "cornflowerblue", "mediumturquoise", "goldenrod", "yellow"]
lambda1 = [10, 11, 12, 13, 14, 15, 20]
lambda2 = [2,  3,  4,  5,  6,  7,  8,  9]
# x, y: position
# x = list(range(len(lambda1)))
# y = list(range(len(lambda2)))
x_tickets = [str(_x) for _x in lambda1]
y_tickets = [str(_x) for _x in lambda2]
y_tickets[0] = 'share'

start = 248000
# color_list = []
# for i in range(len(y)):
#     c = COLOR[i]
#     color_list.append([c] * len(x))
# color_list = np.asarray(color_list)

blue = '#0f6bae'
pink = '#C6CDFF'
color_list = [blue,blue,blue,blue,blue,blue,blue,blue,blue,blue,blue,pink,pink,pink,pink,pink,pink,pink]
xx_flat = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 14, 14, 14, 14, 14, 14, 14]
yy_flat = [ 2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  3,  4,  5,  6,  7,  8,  9]
acc_flat = []
for i in range(1, len(header)):
    acc_flat.append(float(data[6][i]) - start)
print(acc_flat)
acc_min = min(acc_flat)
for i in range(len(acc_flat)):
    if (acc_flat[i] == acc_min):
        color_list[i] = '#964ec2'

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# 座标轴名
# ax.set_ylabel("S1 Shader / 30")
# ax.set_xlabel("S1 Memory Partition / 12")
# ax.set_zlabel("Latency / Cycle")
# ax.set_xticks(lambda1)
# ax.set_xticklabels(x_tickets)
# ax.set_yticks(lambda2)
# ax.set_yticklabels(y_tickets)
# 座标轴范围
# ax.set_zlim(0, 260000)
from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((0,0)) 
ax.zaxis.set_major_formatter(formatter)

ax.bar3d(yy_flat, xx_flat, start, 0.7, 0.7, acc_flat,
            color=color_list,
            edgecolor="black",
            shade=True)

# 保存
plt.tight_layout()
fig.savefig("bar3d.png", bbox_inches='tight', pad_inches=0)
fig.show()
