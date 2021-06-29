import math

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.spatial import distance
# import torch
import os

metric = 'std'
std_lists = []
metric_dir = 'metric_list_npy_dir_0201'
class_items = os.listdir(metric_dir)
for class_item in class_items:
    std_npy = np.load(metric_dir + '/%s/metric_wad_%s_sc_zone_5_zero_list.npy' % (class_item, metric))
    order_arr = np.load(
        metric_dir + '/%s/order_arr_metric_wad_%s_sc_zone_5_zero_list.npy' % (class_item, metric))
    std_list = []
    for i in order_arr:
        std_list.append(std_npy[i])
    std_lists.append(std_list)
    break

inter_plot_height = int(math.sqrt(len(std_lists)))
inter_plot_width = int(len(std_lists) / inter_plot_height)
if inter_plot_height * inter_plot_width < len(std_lists):
    inter_plot_width += 1

axs = []
fig = plt.figure()
for i in range(len(std_lists)):
    axs.append(fig.add_subplot(inter_plot_height, inter_plot_width, i + 1))

x = list(range(len(std_lists[0])))
std_lists = np.array(std_lists)
std_lists = std_lists ** 2.5 / 100
std_lists = std_lists.tolist()
y = std_lists

for i in range(len(std_lists)):
    axs[i].plot(x, y[i], color='black')
    # axs[i].set_title()
    axs[i].set_xlabel(u"节点编号", fontsize=18)
    axs[i].set_ylabel(u"标准差", fontsize=18)

# axs[0].set_title(class_items[0],y=-0.14, fontsize=21)
# axs[1].set_title('(b)'+class_items[1],y=-0.14, fontsize=21)
    plt.yticks(size=16)
    plt.xticks(size=16)
plt.legend(prop={'size': 18})
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.legend()
plt.show()
