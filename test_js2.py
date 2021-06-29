import math
import random

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.spatial import distance
# import torch
import os

metric = 'std'
time = '1'
time2 = ''
std_lists = []
ts_operation = 'gb'
iter_index = ''

metric_dir = 't9102144'
class_name = 'n02128385'
class_items = os.listdir(metric_dir)
deleted_node_set = np.load(
    metric_dir + '/deleted_node_%sd_marker_%s_%s_set_%s.npy' % ('wa', 'toc', ts_operation, class_name))

wa_class_npy_dir = metric_dir + '/wa_class_npy'
wa_class_npy_files = os.listdir(wa_class_npy_dir)
wa_class_npy_file_list = []

keep_node_set = list(set(list(range(9216))) - set(deleted_node_set))
# keep_node_set = np.load(
#                 metric_dir + '/keep_node_set_pools2_wad_sc_time_5_one_list.npy').tolist()
# keep_node_set = np.load(
#     metric_dir + '/%skeep_node_file_list_%s_%s_%s_%s_%s.npy' % (iter_index,
#                                                                 'wa', class_name, ts_operation, 'zone',
#                                                                 'zero')).tolist()
# no_center_node_set = np.load('no_center_node_pools_wad_sc_zone_5_zero_list.npy').tolist()
# print(no_center_node_set)
# keep_node_set = np.load('metric_list_npy_dir_0817_3330/best_node_pools_wad_marker_toc_sc_set_n01728572.npy').tolist()
# keep_node_set = np.load(metric_dir+'/keep_node_pools_wad_marker_sc_set_n01728572.npy').tolist()

# print(keep_node_set.sort())

test_index2 = random.sample(keep_node_set, 1)
test_index = test_index2
test_index = [1572]
print(test_index)

all_wa_npy_file = np.load('t9102144/all_wa_file_list_n02128385_gb_zone_zero.npy')

print(all_wa_npy_file.shape)

for i in range(all_wa_npy_file.shape[0]):
    js_npy = all_wa_npy_file[i][:, test_index[0]]
    wa_class_npy_file_list.append(all_wa_npy_file[i][:, test_index[0]].tolist())
    std_lists.append(js_npy)

# for wa_class_npy_file in wa_class_npy_files:
#     if wa_class_npy_file.__contains__('wa%s_file_list_n02128385-n02128385_' % (time)):
#         wa_class_npy_file_npy = np.load(wa_class_npy_dir + '/' + wa_class_npy_file)
#
#         for i in range(len(test_index)):
#             js_npy = wa_class_npy_file_npy[:, test_index[i]]
#             wa_class_npy_file_list.append(wa_class_npy_file_npy[:, test_index[i]].tolist())
#             std_lists.append(js_npy)

for i in range(len(test_index)):
    js_npy = np.mean(np.array(wa_class_npy_file_list), axis=0)
    print(js_npy.shape)
    std_lists.append(js_npy)

data_list_len = int(len(std_lists))

inter_plot_height = int(math.sqrt(data_list_len))
inter_plot_width = int(data_list_len / inter_plot_height)
if inter_plot_height * inter_plot_width < data_list_len:
    inter_plot_width += 1

axs = []
fig = plt.figure()
for i in range(int(data_list_len)):
    axs.append(fig.add_subplot(inter_plot_height, inter_plot_width, i + 1))

y = std_lists
# o_wa_npy = np.load(
#     metric_dir + '/%so_pools%s_wa_file_list_%s_%s_zone_zero.npy' % (iter_index, time, class_name, ts_operation))
o_wa_npy = np.load(metric_dir + '/%so_wa_file_list_%s_%s_zone_zero.npy' % (iter_index, class_name, ts_operation))
for i in range(int(data_list_len)):
    x = list(range(len(std_lists[i])))
    axs[i].plot(x, y[i], color='red')
    axs[i].plot(x, o_wa_npy[:, test_index[0]], color='blue')
    # axs[i].plot(x, y[int(data_list_len) + i], color='blue')
    # axs[i].set_title(class_items[i])
    # print(np.mean(y[i]), '-', np.mean(y[int(data_list_len) + i]))
    # print()
# o_wa_npy = np.load(metric_dir + '/%so_wa_file_list_%s_%s_zone_zero.npy' % (iter_index, class_name, ts_operation))

y = o_wa_npy[:, test_index[0]]
x = list(range(len(y)))
axs[-1].plot(x, y, color='blue')

plt.legend()
plt.show()
