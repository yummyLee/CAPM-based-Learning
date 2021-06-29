import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.spatial import distance
# import torch


def get_average(m_list):
    m_sum = 0
    for item in m_list:
        m_sum += item
    return m_sum / len(m_list)


step = 11

x = list(range(step))
class_list = []
param2_list = []
param3_list = []
deleted_node_len_list = []
o_acc_list = []
f_acc_list = []
o_top_k_acc_list = []
f_top_k_acc_list = []
dnop_list = []
mt_list = []
tsop_list = []

sub_plot_num = 1

random_class = ['n03982430', 'n03876231', 'n03874599', 'n03775546', 'n03527444', 'n02799071', 'n02877765',
                'n03937543', 'n04447861', 'n12985857', 'n12144580']

with open('acc_analysis_deleted_node2_alexnet_transform_s0606_2204_5_0.00.txt', 'r') as f1:
    infos = f1.readlines()
    for i in range(0, len(infos)):
        info = infos[i].split(' ')
        class_list.append(info[0])
        param2_list.append(int(info[1]))
        param3_list.append(int(info[2]))
        deleted_node_len_list.append(int(info[3]))
        o_acc_list.append(float(info[4]))
        o_top_k_acc_list.append(float(info[5]))
        f_acc_list.append(float(info[6]))
        f_top_k_acc_list.append(float(info[7]))
        dnop_list.append(info[8])
        mt_list.append(info[9])
        tsop_list.append(info[10])

inter_plot_height = int(math.sqrt(sub_plot_num))
inter_plot_width = int(sub_plot_num / inter_plot_height)
if inter_plot_height * inter_plot_width < sub_plot_num:
    inter_plot_width += 1

axs = []
fig = plt.figure()
for i in range(sub_plot_num):
    axs.append(fig.add_subplot(inter_plot_height, inter_plot_width, i + 1))

for sub_plot_index in range(sub_plot_num):
    print(step * (sub_plot_index + 1))

    axs[sub_plot_index].plot(x, o_acc_list[step * sub_plot_index:step * (sub_plot_index + 1)], color='black')
    axs[sub_plot_index].plot(x, f_acc_list[step * sub_plot_index:step * (sub_plot_index + 1)], color='blue')
    # ax1.plot(x, o_top_k_acc_list[step * 0:step * (sub_plot_index+1)], color='black', linestyle="--")
    # ax1.plot(x, f_top_k_acc_list[step * 0:step * (sub_plot_index+1)], color='red', linestyle="--")
    # ax1.plot(x, f_plus_d_top_k_acc_list[step * 0:step * (sub_plot_index+1)], color='blue', linestyle="--")
    axs[sub_plot_index].set_title(
        '%d, %d, %d, %.4f, %.4f %s %s %s ' % (
            param2_list[step * sub_plot_index], param3_list[step * sub_plot_index],
            get_average(deleted_node_len_list[step * sub_plot_index:step * (sub_plot_index + 1)]),
            get_average(o_acc_list[step * sub_plot_index:step * (sub_plot_index + 1)]),
            get_average(f_acc_list[step * sub_plot_index:step * (sub_plot_index + 1)]),
            dnop_list[step * sub_plot_index], mt_list[step * sub_plot_index], tsop_list[step * sub_plot_index]))

# sub_plot_index = 3
#
# axs[0].plot(x, o_acc_list[step * sub_plot_index:step * (sub_plot_index + 1)], color='black', linestyle='--')
# axs[0].plot(x, f_acc_list[step * sub_plot_index:step * (sub_plot_index + 1)], color='blue', linestyle='--')
# axs[0].set_title(
#     '%d, %d, %d, %.4f, %.4f %s %s %s ' % (
#         param2_list[step * sub_plot_index], param3_list[step * sub_plot_index],
#         get_average(deleted_node_len_list[step * sub_plot_index:step * (sub_plot_index + 1)]),
#         get_average(o_acc_list[step * sub_plot_index:step * (sub_plot_index + 1)]),
#         get_average(f_acc_list[step * sub_plot_index:step * (sub_plot_index + 1)]),
#         dnop_list[step * sub_plot_index], mt_list[step * sub_plot_index], tsop_list[step * sub_plot_index]))

plt.legend()
plt.show()

# inter_set_num_list = []
# all_set = set(list(range(9216)))
#
# for random_class_item in random_class:
#
#     for tsop in ['sc']:
#
#         deleted_node_list = []
#
#         for metric in ['js', 'wa']:
#             deleted_node_list.append(
#                 np.load('wa_js_analysis/deleted_node_%sd_marker_toc_%s_set_%s.npy' % (metric, tsop, random_class_item)))
#
#         inter_set_num_list.append(
#             len((all_set - set(deleted_node_list[0])).intersection((all_set - set(deleted_node_list[1])))))
#
#         print(inter_set_num_list[-1])


# 1374
# 1232
# 1221
# 1473
# 1287
# 1380
# 1327
# 1341
# 1393
# 1536
# 473


# 1484
# 1315
# 1293
# 1525
# 1371
# 1441
# 1402
# 1457
# 1511
# 1664
# 1463
