import math
import random

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.spatial import distance
# import torch
import os

metric = 'std'
time = '2'
time2 = ''
std_lists = []
ts_operation = 'sc'
iter_index = ''
combination_type = ''
plot_type = ''

random_class = ['n03982430', 'n03876231', 'n03874599', 'n03775546', 'n03527444', 'n02799071', 'n02877765',
                'n03937543', 'n04447861', 'n12985857', 'n12144580', 'n02128385', 'n02085620', 'n02437616',
                'n02454379',
                'n01667778', 'n01806143', 'n01871265', 'n07714990', 'n11939491',
                'n02398521']

# random_class = ['n02437616', 'n02454379',
#                 'n01667778', 'n01806143', 'n01871265', 'n07714990', 'n11939491',
#                 'n02398521']
#
# random_class = ['n03982430', 'n03876231', 'n03874599', 'n03775546', 'n03527444', 'n02877765',
#                 'n03937543', 'n04447861', 'n12985857', 'n12144580']
# #
# random_class = ['n02229544',
#                 'n02783161',
#                 'n03041632',
#                 'n02098286',
#                 'n02493509',
#                 'n02087394',
#                 'n04591157',
#                 'n07871810',
#                 'n02085782',
#                 'n07613480']

# random_class = []
# butterfly = ['n02276258']  # 6
# cat = ['n02123045']  # 7
# leopard = ['n02128385']  # 3
# dog = ['n02085620']  # 1
# fish = ['n01443537']  # 6
# bird = ['n02002724']  # 8
# random_class.extend(butterfly)
# random_class.extend(cat)
# random_class.extend(leopard)
# random_class.extend(dog)
# random_class.extend(fish)
# random_class.extend(bird)

metric_dir = 't_norm_relu_1014'
class_name = 'n02085782'
#
# class_name = random_class[-1]
# class_name = 'n01530575'
print('class_name: ', class_name)

class_items = os.listdir(metric_dir)
deleted_node_set = np.load(
    metric_dir + '/deleted_node_%sd_marker_%s_%s_set_%s.npy' % ('wa', 'toc', ts_operation, class_name))

output_size = 7009
# output_size = 43264

keep_node_set = list(set(list(range(output_size))) - set(deleted_node_set))

# keep_node_set = np.load(
#     metric_dir + '/keep_node_pools%s_%sd_marker_%s_set_%s.npy' % (
#         str(int(time) - 1), 'wa', ts_operation, class_name)).tolist()
# keep_node_set = np.load(
#     metric_dir + '/best_node_pools%s_%sd_marker_%s_%s_set_%s.npy' % (
#         time, 'wa', 'toc', ts_operation, class_name)).tolist()
# no_center_node_set = np.load('no_center_node_pools_wad_sc_zone_5_zero_list.npy').tolist()
# print(no_center_node_set)
# keep_node_set = np.load('metric_list_npy_dir_0817_3330/best_node_pools_wad_marker_toc_sc_set_n01728572.npy').tolist()
# keep_node_set = np.load(metric_dir+'/keep_node_pools_wad_marker_sc_set_n01728572.npy').tolist()

# print(keep_node_set.sort())

sample_index_set = keep_node_set
if combination_type == 'cov':
    sample_index_set = range(len(keep_node_set))

test_index2 = random.sample(sample_index_set, 1)
# test_index2 = random.sample(range(len(keep_node_set)), 8)
test_index = [300]
# test_index= [6601, 909, 4981, 1572, 5084, 303, 6644, 5606]
# test_index = np.array(keep_node_set)[test_index2]

if time == '1':
    wa_npy = np.load(metric_dir + '/%swa_file_list_%s_%s_zone_zero.npy' % (iter_index, class_name, ts_operation))
    # wa_npy = np.load(metric_dir + '/all_%swa_file_list_%s_%s_zone_zero.npy' % (iter_index, class_name, ts_operation))[12]
    # wa_npy = np.load(metric_dir + '/wa_file_list_n02128385-n02128385_6519_sc_sc_zone_zero.npy')
    o_wa_npy = np.load(metric_dir + '/%so_wa_file_list_%s_%s_zone_zero.npy' % (iter_index, class_name, ts_operation))
    # wa_npy = np.load(metric_dir + '/i_pools%s_wa_file_list_%s_%s_zone_zero.npy' % (time2, class_name, ts_operation))
    # o_wa_npy = np.load(metric_dir + '/o_pools%s_wa_file_list_%s_%s_zone_zero.npy' % (time2, class_name, ts_operation))
else:
    wa_npy = np.load(
        metric_dir + '/%si_pools%s_wa_file_list_%s_%s_zone_zero.npy' % (
            iter_index, str(int(time) - 1), class_name, ts_operation))
    o_wa_npy = np.load(
        metric_dir + '/%so_pools%s_wa_file_list_%s_%s_zone_zero.npy' % (
            iter_index, str(int(time) - 1), class_name, ts_operation))

i_p_wa_npy = np.load(
    metric_dir + '/%si_pools%s_wa_file_list_%s_%s_zone_zero.npy' % (iter_index, time, class_name, ts_operation))
o_p_wa_npy = np.load(
    metric_dir + '/%so_pools%s_wa_file_list_%s_%s_zone_zero.npy' % (iter_index, time, class_name, ts_operation))

if combination_type == 'cov':
    i_p_wa_npy = np.load(
        metric_dir + '/%swa_cov%s_file_list_%s_%s_zone_zero.npy' % (iter_index, time, class_name, ts_operation))
    o_p_wa_npy = np.load(
        metric_dir + '/%so_wa_cov%s_file_list_%s_%s_zone_zero.npy' % (iter_index, time, class_name, ts_operation))

print(keep_node_set)

# for i in keep_node_set:
#     print(np.mean((o_wa_npy[:, i] + 1) / (wa_npy[:, i] + 1)))

print('keep_node_len: ', len(keep_node_set))
i_count = 0
io_count = 0

origin_ratio = []
combination_ratio = []
constant_plus = 1
for index in sample_index_set:
    # for index in range(len(keep_node_set)):
    origin_ratio.append(np.mean(o_wa_npy[:, index] + constant_plus) / (np.mean(wa_npy[:, index] + constant_plus)))
    combination_ratio.append(
        np.mean(o_p_wa_npy[:, index] + constant_plus) / (np.mean(i_p_wa_npy[:, index] + constant_plus)))
    if np.mean(wa_npy[:, index]) < np.mean(o_wa_npy[:, index]):
        i_count += 1
    if np.mean(o_p_wa_npy[:, index]) > np.mean(i_p_wa_npy[:, index]):
        io_count += 1

print(origin_ratio)
print(combination_ratio)

print('i_count: ', i_count)
print('io_count: ', io_count)

print(test_index)
# for class_item in class_items:
#     js_npy = np.load(metric_dir + '/wa_file_list_%s_%szone_zero.npy' % (class_item,ts_operation))[:, 2001][0:100]
#     std_lists.append(js_npy)

if plot_type == 'ratio':
    std_lists.append(origin_ratio)
    std_lists.append(combination_ratio)
else:
    for i in range(len(test_index)):
        js_npy = wa_npy[:,
                 test_index[i]]
        std_lists.append(js_npy)
    for i in range(len(test_index)):
        js_npy = i_p_wa_npy[:,
                 test_index[i]]
        std_lists.append(js_npy)
    for i in range(len(test_index)):
        js_npy = o_wa_npy[:,
                 test_index[i]]
        std_lists.append(js_npy)
    for i in range(len(test_index)):
        js_npy = o_p_wa_npy[:,
                 test_index[i]]
        std_lists.append(js_npy)

# js_npy = np.load(metric_dir + '/wa_file_list_%s_%s_zone_zero.npy' % (class_name, ts_operation))[:, test_index[0]]
# std_lists.append(js_npy)
# js_npy = np.load(metric_dir + '/wa_file_list_%s_%s_zone_zero.npy' % (class_name, ts_operation))[:, test_index[1]]
# std_lists.append(js_npy)
# js_npy = np.load(metric_dir + '/wa_file_list_%s_%s_zone_zero.npy' % (class_name, ts_operation))[:, test_index[2]]
# std_lists.append(js_npy)
#
# # js_npy = np.load(metric_dir + '/i_pools%s_wa_file_list_%s_%s_zone_zero.npy' % (time2, class_name, ts_operation))[:,
# #          test_index[0]]
# # std_lists.append(js_npy)
# # js_npy = np.load(metric_dir + '/i_pools%s_wa_file_list_%s_%s_zone_zero.npy' % (time2, class_name, ts_operation))[:,
# #          test_index[1]]
# # std_lists.append(js_npy)
# # js_npy = np.load(metric_dir + '/i_pools%s_wa_file_list_%s_%s_zone_zero.npy' % (time2, class_name, ts_operation))[:,
# #          test_index[2]]
# # std_lists.append(js_npy)
#
# js_npy = np.load(metric_dir + '/i_pools%s_wa_file_list_%s_%s_zone_zero.npy' % (time, class_name, ts_operation))[:,
#          test_index[0]]
# std_lists.append(js_npy)
# js_npy = np.load(metric_dir + '/i_pools%s_wa_file_list_%s_%s_zone_zero.npy' % (time, class_name, ts_operation))[:,
#          test_index[1]]
# std_lists.append(js_npy)
# js_npy = np.load(metric_dir + '/i_pools%s_wa_file_list_%s_%s_zone_zero.npy' % (time, class_name, ts_operation))[:,
#          test_index[2]]
# std_lists.append(js_npy)
#
# js_npy = np.load(metric_dir + '/o_wa_file_list_%s_%s_zone_zero.npy' % (class_name, ts_operation))[:, test_index[0]]
# std_lists.append(js_npy)
# js_npy = np.load(metric_dir + '/o_wa_file_list_%s_%s_zone_zero.npy' % (class_name, ts_operation))[:, test_index[1]]
# std_lists.append(js_npy)
# js_npy = np.load(metric_dir + '/o_wa_file_list_%s_%s_zone_zero.npy' % (class_name, ts_operation))[:, test_index[2]]
# std_lists.append(js_npy)
#
# # js_npy = np.load(metric_dir + '/o_pools%s_wa_file_list_%s_%s_zone_zero.npy' % ('', class_name, ts_operation))[:,
# #          test_index[0]]
# # std_lists.append(js_npy)
# # js_npy = np.load(metric_dir + '/o_pools%s_wa_file_list_%s_%s_zone_zero.npy' % ('', class_name, ts_operation))[:,
# #          test_index[1]]
# # std_lists.append(js_npy)
# # js_npy = np.load(metric_dir + '/o_pools%s_wa_file_list_%s_%s_zone_zero.npy' % ('', class_name, ts_operation))[:,
# #          test_index[2]]
# # std_lists.append(js_npy)
# # js_npy = np.load(metric_dir + '/o_pools%s_wa_file_list_%s_%s_zone_zero.npy' % (time2, class_name, ts_operation))[:,
# #          test_index[0]]
# # std_lists.append(js_npy)
# # js_npy = np.load(metric_dir + '/o_pools%s_wa_file_list_%s_%s_zone_zero.npy' % (time2, class_name, ts_operation))[:,
# #          test_index[1]]
# # std_lists.append(js_npy)
# # js_npy = np.load(metric_dir + '/o_pools%s_wa_file_list_%s_%s_zone_zero.npy' % (time2, class_name, ts_operation))[:,
# #          test_index[2]]
# # std_lists.append(js_npy)
#
# js_npy = np.load(metric_dir + '/o_pools%s_wa_file_list_%s_%s_zone_zero.npy' % (time, class_name, ts_operation))[:,
#          test_index[0]]
# std_lists.append(js_npy)
# js_npy = np.load(metric_dir + '/o_pools%s_wa_file_list_%s_%s_zone_zero.npy' % (time, class_name, ts_operation))[:,
#          test_index[1]]
# std_lists.append(js_npy)
# js_npy = np.load(metric_dir + '/o_pools%s_wa_file_list_%s_%s_zone_zero.npy' % (time, class_name, ts_operation))[:,
#          test_index[2]]
# std_lists.append(js_npy)
# print(np.mean(std_lists,axis=1))
# js_npy = np.load(metric_dir + '/o_wa_file_list_%s_%s_zone_zero.npy' % ('n03527444', ts_operation))[:, test_index[0]]
# std_lists.append(js_npy)
# js_npy = np.load(metric_dir + '/o_wa_file_list_%s_%s_zone_zero.npy' % ('n03527444', ts_operation))[:, test_index[1]]
# std_lists.append(js_npy)
# js_npy = np.load(metric_dir + '/o_wa_file_list_%s_%s_zone_zero.npy' % ('n03527444', ts_operation))[:, test_index[2]]
# std_lists.append(js_npy)

data_list_len = int(len(std_lists) / 2)

print(data_list_len)

inter_plot_height = int(math.sqrt(data_list_len))
inter_plot_width = int(data_list_len / inter_plot_height)
if inter_plot_height * inter_plot_width < data_list_len:
    inter_plot_width += 1

axs = []
fig = plt.figure()
# for i in range(int(data_list_len)):
#     axs.append(fig.add_subplot(inter_plot_height, inter_plot_width, i + 1))

y = std_lists
x = list(range(len(std_lists[0])))

axs.append(fig.add_subplot(121))
axs[0].plot(x[0:200], y[0][0:200], color='blue', linestyle='--', label=r'$W_{in}$')
axs[0].plot(x[0:200], y[int(data_list_len) + 0][0:200], color='black', linestyle='-', label=r'$W_{out}$')
# axs[i].set_xlabel('')
axs[0].set_ylabel('W 距离', fontsize=21)
axs[0].set_xlabel('图片编号', fontsize=21)
# axs[i].set_xticks(size=18)
# axs[i].set_yticks(size=18)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.legend(prop={'size': 16})
# plt.setp(axs, {'size': 21})
plt.yticks(size=17)
plt.xticks(size=17)
axs.append(fig.add_subplot(122))

axs[1].plot(x[0:200], y[1][0:200], color='blue', linestyle='--', label=r'$W_{in}$')
axs[1].plot(x[0:200], y[int(data_list_len) + 1][0:200], color='black', linestyle='-', label=r'$W_{out}$')
# axs[i].set_xlabel('')
axs[1].set_ylabel('W 距离', fontsize=21)
axs[1].set_xlabel('图片编号', fontsize=21)
# axs[i].set_xticks(size=18)
# axs[i].set_yticks(size=18)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.legend(prop={'size': 16})
# plt.setp(axs, {'size': 21})
plt.yticks(size=17)
plt.xticks(size=17)


axs[0].set_title('(e)节点' + str(test_index[0]) + '组合前', y=-0.2, fontsize=21)
axs[1].set_title('(f)节点' + str(test_index[0]) + '组合后', y=-0.2, fontsize=21)
# axs[i].set_title(class_items[i])
# print(np.mean(y[i]), '-', np.mean(y[int(data_list_len) + i]))
# print()
# plt.axhline(y=1, ls=":", c="black")
# plt.ylim(0,10)

# plt.legend([s1[0]],['组合前'])
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.legend(prop={'size': 21})
# plt.yticks(size=18)
# plt.xticks(size=18)
plt.show()
