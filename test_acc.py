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


# t1 = torch.load('ch5_change-001.pt')
# t2 = torch.load('ch5_change-002.pt')
#
#
# def js_divergence(p, q):
#     M = (p + q) / 2
#     return 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)
#
#
# print(js_divergence(t1, t2))

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

# x = range(9216)

# npy = np.load('test_jsd_t1_11_2212.npy')
# print(npy.shape)
# # ax1.plot(x, npy)
# ax1.plot(x[0:500], npy[0:500])
#
# npy = np.load('test_jsd_t1_11_2238.npy')
# print(npy.shape)
# ax2.plot(x[0:500], npy[0:500])
# # ax2.plot(x, npy)
#
# npy = np.load('test_jsd_t1_11_2243.npy')
# print(npy.shape)
# ax3.plot(x[0:500], npy[0:500])

# npy = np.load('test_jsd_t1_14_2223.npy')
# print(npy.shape)
# ax4.plot(x[0:500], npy[0:500])

# x = range(9216)
# ax1 = fig.add_subplot(221)
# ax2 = fig.add_subplot(222)
# npy = np.load('test_jsd_t1_25_1541.npy')
# plt.plot(x[0:500], npy[0:500])


step = 10
sub_plot = 4
x = range(step)
class_list = []
param2_list = []
param3_list = []
deleted_node_len_list = []
deleted_node_plus_d_len_list = []
o_acc_list = []
f_acc_list = []
f_plus_d_acc_list = []
o_top_k_acc_list = []
f_top_k_acc_list = []
f_plus_d_top_k_acc_list = []

with open('acc_analysis_deleted_node2_alexnet_transform_s0606_1340_5_0.00.txt', 'r') as f1:
    infos = f1.readlines()
    for i in range(0, len(infos)):
        info = infos[i].split(' ')
        class_list.append(info[0])
        param2_list.append(int(info[1]))
        param3_list.append(int(info[2]))
        deleted_node_len_list.append(int(info[3]))
        deleted_node_plus_d_len_list.append(int(info[4]))
        o_acc_list.append(float(info[5]))
        o_top_k_acc_list.append(float(info[6]))
        f_acc_list.append(float(info[7]))
        f_top_k_acc_list.append(float(info[8]))
        f_plus_d_acc_list.append(float(info[9]))
        f_plus_d_top_k_acc_list.append(float(info[10]))

# print(o_acc_list)
ax1.plot(x, o_acc_list[step * 0:step * 1], color='black')
ax1.plot(x, f_acc_list[step * 0:step * 1], color='red')
ax1.plot(x, f_plus_d_acc_list[step * 0:step * 1], color='blue')
# ax1.plot(x, o_top_k_acc_list[step * 0:step * 1], color='black', linestyle="--")
# ax1.plot(x, f_top_k_acc_list[step * 0:step * 1], color='red', linestyle="--")
# ax1.plot(x, f_plus_d_top_k_acc_list[step * 0:step * 1], color='blue', linestyle="--")
ax1.set_title(
    '%d, %d, %d, %d, %.4f, %.4f, %.4f' % (
        param2_list[0], param3_list[step * 0], get_average(deleted_node_len_list[step * 0:step * 1]),
        get_average(deleted_node_plus_d_len_list[step * 0:step * 1]),
        get_average(o_acc_list[step * 0:step * 1]),
        get_average(f_acc_list[step * 0:step * 1]), get_average(f_plus_d_acc_list[step * 0:step * 1])))

if sub_plot >= 2:
    ax2.plot(x, o_acc_list[step * 1:step * 2], color='black')
    ax2.plot(x, f_acc_list[step * 1:step * 2], color='red', )
    ax2.plot(x, f_plus_d_acc_list[step * 1:step * 2], color='blue')
    # ax2.plot(x, o_top_k_acc_list[step * 1:step * 2], color='black', linestyle="--")
    # ax2.plot(x, f_top_k_acc_list[step * 1:step * 2], color='red', linestyle="--")
    # ax2.plot(x, f_plus_d_top_k_acc_list[step * 1:step * 2], color='blue', linestyle="--")
    ax2.set_title(
        '%d, %d, %d, %d, %.4f, %.4f, %.4f' % (
            param2_list[0], param3_list[step * 1], get_average(deleted_node_len_list[step * 1:step * 2]),
            get_average(deleted_node_plus_d_len_list[step * 1:step * 2]),
            get_average(o_acc_list[step * 1:step * 2]),
            get_average(f_acc_list[step * 1:step * 2]), get_average(f_plus_d_acc_list[step * 1:step * 2])))

# ax1.plot(x, o_acc_list[40:50], color='black', linestyle="--")
# ax1.plot(x, f_acc_list[40:50], color='red')
# ax1.plot(x, f_plus_d_acc_list[40:50], color='blue')
# ax1.set_title(
#     '%d, %d, %d, %d, %.4f, %.4f, %.4f' % (param2_list[0], param3_list[40], get_average(deleted_node_len_list[40:50]),
#                                           get_average(deleted_node_plus_d_len_list[40:50]),
#                                           get_average(o_acc_list[40:50]),
#                                           get_average(f_acc_list[40:50]), get_average(f_plus_d_acc_list[40:50])))
#
# ax2.plot(x, o_acc_list[50:60], color='black', linestyle="--")
# ax2.plot(x, f_acc_list[50:60], color='red', )
# ax2.plot(x, f_plus_d_acc_list[50:60], color='blue')
# ax2.set_title(
#     '%d, %d, %d, %d, %.4f, %.4f, %.4f' % (param2_list[0], param3_list[50], get_average(deleted_node_len_list[50:60]),
#                                           get_average(deleted_node_plus_d_len_list[50:60]),
#                                           get_average(o_acc_list[50:60]),
#                                           get_average(f_acc_list[50:60]), get_average(f_plus_d_acc_list[50:60])))

if sub_plot >= 3:
    ax3.plot(x, o_acc_list[step * 2:step * 3], color='black')
    ax3.plot(x, f_acc_list[step * 2:step * 3], color='red')
    ax3.plot(x, f_plus_d_acc_list[step * 2:step * 3], color='blue')
    # ax3.plot(x, o_top_k_acc_list[step * 2:step * 3], color='black', linestyle="--")
    # ax3.plot(x, f_top_k_acc_list[step * 2:step * 3], color='red', linestyle="--")
    # ax3.plot(x, f_plus_d_top_k_acc_list[step * 2:step * 3], color='blue', linestyle="--")
    ax3.set_title(
        '%d, %d, %d, %d, %.4f, %.4f, %.4f' % (
            param2_list[0], param3_list[step * 2], get_average(deleted_node_len_list[step * 2:step * 3]),
            get_average(deleted_node_plus_d_len_list[step * 2:step * 3]),
            get_average(o_acc_list[step * 2:step * 3]),
            get_average(f_acc_list[step * 2:step * 3]), get_average(f_plus_d_acc_list[step * 2:step * 3])))

if sub_plot >= 4:
    ax4.plot(x, o_acc_list[step * 3:step * 4], color='black')
    ax4.plot(x, f_acc_list[step * 3:step * 4], color='red')
    ax4.plot(x, f_plus_d_acc_list[step * 3:step * 4], color='blue')
    # ax4.plot(x, o_top_k_acc_list[step * 3:step * 4], color='black', linestyle="--")
    # ax4.plot(x, f_top_k_acc_list[step * 3:step * 4], color='red', linestyle="--")
    # ax4.plot(x, f_plus_d_top_k_acc_list[step * 3:step * 4], color='blue', linestyle="--")
    ax4.set_title(
        '%d, %d, %d, %d, %.4f, %.4f, %.4f' % (
            param2_list[0], param3_list[step * 3], get_average(deleted_node_len_list[step * 3:step * 4]),
            get_average(deleted_node_plus_d_len_list[step * 3:step * 4]),
            get_average(o_acc_list[step * 3:step * 4]),
            get_average(f_acc_list[step * 3:step * 4]), get_average(f_plus_d_acc_list[step * 3:step * 4])))

plt.yticks(fontproperties='Times New Roman', size=18)
plt.xticks(fontproperties='Times New Roman', size=18)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.legend(prop={'size': 21})
plt.show()

