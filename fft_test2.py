import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
from numpy.fft import fft
from matplotlib.pylab import mpl
import numpy as np
import image_enhance

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint

#
# mode = 'zone'
# phase = 'zero'
# metric_list_np_dir = 'metric_list_jsd_%s_5_%s_npy' % (mode, phase)
# metric_file_name_list = os.listdir(metric_list_np_dir)
# metric_file_list = []
#
# if mode == 'time':
#     for i in range(1080):
#         if i < 10:
#             i = '00' + str(i)
#         elif i < 100:
#             i = '0' + str(i)
#         else:
#             i = str(i)
#         metric_file_list.append(
#             np.array(np.load(metric_list_np_dir + '/metric_list_jsd_%s_5_%s_%s.npy' % (i, mode, phase))))
#
# else:
#     for item in os.listdir('metric_list_jsd_zone_5_zero_npy'):
#         metric_file_list.append(np.load(metric_list_np_dir + '/' + item))
#
# metric_file_list = np.array(metric_file_list)
# metric_jsd_point_value_list = []
#
# print(metric_file_list.shape)
# for i in range(metric_file_list.shape[1]):
#     metric_jsd_point_value_list.append(metric_file_list[:, i])
#
# fig = plt.figure()
# ax1 = fig.add_subplot(221)
# ax2 = fig.add_subplot(222)
# ax3 = fig.add_subplot(223)
# ax4 = fig.add_subplot(224)
#
# x = range(198)
# i = 5435
# y1 = metric_jsd_point_value_list[i]
# y2 = metric_jsd_point_value_list[i + 1]
# y3 = metric_jsd_point_value_list[i + 2]
# y4 = metric_jsd_point_value_list[i + 3]
# ax1.plot(x, y1)
# ax2.plot(x, y2)
# ax3.plot(x, y3)
# ax4.plot(x, y4)
#
# plt.legend()
# plt.show()
#
# # dw_test_node_list = []
# # regular_node_list = []
# # un_regular_node_list = []
# # for i in range(len(metric_jsd_point_value_list)):
# #     dw_test_node_list.append(sm.stats.durbin_watson(metric_jsd_point_value_list[i]))
# #     # print(dw_test_node_list[i])
# #     v = 4 - dw_test_node_list[i]
# #     # print(v)
# #     if v < 1.654 or v > 2.307:
# #         regular_node_list.append(i)
# #     else:
# #         print(i)
# #         un_regular_node_list.append(i)
# #
# # print(len(regular_node_list))
# # print(len(un_regular_node_list))


# test_time_one_jsd_npy = np.load('test_time_one_js.npy')
# test_time_one_abs_d_npy = np.load('test_time_one_abs_d.npy')
# test_time_one_abs_d_npy = test_time_one_abs_d_npy.reshape(test_time_one_jsd_npy.shape)
metric_file_list_npy = np.load('metric_file_list_n02190166_time_zero.npy')
# abs_d_file_list_npy = np.load('abs_d_file_list_n02088364_time_zero.npy').reshape(
#     metric_file_list_npy.shape) * 0.1 + metric_file_list_npy
deleted_node_list_npy = np.load('concept_set_0604_2308/deleted_node_js_marker_toc_set_n02088364.npy')
print(deleted_node_list_npy.shape)
not_deleted_node_list_npy = np.array(
    list(set(list(range(metric_file_list_npy.shape[1]))) - set(deleted_node_list_npy)))

fig = plt.figure()
ax1 = fig.add_subplot(3, 6, 1)
ax2 = fig.add_subplot(3, 6, 2)
ax3 = fig.add_subplot(3, 6, 3)
ax4 = fig.add_subplot(3, 6, 4)
ax5 = fig.add_subplot(3, 6, 5)
ax6 = fig.add_subplot(3, 6, 6)
ax7 = fig.add_subplot(3, 6, 7)
ax8 = fig.add_subplot(3, 6, 8)
ax9 = fig.add_subplot(3, 6, 9)
ax10 = fig.add_subplot(3, 6, 10)
ax11 = fig.add_subplot(3, 6, 11)
ax12 = fig.add_subplot(3, 6, 12)
ax13 = fig.add_subplot(3, 6, 13)
ax14 = fig.add_subplot(3, 6, 14)
ax15 = fig.add_subplot(3, 6, 15)
ax16 = fig.add_subplot(3, 6, 16)
ax17 = fig.add_subplot(3, 6, 17)
ax18 = fig.add_subplot(3, 6, 18)

x = range(847)

i = 300

# y1 = abs_d_file_list_npy[:, deleted_node_list_npy[i]]
# y2 = abs_d_file_list_npy[:, deleted_node_list_npy[i + 11]]
# y3 = abs_d_file_list_npy[:, deleted_node_list_npy[i + 22]]
# y4 = abs_d_file_list_npy[:, deleted_node_list_npy[i + 33]]
y1 = metric_file_list_npy[:, not_deleted_node_list_npy[i]]
y2 = metric_file_list_npy[:, not_deleted_node_list_npy[i + 2]]
y3 = metric_file_list_npy[:, not_deleted_node_list_npy[i + 4]]
y4 = metric_file_list_npy[:, not_deleted_node_list_npy[i + 6]]
y5 = metric_file_list_npy[:, not_deleted_node_list_npy[i + 8]]
y6 = metric_file_list_npy[:, not_deleted_node_list_npy[i + 10]]
y7 = metric_file_list_npy[:, not_deleted_node_list_npy[i + 12]]
y8 = metric_file_list_npy[:, not_deleted_node_list_npy[i + 14]]
y9 = metric_file_list_npy[:, not_deleted_node_list_npy[i + 16]]
# y1 = np.abs(fft(abs_d_file_list_npy[:, not_deleted_node_list_npy[i]], norm="ortho"))
# y2 = np.abs(fft(abs_d_file_list_npy[:, not_deleted_node_list_npy[i + 11]], norm="ortho"))
# y3 = np.abs(fft(abs_d_file_list_npy[:, not_deleted_node_list_npy[i + 22]], norm="ortho"))
# y4 = np.abs(fft(abs_d_file_list_npy[:, not_deleted_node_list_npy[i + 33]], norm="ortho"))
y10 = metric_file_list_npy[:, deleted_node_list_npy[i]]
y11 = metric_file_list_npy[:, deleted_node_list_npy[i + 2]]
y12 = metric_file_list_npy[:, deleted_node_list_npy[i + 4]]
y13 = metric_file_list_npy[:, deleted_node_list_npy[i + 6]]
y14 = metric_file_list_npy[:, deleted_node_list_npy[i + 8]]
y15 = metric_file_list_npy[:, deleted_node_list_npy[i + 10]]
y16 = metric_file_list_npy[:, deleted_node_list_npy[i + 12]]
y17 = metric_file_list_npy[:, deleted_node_list_npy[i + 14]]
y18 = metric_file_list_npy[:, deleted_node_list_npy[i + 16]]
# y1 = metric_file_list_npy[:, not_deleted_node_list_npy[i]]
# y2 = metric_file_list_npy[:, not_deleted_node_list_npy[i + 1]]
# y3 = metric_file_list_npy[:, not_deleted_node_list_npy[i + 2]]
# y4 = metric_file_list_npy[:, not_deleted_node_list_npy[i + 3]]
# y5 = abs_d_file_list_npy[:, not_deleted_node_list_npy[i]]
# y6 = abs_d_file_list_npy[:, not_deleted_node_list_npy[i + 1]]
# y7 = abs_d_file_list_npy[:, not_deleted_node_list_npy[i + 2]]
# y8 = abs_d_file_list_npy[:, not_deleted_node_list_npy[i + 3]]
# print('%.4f\t%.4f\t%.4f' % (np.mean(y1), np.std(y1), np.std(y1) / np.mean(y1)))
# print('%.4f\t%.4f\t%.4f' % (np.mean(y2), np.std(y2), np.std(y2) / np.mean(y2)))
# print('%.4f\t%.4f\t%.4f' % (np.mean(y3), np.std(y3), np.std(y3) / np.mean(y3)))
# print('%.4f\t%.4f\t%.4f' % (np.mean(y4), np.std(y4), np.std(y4) / np.mean(y4)))
# print('%.4f\t%.4f\t%.4f' % (np.mean(y5), np.std(y5), np.std(y5) / np.mean(y5)))
# print('%.4f\t%.4f\t%.4f' % (np.mean(y6), np.std(y6), np.std(y6) / np.mean(y6)))
# print('%.4f\t%.4f\t%.4f' % (np.mean(y7), np.std(y7), np.std(y7) / np.mean(y7)))
# print('%.4f\t%.4f\t%.4f' % (np.mean(y8), np.std(y8), np.std(y8) / np.mean(y8)))

ax1.plot(x, y1)
ax2.plot(x, y2)
ax3.plot(x, y3)
ax4.plot(x, y4)
ax5.plot(x, y5)
ax6.plot(x, y6)
ax7.plot(x, y7)
ax8.plot(x, y8)
ax9.plot(x, y9)
ax10.plot(x, y10)
ax11.plot(x, y11)
ax12.plot(x, y12)
ax13.plot(x, y13)
ax14.plot(x, y14)
ax15.plot(x, y15)
ax16.plot(x, y16)
ax17.plot(x, y17)
ax18.plot(x, y18)

# y1 = np.abs(fft(y1.reshape(1079, ), norm="ortho"))
# y2 = np.abs(fft(y2.reshape(1079, ), norm="ortho"))
# y3 = np.abs(fft(y3.reshape(1079, ), norm="ortho"))
# y4 = np.abs(fft(y4.reshape(1079, ), norm="ortho"))
# # np.save('time_seq_test.npy',y4)
# # print(y4.shape)
# y5 = np.abs(fft(y5.reshape(1079, ), norm="ortho"))
# y6 = np.abs(fft(y6.reshape(1079, ), norm="ortho"))
# y7 = np.abs(fft(y7.reshape(1079, ), norm="ortho"))
# y8 = np.abs(fft(y8.reshape(1079, ), norm="ortho"))
# y9 = np.abs(fft(y9.reshape(1079, ), norm="ortho"))
# y10 = np.abs(fft(y10.reshape(1079, ), norm="ortho"))
# y11 = np.abs(fft(y11.reshape(1079, ), norm="ortho"))
# y12 = np.abs(fft(y12.reshape(1079, ), norm="ortho"))
# y13 = np.abs(fft(y13.reshape(1079, ), norm="ortho"))
# y14 = np.abs(fft(y14.reshape(1079, ), norm="ortho"))
# y15 = np.abs(fft(y15.reshape(1079, ), norm="ortho"))
# y16 = np.abs(fft(y16.reshape(1079, ), norm="ortho"))
# y17 = np.abs(fft(y17.reshape(1079, ), norm="ortho"))
# y18 = np.abs(fft(y18.reshape(1079, ), norm="ortho"))
#
# y1[0:3] = 0
# y2[0:3] = 0
# y3[0:3] = 0
# y4[0:3] = 0
# y5[0:3] = 0
# y6[0:3] = 0
# y7[0:3] = 0
# y8[0:3] = 0
# y9[0:3] = 0
# y10[0:3] = 0
# y11[0:3] = 0
# y12[0:3] = 0
# y13[0:3] = 0
# y14[0:3] = 0
# y15[0:3] = 0
# y16[0:3] = 0
# y17[0:3] = 0
# y18[0:3] = 0
#
# ax1.plot(x[0:100], y1[0:100])
# ax2.plot(x[0:100], y2[0:100])
# ax3.plot(x[0:100], y3[0:100])
# ax4.plot(x[0:100], y4[0:100])
# ax5.plot(x[0:100], y5[0:100])
# ax6.plot(x[0:100], y6[0:100])
# ax7.plot(x[0:100], y7[0:100])
# ax8.plot(x[0:100], y8[0:100])
# ax9.plot(x[0:100], y9[0:100])
# ax10.plot(x[0:100], y10[0:100])
# ax11.plot(x[0:100], y11[0:100])
# ax12.plot(x[0:100], y12[0:100])
# ax13.plot(x[0:100], y13[0:100])
# ax14.plot(x[0:100], y14[0:100])
# ax15.plot(x[0:100], y15[0:100])
# ax16.plot(x[0:100], y16[0:100])
# ax17.plot(x[0:100], y17[0:100])
# ax18.plot(x[0:100], y18[0:100])
# # ax1.plot(x, np.angle(y8))
# # ax2.plot(x, np.abs(y8))
#
# # plt.subplot(211)
# # plt.plot(x, y1)
# # plt.subplot(212)
# # plt.plot(x, y7)

plt.legend()
plt.show()
