import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
# from scipy.fftpack import fft, ifft
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
test_time_zero_jsd_npy = np.load('metric_file_list_n02190166_time_one.npy')
test_time_zero_abs_d_npy = np.load('abs_d_file_time_one_abs_dn04483307.npy').reshape(
    test_time_zero_jsd_npy.shape) * 0.1 + test_time_zero_jsd_npy
new_d_node_list_npy = np.load('deleted_node_plus_d_marker_new_d_set_n04483307.npy')
d_node_list_npy = np.array(
    list(set(np.load('deleted_node_plus_d_marker_d_set_n04483307.npy')) - set(new_d_node_list_npy)))
print(d_node_list_npy.shape)
print(np.load('deleted_node_plus_d_marker_d_set_n04483307.npy').shape)

# test_time_zero_abs_d_npy = np.load('test_time_zero_abs_d.npy')
# test_time_zero_abs_d_npy = test_time_zero_abs_d_npy.reshape(test_time_zero_jsd_npy.shape)

# print(test_time_one_abs_d_npy.shape)
# print(test_time_one_jsd_npy.shape)
# print(test_time_zero_abs_d_npy.shape)
# print(test_time_zero_jsd_npy.shape)
#
fig = plt.figure()
ax1 = fig.add_subplot(241)
ax2 = fig.add_subplot(242)
ax3 = fig.add_subplot(243)
ax4 = fig.add_subplot(244)
ax5 = fig.add_subplot(245)
ax6 = fig.add_subplot(246)
ax7 = fig.add_subplot(247)
ax8 = fig.add_subplot(248)
# # #
x = range(1079)
# # i = 7865
# # y1 = test_time_one_jsd_npy[:, i]
# # y2 = test_time_one_abs_d_npy[:, i]
# # y3 = test_time_one_jsd_npy[:, i] + test_time_one_abs_d_npy[:, i]
# # y4 = test_time_zero_jsd_npy[:, i]
# # y5 = test_time_zero_abs_d_npy[:, i]
# # y6 = test_time_zero_jsd_npy[:, i] + test_time_zero_abs_d_npy[:, i]
# # y7 = fft(y1)
# # y8 = fft(y4)
i = 400
# print(coint_node_list_npy[i])
# print(not_coint_node_list_npy[i])
# y1 = test_time_zero_jsd_npy[:, coint_node_list_npy[i]]
# print(np.std(test_time_zero_jsd_npy[:, coint_node_list_npy[i]]),
#       np.mean(test_time_zero_jsd_npy[:, coint_node_list_npy[i]]))
# y2 = test_time_zero_jsd_npy[:, coint_node_list_npy[i + 1]]
# print(np.std(test_time_zero_jsd_npy[:, coint_node_list_npy[i + 1]]),
#       np.mean(test_time_zero_jsd_npy[:, coint_node_list_npy[i + 1]]))
# y3 = test_time_zero_jsd_npy[:, coint_node_list_npy[i + 2]]
# print(np.std(test_time_zero_jsd_npy[:, coint_node_list_npy[i + 2]]),
#       np.mean(test_time_zero_jsd_npy[:, coint_node_list_npy[i + 2]]))
# y4 = test_time_zero_jsd_npy[:, coint_node_list_npy[i + 3]]
# print(np.std(test_time_zero_jsd_npy[:, coint_node_list_npy[i + 3]]),
#       np.mean(test_time_zero_jsd_npy[:, coint_node_list_npy[i + 3]]))
# y1 = test_time_zero_jsd_npy[:, i]
# print(np.std(test_time_zero_jsd_npy[:, i]),
#       np.mean(test_time_zero_jsd_npy[:, i]))
# y2 = test_time_zero_jsd_npy[:, i + 1]
# print(np.std(test_time_zero_jsd_npy[:, i + 1]),
#       np.mean(test_time_zero_jsd_npy[:, i + 1]))
# y3 = test_time_zero_jsd_npy[:, i + 2]
# print(np.std(test_time_zero_jsd_npy[:, i + 2]),
#       np.mean(test_time_zero_jsd_npy[:, i + 2]))
# y4 = test_time_zero_jsd_npy[:, i + 3]
# print(np.std(test_time_zero_jsd_npy[:, i + 3]),
#       np.mean(test_time_zero_jsd_npy[:, i + 3]))
# y5 = test_time_zero_abs_d_npy[:, i]
# print(np.std(test_time_zero_abs_d_npy[:, i]),
#       np.mean(test_time_zero_abs_d_npy[:, i]))
# y6 = test_time_zero_abs_d_npy[:, i + 1]
# print(np.std(test_time_zero_abs_d_npy[:, i + 1]),
#       np.mean(test_time_zero_abs_d_npy[:, i + 1]))
# y7 = test_time_zero_abs_d_npy[:, i + 2]
# print(np.std(test_time_zero_abs_d_npy[:, i + 2]),
#       np.mean(test_time_zero_abs_d_npy[:, i + 2]))
# y8 = test_time_zero_abs_d_npy[:, i + 3]
# print(np.std(test_time_zero_abs_d_npy[:, i + 3]),
#       np.mean(test_time_zero_abs_d_npy[:, i + 3]))

y1 = test_time_zero_abs_d_npy[:, d_node_list_npy[i]]
y2 = test_time_zero_abs_d_npy[:, d_node_list_npy[i + 11]]
y3 = test_time_zero_abs_d_npy[:, d_node_list_npy[i + 22]]
y4 = test_time_zero_abs_d_npy[:, d_node_list_npy[i + 33]]
# np.save('time_seq_test.npy',y4)
# print(y4.shape)
y5 = test_time_zero_abs_d_npy[:, new_d_node_list_npy[i]]
y6 = test_time_zero_abs_d_npy[:, new_d_node_list_npy[i + 11]]
y7 = test_time_zero_abs_d_npy[:, new_d_node_list_npy[i + 22]]
y8 = test_time_zero_abs_d_npy[:, new_d_node_list_npy[i + 33]]
# y1 = np.diff(test_time_zero_abs_d_npy[:, coint_node_list_npy[i]])
# y2 = np.diff(test_time_zero_abs_d_npy[:, coint_node_list_npy[i + 1]])
# y3 = np.diff(test_time_zero_abs_d_npy[:, coint_node_list_npy[i + 2]])
# y4 = np.diff(test_time_zero_abs_d_npy[:, coint_node_list_npy[i + 3]])
# y5 = fft(y1)
# y6 = fft(y2)
# y7 = fft(y3)
# y8 = fft(y4)
# y5 = np.abs(fft(test_time_zero_abs_d_npy[:, coint_node_list_npy[i]].reshape(1079, ), norm="ortho"))
# y6 = np.abs(fft(test_time_zero_abs_d_npy[:, coint_node_list_npy[i + 11]].reshape(1079, ), norm="ortho"))
# y7 = np.abs(fft(test_time_zero_abs_d_npy[:, not_coint_node_list_npy[i + 22]].reshape(1079, ), norm="ortho"))
# y8 = np.abs(fft(test_time_zero_abs_d_npy[:, not_coint_node_list_npy[i + 33]].reshape(1079, ), norm="ortho"))

# y5[0:4] = 0
# y6[0:4] = 0
# y7[0:4] = 0
# y8[0:4] = 0
#
# print(np.std(y5))
# print(np.std(y6))
# print(np.std(y7))
# print(np.std(y8))
# print(np.mean(y5))
# print(np.mean(y6))
# print(np.mean(y7))
# print(np.mean(y8))
# print(np.std(y5)/np.mean(y5))
# print(np.std(y6)/np.mean(y6))
# print(np.std(y7)/np.mean(y7))
# print(np.std(y8)/np.mean(y8))


y1 = np.abs(fft(y1.reshape(1079, ), norm="ortho"))
y2 = np.abs(fft(y2.reshape(1079, ), norm="ortho"))
y3 = np.abs(fft(y3.reshape(1079, ), norm="ortho"))
y4 = np.abs(fft(y4.reshape(1079, ), norm="ortho"))
# np.save('time_seq_test.npy',y4)
# print(y4.shape)
y5 = np.abs(fft(y5.reshape(1079, ), norm="ortho"))
y6 = np.abs(fft(y6.reshape(1079, ), norm="ortho"))
y7 = np.abs(fft(y7.reshape(1079, ), norm="ortho"))
y8 = np.abs(fft(y8.reshape(1079, ), norm="ortho"))


y1[0:3] = 0
y2[0:3] = 0
y3[0:3] = 0
y4[0:3] = 0
y5[0:3] = 0
y6[0:3] = 0
y7[0:3] = 0
y8[0:3] = 0

ax1.plot(x[0:100], y1[0:100])
ax2.plot(x[0:100], y2[0:100])
ax3.plot(x[0:100], y3[0:100])
ax4.plot(x[0:100], y4[0:100])
ax5.plot(x[0:100], y5[0:100])
ax6.plot(x[0:100], y6[0:100])
ax7.plot(x[0:100], y7[0:100])
ax8.plot(x[0:100], y8[0:100])
# # ax1.plot(x, np.angle(y8))
# # ax2.plot(x, np.abs(y8))
#
# # plt.subplot(211)
# # plt.plot(x, y1)
# # plt.subplot(212)
# # plt.plot(x, y7)
plt.legend()
plt.show()

# n = range(0, 1080)
# gb_seq = image_enhance.gen_gb_seq_no_noise()[1:1080]
# gb_seq_no_linear = image_enhance.gen_gb_seq_no_linear_no_noise()[1:1080]
# gb_seq = image_enhance.gen_gb_seq()[1:1080]
# adf_gb_seq = adfuller(np.diff(gb_seq_no_linear))
# print(adf_gb_seq)
# adf_gb_seq_no_linear = adfuller(gb_seq_no_linear)
# print(adf_gb_seq_no_linear)
# print(adfuller(y1))
# print(adfuller(y5))
# print(coint(y1, gb_seq_no_linear))
# print(coint(y1, gb_seq_no_linear, trend='ct'))
# print(coint(y5, gb_seq_no_linear))
# print(coint(y5, gb_seq_no_linear, trend='ct'))
#
# coint_node_js_list = []
# not_coint_node_js_list = []
#
# for i in coint_node_list_npy:
#
#     print('=== %d ===' % i)
#
#     node_metric = test_time_zero_jsd_npy[:, i]
#     node_metric = node_metric.reshape(gb_seq_no_linear.shape)
#     adf_node_metric = adfuller(np.diff(node_metric))
#     print(adf_node_metric)
#     # print(node_metric)
#
#     coint_node_metric1, coint_node_metric2, coint_node_metric3 = coint(node_metric, gb_seq_no_linear)
#
#     if i % 2500 == 0 and i != 0:
#         print('=== cal process: %d ===' % i)
#     # node_js_plus_abs_d = metric_file_list[:, i] + abs_d_file_list[:, i] * 0.1
#     node_js = test_time_zero_jsd_npy[:, i]
#     # adf_node_metric = adfuller(node_metric)
#
#     confidence_value_js, p_value_js, confidence_bound_js = coint(node_js, gb_seq_no_linear)
#     print(confidence_value_js, p_value_js, confidence_bound_js)
#     if confidence_value_js < confidence_bound_js[1] and p_value_js < 0.05:
#         coint_node_js_list.append(i)
#         print(1)
#     else:
#         not_coint_node_js_list.append(i)
#         print(2)
#
# np.save('test_coint_maker_d_1.npy', np.array(coint_node_js_list))
# np.save('test_not_coint_maker_d_1.npy', np.array(not_coint_node_js_list))

# print(not_coint_node_list_npy.shape)
# for i in coint_node_list_npy:
#
#     if i % 2500 == 0 and i != 0:
#         print('=== cal process: %d ===' % i)
#     # node_js_plus_abs_d = test_time_zero_jsd_npy[:, i] + test_time_zero_abs_d_npy[:, i] * 0.1
#     node_js = test_time_zero_jsd_npy[:, i]
#     adf_node_metric = adfuller(node_js)
#     # print(type(adf_node_metric))
#     print(adf_node_metric)
#     if adf_node_metric[0] > list(adf_node_metric[4].values())[0]:
#         print(1)

# adf_node_metric = adfuller(node_metric)

# confidence_value_js, p_value_js, confidence_bound_js = coint(node_js, gb_seq_no_linear)
# print(coint(node_js, gb_seq_no_linear))
# if adf_node_metric[]
# if p_value_js > 0.05:
#     print(p_value_js)
# if confidence_value_js < confidence_bound_js[0] and p_value_js < 0.05:
#     pass
# else:
#     print(2)

# file = r'test_marker_d_1.txt'
# with open(file, 'a+') as f:
#     if coint_node_metric1 < coint_node_metric3[1] and coint_node_metric2 < 0.05:
#         f.write(str(i) + '\n')

# y = np.abs(fft(gb_seq))  # fft??????????????????
# plt.subplot(211)
# plt.plot(x, gb_seq_no_linear)
# plt.subplot(212)
# plt.plot(n, y)
#
# plt.legend()
# plt.show()
