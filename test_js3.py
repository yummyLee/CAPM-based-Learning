import multiprocessing
import random
import sys
import threading
# import skimage
# import torch
# import torchvision.models as models
import numpy as np

# def test_array_all_the_same(arr):
#     # print('arr: ', arr)
#     first = arr[0]
#     for item in arr:
#         if not item == first:
#             # print(False)
#             return False
#     # print(True)
#     return True
#
#
# o_metric_file_list = np.load('makerc_out.npy').transpose()
# metric_file_list = np.load('makerc.npy').transpose()
#
# print(metric_file_list.shape)
#
# o_metric_file_list_del = []
# metric_file_list_del = []
# io_metric_file_list_del = []
# log_io_metric_file_list_del = []
#
# metric_file_list[metric_file_list == 0] = 1
#
# for i in range(o_metric_file_list.shape[1]):
#     o_metric_file_list_del.append((o_metric_file_list[:, i]))
#     metric_file_list_del.append((metric_file_list[:, i]))
#     io_metric_file_list_del.append(
#         (o_metric_file_list[:, i] + 1) / (metric_file_list[:, i] + 1))
#     # log_io_metric_file_list_del.append(
#     #     np.log(((o_metric_file_list[:, i] + 1) / (metric_file_list[:, i] + 1))) / std_npy[
#     #         i])
#     # io_metric_file_list_del.append(
#     #     (o_metric_file_list[:, i] + 10e-7) / (metric_file_list[:, i] + 10e-7))
#     # log_io_metric_file_list_del.append(
#     #     np.log(((o_metric_file_list[:, i] + 10e-7) / (metric_file_list[:, i] + 10e-7))) / std_npy[
#     #         i])
#     log_io_metric_file_list_del.append(
#         np.log(((o_metric_file_list[:, i] + 1) / (metric_file_list[:, i] + 1))))
#
# log_io_metric_file_list_del = np.array(log_io_metric_file_list_del)
# io_metric_file_list_del = np.array(io_metric_file_list_del)
#
# er_list = np.mean(io_metric_file_list_del, axis=1)
# log_er_list = np.mean(log_io_metric_file_list_del, axis=1)
#
# print('io_m ', io_metric_file_list_del.shape)
# print('er_list ', er_list.shape)
#
# t = 'wzy'
# mt = 'wa'
#
# # np.save(t + '/io_%s_file_list.npy' % (mt), io_metric_file_list_del)
# # np.save(t + '/log_io_%s_file_list.npy' % (mt), log_io_metric_file_list_del)
# # np.save(t + '/io_%s_er_file_list.npy' % (mt), io_er_list)
# # np.save(t + '/log_io_%s_er_file_list.npy' % (mt), log_io_er_list)
#
# log_metric_cov = np.cov(log_io_metric_file_list_del)
# metric_cov = np.cov(io_metric_file_list_del)
# log_metric_cof = np.corrcoef(log_io_metric_file_list_del)
# metric_cof = np.corrcoef(io_metric_file_list_del)
#
# # np.save(t + '/io_%s_cov_file_list.npy' % (mt), metric_cov)
# # np.save(t + '/log_io_%s_cov_file_list.npy' % (mt), log_metric_cov)
# # np.save(t + '/io_%s_cof_file_list.npy' % (mt), metric_cof)
# # np.save(t + '/log_io_%s_cof_file_list.npy' % (mt), log_metric_cof)
#
#
# io_metric_file_list = io_metric_file_list_del
# log_io_metric_file_list = log_io_metric_file_list_del
#
# keep_node_set = np.array(list(range(metric_cov.shape[0])))
#
# keep_node_pools = []
# keep_node_pools_weight = []
# no_center_node_list = []
#
# det_0_count = 0
#
# # print(er_list[r_i_list_topk_index])
# # print(keep_node_set[r_i_list_topk_index])
# # for j in range(0):
#
# for j in range(len(keep_node_set)):
#
#     # if keep_node_set[j] != 9215:
#     #     continue
#
#     indexes = []
#
#     # htopk_v, htopk_index = torch.from_numpy(log_metric_cof[j, r_i_list_topk_index]).topk(10)
#     ltopk_v, htopk_index = torch.from_numpy(log_metric_cof[j, :]).topk(5,
#                                                                        largest=False)
#     ltopk_v, ltopk_index = torch.from_numpy(np.abs(log_metric_cof[j, :])).topk(5,
#                                                                                largest=False)
#
#     # print(io_metric_file_list.shape)
#     # print(er_list[j])
#     for index in ltopk_index.numpy():
#         # print(er_list[index])
#         if not test_array_all_the_same(io_metric_file_list[index, :]):
#             # if er_list[index] > er_list[j] and not test_array_all_the_same(io_metric_file_list[index, :]):
#             indexes.append(index)
#     for index in htopk_index.numpy():
#         if not test_array_all_the_same(io_metric_file_list[index, :]):
#             # if er_list[index] > er_list[j] and not test_array_all_the_same(io_metric_file_list[index, :]):
#             indexes.append(index)
#
#     # print()
#     if not test_array_all_the_same(io_metric_file_list[j, :]):
#         indexes.append(j)
#     else:
#         print('no_center')
#         no_center_node_list.append(j)
#
#     print('indexs: ', indexes)
#     indexes = list(set(indexes))
#     # indexes.append(j)
#
#     indexes_big = []
#
#     indexes = np.array(indexes)
#
#     if indexes.shape[0] <= 1:
#         if indexes.shape[0] == 0:
#             keep_node_pools.append(np.array([j]))
#             keep_node_pools_weight.append(np.array([1]))
#         else:
#             keep_node_pools.append(keep_node_set[indexes])
#             keep_node_pools_weight.append(np.array([1]))
#         det_0_count += 1
#         print('lt 1')
#         continue
#
#     market_er_list = log_er_list[indexes].reshape(-1, 1)
#     # print('max_marker: ',np.max(market_er_list))
#
#     ee = np.ones(market_er_list.shape).reshape(-1, 1)
#
#     pools_wa_list = log_io_metric_file_list[indexes]
#     pools_wa_cov = np.cov(pools_wa_list)
#     # print(pools_wa_cov)
#
#     wa_cov_inv = np.linalg.inv(pools_wa_cov)
#
#     a = np.dot(np.dot(market_er_list.transpose(), wa_cov_inv), market_er_list)[0][0] + 0.0000001
#     b = np.dot(np.dot(market_er_list.transpose(), wa_cov_inv), ee)[0][0] + 0.0000001
#     c = np.dot(np.dot(ee.transpose(), wa_cov_inv), ee)[0][0] + 0.0000001
#     b2 = np.dot(np.dot(ee.transpose(), wa_cov_inv), market_er_list)[0][0] + 0.0000001
#
#     # print('a: ', a)
#     print('b: ', b)
#     # print('c: ', c)
#     print('b2: ', b2)
#
#     # miu_p = (a * a) / (b * b2)
#     miu_p = np.sqrt(a * a / (b * b2))
#
#     d = a * c - b * b2 + 0.0000001
#     a_inv = np.ones((2, 2))
#     a_inv[0][0] = c
#     a_inv[0][1] = -b
#     a_inv[1][0] = -b2
#     a_inv[1][1] = a
#     # print(a_inv.shape)
#     a_inv *= (1 / d)
#
#     m1 = np.dot(wa_cov_inv, np.concatenate((market_er_list, ee), axis=1))
#     m2 = np.dot(m1, a_inv)
#     m3 = np.dot(m2, np.concatenate(([miu_p], [1]), axis=0))
#
#     weight = m3
#     # weight[weight < 0] = 0
#     # print('indexes: ', keep_node_set[indexes])
#     # print(weight)
#     # if keep_node_set[j] == 288:
#     #     sys.exit(0)
#     # print('miu_p_w: ', np.dot(weight.transpose(), market_er_list))
#
#     # weight = tensor_array_normalization(m3)
#     # print(np.sum(weight))
#     print('j: ', keep_node_set[j])
#     # print('er:', er_list[indexes])
#     print('log_er:', log_er_list[indexes])
#     print('weight: ', weight)
#     print('miu_p: ', miu_p)
#     # if keep_node_set[j] == 5887:
#     #     break
#     # weight = weight / np.sum(weight)
#     # print(np.sum(weight))
#     # print(tensor_array_normalization(m3))
#
#     # print(tensor_array_normalization(m3)/np.sum(m3))
#     # print(np.sum(weight))
#
#     keep_node_pools.append(keep_node_set[indexes])
#     keep_node_pools_weight.append(weight)
#     # print('shape indexes: ', indexes.shape)
#     # print('shape weights: ', weight.shape)
#     # if j==2:
#     #     print('shape indexes: ',keep_node_pools[j].shape)
#     #     print('shape weights: ',weight.shape)
#     #     print(j)
#     # break
#
#     print('======================')
#     # break
#
# keep_node_pools = np.array(keep_node_pools)
# keep_node_pools_weight = np.array(keep_node_pools_weight)
#
# print(keep_node_pools.shape)
# print(keep_node_pools_weight.shape)
#
# np.save(t + '/keep_node_pools.npy', keep_node_pools)
# np.save(t + '/keep_node_pools_weight.npy', keep_node_pools_weight)


# def cal_inter(a, b):
#     return len(set(a).intersection(set(b)))
#
#
# best1 = np.load('t_942005/best_node_pools1_wad_marker_toc_sc_set_n02128385.npy')
# best2 = np.load('t_942006/best_node_pools1_wad_marker_toc_sc_set_n02128385.npy')
#
# print(best1.shape)
# print(best2.shape)
#
# print(cal_inter(best1, best2))


a = [[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[1, 1, 1], [2, 2, 2], [3, 3, 3]]]

a = np.array(a)

print(a)

print(a.shape)

print(np.mean(a, axis=1))
