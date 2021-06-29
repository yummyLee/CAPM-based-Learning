# from collections import Counter
#
import numpy as np
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
import cv2
import scipy.ndimage
from PIL import Image
import matplotlib.pyplot as plt
import os
import math
import xlrd
from xlrd import open_workbook
from xlutils.copy import copy
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# from statsmodels.tsa.stattools import coint

#
# test_list = [1, 2, 3, 4]
# test_list2 = [1, 2, 3, 4]
# # print(np.append(np.array(test_list), np.array(test_list)))
# print(np.array(test_list) / np.array(test_list2) * 0.1)
#
# test_list = [9, 9, 15]
# print(np.std(np.array(test_list)))
#
# index_counts = Counter(test_list)
# # print(index_counts.most_common(5)[0][1])
# top_k_correct_item_num = 0
# for i in index_counts.most_common(5):
#     top_k_correct_item_num += i[1]
#
# print(top_k_correct_item_num)


# test_list = np.load('deleted_node_js_marker_toc_set_n02088364.npy')
# test_list_2 = np.ones((9216,))
# test_list_2[test_list] = 0
# test_list_2 = test_list_2.reshape((256, 6, 6))
#
# for i in range(test_list_2.shape[0]):
#     print(test_list_2[i])

# print(test_list.shape)


# img1 = Image.open('n02190166_11747.JPEG')
# img1_shape = np.asarray(img1).shape
# h = img1_shape[0]
# w = img1_shape[1]
#
# window_h = h / 1.45
# window_w = w / 1.45
#
# print(window_w)
# # print(w - window_h / 2)
#
# center_bound_points = [[window_h / 2, window_w / 2], [window_h / 2, w - window_w / 2],
#                        [h - window_h / 2, w - window_w / 2], [h - window_h / 2, window_w / 2]]
#
# center_points = []
#
# for i in np.arange(center_bound_points[0][1], center_bound_points[1][1],
#                    (center_bound_points[1][1] - center_bound_points[0][1]) / 20):
#     center_points.append([center_bound_points[0][0], i])
#
# for i in np.arange(center_bound_points[1][0], center_bound_points[2][0],
#                    (center_bound_points[2][0] - center_bound_points[1][0]) / 20):
#     center_points.append([i, center_bound_points[1][1]])
#
# for i in np.arange(center_bound_points[2][1], center_bound_points[3][1],
#                    (center_bound_points[3][1] - center_bound_points[2][1]) / 20):
#     center_points.append([center_bound_points[2][0], i])
#
# for i in np.arange(center_bound_points[3][0], center_bound_points[0][0],
#                    (center_bound_points[0][0] - center_bound_points[3][0]) / 20):
#     center_points.append([i, center_bound_points[3][1]])
#
# frame_points = []
#
# for i in center_points:
#     frame_points.append(
#         (int(i[1] - window_w / 2), int(i[0] - window_h / 2), int(i[1] + window_w / 2), int(i[0] + window_h / 2)))
#
# for i in range(len(frame_points)):
#     cropped = img1.crop(frame_points[i])
#     cropped.save('crop_test/%d.jpg' % i)
