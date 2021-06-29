import math

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.spatial import distance
import torch
import os

cov_dir = 'cov_npy_dir'

# for cov_file in os.listdir(cov_dir):
#     cov_i_list = np.load(cov_dir + '/' + cov_file)
#     count = 0
#     print('=== %s ===' % cov_file)
#     print(len(cov_i_list))
#     for i in range(len(cov_i_list)):
#         if cov_i_list[i][0][1] > 0:
#             count += 1
#     print(count)


arr = [[1, 2, 3, 4],
       [2, 34, 21, 54],
       [3, 4, 5, 6],
       [31, 4, 5, 6],
       [5, 6, 7, 8],
       [4,215,213,2]]

print(np.cov(arr))
