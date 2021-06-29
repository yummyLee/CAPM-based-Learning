# 226
import multiprocessing
import random
import sys
import threading
import skimage
import torch
import torchvision.models as models
import numpy as np
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as datasets
from PIL import Image
from torchvision.transforms import transforms
# from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
# from sklearn.neighbors.kde import KernelDensity
import scipy.signal
from collections import Counter, OrderedDict
import math
import argparse
import os
import random
import shutil
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.spatial.distance
import scipy.stats
import image_enhance
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.stattools import coint
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
from sklearn import metrics
from image_enhance import transform, random_transform_one_img, transform_random_one_image2
import copy
import sklearn
import net
import warnings

# from pic_gen import transform_random_image

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# import warnings
#
# warnings.filterwarnings('ignore', '.*output shape of zoom.*')

parser = argparse.ArgumentParser()
parser.add_argument('-b', type=int, default=256, help='batch size')
parser.add_argument('-dir', type=str, help='image to be operated')
parser.add_argument('-arch', default='alexnet', help='type of net')
parser.add_argument('-op', type=str, help='operation')
parser.add_argument('-g', type=str, default='psnr', help='grades type')
parser.add_argument('-gn', type=str, default='y', help='normalization or not')
parser.add_argument('-f', type=int, default=-1, help='data to next layer')
parser.add_argument('-sl', type=int, default=1000, help='len of generated series')
parser.add_argument('-ts', type=str, default='none', help='trans before input')
parser.add_argument('-tm', type=str, default='n', help='model been trained')
parser.add_argument('-l', type=int, default=-1, help='layer to be operated')
parser.add_argument('-cdir', type=str, default='dog', help='dir to be compared')
parser.add_argument('-llen', type=int, default=-1, help='layer len')
parser.add_argument('-thres', type=float, default=1.0, help='threshold')
parser.add_argument('-vdir', type=str, help='image to be validated')
parser.add_argument('-thresop', type=str, help='image to be validated')
parser.add_argument('-xrate', type=float, help='rate for node to multiple', default=0.000000001)
parser.add_argument('-sdir', type=str, help='batch size')
parser.add_argument('-tdir', type=str, default='', help='image to be operated')
parser.add_argument('-crate', type=float, help='image to be operated')
parser.add_argument('-ec', type=str, default='none', help='except class')
parser.add_argument('-param', type=str, default='all_channel_change', help='any way')
parser.add_argument('-param2', type=int, default=0, help='any way')
parser.add_argument('-param3', type=int, default=0, help='any way')
parser.add_argument('-param4', type=int, default=0, help='any way')
parser.add_argument('-param5', type=int, default=0, help='any way')
parser.add_argument('-param6', type=int, default=0, help='any way')
parser.add_argument('-param7', type=int, default=0, help='any way')
parser.add_argument('-tsop', type=str, default='none', help='transform_type')
parser.add_argument('-dnop', type=str, default='none', help='delete node strategy')
parser.add_argument('-cb', type=int, default=None, help='confidence bound of the coint')
parser.add_argument('-pb', type=float, default=None, help='p_value')
parser.add_argument('-phase', type=str, default=None, help='phase')
parser.add_argument('-mt', type=str, default='js', help='metric_type')
parser.add_argument('-rs', type=int, default=1, help='metric_type')
parser.add_argument('-rt', type=int, default=1, help='metric_type')
parser.add_argument('-ioi', type=str, default='i', help='ioi')
parser.add_argument('-tt', type=str, default='1', help='time type')
parser.add_argument('-cp', type=str, default='one', help='compare to one or to all')
parser.add_argument('-isall', type=str, default='all', help='only_best')
parser.add_argument('-tv', type=str, default='train', help='train or validate')

args = parser.parse_args()

if args.arch == 'resnet50':
    output_size = 2048
else:
    output_size = 43264
    # output_size = 9216


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def mkdir(path):
    import os
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)

    if not is_exists:
        os.makedirs(path)
        # print(path + ' success')
        return True
    else:
        # print(path + ' existed')
        return False


def tensor_array_normalization(wait_norm_tensors, ymin=0.0, ymax=1.0):
    # print('---- TENSOR ARRAY NORMALIZTION ----')

    max_i = np.max(wait_norm_tensors)
    min_i = np.min(wait_norm_tensors)

    # return (wait_norm_tensors - min_i) / (max_i - min_i)
    return (ymax - ymin) * (wait_norm_tensors - min_i) / (max_i - min_i) + ymin


def two_tensor_array_normalization(wait_norm_tensors1, wait_norm_tensors2, y_min=0.0, y_max=1.0):
    # print('---- TENSOR ARRAY NORMALIZTION ----')

    std1 = np.std(wait_norm_tensors1)
    avg1 = np.average(wait_norm_tensors1)
    max_o1 = np.max(wait_norm_tensors1)
    min_o1 = np.min(wait_norm_tensors1)
    max_i1 = min(avg1 + 2 * std1, max_o1)
    min_i1 = max(avg1 - 2 * std1, min_o1)

    # set -1 for outlier
    wait_norm_tensors1[wait_norm_tensors1 > max_i1] = max_i1
    wait_norm_tensors1[wait_norm_tensors1 < min_i1] = min_i1

    std2 = np.std(wait_norm_tensors2)
    avg2 = np.average(wait_norm_tensors2)
    max_o2 = np.max(wait_norm_tensors2)
    min_o2 = np.min(wait_norm_tensors2)
    max_i2 = min(avg2 + 2 * std2, max_o2)
    min_i2 = max(avg2 - 2 * std2, min_o2)

    # set -1 for outlier
    wait_norm_tensors2[wait_norm_tensors2 > max_i2] = max_i2
    wait_norm_tensors2[wait_norm_tensors2 < min_i2] = min_i2

    max_bound = max(max_i1, max_i2)
    min_bound = min(min_i1, min_i2)

    return (y_max - y_min) * (wait_norm_tensors1 - min_bound) / (max_bound - min_bound) + y_min, (y_max - y_min) * (
            wait_norm_tensors2 - min_bound) / (max_bound - min_bound) + y_min


def two_tensor_array_normalization2(wait_norm_tensors1, wait_norm_tensors2, y_min=0.0, y_max=1.0):
    # print('---- TENSOR ARRAY NORMALIZTION ----')

    std1 = np.std(wait_norm_tensors1)
    avg1 = np.average(wait_norm_tensors1)
    max_o1 = np.max(wait_norm_tensors1)
    min_o1 = np.min(wait_norm_tensors1)
    max_i1 = min(avg1 + 2 * std1, max_o1)
    min_i1 = max(avg1 - 2 * std1, min_o1)

    # set -1 for outlier
    wait_norm_tensors1[wait_norm_tensors1 > max_i1] = max_i1
    wait_norm_tensors1[wait_norm_tensors1 < min_i1] = min_i1

    std2 = np.std(wait_norm_tensors2)
    avg2 = np.average(wait_norm_tensors2)
    max_o2 = np.max(wait_norm_tensors2)
    min_o2 = np.min(wait_norm_tensors2)
    max_i2 = min(avg2 + 2 * std2, max_o2)
    min_i2 = max(avg2 - 2 * std2, min_o2)

    # set -1 for outlier
    wait_norm_tensors2[wait_norm_tensors2 > max_i1] = max_i2
    wait_norm_tensors2[wait_norm_tensors2 < min_i2] = min_i2

    max_bound = max(max_i1, max_i2) + 0.0000001
    min_bound = max(min(min_i1, min_i2), 0)

    return (y_max - y_min) * (wait_norm_tensors1 - min_bound) / (max_bound - min_bound) + y_min, (y_max - y_min) * (
            wait_norm_tensors2 - min_bound) / (max_bound - min_bound) + y_min


def tensor_array_normalization_std(wait_norm_tensors):
    # print('---- TENSOR ARRAY NORMALIZTION ----')

    std = np.std(wait_norm_tensors)

    max_i = np.average(wait_norm_tensors) + 2 * std
    min_i = np.average(wait_norm_tensors) - 2 * std

    return (wait_norm_tensors - min_i) / (max_i - min_i)


def two_tensor_array_normalization_std(wait_norm_tensors1, wait_norm_tensors2, y_min=0.0, y_max=1.0):
    # print('---- TENSOR ARRAY NORMALIZTION ----')

    max_i = max(np.max(wait_norm_tensors1), np.max(wait_norm_tensors2))
    min_i = min(np.min(wait_norm_tensors1), np.min(wait_norm_tensors2))

    # return (wait_norm_tensors - min_i) / (max_i - min_i)
    return (y_max - y_min) * (wait_norm_tensors1 - min_i) / (max_i - min_i) + y_min, (y_max - y_min) * (
            wait_norm_tensors2 - min_i) / (max_i - min_i) + y_min


def js_divergence(p, q):
    M = (p + q) / 2
    # print(p.shape)
    # print(q.shape)
    # print(M.shape)
    return 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)


def cal_jsd2(targ, ref):
    """
    :param targ: dim: 200 or 1080
    :param ref: dim: 200 or 1080
    :return:
    """
    targ = targ.reshape(-1, 1)
    ref = ref.reshape(-1, 1)

    # normalization and delete the outlier
    targ, ref = two_tensor_array_normalization2(targ, ref)

    bin_edges = np.arange(0.00, 1.01, 0.1)
    t_count = np.histogram(targ, bin_edges)[0] + 0.0001
    r_count = np.histogram(ref, bin_edges)[0] + 0.0001

    t_count = t_count / (np.sum(t_count))
    r_count = r_count / (np.sum(r_count))

    JS = js_divergence(t_count, r_count)

    return JS, -1


def cal_jsd3(tar, ref):
    """
    :param tar: dim: 200 or 1080
    :param ref: dim: 200 or 1080
    :return:
    """

    tar = tar.reshape(-1, 1) + 0.0000001
    ref = ref.reshape(-1, 1) + 0.0000001

    tar_norm, ref_norm = two_tensor_array_normalization2(tar, ref)

    tar_none_zero = tar[tar > 0]
    tar_none_zero_avg = np.average(tar_none_zero)
    tar_none_zero_std = np.std(tar_none_zero)
    tar_max_bound = tar_none_zero_avg + 2 * tar_none_zero_std
    tar[tar > tar_max_bound] = tar_max_bound
    tar = tar / np.sum(tar)

    ref_none_zero = tar[tar > 0]
    ref_none_zero_avg = np.average(ref_none_zero)
    ref_none_zero_std = np.std(ref_none_zero)
    ref_max_bound = ref_none_zero_avg + 2 * ref_none_zero_std
    ref[ref > ref_max_bound] = ref_max_bound
    ref = ref / np.sum(ref)

    # JS = js_divergence(tar, ref)
    min_shape = min(tar.shape[0], ref.shape[0])
    tar = tar[0:min_shape] + 0.0000001
    ref = ref[0:min_shape] + 0.0000001
    # JS = distance.jensenshannon(tar, ref)
    # JS = JS * JS
    JS = js_divergence(tar, ref)

    return JS, abs(np.average(tar_norm) - np.average(ref_norm))


def cal_middle_output(m_model, arch, input_tensor, layer_index):
    output = None
    if arch == 'alexnet' or arch == 'm_alexnet':
        model_f = m_model.features
        model_a = m_model.avgpool
        model_c = m_model.classifier
        #
        # print(input_tensor.shape)
        # print(layer_index)

        if layer_index == 1:
            output = model_f[2](model_f[1](model_f[0](input_tensor.cpu()).cpu()).cpu()).cpu()
        elif layer_index == 2:
            output = model_f[5](model_f[4](model_f[3](input_tensor.cpu()).cpu()).cpu()).cpu()
        elif layer_index == 3:
            output = model_f[7](model_f[6](input_tensor.cpu()).cpu()).cpu()
        elif layer_index == 4:
            output = model_f[9](model_f[8](input_tensor.cpu()).cpu()).cpu()
        elif layer_index == 5:
            output = model_f[10](input_tensor.cpu()).cpu()
            output = model_f[11](output).cpu()
            output = model_f[12](output).cpu()
            output = model_a(output)
            print(output.size())
        elif layer_index == 6:
            # output = model_f[12](input_tensor.cpu()).cpu()
            # output = model_f[12](input_tensor.cpu()).cpu()
            # output = model_a(output)
            input_flat = output.view(-1, output_size)
            input_flat = input_tensor.cpu().view(-1, output_size)
            output = model_c[2](model_c[1](model_c[0](input_flat).cpu()).cpu())
        elif layer_index == 7:
            output = model_c[5](model_c[4](model_c[3](input_tensor.cpu()).cpu()).cpu()).cpu()
        elif layer_index == 8:
            output = model_c[6](input_tensor.cpu()).cpu()

        # if layer_index == 1:
        #     output = model_f[2](model_f[1](model_f[0](input_tensor)))
        # elif layer_index == 2:
        #     output = model_f[5](model_f[4](model_f[3](input_tensor)))
        # elif layer_index == 3:
        #     output = model_f[7](model_f[6](input_tensor))
        # elif layer_index == 4:
        #     output = model_f[9](model_f[8](input_tensor))
        # elif layer_index == 5:
        #     output = model_f[12](model_f[11](model_f[10](input_tensor)))
        # elif layer_index == 6:
        #     input_flat = input_tensor.view(-1, output_size)
        #     output = model_c[2](model_c[1](model_c[0](input_flat)))
        # elif layer_index == 7:
        #     output = model_c[5](model_c[4](model_c[3](input_tensor)))
        # elif layer_index == 8:
        #     output = model_c[6](input_tensor)

    if arch == 'resnet50':
        model_conv1 = m_model.conv1
        model_bn1 = m_model.bn1
        model_relu = m_model.relu
        model_m = m_model.maxpool
        model_layer1 = m_model.layer1
        model_layer2 = m_model.layer2
        model_layer3 = m_model.layer3
        model_layer4 = m_model.layer4
        model_a = m_model.avgpool
        model_fc = m_model.fc
        # print(input_tensor.shape)
        # print(layer_index)

        if layer_index == 1:
            output = model_conv1(input_tensor.cpu())
        elif layer_index == 2:
            output = model_bn1(input_tensor.cpu())
        elif layer_index == 3:
            output = model_relu(input_tensor.cpu())
        elif layer_index == 4:
            output = model_m(input_tensor.cpu())
        elif layer_index == 5:
            output = model_layer1(input_tensor.cpu())
        elif layer_index == 6:
            output = model_layer2(input_tensor.cpu())
        elif layer_index == 7:
            output = model_layer3(input_tensor.cpu())
        elif layer_index == 8:
            output = model_layer4(input_tensor.cpu())
        elif layer_index == 9:
            output = model_a(input_tensor.cpu())
        elif layer_index == 10:
            output = model_fc(input_tensor.cpu())

    if arch == 'vgg16':

        model_f = m_model.features
        model_a = m_model.avgpool
        model_c = m_model.classifier

        if layer_index == 1:
            output = model_f[1](model_f[0](input_tensor.cpu()).cpu()).cpu()
            # print(output.shape)
        elif layer_index == 2:
            # print(input_tensor.shape)
            output = model_f[2](input_tensor.cpu())
            # print(output.shape)
            output = model_f[3](output.cpu())
            # print(output.shape)
            output = model_f[4](output.cpu())
        elif layer_index == 3:
            output = model_f[6](model_f[5](input_tensor.cpu()).cpu()).cpu()
        elif layer_index == 4:
            output = model_f[9](model_f[8](model_f[7](input_tensor.cpu()).cpu()).cpu()).cpu()
        elif layer_index == 5:
            output = model_f[11](model_f[10](input_tensor.cpu()).cpu()).cpu()
        elif layer_index == 6:
            output = model_f[13](model_f[12](input_tensor.cpu()).cpu()).cpu()
        elif layer_index == 7:
            output = model_f[16](model_f[15](model_f[14](input_tensor.cpu()).cpu()).cpu()).cpu()
        elif layer_index == 8:
            output = model_f[18](model_f[17](input_tensor.cpu()).cpu()).cpu()
        elif layer_index == 9:
            output = model_f[20](model_f[19](input_tensor.cpu()).cpu()).cpu()
        elif layer_index == 10:
            output = model_f[23](model_f[22](model_f[21](input_tensor.cpu()).cpu()).cpu()).cpu()
        elif layer_index == 11:
            output = model_f[25](model_f[24](input_tensor.cpu()).cpu()).cpu()
        elif layer_index == 12:
            output = model_f[27](model_f[26](input_tensor.cpu()).cpu()).cpu()
        elif layer_index == 13:
            output = model_a(model_f[30](model_f[29](model_f[28](input_tensor.cpu()).cpu()).cpu()).cpu()).cpu()
        elif layer_index == 14:
            input_flat = input_tensor.cpu().view(-1, 7 * 7 * 512)
            output = model_c[1](model_c[0](input_flat).cpu()).cpu()
        elif layer_index == 15:
            output = model_c[4](model_c[3](model_c[2](input_tensor.cpu()).cpu()).cpu()).cpu()
        elif layer_index == 16:
            output = model_c[6](model_c[5](input_tensor.cpu()).cpu()).cpu()

        pass

    return output


def cal_middle_output_cuda(m_model, arch, input_tensor, layer_index):
    output = None
    if arch == 'alexnet' or arch == 'm_alexnet':
        model_f = m_model.features
        model_a = m_model.avgpool
        model_c = m_model.classifier
        #
        # print(input_tensor.shape)
        # print(layer_index)

        if layer_index == 1:
            output = model_f(input_tensor)
        elif layer_index == 2:
            output = input_tensor
        elif layer_index == 3:
            output = input_tensor
        elif layer_index == 4:
            output = input_tensor
        elif layer_index == 5:
            output = model_a(input_tensor)
            # print(output.size())
        elif layer_index == 6:
            # output = model_f[12](input_tensor)
            # output = model_f[12](input_tensor)
            # output = model_a(output)
            input_flat = output.view(-1, output_size)
            input_flat = input_tensor.view(-1, output_size)
            output = model_c[2](model_c[1](model_c[0](input_flat)))
        elif layer_index == 7:
            output = model_c[5](model_c[4](model_c[3](input_tensor)))
        elif layer_index == 8:
            output = model_c[6](input_tensor)

        # if layer_index == 1:
        #     output = model_f[2](model_f[1](model_f[0](input_tensor)))
        # elif layer_index == 2:
        #     output = model_f[5](model_f[4](model_f[3](input_tensor)))
        # elif layer_index == 3:
        #     output = model_f[7](model_f[6](input_tensor))
        # elif layer_index == 4:
        #     output = model_f[9](model_f[8](input_tensor))
        # elif layer_index == 5:
        #     output = model_f[10](input_tensor)
        #     output = model_f[11](output)
        #     output = model_f[12](output)
        #     output = model_a(output)
        #     # print(output.size())
        # elif layer_index == 6:
        #     # output = model_f[12](input_tensor)
        #     # output = model_f[12](input_tensor)
        #     # output = model_a(output)
        #     input_flat = output.view(-1, output_size)
        #     input_flat = input_tensor.view(-1, output_size)
        #     output = model_c[2](model_c[1](model_c[0](input_flat)))
        # elif layer_index == 7:
        #     output = model_c[5](model_c[4](model_c[3](input_tensor)))
        # elif layer_index == 8:
        #     output = model_c[6](input_tensor)

        # if layer_index == 1:
        #     output = model_f[2](model_f[1](model_f[0](input_tensor)))
        # elif layer_index == 2:
        #     output = model_f[5](model_f[4](model_f[3](input_tensor)))
        # elif layer_index == 3:
        #     output = model_f[7](model_f[6](input_tensor))
        # elif layer_index == 4:
        #     output = model_f[9](model_f[8](input_tensor))
        # elif layer_index == 5:
        #     output = model_f[12](model_f[11](model_f[10](input_tensor)))
        # elif layer_index == 6:
        #     input_flat = input_tensor.view(-1, output_size)
        #     output = model_c[2](model_c[1](model_c[0](input_flat)))
        # elif layer_index == 7:
        #     output = model_c[5](model_c[4](model_c[3](input_tensor)))
        # elif layer_index == 8:
        #     output = model_c[6](input_tensor)

    if arch == 'resnet50':
        model_conv1 = m_model.conv1
        model_bn1 = m_model.bn1
        model_relu = m_model.relu
        model_m = m_model.maxpool
        model_layer1 = m_model.layer1
        model_layer2 = m_model.layer2
        model_layer3 = m_model.layer3
        model_layer4 = m_model.layer4
        model_a = m_model.avgpool
        model_fc = m_model.fc
        # print(input_tensor.shape)
        # print(layer_index)

        if layer_index == 1:
            output = model_conv1(input_tensor)
        elif layer_index == 2:
            output = model_bn1(input_tensor)
        elif layer_index == 3:
            output = model_relu(input_tensor)
        elif layer_index == 4:
            output = model_m(input_tensor)
        elif layer_index == 5:
            output = model_layer1(input_tensor)
        elif layer_index == 6:
            output = model_layer2(input_tensor)
        elif layer_index == 7:
            output = model_layer3(input_tensor)
        elif layer_index == 8:
            output = model_layer4(input_tensor)
        elif layer_index == 9:
            output = model_a(input_tensor)
        elif layer_index == 10:
            output = model_fc(input_tensor)

    if arch == 'vgg16':

        model_f = m_model.features
        model_a = m_model.avgpool
        model_c = m_model.classifier

        if layer_index == 1:
            output = model_f[1](model_f[0](input_tensor.cpu()).cpu()).cpu()
            # print(output.shape)
        elif layer_index == 2:
            # print(input_tensor.shape)
            output = model_f[2](input_tensor.cpu())
            # print(output.shape)
            output = model_f[3](output.cpu())
            # print(output.shape)
            output = model_f[4](output.cpu())
        elif layer_index == 3:
            output = model_f[6](model_f[5](input_tensor.cpu()).cpu()).cpu()
        elif layer_index == 4:
            output = model_f[9](model_f[8](model_f[7](input_tensor.cpu()).cpu()).cpu()).cpu()
        elif layer_index == 5:
            output = model_f[11](model_f[10](input_tensor.cpu()).cpu()).cpu()
        elif layer_index == 6:
            output = model_f[13](model_f[12](input_tensor.cpu()).cpu()).cpu()
        elif layer_index == 7:
            output = model_f[16](model_f[15](model_f[14](input_tensor.cpu()).cpu()).cpu()).cpu()
        elif layer_index == 8:
            output = model_f[18](model_f[17](input_tensor.cpu()).cpu()).cpu()
        elif layer_index == 9:
            output = model_f[20](model_f[19](input_tensor.cpu()).cpu()).cpu()
        elif layer_index == 10:
            output = model_f[23](model_f[22](model_f[21](input_tensor.cpu()).cpu()).cpu()).cpu()
        elif layer_index == 11:
            output = model_f[25](model_f[24](input_tensor.cpu()).cpu()).cpu()
        elif layer_index == 12:
            output = model_f[27](model_f[26](input_tensor.cpu()).cpu()).cpu()
        elif layer_index == 13:
            output = model_a(model_f[30](model_f[29](model_f[28](input_tensor.cpu()).cpu()).cpu()).cpu()).cpu()
        elif layer_index == 14:
            input_flat = input_tensor.cpu().view(-1, 7 * 7 * 512)
            output = model_c[1](model_c[0](input_flat).cpu()).cpu()
        elif layer_index == 15:
            output = model_c[4](model_c[3](model_c[2](input_tensor.cpu()).cpu()).cpu()).cpu()
        elif layer_index == 16:
            output = model_c[6](model_c[5](input_tensor.cpu()).cpu()).cpu()

        pass

    return output


def cal_middle_output2(m_model, arch, input_tensor, layer_index, get_layer_dir, save_dir_pre):
    def get_features_hook(self, input, output):
        np.save(save_dir_pre + '/%s/%s_%s_layer%d.npy' % (
            (arch + '_' + get_layer_dir + '_mid_res'), arch, get_layer_dir, layer_index),
                output.detach().cpu().numpy())

    handle_feat = None

    if arch == 'alexnet' or arch == 'm_alexnet':
        model_f = m_model.features
        model_c = m_model.classifier
        if layer_index == 1:
            handle_feat = m_model.features[2].register_forward_hook(get_features_hook)
        elif layer_index == 2:
            handle_feat = m_model.features[5].register_forward_hook(get_features_hook)
        elif layer_index == 3:
            handle_feat = m_model.features[7].register_forward_hook(get_features_hook)
        elif layer_index == 4:
            handle_feat = m_model.features[9].register_forward_hook(get_features_hook)
        elif layer_index == 5:
            handle_feat = m_model.features[12].register_forward_hook(get_features_hook)
        elif layer_index == 6:
            handle_feat = m_model.classifier[3].register_forward_hook(get_features_hook)
        elif layer_index == 7:
            handle_feat = m_model.classifier[5].register_forward_hook(get_features_hook)
        elif layer_index == 8:
            handle_feat = m_model.classifier[6].register_forward_hook(get_features_hook)

    if arch == 'vgg16':

        if layer_index == 1:
            handle_feat = m_model.features[1].register_forward_hook(get_features_hook)
        elif layer_index == 2:
            handle_feat = m_model.features[4].register_forward_hook(get_features_hook)
        elif layer_index == 3:
            handle_feat = m_model.features[6].register_forward_hook(get_features_hook)
        elif layer_index == 4:
            handle_feat = m_model.features[9].register_forward_hook(get_features_hook)
        elif layer_index == 5:
            handle_feat = m_model.features[11].register_forward_hook(get_features_hook)
        elif layer_index == 6:
            handle_feat = m_model.features[13].register_forward_hook(get_features_hook)
        elif layer_index == 7:
            handle_feat = m_model.features[16].register_forward_hook(get_features_hook)
        elif layer_index == 8:
            handle_feat = m_model.features[18].register_forward_hook(get_features_hook)
        elif layer_index == 9:
            handle_feat = m_model.features[20].register_forward_hook(get_features_hook)
        elif layer_index == 10:
            handle_feat = m_model.features[23].register_forward_hook(get_features_hook)
        elif layer_index == 11:
            handle_feat = m_model.features[25].register_forward_hook(get_features_hook)
        elif layer_index == 12:
            handle_feat = m_model.features[27].register_forward_hook(get_features_hook)
        elif layer_index == 13:
            handle_feat = m_model.features[30].register_forward_hook(get_features_hook)
        elif layer_index == 14:
            handle_feat = m_model.classifier[1].register_forward_hook(get_features_hook)
        elif layer_index == 15:
            handle_feat = m_model.classifier[4].register_forward_hook(get_features_hook)
        elif layer_index == 16:
            handle_feat = m_model.classifier[6].register_forward_hook(get_features_hook)

        pass

    return handle_feat


def generate_first_order_difference(wait_to_cal_array, series_len):
    num_of_image = wait_to_cal_array.shape[0]
    random_indexs = np.random.randint(num_of_image, size=series_len)
    new_array = []
    m_cur_image = wait_to_cal_array[random_indexs[np.random.randint(series_len)]]
    for random_index in random_indexs:
        new_array.append(wait_to_cal_array[random_index] - m_cur_image)
        m_cur_image = wait_to_cal_array[random_index]

    return np.array(new_array)


def print_layer_grades_info(m_layer, m_grades_type, m_grade_avg, m_grade_std, m_grade_min, m_grade_max, m_zero_rate,
                            m_transform_type):
    print('--- LAYER %d avg %s is %f, std is %f, min is %f, max is %f, zero_rate is %f (transform=%s) ---' % (
        m_layer, m_grades_type, m_grade_avg, m_grade_std, m_grade_min, m_grade_max, m_zero_rate, m_transform_type))


def get_model():
    model = None
    model_pth_name = 'cat_dog_alexnet_model_best.pth'

    if args.arch == 'alexnet':
        model = models.__dict__[args.arch](pretrained=True)
        num_of_layer = 8
        # print('---- ARCH IS ALEXNET ----')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    elif args.arch == 'vgg16':

        model = models.__dict__[args.arch](pretrained=True)
        # model2 = models.googlenet()

        num_of_layer = 17
        print('---- ARCH IS VGG16 ----')
        normalize = transforms.Normalize(mean=[0, 0, 0],
                                         std=[1, 1, 1])
    elif args.arch == 'm_alexnet':

        num_of_layer = 8

        # print('---- ARCH IS ALEXNET ----')
        # print('---- TRAINED ----')

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        model = models.alexnet()

        state_dict = torch.load(model_pth_name)['state_dict']

        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            k = k.replace('module.', '')
            new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
    else:
        model = models.__dict__[args.arch](pretrained=True)

    return model.eval()


def gen_sin(num):
    angle_list = np.arange(0, math.pi, math.pi / num)
    angle_list_sin = []
    for angle in angle_list:
        angle_list_sin.append(math.sin(angle))

    # plt.figure()
    # plt.plot(angle_list, angle_list_sin)
    # plt.show()

    return angle_list_sin


def from_mid_to_end(m_model, arch, input_tensor, tensor_layer_index):
    m_outputs = input_tensor

    if arch == 'alexnet' or arch == 'm_alexnet':

        for m_layer_id in range(tensor_layer_index, 8):
            m_outputs = cal_middle_output(m_model, arch, m_outputs, m_layer_id + 1)
    elif arch == 'vgg16':
        for m_layer_id in range(tensor_layer_index, 16):
            m_outputs = cal_middle_output(m_model, arch, m_outputs, m_layer_id + 1)
    elif arch == 'resnet50':
        for m_layer_id in range(tensor_layer_index, 10):
            m_outputs = cal_middle_output(m_model, arch, m_outputs, m_layer_id + 1)

    return m_outputs


def random_int_list(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))
    return random_list


def bubbleSort(arr):
    order_arr = np.array(list(range(1000)))
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                order_arr[j], order_arr[j + 1] = order_arr[j + 1], order_arr[j]
    return order_arr


def partition(arr, low, high, order_arr):
    i = (low - 1)
    pivot = arr[high]

    for j in range(low, high):

        if arr[j] <= pivot:
            i = i + 1
            arr[i], arr[j] = arr[j], arr[i]
            order_arr[i], order_arr[j] = order_arr[j], order_arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    order_arr[i + 1], order_arr[high] = order_arr[high], order_arr[i + 1]
    return i + 1


def quick_sort(arr, low, high, order_arr):
    if low < high:
        pi = partition(arr, low, high, order_arr)

        quick_sort(arr, low, pi - 1, order_arr)
        quick_sort(arr, pi + 1, high, order_arr)
    # print(arr)
    return order_arr


def imagenet_class_index_dic():
    with open('imagenet_class_index.txt', 'r') as f:
        index_list = f.readlines()

    index_dic = {}
    for i in range(len(index_list)):
        index_list[i] = index_list[i].strip('\n')
        index_split = index_list[i].split(' ')
        index_dic[index_split[0]] = index_split[1]

    return index_dic


def select_random_file(file_dir, tar_dir, rate):
    pathDir = os.listdir(file_dir)

    for path in pathDir:

        path2 = os.listdir(file_dir + '/' + path)

        filenumber = len(path2)

        picknumber = int(filenumber * rate)

        sample = random.sample(path2, picknumber)

        # print(sample)

        for name in sample:
            if not os.path.exists(tar_dir + '/' + path):
                os.makedirs(tar_dir + '/' + path)
            shutil.copy(file_dir + '/' + path + '/' + name, tar_dir + '/' + path + '/' + name)


def select_random_dir(file_dir, tar_dir, val_dir, rate):
    path_dir = os.listdir(file_dir)
    # print(path_dir)
    dir_number = len(path_dir)
    pick_number = int(dir_number * rate)
    sample = random.sample(path_dir, pick_number)
    # sample = ['n01443537', 'n01530575']
    # print(sample)
    for name in sample:
        shutil.copytree(file_dir + '/' + name, tar_dir + '/' + args.tdir + '/' + name)
        shutil.copytree(file_dir + '/' + name, tar_dir + '/' + 'train_' + name + '/' + name)
        shutil.copytree(val_dir + '/' + name, tar_dir + '/' + 'val_' + name + '/' + name)
    return sample


def get_layer(get_layer_dir, arch, single, save_dir_pre, model):
    print('=== OP IS GET_LAYER ===%s' % get_layer_dir)

    if os.path.exists(save_dir_pre + '/%s/%s_%s_layer%d.npy' % (
            (arch + '_' + get_layer_dir.replace('/', '-') + '_mid_res'), arch, get_layer_dir.replace('/', '-'),
            single)):
        print('--- GET_LAYER EXISTS: %s ---' % (save_dir_pre + '/%s/%s_%s_layer%d.npy' % (
            (arch + '_' + get_layer_dir.replace('/', '-') + '_mid_res'), arch, get_layer_dir.replace('/', '-'),
            single)))
        return

    # print(model)

    num_of_layer = 0
    # model = get_model()

    normalize = transforms.Normalize(mean=[0, 0, 0],
                                     std=[1, 1, 1])
    num_class = 1000

    if arch == 'alexnet' or arch == 'm_alexnet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        num_of_layer = 8
    elif arch == 'vgg16':
        num_of_layer = 17
    elif arch == 'resnet50':
        num_of_layer = 10
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    transform_data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(get_layer_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])),
        batch_size=args.b,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    device = 'cpu'

    outputs_multi_layer = []

    get_layer_dir = get_layer_dir.replace('/', '-')

    mkdir(save_dir_pre + '/' + arch + '_' + get_layer_dir + '_mid_res')

    outputs_list = []

    outputs = None

    for i, (image_input, target) in enumerate(transform_data_loader):
        image_input, target = image_input.to(device), target.to(device)
        # print('target is ', target)
        inputs = image_input.cpu()
        if i == 0:
            outputs = inputs
        else:
            outputs = torch.cat((outputs, inputs))
    if single == 0:
        if os.path.exists(save_dir_pre + '/%s/%s_%s_layer%d.npy' % (
                (arch + '_' + get_layer_dir + '_mid_res'), arch, get_layer_dir, 0)) is not True:
            np.save(
                save_dir_pre + '/%s/%s_%s_layer%d.npy' % (
                    (arch + '_' + get_layer_dir + '_mid_res'), arch, get_layer_dir, 0),
                outputs.numpy())

        # print('%s/%s_%s_layer%d.npy saved' % ((arch + '_' + get_layer_dir + '_mid_res'), arch, get_layer_dir, 0))

    # outputs = inputs

    for layer_id in range(1, num_of_layer + 1):

        outputs = cal_middle_output(model, arch, outputs, layer_id)
        # if single == layer_id:
        #     outputs_list.append(outputs)
        #     break
        if single == 0:
            np.save(save_dir_pre + '/%s/%s_%s_layer%d.npy' % (
                (arch + '_' + get_layer_dir + '_mid_res'), arch, get_layer_dir, layer_id),
                    outputs.detach().cpu().numpy())
        elif layer_id == single:
            np.save(save_dir_pre + '/%s/%s_%s_layer%d.npy' % (
                (arch + '_' + get_layer_dir + '_mid_res'), arch, get_layer_dir, layer_id),
                    outputs.detach().cpu().numpy())
            print('--- GET LAYER %d SUCCESS ---' % single, get_layer_dir)
            break
    # print('%s/%s_%s_layer%d.npy saved' % (
    #     (arch + '_' + get_layer_dir + '_mid_res'), arch, get_layer_dir, layer_id))

    # outputs_cat = outputs_list[0]
    # for i in range(1, len(outputs_list)):
    #     outputs_cat = torch.cat((outputs_cat, outputs_list[i]))
    #
    #     np.save(save_dir_pre + '/%s/%s_%s_layer%d.npy' % (
    #         (arch + '_' + get_layer_dir + '_mid_res'), arch, get_layer_dir, single),
    #             outputs_cat.detach().cpu().numpy())
    #     print('--- GET LAYER %d SUCCESS ---' % single, get_layer_dir)


def get_layer_cuda(get_layer_dir, arch, single, save_dir_pre, model):
    # print('=== OP IS GET_LAYER ===%s' % get_layer_dir)

    if os.path.exists(save_dir_pre + '/%s/%s_%s_layer%d.npy' % (
            (arch + '_' + get_layer_dir.replace('/', '-') + '_mid_res'), arch, get_layer_dir.replace('/', '-'),
            single)):
        # print('--- GET_LAYER EXISTS: %s ---' % (save_dir_pre + '/%s/%s_%s_layer%d.npy' % (
        #     (arch + '_' + get_layer_dir.replace('/', '-') + '_mid_res'), arch, get_layer_dir.replace('/', '-'),
        #     single)))
        return

    # print(model)

    num_of_layer = 0
    # model = get_model()

    normalize = transforms.Normalize(mean=[0, 0, 0],
                                     std=[1, 1, 1])
    num_class = 1000

    if arch == 'alexnet' or arch == 'm_alexnet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        num_of_layer = 8
    elif arch == 'vgg16':
        num_of_layer = 17
    elif arch == 'resnet50':
        num_of_layer = 10
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    transform_data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(get_layer_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])),
        batch_size=args.b,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #
    # torch.cuda.set_device(1)
    #
    # device = torch.device("cuda:1")
    #
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

    # print(device)
    # device = 'cpu'

    outputs_multi_layer = []

    get_layer_dir = get_layer_dir.replace('/', '-')

    mkdir(save_dir_pre + '/' + arch + '_' + get_layer_dir + '_mid_res')

    outputs_list = []

    outputs = None

    for i, (image_input, target) in enumerate(transform_data_loader):
        image_input, target = image_input.to(device), target.to(device)
        # print('target is ', target)
        inputs = image_input
        if i == 0:
            outputs = inputs
        else:
            outputs = torch.cat((outputs, inputs))
    if single == 0:
        if os.path.exists(save_dir_pre + '/%s/%s_%s_layer%d.npy' % (
                (arch + '_' + get_layer_dir + '_mid_res'), arch, get_layer_dir, 0)) is not True:
            np.save(
                save_dir_pre + '/%s/%s_%s_layer%d.npy' % (
                    (arch + '_' + get_layer_dir + '_mid_res'), arch, get_layer_dir, 0),
                outputs.numpy())

        # print('%s/%s_%s_layer%d.npy saved' % ((arch + '_' + get_layer_dir + '_mid_res'), arch, get_layer_dir, 0))

    # outputs = inputs

    for layer_id in range(1, num_of_layer + 1):

        outputs = cal_middle_output_cuda(model, arch, outputs, layer_id)
        # if single == layer_id:
        #     outputs_list.append(outputs)
        #     break
        if single == 0:
            np.save(save_dir_pre + '/%s/%s_%s_layer%d.npy' % (
                (arch + '_' + get_layer_dir + '_mid_res'), arch, get_layer_dir, layer_id),
                    outputs.cuda().data.cpu().numpy())
        elif layer_id == single:
            np.save(save_dir_pre + '/%s/%s_%s_layer%d.npy' % (
                (arch + '_' + get_layer_dir + '_mid_res'), arch, get_layer_dir, layer_id),
                    outputs.cuda().data.cpu().numpy())
            print('--- GET LAYER %d SUCCESS ---' % single, get_layer_dir)
            break
    # print('%s/%s_%s_layer%d.npy saved' % (
    #     (arch + '_' + get_layer_dir + '_mid_res'), arch, get_layer_dir, layer_id))

    # outputs_cat = outputs_list[0]
    # for i in range(1, len(outputs_list)):
    #     outputs_cat = torch.cat((outputs_cat, outputs_list[i]))
    #
    #     np.save(save_dir_pre + '/%s/%s_%s_layer%d.npy' % (
    #         (arch + '_' + get_layer_dir + '_mid_res'), arch, get_layer_dir, single),
    #             outputs_cat.detach().cpu().numpy())
    #     print('--- GET LAYER %d SUCCESS ---' % single, get_layer_dir)


def get_layer2(get_layer_dir, arch, single, save_dir_pre):
    print('=== OP IS GET_LAYER ===%s' % get_layer_dir)

    if os.path.exists(save_dir_pre + '/%s/%s_%s_layer%d.npy' % (
            (arch + '_' + get_layer_dir.replace('/', '-') + '_mid_res'), arch, get_layer_dir.replace('/', '-'),
            single)):
        print('--- GET_LAYER EXISTS ---')
        return

        # print(model)

    num_of_layer = 0
    model = get_model()

    normalize = transforms.Normalize(mean=[0, 0, 0],
                                     std=[1, 1, 1])
    num_class = 1000

    if arch == 'alexnet' or arch == 'm_alexnet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        num_of_layer = 8
    elif arch == 'vgg16':
        num_of_layer = 17

    elif arch == 'resnet34':
        num_of_layer = 18

    transform_data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(get_layer_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])),
        batch_size=args.b,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'

    outputs_multi_layer = []

    get_layer_dir = get_layer_dir.replace('/', '-')

    mkdir(save_dir_pre + '/' + arch + '_' + get_layer_dir + '_mid_res')

    outputs = None

    for i, (image_input, target) in enumerate(transform_data_loader):
        image_input, target = image_input.to(device), target.to(device)
        # print('target is ', target)
        inputs = image_input.cpu()
        if i == 0:
            outputs = inputs
        else:
            outputs = torch.cat((outputs, inputs))

    if single == 0:
        if os.path.exists(save_dir_pre + '/%s/%s_%s_layer%d.npy' % (
                (arch + '_' + get_layer_dir + '_mid_res'), arch, get_layer_dir, 0)) is not True:
            np.save(
                save_dir_pre + '/%s/%s_%s_layer%d.npy' % (
                    (arch + '_' + get_layer_dir + '_mid_res'), arch, get_layer_dir, 0),
                outputs.numpy())

    for layer_id in range(1, num_of_layer + 1):

        if layer_id != 0 and layer_id != single:
            continue

        handle_feat = cal_middle_output2(model, arch, outputs, layer_id, get_layer_dir, save_dir_pre)
        a = model(outputs)
        # print('a.shape: ', a.shape)
        a.backward(torch.ones(outputs.shape[0], num_class).cuda())
        handle_feat.remove()

        # print('%s/%s_%s_layer%d.npy saved' % (
        #     (arch + '_' + get_layer_dir + '_mid_res'), arch, get_layer_dir, layer_id))
        print('--- GET LAYER %d SUCCESS ---' % single, get_layer_dir)


def trans_tensor_from_image(dir, arch):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if arch == 'alexnet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    transform_data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])),
        batch_size=args.b,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # transform_data_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(dir, transforms.Compose([  # [1]
    #         transforms.Resize(256),  # [2]
    #         transforms.CenterCrop(227),  # [3]
    #         transforms.ToTensor(),  # [4]
    #     ])),
    #     batch_size=args.b, shuffle=False,
    #     num_workers=2, pin_memory=True)

    outputs = None

    with torch.no_grad():
        # print('2')
        for it_loader_batch_i, (image_input, target) in enumerate(transform_data_loader):
            image_input, target = image_input.to(device), target.to(device)

            # print('target is ', target)
            inputs = image_input.cpu()

            if it_loader_batch_i == 0:
                outputs = inputs
            else:
                outputs = torch.cat((outputs, inputs))

    return outputs


def get_accuracy_from_output(m_outputs, class_index):
    index_list = []

    m_outputs = m_outputs.detach().numpy()

    for j in range(m_outputs.shape[0]):
        index_list.append(np.argmax(m_outputs[j]))

    index_counts = Counter(index_list)
    correct_item_num = index_counts[int(class_index)]
    val_item_num = m_outputs.shape[0]
    val_acc = correct_item_num / val_item_num

    top_k_correct_item_num = 0
    top_k_contain = False
    for i in index_counts.most_common(5):
        if i[0] == int(class_index):
            top_k_contain = True
        top_k_correct_item_num += i[1]

    if not top_k_contain:
        top_k_correct_item_num = 0

    top_k_val_acc = top_k_correct_item_num / val_item_num

    return val_acc, correct_item_num, val_item_num, top_k_val_acc, top_k_correct_item_num


def do_cal_jsd_list_between_tensors(param):
    # print('=====do=====', param[2])
    cal_jsd_list_between_tensors(param[0], param[1], param[2], param[3], param[4], param[5])


def do_cal_jsd_list_between_tensors3(param):
    # print('=====do=====', param[2])
    # t0 = get_min_pool_npy(param[0], '', param[5], param[6], param[7], param[8], param[10], param[11], is_save=False)
    # t1 = get_min_pool_npy(param[1], '', param[5], param[6], param[7], param[9], param[10], param[11])
    t0 = np.load(param[8].split('.')[0] + '_' + param[10] + '_' + param[11] + '.npy')
    t1 = np.load(param[9].split('.')[0] + '_' + param[10] + '_' + param[11] + '.npy')
    cal_jsd_list_between_tensors(t0, t1, param[2], param[3], param[4])


def do_cal_jsd_list_between_tensors_avg(param):
    # print('=====do=====', param[2])
    # if os.path.exists(param[2]):
    #     print('--- DS2 EXISTS: %s ---' % (param[2]))
    #     return
    t0 = get_min_pool_npy(param[0], '', param[5], param[6], param[7], param[8], param[10], param[11], is_save=False)
    # t1 = get_min_pool_npy(param[1], '', param[5], param[6], param[7], param[9], param[10], param[11])
    # t0 = np.load(param[8].split('.')[0] + '_' + param[10] + '_' + param[11] + '.npy')
    t1 = param[9].split('.')[0] + '_' + param[10] + '_' + param[11] + '.npy'
    # print('t1', t1)
    cal_jsd_list_between_tensors(t0, t1, param[2], param[3], param[4], param[-1])


def do_get_min_pool_npy(param):
    get_min_pool_npy(param[0], param[1], param[2], param[3], param[4], param[5], param[6], param[7], param[8], param[9])


def do_get_min_pool_npy2(param):
    tensor_array = param[4]
    # print(tensor_array.shape)

    for time_index in range(1, int(param[0]) + 1):
        keep_node_pools = param[1][time_index - 1]
        keep_node_pools_weight = param[2][time_index - 1]
        keep_node_set = param[3]
        tensor_array = get_min_pool_npy(tensor_array, '', keep_node_pools, keep_node_pools_weight, keep_node_set, '',
                                        '', '', False)

        # print(time_index==int(param[0]))
        # if time_index == int(param[0]):
    deleted_node_set = list(set(list(range(output_size))) - set(param[3]))
    # print(tensor_array.shape)
    # tensor_array[:, deleted_node_set] = 0
    np.save(param[-2], tensor_array)
    print('---%d: completed %s---' % (param[-1], param[-2]))


def cal_jsd_list_between_tensors(tensor_a, tensor_b, save_path, save_path2, mt, count=-1):
    """
    :param mt:
    :param save_path2: store the abs d file
    :param tensor_a: dim: 200(1080)*output_size
    :param tensor_b: dim: 200(1080)*output_size
    :param save_path: store the js file
    :return:
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #
    # torch.cuda.set_device(1)
    #
    # device = torch.device("cuda:1")
    #
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

    # print('=====started=====' + save_path)

    if isinstance(tensor_a, str):
        tensor_a = np.load(tensor_a)

    if isinstance(tensor_b, str):
        tensor_b = np.load(tensor_b)

    tensor_a = tensor_a.reshape(tensor_a.shape[0], -1)
    tensor_b = tensor_b.reshape(tensor_b.shape[0], -1)

    tensor_a, tensor_b = two_tensor_array_normalization(tensor_a, tensor_b)

    metric_list = []
    if mt == 'js':
        abs_d_list = []
        for i in range(min(tensor_a.shape[1], tensor_b.shape[1])):
            # cal the js between two vector, dim: 200 or 1080
            metric, abs_d = cal_jsd3(tensor_a[:, i], tensor_b[:, i])
            metric_list.append(np.asarray(metric))
            abs_d_list.append(abs_d)

        metric_list_np = np.array(metric_list)
        abs_d_list_np = np.array(abs_d_list)
        np.save(save_path, metric_list_np)
        np.save(save_path2, abs_d_list_np)

    if mt == 'wa':
        for i in range(min(tensor_a.shape[1], tensor_b.shape[1])):
            # cal the js between two vector, dim: 200 or 1080

            # print(index_arr.shape, tensor_a[:, i].shape)

            # if i != 4029:
            #     continue
            # metric = 0
            # try:
            # print(i)
            # print(tensor_a[:, i])
            # print(tensor_b[:, i])
            # print(save_path)

            max_img = max(tensor_a[:, i].shape[0], tensor_b[:, i].shape[0])
            min_img = min(tensor_a[:, i].shape[0], tensor_b[:, i].shape[0])
            index_arr = np.array(list(range(min_img - 1)))
            # print(tensor_a[:, i][-min_img:-1])
            # print(tensor_b[:, i][-min_img:-1])
            # print('a', np.count_nonzero(tensor_a[:, i]))
            # print('b', np.count_nonzero(tensor_b[:, i]))
            metric = None
            try:
                metric = wasserstein_distance(index_arr, index_arr, tensor_a[:, i][-min_img:-1] + 0.0000001,
                                              tensor_b[:, i][-min_img:-1] + 0.0000001)
            except ValueError:
                print(save_path)
                return
            # except ValueError:
            #     # print(ValueError)
            #     print(i)
            #     print(tensor_a[:, i])
            #     print(tensor_b[:, i])
            #     print(save_path)
            # if metric >60:
            #     print(metric)
            #     print(tensor_a[:, i])
            #     print(tensor_b[:, i])
            #     sys.exit(0)
            # metric = wasserstein_distance(tensor_a[:, i] + 0.0000001, tensor_b[:, i] + 0.0000001, index_arr, index_arr)
            metric_list.append(np.asarray(metric))

        metric_list_np = np.array(metric_list)
        np.save(save_path, metric_list_np)
        # print('==========')
        # print(metric_list_np[0])
        # print(metric_list_np[1])
        # print(metric_list_np[2])
        # print('==========')

    print('%d: =====completed=====' % (count) + save_path)


def cal_jsd_list_between_tensors2(tensor_a, tensor_b, save_path):
    """
    :param tensor_a: dim: 200(1080)*output_size
    :param tensor_b: dim: 200(1080)*output_size
    :param save_path: store the js file
    :return:
    """

    # print('=====started=====' + save_path)

    tensor_a = tensor_a.reshape(tensor_a.shape[0], -1)
    tensor_b = tensor_b.reshape(tensor_b.shape[0], -1)

    # tensor_a, tensor_b = two_tensor_array_normalization2(tensor_a, tensor_b)

    metric_list = []
    for i in range(min(tensor_a.shape[1], tensor_b.shape[1])):
        # cal the js between two vector, dim: 200 or 1080
        metric, abs_d = cal_jsd2(tensor_a[:, i], tensor_b[:, i])
        metric_list.append(np.asarray(metric))

    metric_list_np = np.array(metric_list)
    np.save(save_path, metric_list_np)

    print('=====completed=====' + save_path)


def get_deleted_node_output(img_dir, class_index, deleted_node_set, arch, layer_index, model, xrate, s_img_tensor=None):
    save_dir_pre = args.tdir + '_layer_npy'

    # print('=== get_deleted_node_output2 ===')

    if s_img_tensor is None:
        train_dir = img_dir.replace('/', '-')
        s_img_tensor = torch.from_numpy(np.load(
            save_dir_pre + '/%s/%s_%s_layer%d.npy' % (
                (args.arch + '_' + train_dir + '_mid_res'), args.arch, train_dir, layer_index)))

    origin_shape = s_img_tensor.shape
    # print('=== get_deleted_node_output21 ===')
    s_img_tensor = s_img_tensor.reshape(s_img_tensor.shape[0], -1)
    # print('=== get_deleted_node_output22 ===')
    s_img_tensor[:, deleted_node_set] = s_img_tensor[:, deleted_node_set] * xrate

    # print('=== get_deleted_node_output3 ===')

    s_img_tensor = s_img_tensor.reshape(origin_shape)
    s_img_tensor = from_mid_to_end(model, arch, s_img_tensor, layer_index)

    # print('=== get_deleted_node_output4 ===')

    print(s_img_tensor)

    return s_img_tensor


def get_deleted_node_accuracy(img_dir, class_index, deleted_node_set, arch, layer_index, model, xrate,
                              return_output=False):
    img_tensor = trans_tensor_from_image(img_dir, arch)

    num_of_layer = 8

    if args.arch == 'vgg16':
        num_of_layer = 16

    if args.arch == 'resnet50':
        num_of_layer = 10

    s_img_tensor = None

    for i in range(1, num_of_layer + 1):
        img_tensor = cal_middle_output(model, arch, img_tensor, i)
        if i == layer_index:
            s_img_tensor = img_tensor

    # print(img_tensor.shape)
    # print('output_sum: ', np.sum(img_tensor.detach().cpu().numpy()[0]))

    origin_shape = s_img_tensor.shape
    origin_acc, o_acc_item, o_item_num, origin_top_k_acc, o_top_k_acc_item = get_accuracy_from_output(img_tensor,
                                                                                                      class_index)

    s_img_tensor = s_img_tensor.reshape(s_img_tensor.shape[0], -1)
    s_img_tensor[:, deleted_node_set] = s_img_tensor[:, deleted_node_set] * xrate

    s_img_tensor = s_img_tensor.reshape(origin_shape)
    s_img_tensor = from_mid_to_end(model, arch, s_img_tensor, layer_index)

    final_acc, f_acc_item, f_item_num, final_top_k_acc, f_top_k_acc_item = get_accuracy_from_output(s_img_tensor,
                                                                                                    class_index)

    if return_output:
        return origin_acc, o_acc_item, origin_top_k_acc, o_top_k_acc_item, o_item_num, final_acc, f_acc_item, final_top_k_acc, f_acc_item, f_item_num, s_img_tensor

    return origin_acc, o_acc_item, origin_top_k_acc, o_top_k_acc_item, o_item_num, final_acc, f_acc_item, final_top_k_acc, f_acc_item, f_item_num


def get_min_pool_npy(npy, npy_name, keep_node_pools, keep_node_pools_weight, keep_node_set, t_save_path, time_type,
                     class_name, is_save=True, count=-1):
    # print(npy.shape)
    # print(keep_node_pools.shape)
    # print(keep_node_pools_weight.shape)
    # print('len(keep_node_set): ', len(keep_node_set))

    # relu = nn.ReLU()

    # print('%d: started %s' % (count, t_save_path))
    # print(11111111111)

    if isinstance(npy, str):
        npy = np.load(t_save_path)
    npy = npy.reshape(npy.shape[0], -1)
    # npy = torch.from_numpy(npy).double().cuda()

    # res = []
    # for j in range(len(keep_node_pools)):
    #     res.append(npy[:, keep_node_pools[j][1]] * np.array(keep_node_pools_weight[j]))
    #
    # # print(res)
    #
    # for i in range(len(keep_node_pools)):
    #     for j in range(npy.shape[0]):
    #         npy[i, keep_node_pools[j][0]] = np.sum(res[j][i])

    # npy_res = torch.from_numpy(np.zeros(npy.shape)).double().cuda()
    npy_res = np.zeros(npy.shape)

    for i in range(npy.shape[0]):
        for j in range(len(keep_node_pools)):
            npy_res[i][keep_node_pools[j][0]] = np.dot(
                npy[i][keep_node_pools[j][1]], np.array(keep_node_pools_weight[j]).transpose())
            # npy_res[i][keep_node_pools[j][0]] = npy[i][keep_node_pools[j][1]].mul(torch.from_numpy(np.array(keep_node_pools_weight[j])).cuda()).sum()

    npy_res[npy_res < 0] = 0

    # print(np.count_nonzero(npy))

    if is_save:
        # np.save(t_save_path.split('.')[0] + '_' + time_type + '_' + class_name + '.npy', npy_res.detach().cpu().numpy())
        np.save(t_save_path.split('.')[0] + '_' + time_type + '_' + class_name + '.npy', npy_res)
        print(str(count), ': ', t_save_path.split('.')[0] + '_' + time_type + '_' + class_name + '.npy')
    return npy_res


def get_min_npy_pool_npy(npy, npy_name, keep_node_pools, keep_node_pools_weight, keep_node_set, t_save_path, time_type,
                         class_name, is_save=True):
    # print(npy.shape)
    # print(keep_node_pools.shape)
    # print(keep_node_pools_weight.shape)
    # print('len(keep_node_set): ', len(keep_node_set))

    # relu = nn.ReLU()
    npy = npy.reshape(npy.shape[0], -1)

    npy_res = np.zeros(npy.shape)

    for j in range(len(keep_node_pools)):
        # print(npy[i][keep_node_set[j]] )
        # print(np.max(npy[i][keep_node_pools[j]]))
        # npy[i][keep_node_set[j]] = npy[i][keep_node_set[j]] + np.sum(
        #     npy[i][keep_node_pools[j]] * np.array(keep_node_pools_weight[j]))
        # print(npy[i][keep_node_set[j]])
        # print(keep_node_pools[j])
        # print(j)
        npy_res[:, keep_node_pools[j][0]] = np.sum(npy[:, keep_node_pools[j][1]] * np.array(keep_node_pools_weight[j]))

    npy_res[npy_res < 0] = 0

    if is_save:
        np.save(t_save_path.split('.')[0] + '_' + time_type + '_' + class_name + '.npy', npy_res)
        print(t_save_path.split('.')[0] + '_' + time_type + '_' + class_name + '.npy')
    return npy_res


def get_best_pool_npy(npy, npy_name, best_node_pools, best_node_pools_weight, best_node_set, t_save_path, time_type,
                      class_name, is_save=True):
    npy = np.zeros(npy.shape)
    for i in range(npy.shape[0]):
        for j in range(len(best_node_set)):
            npy[i][best_node_set[j]] = np.sum(
                npy[i][best_node_pools[j]] * np.array(best_node_pools_weight[j]))
            pass

    npy[npy < 0] = 0

    if is_save:
        np.save(t_save_path.split('.')[0] + '_' + time_type + '_' + class_name + '.npy', npy)

    return npy


def get_min_pool_accuracy(img_dir, class_index, deleted_node_set, keep_node_pools, keep_node_pools_weight, arch,
                          layer_index, model, xrate,
                          return_output=False):
    img_tensor = trans_tensor_from_image(img_dir, arch)

    num_of_layer = 8

    if args.arch == 'vgg16':
        num_of_layer = 16

    s_img_tensor = None

    for i in range(1, num_of_layer + 1):
        img_tensor = cal_middle_output(model, arch, img_tensor, i)
        if i == layer_index:
            s_img_tensor = img_tensor

    keep_node_set = np.array(list(set(list(range(output_size))) - set(deleted_node_set)))

    # print(img_tensor.shape)
    # print('output_sum: ', np.sum(img_tensor.detach().cpu().numpy()[0]))

    origin_shape = s_img_tensor.shape
    origin_acc, o_acc_item, o_item_num, origin_top_k_acc, o_top_k_acc_item = get_accuracy_from_output(img_tensor,
                                                                                                      class_index)

    s_img_tensor = s_img_tensor.reshape(s_img_tensor.shape[0], -1)
    s_img_tensor[:, deleted_node_set] = s_img_tensor[:, deleted_node_set] * xrate

    s_img_tensor = s_img_tensor.detach().numpy()

    for i in range(s_img_tensor.shape[0]):
        for j in range(len(keep_node_set)):
            # print(j)
            # print(keep_node_pools)
            # print(keep_node_set[keep_node_pools])
            # print(s_img_tensor[i][keep_node_pools[j]])
            # print(np.array(keep_node_pools_weight[j]))
            s_img_tensor[i][keep_node_set[j]] = np.sum(
                s_img_tensor[i][keep_node_pools[j]] * np.array(keep_node_pools_weight[j]))
            pass

    mkdir('metric_list_npy_dir/' + class_index)
    np.save('metric_list_npy_dir/' + class_index + '/pool_%s_file_list_%s.npy' % (
        args.mt, class_index), s_img_tensor)

    print('s_img_tensor shape: ', s_img_tensor.shape)

    s_img_tensor = torch.from_numpy(s_img_tensor)

    s_img_tensor = s_img_tensor.reshape(origin_shape)
    s_img_tensor = from_mid_to_end(model, arch, s_img_tensor, layer_index)

    final_acc, f_acc_item, f_item_num, final_top_k_acc, f_top_k_acc_item = get_accuracy_from_output(s_img_tensor,
                                                                                                    class_index)

    if return_output:
        return origin_acc, o_acc_item, origin_top_k_acc, o_top_k_acc_item, o_item_num, final_acc, f_acc_item, final_top_k_acc, f_acc_item, f_item_num, s_img_tensor

    return origin_acc, o_acc_item, origin_top_k_acc, o_top_k_acc_item, o_item_num, final_acc, f_acc_item, final_top_k_acc, f_acc_item, f_item_num


def do_transform(param):
    img_mode = transform(param[0], param[1], param[2], param[3])
    print('--- TRANSFORM SUCCESS ---', param[1])


def do_random_transform_one_img(param):
    transform_random_one_image2(param[0], param[1], param[2])


def do_get_layer(param):
    get_layer(param[0], param[1], param[2], param[3], param[4])


def do_get_layer_cuda(param):
    get_layer_cuda(param[0], param[1], param[2], param[3], param[4])


def do_shutil_copy(param):
    shutil.copy(param[0], param[1])
    # print(param[1])


# def do_random_transform_one_img(param):
#     print('random_transform: %s' % param[0])
#     mkdir(param[1])
#     random_transform_one_img(param[0], param[1])
#     print('completed random_transform: %s' % param[0])


def do_coint(param):
    return cal_coint(param[0], param[1], param[2], param[3], param[4])


def do_granger(param):
    return cal_granger(param[0], param[1], param[2], param[3], param[4])


def cal_granger(multi_class_item, t, operation, mode, phase):
    confidence_bound = args.cb
    p_value_bound = args.pb

    # if os.path.exists(
    #         t + '/'+args.arch+'/metric_list_npy_dir/' + multi_class_item + '/coint_metric_plus_abs_d_%s_%s_%d_%s_c%d_%.2f_list.npy' % (
    #                 operation, mode, args.l, phase, confidence_bound, p_value_bound)):
    #     print('=== %s exists ===' % ('coint_metric_plus_abs_d_%s_%s_%d_%s_c%d_%.2f_list.npy' % (
    #         operation, mode, args.l, phase, confidence_bound, p_value_bound)))
    #     return

    metric_file_name_list = []
    abs_d_file_name_list = []

    metric_list_np_dir = t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/metric_list_jsd_%s_%s_%d_%s_npy' % (
        operation, mode, args.l, phase)
    abs_d_list_np_dir = t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/metric_list_abs_d_%s_%s_%d_%s_npy' % (
        operation, mode, args.l, phase)

    t_num = len(os.listdir(metric_list_np_dir)) + 1

    for seq_index in range(1, t_num):
        if seq_index < 10:
            seq_index = '000' + str(seq_index)
        elif seq_index < 100:
            seq_index = '00' + str(seq_index)
        elif seq_index < 1000:
            seq_index = '0' + str(seq_index)
        else:
            seq_index = str(seq_index)

        metric_file_name_list.append(
            'metric_list_jsd_%s_%s_%d_%s_' % (operation, mode, args.l, phase) + seq_index + '.npy')
        abs_d_file_name_list.append(
            'metric_list_abs_d_%s_%s_%d_%s_' % (operation, mode, args.l, phase) + seq_index + '.npy')

    metric_file_list = []
    abs_d_file_list = []

    for metric_file in metric_file_name_list:
        metric_file_list.append(np.array(np.load(metric_list_np_dir + '/' + metric_file)))
    for abs_d_file in abs_d_file_name_list:
        abs_d_file_list.append(np.array(np.load(abs_d_list_np_dir + '/' + abs_d_file)))

    metric_file_list = np.array(metric_file_list)
    abs_d_file_list = np.array(abs_d_file_list)
    abs_d_file_list = abs_d_file_list.reshape(metric_file_list.shape)

    np.save(
        t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/metric_file_time_zero_js_%s.npy' % multi_class_item,
        metric_file_list)
    np.save(
        t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/abs_d_file_time_zero_abs_d_%s.npy' % multi_class_item,
        abs_d_file_list)

    # for i in range(metric_file_list.shape[1]):
    #     # print(metric_file_list[:, i].shape)
    #     metric_jsd_point_value_list.append(metric_file_list[:, i])
    #     metric_jsd_plus_abs_d_point_value_list.append(metric_file_list[:, i] + abs_d_file_list[:, i] * 0.1)

    gb_seq_no_linear = image_enhance.gen_gb_seq_no_linear_no_noise()[1:t_num]

    granger_node_js_list = []
    not_granger_node_js_list = []
    granger_node_plus_abs_d_list = []
    not_granger_node_plus_abs_d_list = []

    print('=== cal granger %s ===' % multi_class_item)
    for i in range(metric_file_list.shape[1]):
        if i % 1000 == 0 and i != 0:
            print('=== %s cal process: %d ===' % (multi_class_item, i))
        node_js_plus_abs_d = metric_file_list[:, i] + abs_d_file_list[:, i] * 0.1
        node_js = metric_file_list[:, i]
        # adf_js = adfuller(node_js)
        # adf_plus_abs_d = adfuller(node_js_plus_abs_d)

        confidence_value_js, p_value_js, confidence_bound_js = grangercausalitytests(
            (node_js.T + 0.0000001).vstack(gb_seq_no_linear.T))
        # if confidence_value_js < confidence_bound_js[confidence_bound] and p_value_js < p_value_bound and \
        #         adf_js[0] < list(adf_js[4].values())[0]:
        if confidence_value_js < confidence_bound_js[confidence_bound] and p_value_js < p_value_bound:
            granger_node_js_list.append(i)

        confidence_value_js, p_value_js, confidence_bound_js = grangercausalitytests(node_js_plus_abs_d,
                                                                                     gb_seq_no_linear)
        # if confidence_value_js < confidence_bound_js[confidence_bound] and p_value_js < p_value_bound and \
        #         adf_plus_abs_d[0] < list(adf_plus_abs_d[4].values())[0]:
        if confidence_value_js < confidence_bound_js[confidence_bound] and p_value_js < p_value_bound:
            granger_node_plus_abs_d_list.append(i)

    print('=== granger node js list len: %d ===' % len(granger_node_js_list))
    print('=== granger node plus abs d list len: %d ===' % len(granger_node_plus_abs_d_list))

    np.save(
        t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/granger_metric_js_%s_%s_%d_%s_c%d_%.2f_list.npy' % (
            operation, mode, args.l, phase, confidence_bound, p_value_bound), np.array(granger_node_js_list))
    np.save(
        t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/granger_metric_plus_abs_d_%s_%s_%d_%s_c%d_%.2f_list.npy' % (
            operation, mode, args.l, phase, confidence_bound, p_value_bound),
        np.array(granger_node_plus_abs_d_list))
    pass


def cal_coint(multi_class_item, t, operation, mode, phase):
    confidence_bound = args.cb
    p_value_bound = args.pb

    # if os.path.exists(
    #         t + '/'+args.arch+'/metric_list_npy_dir/' + multi_class_item + '/coint_metric_plus_abs_d_%s_%s_%d_%s_c%d_%.2f_list.npy' % (
    #                 operation, mode, args.l, phase, confidence_bound, p_value_bound)):
    #     print('=== %s exists ===' % ('coint_metric_plus_abs_d_%s_%s_%d_%s_c%d_%.2f_list.npy' % (
    #         operation, mode, args.l, phase, confidence_bound, p_value_bound)))
    #     return

    metric_file_name_list = []
    abs_d_file_name_list = []

    metric_list_np_dir = t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/metric_list_jsd_%s_%s_%d_%s_npy' % (
        operation, mode, args.l, phase)
    abs_d_list_np_dir = t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/metric_list_abs_d_%s_%s_%d_%s_npy' % (
        operation, mode, args.l, phase)

    t_num = len(os.listdir(metric_list_np_dir)) + 1

    for seq_index in range(1, t_num):
        if seq_index < 10:
            seq_index = '000' + str(seq_index)
        elif seq_index < 100:
            seq_index = '00' + str(seq_index)
        elif seq_index < 1000:
            seq_index = '0' + str(seq_index)
        else:
            seq_index = str(seq_index)

        metric_file_name_list.append(
            'metric_list_jsd_%s_%s_%d_%s_' % (operation, mode, args.l, phase) + seq_index + '.npy')
        abs_d_file_name_list.append(
            'metric_list_abs_d_%s_%s_%d_%s_' % (operation, mode, args.l, phase) + seq_index + '.npy')

    metric_file_list = []
    abs_d_file_list = []

    for metric_file in metric_file_name_list:
        metric_file_list.append(np.array(np.load(metric_list_np_dir + '/' + metric_file)))
    for abs_d_file in abs_d_file_name_list:
        abs_d_file_list.append(np.array(np.load(abs_d_list_np_dir + '/' + abs_d_file)))

    metric_file_list = np.array(metric_file_list)
    abs_d_file_list = np.array(abs_d_file_list)
    abs_d_file_list = abs_d_file_list.reshape(metric_file_list.shape)

    np.save(
        t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/metric_file_time_zero_js_%s.npy' % multi_class_item,
        metric_file_list)
    np.save(
        t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/abs_d_file_time_zero_abs_d_%s.npy' % multi_class_item,
        abs_d_file_list)

    # for i in range(metric_file_list.shape[1]):
    #     # print(metric_file_list[:, i].shape)
    #     metric_jsd_point_value_list.append(metric_file_list[:, i])
    #     metric_jsd_plus_abs_d_point_value_list.append(metric_file_list[:, i] + abs_d_file_list[:, i] * 0.1)

    gb_seq_no_linear = image_enhance.gen_gb_seq_no_linear_no_noise()[1:t_num]

    coint_node_js_list = []
    not_coint_node_js_list = []
    coint_node_plus_abs_d_list = []
    not_coint_node_plus_abs_d_list = []

    print('=== cal coint %s ===' % multi_class_item)
    for i in range(metric_file_list.shape[1]):
        if i % 1000 == 0 and i != 0:
            print('=== %s cal process: %d ===' % (multi_class_item, i))
        node_js_plus_abs_d = metric_file_list[:, i] + abs_d_file_list[:, i] * 0.1
        node_js = metric_file_list[:, i]
        # adf_js = adfuller(node_js)
        # adf_plus_abs_d = adfuller(node_js_plus_abs_d)

        confidence_value_js, p_value_js, confidence_bound_js = coint(node_js + 0.0000001, gb_seq_no_linear)
        # if confidence_value_js < confidence_bound_js[confidence_bound] and p_value_js < p_value_bound and \
        #         adf_js[0] < list(adf_js[4].values())[0]:
        if confidence_value_js < confidence_bound_js[confidence_bound] and p_value_js < p_value_bound:
            coint_node_js_list.append(i)

        confidence_value_js, p_value_js, confidence_bound_js = coint(node_js_plus_abs_d, gb_seq_no_linear)
        # if confidence_value_js < confidence_bound_js[confidence_bound] and p_value_js < p_value_bound and \
        #         adf_plus_abs_d[0] < list(adf_plus_abs_d[4].values())[0]:
        if confidence_value_js < confidence_bound_js[confidence_bound] and p_value_js < p_value_bound:
            coint_node_plus_abs_d_list.append(i)

    print('=== coint node js list len: %d ===' % len(coint_node_js_list))
    print('=== coint node plus abs d list len: %d ===' % len(coint_node_plus_abs_d_list))

    np.save(
        t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/coint_metric_js_%s_%s_%d_%s_c%d_%.2f_list.npy' % (
            operation, mode, args.l, phase, confidence_bound, p_value_bound), np.array(coint_node_js_list))
    np.save(
        t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/coint_metric_plus_abs_d_%s_%s_%d_%s_c%d_%.2f_list.npy' % (
            operation, mode, args.l, phase, confidence_bound, p_value_bound),
        np.array(coint_node_plus_abs_d_list))


def do_get_deleted_node_output(param):
    # if os.path.exists(param[7]):
    #     print('==exists %s ==' % param[7])
    #     return

    print('==cal output %s ==' % param[7])

    np.save(param[7], get_deleted_node_output(param[0], param[1], param[2], param[3], param[4], param[5],
                                              param[6], param[8]).detach().cpu().numpy())
    print('==save %s ==' % param[7])


def get_classify_vector(img_dir, class_index, deleted_node_set, arch, l, model, xrate):
    m_outputs = get_deleted_node_output(
        img_dir, class_index, deleted_node_set, arch, args.l,
        model, xrate).detach().cpu().numpy()

    classify_vector = []
    for j in range(1, m_outputs.shape[0]):
        if np.argmax(m_outputs[j]) == class_index:
            classify_vector.append(j)
        # else:
        #     classify_vector.append()
    return classify_vector


def test_array_all_the_same(arr):
    # print('arr: ', arr)
    first = arr[0]
    for item in arr:
        if not item == first:
            # print(False)
            return False
    # print(True)
    return True


def cal_cos(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim


def cal_norm(vector_a, vector_b):
    sim = np.sqrt(np.sum(np.square(vector_a - vector_b)))
    # sim = np.linalg.norm(vector_a - vector_b)
    return sim


def do_cal_cov(param):
    t = param[0]
    ts_operation = param[1]
    mode = param[2]
    metric = param[3]
    time_type = param[4]
    phase = param[5]
    point_times = param[6]
    multi_class_item = param[7]
    count = param[8]

    # if os.path.exists(
    #         t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/log_io%s_%s_file_list_cof_%s_%s_%s_%s.npy' % (
    #                 time_type, args.mt, multi_class_item, ts_operation, 'zone', 'zero')):
    #     return

    deleted_node_set = None
    keep_node_set = None
    o_metric_file_list = None
    metric_file_list = None
    std_npy = None

    deleted_node_set = np.load(
        t + '/' + args.arch + '/deleted_node_npy_dir/deleted_node_%sd_marker_%s_%s_set_%s.npy' % (
            args.mt, args.dnop, ts_operation, multi_class_item))

    keep_node_set = list(set(list(range(output_size))) - set(deleted_node_set))

    # print(len(keep_node_set))

    if time_type == '1':

        all_o_metric_file_list = np.load(
            t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/all_o_%s_file_list_%s_%s_%s_%s.npy' % (
                args.mt, multi_class_item, ts_operation, 'zone', 'zero'))

        all_metric_file_list = np.load(
            t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/all_%s_file_list_%s_%s_%s_%s.npy' % (
                args.mt, multi_class_item, ts_operation, 'zone', 'zero'))

        std_npy = np.load(
            t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/metric_wad_%s_%s_zone_%d_zero_list.npy' % (
                metric, ts_operation, args.l))
    else:

        # keep_node_set = np.load(
        #     t + '/' + args.arch + '/pools_keep_node_npy_dir/keep_node_pools%s_%sd_marker_%s_set_%s.npy' % (
        #         str(int(time_type) - 1), args.mt, ts_operation, multi_class_item))

        all_o_metric_file_list = np.load(
            t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/all_o_pools%s_%s_file_list_%s_%s_%s_%s.npy' % (
                str(int(time_type) - 1), args.mt, multi_class_item, ts_operation, mode, phase))

        all_metric_file_list = np.load(
            t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/all_i_pools%s_%s_file_list_%s_%s_%s_%s.npy' % (
                str(int(time_type) - 1), args.mt, multi_class_item, ts_operation, mode, phase))

        std_npy = np.load(
            t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/metric_i_pools%s_%sd_std_%s_%s_%d_zero_list.npy' % (
                str(int(time_type) - 1), args.mt, ts_operation, mode, args.l))

    # metric_file_list[metric_file_list == 0] = 1
    io_er_list = []
    log_io_er_list = []

    for inner_index in range(all_o_metric_file_list.shape[0]):

        o_metric_file_list = all_o_metric_file_list[inner_index]
        metric_file_list = all_metric_file_list[inner_index]

        # print(o_metric_file_list.shape)

        o_metric_file_list_del_index = []
        metric_file_list_del_index = []
        io_metric_file_list_del_index = []
        log_io_metric_file_list_del_index = []

        # metric_file_list[metric_file_list == 0] = 1

        for i in keep_node_set:
            # if i == 5842 or i == 3482:
            #     print(i)
            #     print(o_metric_file_list[:, i])
            #     print(metric_file_list[:, i])
            #
            # else:
            #     continue

            constant_plus = 1

            o_metric_file_list_del_index.append((o_metric_file_list[:, i]))
            metric_file_list_del_index.append((metric_file_list[:, i]))
            io_metric_file_list_del_index.append(
                (o_metric_file_list[:, i] + constant_plus) / (np.mean(metric_file_list[:, i] + constant_plus)))
            log_io_metric_file_list_del_index.append(
                np.log(
                    ((o_metric_file_list[:, i] + constant_plus) / (
                        np.mean(metric_file_list[:, i] + constant_plus)))) /
                std_npy[
                    i])
        # io_metric_file_list_del.append(
        #     (o_metric_file_list[:, i] + 10e-7) / (metric_file_list[:, i] + 10e-7))
        # log_io_metric_file_list_del.append(
        #     np.log(((o_metric_file_list[:, i] + 10e-7) / (metric_file_list[:, i] + 10e-7))) / std_npy[
        #         i])
        # log_io_metric_file_list_del.append(
        #     np.log(((o_metric_file_list[:, i] + 1) / (metric_file_list[:, i] + 1))))

        # print('np.array(io_metric_file_list_del): ', np.array(io_metric_file_list_del_index).shape)
        # io_er_list_inner = np.mean(io_metric_file_list_del, axis=1)
        # log_io_er_list_inner = np.mean(log_io_metric_file_list_del, axis=1)
        # io_er_list.append(io_er_list_inner)
        # log_io_er_list.append(log_io_er_list_inner)
        io_er_list.append(io_metric_file_list_del_index)
        log_io_er_list.append(log_io_metric_file_list_del_index)

    # print('np.array(io_er_list).shape: ', np.array(io_er_list).shape)

    io_metric_file_list_del = np.mean(np.array(io_er_list), axis=0)
    log_io_metric_file_list_del = np.mean(np.array(log_io_er_list), axis=0)

    # print(log_io_metric_file_list_del.shape)

    io_er_list = np.mean(np.array(io_metric_file_list_del), axis=1)
    log_io_er_list = np.mean(np.array(io_metric_file_list_del), axis=1)

    # print(metric_file_list[:,9034])
    # print(o_metric_file_list[:, 9034])

    # print(log_io_er_list[-1])

    # print(er_list)

    np.save(
        t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/io%s_%s_file_list_%s_%s_%s_%s.npy' % (
            time_type, args.mt, multi_class_item, ts_operation, 'zone', 'zero'), io_metric_file_list_del)
    np.save(
        t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/log_io%s_%s_file_list_%s_%s_%s_%s.npy' % (
            time_type, args.mt, multi_class_item, ts_operation, 'zone', 'zero'), log_io_metric_file_list_del)

    np.save(
        t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/io%s_%s_er_list_%s_%s_%s_%s.npy' % (
            time_type, args.mt, multi_class_item, ts_operation, 'zone', 'zero'), io_er_list)

    np.save(
        t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/log_io%s_%s_er_list_%s_%s_%s_%s.npy' % (
            time_type, args.mt, multi_class_item, ts_operation, 'zone', 'zero'), log_io_er_list)

    r_i_list_topk_value, r_i_list_topk_index = torch.from_numpy(io_er_list).topk(
        point_times * int(np.sqrt(len(keep_node_set))))

    r_i_list_topk_index = r_i_list_topk_index.numpy()
    keep_node_set = np.array(keep_node_set)

    mkdir(t + '/' + args.arch + '/best_node_pools_npy_dir')
    np.save(t + '/' + args.arch + '/best_node_pools_npy_dir/best_node_pools%s_%sd_marker_%s_%s_set_%s.npy' % (
        args.tt, args.mt, args.dnop, ts_operation, multi_class_item), keep_node_set[r_i_list_topk_index])

    if args.isall != 'all':
        log_io_metric_file_list_del = log_io_metric_file_list_del[r_i_list_topk_index]
        io_metric_file_list_del = io_metric_file_list_del[r_i_list_topk_index]

    # metric_cov = np.cov(io_metric_file_list_del)
    # log_metric_cov = np.cov(log_io_metric_file_list_del)
    # metric_cof = np.corrcoef(io_metric_file_list_del)
    log_metric_cof = np.corrcoef(log_io_metric_file_list_del)

    # print(multi_class_item, '-', metric_cov.shape)
    # print(keep_node_set[r_i_list_topk_index])

    # np.save(
    #     t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/io%s_%s_file_list_cov_%s_%s_%s_%s.npy' % (
    #         time_type, args.mt, multi_class_item, ts_operation, 'zone', 'zero'), metric_cov)
    # np.save(
    #     t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/log_io%s_%s_file_list_cov_%s_%s_%s_%s.npy' % (
    #         time_type, args.mt, multi_class_item, ts_operation, 'zone', 'zero'), log_metric_cov)
    # np.save(
    #     t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/io%s_%s_file_list_cof_%s_%s_%s_%s.npy' % (
    #         time_type, args.mt, multi_class_item, ts_operation, 'zone', 'zero'), metric_cof)
    np.save(
        t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/log_io%s_%s_file_list_cof_%s_%s_%s_%s.npy' % (
            time_type, args.mt, multi_class_item, ts_operation, 'zone', 'zero'), log_metric_cof)

    print('===completed cal cov %s: %s-%s===' % (count, time_type, multi_class_item))


def do_analysis_cov(param):
    t = param[0]
    ts_operation = param[1]
    mode = param[2]
    phase = param[3]
    time_type = param[4]
    point_times = param[5]
    related_point_number = param[6]
    multi_class_item = param[7]
    count = param[8]

    print('=== %s ===' % (multi_class_item))

    io_metric_file_list = np.load(
        t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/io%s_%s_file_list_%s_%s_%s_%s.npy' % (
            time_type, args.mt, multi_class_item, ts_operation, 'zone', 'zero'))
    log_io_metric_file_list = np.load(
        t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/log_io%s_%s_file_list_%s_%s_%s_%s.npy' % (
            time_type, args.mt, multi_class_item, ts_operation, 'zone', 'zero'))

    # metric_cov = np.load(
    #     t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/io%s_%s_file_list_cov_%s_%s_%s_%s.npy' % (
    #         time_type, args.mt, multi_class_item, ts_operation, 'zone', 'zero'))
    # log_metric_cov = np.load(
    #     t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/log_io%s_%s_file_list_cov_%s_%s_%s_%s.npy' % (
    #         time_type, args.mt, multi_class_item, ts_operation, 'zone', 'zero'))
    # metric_cof = np.load(
    #     t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/io%s_%s_file_list_cof_%s_%s_%s_%s.npy' % (
    #         time_type, args.mt, multi_class_item, ts_operation, 'zone', 'zero'))
    log_metric_cof = np.load(
        t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/log_io%s_%s_file_list_cof_%s_%s_%s_%s.npy' % (
            time_type, args.mt, multi_class_item, ts_operation, 'zone', 'zero'))

    er_list = np.load(
        t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/io%s_%s_er_list_%s_%s_%s_%s.npy' % (
            time_type, args.mt, multi_class_item, ts_operation, 'zone', 'zero')).reshape(-1)

    log_er_list = np.load(
        t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/log_io%s_%s_er_list_%s_%s_%s_%s.npy' % (
            time_type, args.mt, multi_class_item, ts_operation, 'zone', 'zero')).reshape(-1)

    deleted_node_set = np.load(
        t + '/' + args.arch + '/deleted_node_npy_dir/deleted_node_%sd_marker_%s_%s_set_%s.npy' % (
            args.mt, args.dnop, ts_operation, multi_class_item))

    keep_node_set = list(set(list(range(output_size))) - set(deleted_node_set))
    keep_node_set = np.array(keep_node_set)

    # if time_type != '1':
    #     keep_node_set = np.load(
    #         t + '/' + args.arch + '/pools_keep_node_npy_dir/keep_node_pools%s_%sd_marker_%s_set_%s.npy' % (
    #             str(int(time_type) - 1), args.mt, ts_operation, multi_class_item))

    # print('keep node set len: %d' % (len(keep_node_set)))

    r_i_list_topk_value, r_i_list_topk_index = torch.from_numpy(er_list).topk(
        point_times * int(np.sqrt(len(keep_node_set))))

    r_i_list_topk_index = r_i_list_topk_index.numpy()

    mkdir(t + '/' + args.arch + '/best_node_pools_npy_dir')
    np.save(t + '/' + args.arch + '/best_node_pools_npy_dir/best_node_pools%s_%sd_marker_%s_%s_set_%s.npy' % (
        args.tt, args.mt, args.dnop, ts_operation, multi_class_item), keep_node_set[r_i_list_topk_index])
    # mkdir(t + '/' + args.arch + '/keep_node_npy_dir')
    # np.save(
    #     t + '/' + args.arch + '/keep_node_npy_dir' + '/keep_node_file_list_%s_%s_%s_%s_%s.npy' % (
    #         args.mt, multi_class_item, ts_operation, 'zone', 'zero'), keep_node_set[r_i_list_topk_index])

    # metric = 'std'
    # std_npy = np.load(
    #     t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/metric_wad_%s_%s_time_5_one_list.npy' % (
    #         metric,ts_operation))

    keep_node_pools = []
    keep_node_pools_weight = []
    no_center_node_list = []

    best_node_index = []
    best_node_pools = []
    best_node_pools_weight = []

    det_0_count = 0

    # print(er_list[r_i_list_topk_index])
    # print(keep_node_set[r_i_list_topk_index])
    # for j in range(0):

    if not args.isall == 'all':
        keep_node_set = keep_node_set[r_i_list_topk_index]
        io_metric_file_list = io_metric_file_list[r_i_list_topk_index]
        log_er_list = r_i_list_topk_value
        log_io_metric_file_list = log_io_metric_file_list[r_i_list_topk_index]

    for j in range(len(keep_node_set)):

        # if keep_node_set[j] != 11:
        #     continue

        # if j not in r_i_list_topk_index:
        #     continue

        indexes = []

        if args.isall == 'all':

            # htopk_v, htopk_index = torch.from_numpy(log_metric_cof[j, r_i_list_topk_index]).topk(10)
            ltopk_v, htopk_index = torch.from_numpy(log_metric_cof[j, r_i_list_topk_index]).topk(
                related_point_number,
                largest=False)
            ltopk_v, ltopk_index = torch.from_numpy(np.abs(log_metric_cof[j, r_i_list_topk_index])).topk(
                related_point_number,
                largest=False)

            for index in r_i_list_topk_index[ltopk_index.numpy()]:
                # print(er_list[index])
                if not test_array_all_the_same(io_metric_file_list[index, :]):
                    # if er_list[index] > er_list[j] and not test_array_all_the_same(io_metric_file_list[index, :]):
                    indexes.append(index)
            for index in r_i_list_topk_index[htopk_index.numpy()]:
                if not test_array_all_the_same(io_metric_file_list[index, :]):
                    # if er_list[index] > er_list[j] and not test_array_all_the_same(io_metric_file_list[index, :]):
                    indexes.append(index)
        else:

            # print(log_metric_cof.shape)

            ltopk_v, htopk_index = torch.from_numpy(log_metric_cof[j, :]).topk(related_point_number,
                                                                               largest=False)
            ltopk_v, ltopk_index = torch.from_numpy(np.abs(log_metric_cof[j, :])).topk(related_point_number,
                                                                                       largest=False)

            # print(io_metric_file_list.shape)

            for index in ltopk_index.numpy():
                # print(er_list[index])
                if not test_array_all_the_same(io_metric_file_list[index, :]):
                    # if er_list[index] > er_list[j] and not test_array_all_the_same(io_metric_file_list[index, :]):
                    indexes.append(index)
            for index in htopk_index.numpy():
                if not test_array_all_the_same(io_metric_file_list[index, :]):
                    # if er_list[index] > er_list[j] and not test_array_all_the_same(io_metric_file_list[index, :]):
                    indexes.append(index)
            # print('log_er:', r_i_list_topk_value[indexes])
        # print(io_metric_file_list.shape)
        # print(er_list[j])

        # print()
        if not test_array_all_the_same(io_metric_file_list[j, :]):
            indexes.append(j)
        else:
            print('no_center')
            no_center_node_list.append(j)
        # print('index:', indexes)
        # print('indexs: ', keep_node_set[indexes])
        indexes = list(set(indexes))
        # indexes.append(j)

        indexes_big = []

        indexes = np.array(indexes)

        # if indexes.shape[0] <= 1:
        #     if indexes.shape[0] == 0:
        #         keep_node_pools.append(np.array([j]))
        #         keep_node_pools_weight.append(np.array([1]))
        #     else:
        #         keep_node_pools.append(keep_node_set[indexes])
        #         keep_node_pools_weight.append(np.array([1]))
        #     det_0_count += 1
        #     print('lt 1')
        #     continue

        market_er_list = log_er_list[indexes].reshape(-1, 1)
        if args.isall == 'best':
            market_er_list = market_er_list.numpy()
        # print('max_marker: ',np.max(market_er_list))

        ee = np.ones(market_er_list.shape).reshape(-1, 1)

        pools_wa_list = log_io_metric_file_list[indexes]
        pools_wa_cov = np.cov(pools_wa_list)
        # print(pools_wa_list)

        wa_cov_inv = np.linalg.inv(pools_wa_cov)

        # print(market_er_list.shape)

        a = np.dot(np.dot(market_er_list.transpose(), wa_cov_inv), market_er_list)[0][0] + 0.0000001
        b = np.dot(np.dot(market_er_list.transpose(), wa_cov_inv), ee)[0][0] + 0.0000001
        c = np.dot(np.dot(ee.transpose(), wa_cov_inv), ee)[0][0] + 0.0000001
        b2 = np.dot(np.dot(ee.transpose(), wa_cov_inv), market_er_list)[0][0] + 0.0000001

        # print('a: ', a)
        # print('b: ', b, ' = ', b2)
        # print('c: ', c)
        # print('b2: ', b2)

        # miu_p = (a * a) / (b * b2)
        miu_p = np.sqrt(a * a / (b * b2))

        d = a * c - b * b2 + 0.0000001
        a_inv = np.ones((2, 2))
        a_inv[0][0] = c
        a_inv[0][1] = -b
        a_inv[1][0] = -b2
        a_inv[1][1] = a
        # print(a_inv.shape)
        a_inv *= (1 / d)

        m1 = np.dot(wa_cov_inv, np.concatenate((market_er_list, ee), axis=1))
        m2 = np.dot(m1, a_inv)
        m3 = np.dot(m2, np.concatenate(([miu_p], [1]), axis=0))

        weight = m3
        # weight[weight < 0] = 0
        # print('indexes: ', keep_node_set[indexes])
        # print(weight)
        # if keep_node_set[j] == 288:
        #     sys.exit(0)
        # print('miu_p_w: ', np.dot(weight.transpose(), market_er_list))
        # weight = weight / np.sum(weight)

        # weight = tensor_array_normalization(m3)
        # print(np.sum(weight))
        # print('j: ', keep_node_set[j])
        # print('er:', er_list[indexes])

        # print('weight: ', weight)
        # print('miu_p: ', miu_p)
        # if keep_node_set[j] == 8073:
        #     break
        # weight = weight / np.sum(weight)
        # print(np.sum(weight))
        # print(tensor_array_normalization(m3))

        # print(tensor_array_normalization(m3)/np.sum(m3))
        # print(np.sum(weight))

        keep_node_pools.append((keep_node_set[j], keep_node_set[indexes]))
        keep_node_pools_weight.append(weight)

        if j in r_i_list_topk_index:
            best_node_index.append(keep_node_set[j])
            best_node_pools.append((keep_node_set[j], keep_node_set[indexes]))
            best_node_pools_weight.append(weight)

        # print('shape indexes: ', indexes.shape)
        # print('shape weights: ', weight.shape)
        # if j==2:
        #     print('shape indexes: ',keep_node_pools[j].shape)
        #     print('shape weights: ',weight.shape)
        #     print(j)
        # break

        # print('======================')
        # break

    # print('det_0_count: ', det_0_count)
    # print(keep_node_set)
    # print('%s-keep_node_pools_weight: ' % (multi_class_item), len(keep_node_pools_weight))

    np.save(
        t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/keep_node_pools%s_wad_%s_%s_5_%s_list.npy' % (
            time_type, ts_operation, mode, phase),
        np.array(keep_node_pools))

    np.save(
        t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/no_center_node_pools%s_wad_%s_%s_5_%s_list.npy' % (
            time_type, ts_operation, mode, phase),
        np.array(no_center_node_list))

    np.save(
        t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/keep_node_pools%s_weight_wad_%s_%s_5_%s_list.npy' % (
            time_type, ts_operation, mode, phase),
        np.array(keep_node_pools_weight))

    np.save(
        t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/best_node_index%s_wad_%s_%s_5_%s_list.npy' % (
            time_type, ts_operation, mode, phase),
        np.array(best_node_index))

    np.save(
        t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/best_node_pools%s_wad_%s_%s_5_%s_list.npy' % (
            time_type, ts_operation, mode, phase),
        np.array(best_node_pools))

    np.save(
        t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/best_node_pools%s_weight_wad_%s_%s_5_%s_list.npy' % (
            time_type, ts_operation, mode, phase),
        np.array(best_node_pools_weight))

    print('===completed ana cov %s: %s-%s===' % (count, time_type, multi_class_item))

    pass


def do_analysis_pools(param):
    t = param[0]
    ts_operation = param[1]
    img_dir = param[2]
    mode = param[3]
    phase = param[4]
    time_type = param[5]
    multi_class_item = param[6]
    count = param[8]
    compare_index = param[7]

    range_list_inner = os.listdir(img_dir + '/' + multi_class_item)

    metric_file_list_list = []

    random_cp_classes = np.load(t + '/inner_cp_classes/%s_cp_classes.npy' % (multi_class_item)).tolist()
    random_cp_classes.append(multi_class_item)

    for multi_class_item_cp in random_cp_classes:

        if multi_class_item_cp == 'n00000000' or multi_class_item == 'n00000000':
            continue

        if args.ioi == 'i' and multi_class_item_cp != multi_class_item:
            continue
        if args.ioi == 'o' and multi_class_item_cp == multi_class_item:
            continue

        # print(multi_class_item + '-' + multi_class_item_cp)

        metric_list_np_dir = t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/metric_list_%sd_%s_%s_%d_zero_npy' % (
            args.mt, ts_operation, mode, args.l)
        abs_d_list_np_dir = t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/metric_list_abs_d_%s_%s_%d_zero_npy' % (
            ts_operation, mode, args.l)

        metric_file_name_list = []
        abs_d_file_name_list = []

        if mode == 'zone':
            metric_file_name_list = os.listdir(metric_list_np_dir)
            if args.mt == 'js':
                abs_d_file_name_list = os.listdir(abs_d_list_np_dir)

        metric_file_list = []
        abs_d_file_list = []

        if args.ioi == 'i':

            # print(range_list_inner)

            iter_list = []
            if args.cp == 'one':
                iter_list = range_list_inner[compare_index:compare_index + 1]
            elif args.cp == 'all':
                iter_list = range_list_inner[0:len(range_list_inner)]
                iter_list = ['avg']

            for transform_index_inner in iter_list:
                # for transform_img_index in range_list[0:len(range_list)]:
                # if transform_img_index == iter_list[compare_index]:
                #     continue

                for transform_index_cp in range_list_inner:

                    if transform_index_cp == range_list_inner[compare_index]:
                        continue

                    metric_list_np_name = metric_list_np_dir + '/metric_list_%s_pools%s_cpt_%s-%s-%s-%s_%sd_%s_%s_%d_zero_' % (
                        args.ioi, time_type, transform_index_inner, multi_class_item,
                        multi_class_item_cp,
                        transform_index_cp,
                        args.mt, ts_operation, mode,
                        args.l) + '.npy'

                    metric_file_np = np.load(metric_list_np_name)
                    # print(metric_file_np.shape)
                    metric_file_list.append(metric_file_np.reshape(metric_file_np.shape[0], ).tolist())

                if len(metric_file_list) == 0:
                    continue

                # print(np.array(metric_file_list).shape)
                metric_file_list_list.append(metric_file_list)
                np.save(
                    t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/wa_class_npy/%s%s_file_list_%s-%s_%s_%s_%s.npy' % (
                        args.mt, time_type, multi_class_item, transform_index_inner, ts_operation, mode,
                        phase),
                    metric_file_list)
                metric_file_list = []
            # metric_file_list = np.array(metric_file_list)
            # np.save(
            #     t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/%s_pools%s_%s_file_list_%s_%s_%s_%s.npy' % (
            #         args.ioi, time_type, args.mt, multi_class_item, ts_operation, mode, phase),
            #     metric_file_list)

        if args.ioi == 'o':

            range_list_cp = os.listdir(img_dir + '/' + multi_class_item_cp)
            # print(range_list)

            # iter_list = []
            # if args.cp == 'one':
            #     iter_list = range_list_cp[compare_index:compare_index + 1]
            # elif args.cp == 'all':
            #     iter_list = range_list_cp[0:len(range_list_cp)]

            metric_file_cp = []

            iter_list = []

            if args.cp == 'one':
                iter_list = range_list_inner[compare_index:compare_index + 1]
            elif args.cp == 'all':
                iter_list = range_list_inner[0:len(range_list_inner)]
                iter_list = ['avg']

            for transform_index_inner in iter_list:

                metric_file_index = []

                for transform_index_cp in range_list_cp:

                    if transform_index_cp == range_list_cp[compare_index]:
                        continue

                    # print(transform_index_cp, '-', range_list_cp[compare_index])

                    metric_list_np_name = metric_list_np_dir + '/metric_list_%s_pools%s_cpt_%s-%s-%s-%s_%sd_%s_%s_%d_zero_' % (
                        args.ioi, time_type, transform_index_inner, multi_class_item,
                        multi_class_item_cp,
                        transform_index_cp,
                        args.mt, ts_operation, mode,
                        args.l) + '.npy'
                    metric_file_np = np.load(metric_list_np_name, allow_pickle=True)
                    metric_file_index.append(metric_file_np.tolist())
                # print(np.array(metric_file_index).shape)
                # 25 * 24 * output_size
                metric_file_cp.append(metric_file_index)
                # metric_file_list_list.append(metric_file_index)
                # metric_file_list.append(metric_file_np.reshape(metric_file_np.shape[0], ).tolist())
            # metric_file_list_list.append(metric_file_list)
            # 20 * 25 * 24 * output_size
            metric_file_list_list.append(metric_file_cp)
    # if args.ioi == 'o':
    # print(np.array(metric_file_list_list).shape)

    all_pools = np.array(metric_file_list_list)
    if args.ioi == 'o':
        all_pools = np.mean(np.array(metric_file_list_list), axis=0)

    np.save(
        t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/all_%s_pools%s_%s_file_list_%s_%s_%s_%s.npy' % (
            args.ioi, time_type, args.mt, multi_class_item, ts_operation, mode, phase), all_pools)

    pools = np.mean(all_pools, axis=0)

    np.save(
        t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/%s_pools%s_%s_file_list_%s_%s_%s_%s.npy' % (
            args.ioi, time_type, args.mt, multi_class_item, ts_operation, mode, phase), pools)

    metric_file_io = np.mean(np.array(metric_file_list_list), axis=0)

    metric_std_list = []
    metric_mean_list = []
    metric_cv_list = []
    metric_kurt_list = []

    metric_plus_abs_d_std_list = []
    metric_plus_abs_d_mean_list = []
    metric_plus_abs_d_cv_list = []
    metric_plus_abs_d_kurt_list = []

    for i in range(metric_file_io.shape[1]):
        # if i % 10000 == 0:
        #     print('std cal process: %d' % i)

        mean = np.mean(metric_file_io[:, i]) + 0.0000001
        std = np.std(metric_file_io[:, i]) + 0.0000001
        cv = std / mean
        kurt = np.mean((metric_file_io[:, i] - mean) ** 4) / pow(std * std, 2)
        # kurt = scipy.stats.kurtosis(metric_file_list[:, i])
        metric_std_list.append(std)
        metric_mean_list.append(mean)
        metric_cv_list.append(cv)
        metric_kurt_list.append(kurt)

    np.save(
        t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/metric_%s_pools%s_%sd_std_%s_%s_%d_zero_list.npy' % (
            args.ioi, time_type, args.mt, ts_operation, mode, args.l), np.array(metric_std_list))

    print('===completed analysis pools %s: %s-%s===' % (count, time_type, multi_class_item))

    pass


def do_partition_train_vector_by_all_sim3(param):
    cos_vector_dir = param[0]
    multi_class_item = param[1]
    comb_vector_dir = param[2]
    weight_vector_dir = param[3]
    t = param[4]
    time_type = param[5]
    multi_classes = param[6]

    class_vector_dir = cos_vector_dir + '/' + multi_class_item

    mkdir(class_vector_dir)

    class_comb_vector_dir = comb_vector_dir + '/' + multi_class_item

    img_index = 0

    for tensor_array_index in os.listdir(class_comb_vector_dir + '/' + multi_class_item):

        tensor_array_item_list = []
        # random_cp_classes = np.load(
        #     t + '/inner_cp_classes/%s_cp_classes.npy' % (multi_class_item_inner)).tolist()

        top_weight = np.load(
            weight_vector_dir + '/' + multi_class_item + '/top_weight_%s.npy' % (tensor_array_index))

        multi_class_item2_index = 0

        for multi_class_item2 in multi_classes:
            # print(multi_class_item2)

            if not multi_class_item2_index in top_weight[:, 0]:
                multi_class_item2_index += 1
                continue
            else:
                multi_class_item2_index += 1

            tensor_array_item = np.load(
                class_comb_vector_dir + '/' + multi_class_item2 + '/' + tensor_array_index)

            # print(tensor_array_item.shape)
            # tensor_array_item = tensor_array_item.reshape(1, tensor_array_item.shape[0])
            tensor_array_item_list.append(tensor_array_item)

        o_train_vector = np.load(
            t + '/vector_dir/' + time_type + '/o_%s/' % (args.tv) + multi_class_item + '/%s-%s-%s.npy' % (
                multi_class_item, img_index, time_type)).reshape(1, -1)

        weight = top_weight[:, 1]
        # print('weight: ', weight[0].shape)

        tensor_array_item_list = np.array(tensor_array_item_list)
        tensor_array_item_weight_sum = np.zeros(tensor_array_item_list[0].shape)

        # print(weight.shape)
        for weight_index in range(weight.shape[0]):
            tensor_array_item_weight_sum += weight[weight_index] * tensor_array_item_list[weight_index]

        # print(ss_max_class)
        tensor_array_item_weight_sum = tensor_array_item_weight_sum.reshape(-1, )

        np.save(class_vector_dir + '/%s-%s-%s.npy' % (multi_class_item, img_index, time_type),
                tensor_array_item_weight_sum + o_train_vector.reshape(-1, ))
        #
        img_index += 1


def main():
    f = 'imagenet_2012/train'
    v = 'imagenet_2012/val'
    t = 'imagenet_2012/' + args.tdir
    r = 25

    num_of_layer = 8
    if args.arch == 'restnet34':
        num_of_layer = 18
    elif args.arch == 'vgg16':
        num_of_layer = 17

    op = args.op
    arch = args.arch
    xrate = args.xrate
    # model = get_model()

    # random_class = select_random_dir(f, t, v, r)

    save_dir_pre = args.tdir + '_layer_npy'

    single_train_dir = 'imagenet_2012/single_train'
    single_test_train_dir = 'imagenet_2012/single_test_train'
    single_val_dir = 'imagenet_2012/single_val'

    origin_image_dir = t + '/origin_images'

    transform_dir = t + '/transform_images'
    transform_dir_t = t + '/transform_images_t'

    # class_dic = imagenet_class_index_dic()

    params = []

    compare_index = 0

    out_classes_size = 20

    if args.op == 'gen_transform_origin_images_zhiyong':

        # pool table, paintbrush, padlock, bowl, holster, baseball, bottlecap, pll bottle, toilet seat,coral fungus,corn
        random_class = ['n01443537', 'n01484850', 'n01491361', 'n01494475', 'n01496331', 'n01498041', 'n02085620',
                        'n02085782', 'n02085936', 'n02086079', 'n02086240', 'n02086646', 'n02086910', 'n02087046',
                        'n02087394', 'n02088094', 'n02123045', 'n02123159', 'n02123394', 'n02123597', 'n02124075',
                        'n02125311', 'n02127052']

        # random_class = ['n03937543']

        mkdir(origin_image_dir)

        for class_index in range(len(random_class)):
            one_class_img_dir = '/data/zhiyong/cnn_size/' + random_class[class_index] + '/source/orig'
            one_class_imgs = os.listdir(one_class_img_dir)
            one_class_img_origin_dir = origin_image_dir + '/' + random_class[class_index] + '/' + random_class[
                class_index]
            mkdir(one_class_img_origin_dir)

            # sample = random.sample(one_class_imgs, 1)

            for item in one_class_imgs:
                shutil.copy(one_class_img_dir + '/' + item, one_class_img_origin_dir + '/' + item)

    if args.op == 'gen_transform_images':

        model = get_model().cuda()

        ts_operation = args.tsop
        pick_number = 20

        # random_class = ['n02917067', 'n02802426', 'n02483362', 'n02487347', 'n02494079', 'n02486410', 'n13054560',
        #                 'n12998815', 'n02835271', 'n03792782', 'n04482393']

        # snake, butterfly, cat, leopard, dog
        # random_class = ['n01728572', 'n01 728920', 'n01729322', 'n01729977', 'n01734418', 'n01735189', 'n01737021',
        #                 'n01739381', 'n01740131', 'n01742172', 'n01744401', 'n01748264', 'n01749939', 'n01751748',
        #                 'n01753488', 'n01755581', 'n01756291', 'n02276258', 'n02277742', 'n02279972', 'n02280649',
        #                 'n02281406', 'n02281787', 'n02123045', 'n02123159', 'n02123394', 'n02123597', 'n02124075',
        #                 'n02125311', 'n02127052', 'n02128385', 'n02128757', 'n02128925', 'n02085620', 'n02085782',
        #                 'n02085936', 'n02086079', 'n02086240', 'n02086646', 'n02086910', 'n02087046', 'n02087394',
        #                 'n02088094']

        # bird, spider, fish
        # random_class = ['n02002724', 'n02006656', 'n02007558', 'n02009229', 'n02009912', 'n02011460', 'n02012849',
        #                 'n02013706', 'n01773157', 'n01773549', 'n01773797', 'n01774384', 'n01774750', 'n01775062',
        #                 'n01776313', 'n01443537', 'n01484850', 'n01491361', 'n01494475', 'n01496331', 'n01498041']

        # random_class = ['n01675722', 'n01677366', 'n01682714', 'n01685808', 'n01687978', 'n01688243', 'n01689811',
        #                 'n01692333', 'n01693334', 'n01694178', 'n01695060', 'n01629819', 'n01630670', 'n01631663',
        #                 'n01632458', 'n01632777', 'n02119022', 'n02119789', 'n02120079', 'n02120505', 'n02441942',
        #                 'n02442845', 'n02443114', 'n02443484', 'n02444819', 'n02445715', 'n02447366', 'n02403003',
        #                 'n02408429', 'n02410509', 'n02412080', 'n02415577', 'n02417914', 'n02422106', 'n02422699',
        #                 'n02423022', 'n12985857', 'n12998815', 'n13037406', 'n13040303', 'n13044778', 'n13052670',
        #                 'n13054560']
        #

        # random_class = ['n01728572', 'n01728920', 'n01729322', 'n01729977', 'n01734418', 'n01735189', 'n01737021',
        #                 'n01739381', 'n01740131', 'n01742172', 'n01744401', 'n01748264', 'n01749939', 'n01751748',
        #                 'n01753488', 'n01755581', 'n01756291', 'n02276258', 'n02277742', 'n02279972', 'n02280649',
        #                 'n02281406', 'n02281787', 'n02123045', 'n02123159', 'n02123394', 'n02123597', 'n02124075',
        #                 'n02125311', 'n02127052', 'n02128385', 'n02128757', 'n02128925', 'n02085620', 'n02085782',
        #                 'n02085936', 'n02086079', 'n02086240', 'n02086646', 'n02086910', 'n02087046', 'n02087394',
        #                 'n02088094', 'n01443537', 'n01484850', 'n01491361', 'n01494475', 'n01496331', 'n01498041',
        #                 'n02002724', 'n02006656', 'n02007558', 'n02009229', 'n02009912', 'n02011460', 'n02012849',
        #                 'n02013706', 'n01773157', 'n01773549', 'n01773797', 'n01774384', 'n01774750', 'n01775062',
        #                 'n01776313']

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

        # pool table, paintbrush, padlock, bowl, holster, baseball, bottlecap, pll bottle, toilet seat,coral fungus,corn
        # random_class = ['n03982430', 'n03876231', 'n03874599', 'n03775546', 'n03527444', 'n02799071', 'n02877765',
        #                 'n03937543', 'n04447861', 'n12985857', 'n12144580']

        # random_class = ['n12144580']

        # random_class = ['n01728572', 'n02276258', 'n02123045', 'n02128385', 'n02085620', 'n01443537', 'n02002724',
        #                 'n01773157',
        #                 'n02483362']

        # random_class = ['n02437616', 'n02454379', 'n01667778', 'n01806143', 'n01871265', 'n07714990', 'n11939491',
        #                 'n02398521', 'n03982430', 'n03876231', 'n03874599', 'n03775546', 'n03527444', 'n02799071',
        #                 'n02877765', 'n03937543', 'n04447861', 'n12985857', 'n12144580']

        # random_class = ['n03982430', 'n03876231', 'n03874599', 'n03775546', 'n03527444', 'n02799071', 'n02877765',
        #                 'n03937543', 'n04447861', 'n12985857', 'n12144580', 'n02128385', 'n02085620', 'n02437616',
        #                 'n02454379',
        #                 'n01667778', 'n01806143', 'n01871265', 'n07714990', 'n11939491',
        #                 'n02398521']
        #
        # random_class = ['n03982430', 'n03876231', 'n03874599', 'n03775546', 'n03527444', 'n02877765',
        #                 'n03937543', 'n04447861', 'n12985857', 'n12144580']

        # random_class = ['n03982430', 'n03876231', 'n03874599']

        # random_class = ['n02437616', 'n02454379',
        #                 'n01667778', 'n01806143', 'n01871265', 'n07714990', 'n11939491',
        #                 'n02398521']
        #
        # random_class = ['n02793495', 'n13052670',
        #                 'n04548362', 'n04154565', 'n01644900', 'n01755581', 'n02256656',
        #                 'n02870880', 'n04067472', 'n02101556']

        # random_class = ['n01443537', 'n01484850', 'n01491361', 'n01494475', 'n01496331', 'n01498041', 'n02085620',
        #                 'n02085782', 'n02085936', 'n02086079', 'n02086240', 'n02086646', 'n02086910', 'n02087046',
        #                 'n02087394', 'n02088094', 'n02123045', 'n02123159', 'n02123394', 'n02123597', 'n02124075',
        #                 'n02125311', 'n02127052']

        # for random_class_item in random_class:
        #     if not os.path.exists(t + '/transform_images/' + random_class_item):
        #         mkdir(t + '/transform_images/' + random_class_item)

        print('======transform img======')

        # for class_index in range(len(random_class)):
        #     multi_class_except = t + '/multi_class_except_' + random_class[
        #         class_index] + '/multi_class_except_' + \
        #                          random_class[class_index]
        #     if not os.path.exists(multi_class_except):
        #         os.makedirs(multi_class_except)

        if args.rs == 0:
            random_class = os.listdir(origin_image_dir)
        else:
            # random_class = np.load('t_norm_relu_alex_avg_1111_origin_classes.npy')
            random_class = random.sample(os.listdir(single_train_dir), r)

        for class_index in range(len(random_class)):

            if args.ec != 'none' and random_class[class_index] != args.ec:
                continue

            if args.rs == 1:
                one_class_img_dir = single_train_dir + '/' + random_class[class_index] + '/' + random_class[class_index]
                one_class_imgs = os.listdir(one_class_img_dir)
                one_class_img_origin_dir = origin_image_dir + '/' + random_class[class_index] + '/' + random_class[
                    class_index]
                mkdir(one_class_img_origin_dir)

                samples = random.sample(one_class_imgs, pick_number)

                print('=====copy sample to origin_image_dir======')

                for sample in samples:
                    shutil.copy(one_class_img_dir + '/' + sample, one_class_img_origin_dir + '/' + sample)

                # print('=====copy sample to multi_class_except_dir======')
                # for class_index2 in range(len(random_class)):
                #     if class_index2 == class_index:
                #         continue
                #     shutil.copy(one_class_img_dir + '/' + sample,
                #                 t + '/multi_class_except_' + random_class[class_index2] + '/multi_class_except_' +
                #                 random_class[class_index2] + '/' + sample)

            get_layer_cuda(origin_image_dir + '/' + random_class[class_index], args.arch, args.l, save_dir_pre, model)

        # ==================
        # re transform
        # ===================

        if args.rt == 1:

            for class_index in range(len(random_class)):

                if args.ec != 'none' and random_class[class_index] != args.ec:
                    continue

                mkdir(t + '/transform_images_%s_noise/' % ts_operation + random_class[class_index])
                # get_layer(t + '/multi_class_except_' + random_class[class_index], args.arch, args.l, save_dir_pre)

                params = []

                for one_class_img in os.listdir(
                        origin_image_dir + '/' + random_class[class_index] + '/' + random_class[class_index]):
                    # =======================================
                    # transform every image
                    # =======================================

                    one_class_img_no_format = one_class_img.split('.')[0]
                    # transform_img_save_path = t + '/transform_images_gb_noise' + '/' + random_class[
                    #     class_index] + '/' + one_class_img_no_format + '_' + operation + '/' + one_class_img_no_format + '_' + operation
                    transform_img_save_path = t + '/transform_images_%s_noise' % ts_operation + '/' + random_class[
                        class_index] + '/' + one_class_img_no_format + '_' + ts_operation + '/' + one_class_img_no_format + '_' + ts_operation

                    params.append((origin_image_dir + '/' + random_class[class_index] + '/' + random_class[
                        class_index] + '/' + one_class_img, transform_img_save_path, ts_operation, args.param5))

                p = multiprocessing.Pool(15)
                p.map(do_transform, params)
                p.close()
                p.join()

    if args.op == 'get_transform_images_mid_res':

        model = get_model().cuda()
        device_count = torch.cuda.device_count()
        print(device_count)
        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model, device_ids=[0, 1, 2])

        save_dir_pre = args.tdir + '_layer_npy'
        for ts_operation in [args.tsop]:
            img_dirs = []
            if args.param == 'time':
                img_dirs.append(t + '/transform_images_t_%s_noise' % ts_operation)
            elif args.param == 'zone':
                img_dirs.append(t + '/transform_images_%s_noise' % ts_operation)
            else:
                img_dirs.append(t + '/transform_images_%s_noise' % ts_operation)
                img_dirs.append(t + '/transform_images_t_%s_noise' % ts_operation)

            for img_dir in img_dirs:
                np.save(t + '/all_classes.npy', os.listdir(img_dir))
                for class_item in os.listdir(img_dir):

                    if args.ec != 'none' and class_item != args.ec:
                        continue
                    params = []
                    for transform_class_item in os.listdir(img_dir + '/' + class_item):
                        # print('=====get layer=====' + img_dir + '/' + class_item + '/' + transform_class_item)
                        # if not os.path.exists(img_dir + '/' + class_item + '/' + transform_class_item):
                        #     continue
                        params.append(
                            (
                                img_dir + '/' + class_item + '/' + transform_class_item, arch, args.l, save_dir_pre,
                                model))
                        get_layer_cuda(img_dir + '/' + class_item + '/' + transform_class_item, arch, args.l,
                                       save_dir_pre,
                                       model)
                    # print('params len: ', len(params))
                    # p = multiprocessing.Pool(15)
                    # p.map(do_get_layer_cuda, params)
                    # p.close()
                    # p.join()

    if args.op == 'cal_jsd_marker_cd_zone_time_zero2_avg':

        # ==================
        # cal the js for zone in zero phase
        # ==================
        # the operation used to transform images, eg. gaussian blur: gb, all channel change: ac
        ts_operation = args.tsop
        img_dir = t + '/transform_images_%s_noise' % ts_operation
        mode = 'zone'
        save_dir_pre = args.tdir + '_layer_npy'

        # ==================
        # cal the js for zone in zero phase
        # ==================
        if args.param == 'time':
            img_dir = t + '/transform_images_t_%s_noise' % ts_operation
            mode = 'time'

        multi_classes = os.listdir(img_dir)

        count = 0
        params = []
        for multi_class_item_inner in multi_classes:

            if multi_class_item_inner != args.ec and args.ec != 'none' or multi_class_item_inner == 'n00000000':
                continue

            range_list_inner = os.listdir(img_dir + '/' + multi_class_item_inner)
            print(range_list_inner)
            iter_list = []
            if args.cp == 'one':
                iter_list = range_list_inner[compare_index:compare_index + 1]
            elif args.cp == 'all':
                iter_list = range_list_inner[0:len(range_list_inner)]

            tensor_array_list_zero_list = []
            tensor_array_list = [None, None]

            for transform_index_inner in iter_list:
                # for transform_index_innner in range_list[compare_index:compare_index + 1]:
                # for transform_index_innner in range_list[0:len(range_list)]:

                train_dir = img_dir + '/' + multi_class_item_inner + '/' + transform_index_inner
                train_dir = train_dir.replace('/', '-')
                tensor_array_list[0] = np.load(
                    save_dir_pre + '/%s/%s_%s_layer%d.npy' % (
                        (args.arch + '_' + train_dir + '_mid_res'), args.arch, train_dir, args.l))
                tensor_array_list_zero_list.append(tensor_array_list[0])

            tensor_array_list[0] = np.zeros(tensor_array_list_zero_list[0].shape)

            for item in tensor_array_list_zero_list:
                tensor_array_list[0] += item

            tensor_array_list[0] = tensor_array_list[0] / len(tensor_array_list_zero_list)

            for test_layer in range(args.l, args.l + 1):
                metric_list_np_dir = t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item_inner + '/metric_list_%sd_%s_%s_%d_zero_npy' % (
                    args.mt, ts_operation, mode, args.l)
                abs_d_list_np_dir = t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item_inner + '/metric_list_abs_d_%s_%s_%d_zero_npy' % (
                    ts_operation, mode, args.l)

                mkdir(metric_list_np_dir)
                mkdir(abs_d_list_np_dir)

                # ==================
                # cal the js for zone in zero phase
                # ==================

                print('=====continue cal jsd=====')

                transform_index_avg = 'avg'

                for transform_index_cp in range_list_inner[0:len(range_list_inner)]:

                    if transform_index_cp == range_list_inner[compare_index]:
                        continue

                    count += 1

                    train_dir = img_dir + '/' + multi_class_item_inner + '/' + transform_index_cp
                    train_dir = train_dir.replace('/', '-')

                    # print(save_dir_pre + '/%s/%s_%s_layer%d.npy' % (
                    #     (args.arch + '_' + train_dir + '_mid_res'), args.arch, train_dir, test_layer))
                    tensor_array_list[1] = save_dir_pre + '/%s/%s_%s_layer%d.npy' % (
                        (args.arch + '_' + train_dir + '_mid_res'), args.arch, train_dir, test_layer)

                    # for i in range(len(tensor_array_list)):
                    #     tensor_array_list[i] = tensor_array_list[i].reshape(tensor_array_list[i].shape[0], -1)

                    metric_list_np_name = metric_list_np_dir + '/metric_list_%s-%s_%sd_%s_%s_%d_zero_' % (
                        transform_index_avg, transform_index_cp, args.mt, ts_operation, mode,
                        args.l) + '.npy'

                    abs_d_list_np_name = abs_d_list_np_dir + '/metric_list_%s-%s_abs_d_%s_%s_%d_zero_' % (
                        transform_index_avg, transform_index_cp, ts_operation, mode,
                        args.l) + '.npy'

                    print('%d: ' % count, metric_list_np_name)

                    # if os.path.exists(metric_list_np_name):
                    #     print('--- DS EXISTS: %s ---' % (metric_list_np_name))
                    #     continue
                    params.append(
                        (tensor_array_list[0], tensor_array_list[1], metric_list_np_name, abs_d_list_np_name,
                         args.mt, count))

                # break
                # multi thread
        p = multiprocessing.Pool(15)
        p.map(do_cal_jsd_list_between_tensors, params)
        p.close()
        p.join()

    if args.op == 'cal_jsd_marker_cd_zone_time_zero_other_inner_all_avg':

        # ==================
        # cal the js for zone in zero phase
        # ==================
        # the operation used to transform images, eg. gaussian blur: gb, all channel change: ac
        ts_operation = args.tsop
        img_dir = t + '/transform_images_%s_noise' % ts_operation
        mode = 'zone'
        save_dir_pre = args.tdir + '_layer_npy'

        # ==================
        # cal the js for zone in zero phase
        # ==================
        if args.param == 'time':
            img_dir = t + '/transform_images_t_%s_noise' % ts_operation
            mode = 'time'

        multi_classes = os.listdir(img_dir)

        inner_cp_classes_dir = t + '/inner_cp_classes'

        mkdir(inner_cp_classes_dir)

        count = 0
        params = []
        for multi_class_item_inner in multi_classes:

            if multi_class_item_inner != args.ec and args.ec != 'none':
                continue

            tensor_array_list = [None, None]

            print('=====continue cal jsd=====')

            range_list_inner = []

            range_list_inner = os.listdir(img_dir + '/' + multi_class_item_inner)

            iter_list = []

            if args.cp == 'one':
                iter_list = range_list_inner[compare_index:compare_index + 1]
            elif args.cp == 'all':
                iter_list = range_list_inner[0:len(range_list_inner)]

            tensor_array_list_zero_list = []

            for transform_img_index_inner in iter_list:
                train_dir = img_dir + '/' + multi_class_item_inner + '/' + transform_img_index_inner
                train_dir = train_dir.replace('/', '-')
                tensor_array_list_zero_list.append(np.load(
                    save_dir_pre + '/%s/%s_%s_layer%d.npy' % (
                        (args.arch + '_' + train_dir + '_mid_res'), args.arch, train_dir, args.l)))

            tensor_array_list[0] = np.zeros(tensor_array_list_zero_list[0].shape)

            for item in tensor_array_list_zero_list:
                tensor_array_list[0] += item

            tensor_array_list[0] = tensor_array_list[0] / len(tensor_array_list_zero_list)

            # random_cp_class = random.sample(multi_classes, int(len(multi_classes)/10)+1)
            # if multi_class_item_inner in random_cp_class:
            #     random_cp_class.remove(multi_class_item_inner)
            # else:
            #     random_cp_class = random_cp_class[0:int(len(multi_classes)/10)]

            random_cp_class = random.sample(multi_classes, out_classes_size + 1)
            if multi_class_item_inner in random_cp_class:
                random_cp_class.remove(multi_class_item_inner)
            else:
                random_cp_class = random_cp_class[0:out_classes_size]

            np.save(inner_cp_classes_dir + '/%s_cp_classes.npy' % multi_class_item_inner, random_cp_class)

            for multi_class_item_cp in random_cp_class:

                # if multi_class_item_cp != 'n12985857':
                #     continue

                if multi_class_item_cp == 'n00000000' or multi_class_item_inner == 'n00000000' or multi_class_item_inner == multi_class_item_cp:
                    continue

                for test_layer in range(args.l, args.l + 1):
                    metric_list_np_dir = t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item_inner + '/metric_list_o_%sd_%s_%s_%d_zero_npy' % (
                        args.mt, ts_operation, mode, args.l)
                    abs_d_list_np_dir = t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item_inner + '/metric_list_o_abs_d_%s_%s_%d_zero_npy' % (
                        ts_operation, mode, args.l)

                    mkdir(metric_list_np_dir)
                    mkdir(abs_d_list_np_dir)

                    # ==================
                    # cal the js for zone in zero phase
                    # ==================

                    range_list_cp = os.listdir(img_dir + '/' + multi_class_item_cp)
                    transform_img_index_avg = 'avg'

                    for transform_index_cp in range_list_cp[0:len(range_list_cp)]:

                        if transform_index_cp == range_list_cp[compare_index]:
                            continue

                        count += 1

                        train_dir = img_dir + '/' + multi_class_item_cp + '/' + transform_index_cp
                        train_dir = train_dir.replace('/', '-')

                        try:
                            print(save_dir_pre + '/%s/%s_%s_layer%d.npy' % (
                                (args.arch + '_' + train_dir + '_mid_res'), args.arch, train_dir, test_layer))
                            tensor_array_list[1] = save_dir_pre + '/%s/%s_%s_layer%d.npy' % (
                                (args.arch + '_' + train_dir + '_mid_res'), args.arch, train_dir, test_layer)
                        except FileNotFoundError:
                            print('file_not_found: ', transform_index_cp)
                            continue

                        metric_list_np_name = metric_list_np_dir + '/metric_list_o_cpt_%s-%s-%s-%s_%sd_%s_%s_%d_zero_' % (
                            transform_img_index_avg, multi_class_item_inner, multi_class_item_cp,
                            transform_index_cp, args.mt, ts_operation,
                            mode,
                            args.l) + '.npy'

                        abs_d_list_np_name = abs_d_list_np_dir + '/metric_list_o_cpt_%s-%s-%s-%s_abs_d_%s_%s_%d_zero_' % (
                            transform_img_index_avg, multi_class_item_inner, multi_class_item_cp,
                            transform_index_cp, ts_operation, mode,
                            args.l) + '.npy'

                        # if os.path.exists(metric_list_np_name):
                        #     continue

                        print('%d: ' % count, metric_list_np_name)

                        # if os.path.exists(metric_list_np_name):
                        #     print('--- DS EXISTS: %s ---' % (metric_list_np_name))
                        #     continue
                        params.append(
                            (tensor_array_list[0], tensor_array_list[1], metric_list_np_name, abs_d_list_np_name,
                             args.mt, count))

                    # multi thread
        p = multiprocessing.Pool(15)
        p.map(do_cal_jsd_list_between_tensors, params)
        p.close()
        p.join()

    if args.op == 'cal_pools_jsd_marker_cd_zone_time_zero_inner_all_avg':

        # ==================
        # cal the js for zone in zero phase
        # ==================
        # the operation used to transform images, eg. gaussian blur: gb, all channel change: ac
        ts_operation = args.tsop
        img_dir = t + '/transform_images_%s_noise' % ts_operation
        mode = 'zone'
        save_dir_pre = args.tdir + '_layer_npy'
        i_o_index = args.ioi
        time_type = args.tt
        phase = 'zero'

        # ==================
        # cal the js for zone in zero phase
        # ==================
        if args.param == 'time':
            img_dir = t + '/transform_images_t_%s_noise' % ts_operation
            mode = 'time'

        multi_classes = os.listdir(img_dir)
        count = 0

        params = []

        for multi_class_item in multi_classes:

            # if args.ioi == 'o':
            #     params = []

            if multi_class_item != args.ec and args.ec != 'none' or multi_class_item == 'n00000000':
                continue

            print('===%s===' % (multi_class_item))

            layer_time = ''
            if time_type == '2':
                layer_time = '_' + str(int(time_type) - 1) + '_' + multi_class_item
            elif time_type == '3':
                layer_time = '_' + str(int(time_type) - 2) + '_' + multi_class_item + '_' + str(
                    int(time_type) - 1) + '_' + multi_class_item
            elif time_type == '4':
                layer_time = '_' + str(int(time_type) - 3) + '_' + multi_class_item + '_' + str(
                    int(time_type) - 2) + '_' + multi_class_item + '_' + str(
                    int(time_type) - 1) + '_' + multi_class_item

            keep_node_pools = np.load(
                t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/keep_node_pools%s_wad_%s_%s_5_%s_list.npy' % (
                    time_type, ts_operation, mode, phase), allow_pickle=True)

            keep_node_pools_weight = np.load(
                t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/keep_node_pools%s_weight_wad_%s_%s_5_%s_list.npy' % (
                    time_type, ts_operation, mode, phase), allow_pickle=True)
            deleted_node_set = np.load(
                t + '/' + args.arch + '/deleted_node_npy_dir/deleted_node_%sd_marker_%s_%s_set_%s.npy' % (
                    args.mt, args.dnop, ts_operation, multi_class_item))

            keep_node_set = list(set(list(range(output_size))) - set(deleted_node_set))

            # if time_type != '1':
            #     keep_node_set = np.load(
            #         t + '/' + args.arch + '/pools_keep_node_npy_dir/keep_node_pools%s_%sd_marker_%s_set_%s.npy' % (
            #             str(int(time_type) - 1), args.mt, ts_operation, multi_class_item))

            for test_layer in range(args.l, args.l + 1):
                metric_list_np_dir = t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/metric_list_%sd_%s_%s_%d_zero_npy' % (
                    args.mt, ts_operation, mode, args.l)
                abs_d_list_np_dir = t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/metric_list_abs_d_%s_%s_%d_zero_npy' % (
                    ts_operation, mode, args.l)

                # ==================
                # cal the js for zone in zero phase
                # ==================
                tensor_array_list = [None, None]

                # print('=====continue cal jsd=====')

                range_list_inner = []

                t0_name = ''
                t1_name = ''

                range_list_inner = os.listdir(img_dir + '/' + multi_class_item)

                iter_list = []

                if args.cp == 'one':
                    iter_list = range_list_inner[compare_index:compare_index + 1]
                elif args.cp == 'all':
                    iter_list = range_list_inner[0:len(range_list_inner)]

                tensor_array_list_zero_list = []

                for transform_img_index_inner in iter_list:
                    # print(transform_img_index_inner)
                    train_dir = img_dir + '/' + multi_class_item + '/' + transform_img_index_inner
                    train_dir = train_dir.replace('/', '-')

                    t0_name = save_dir_pre + '/%s/%s_%s_layer%d%s.npy' % (
                        (args.arch + '_' + train_dir + '_mid_res'), args.arch, train_dir, test_layer, layer_time)
                    tensor_array_list[0] = np.load(t0_name)

                    tensor_array_list[0] = tensor_array_list[0].reshape(tensor_array_list[0].shape[0], -1)

                    tensor_array_list_zero_list.append(tensor_array_list[0])

                tensor_array_list[0] = np.zeros(tensor_array_list_zero_list[0].shape)

                for item in tensor_array_list_zero_list:
                    tensor_array_list[0] += item

                tensor_array_list[0] = tensor_array_list[0] / len(tensor_array_list_zero_list)

                transform_img_index_avg = 'avg'

                random_cp_classes = np.load(
                    t + '/inner_cp_classes/%s_cp_classes.npy' % (multi_class_item)).tolist()

                random_cp_classes.append(multi_class_item)

                for multi_class_item_cp in random_cp_classes:

                    if multi_class_item_cp == 'n00000000':
                        continue
                    if args.ioi == 'i' and multi_class_item_cp != multi_class_item:
                        continue
                    if args.ioi == 'o' and multi_class_item_cp == multi_class_item:
                        continue

                    range_list_cp = os.listdir(img_dir + '/' + multi_class_item_cp)

                    # print(multi_class_item_cp)

                    for transform_index_cp in range_list_cp:

                        if transform_index_cp == range_list_cp[compare_index]:
                            continue

                        # if transform_index_cp != 'n03937543_4835_sc':
                        #     continue

                        metric_list_np_name = metric_list_np_dir + '/metric_list_%s_pools%s_cpt_%s-%s-%s-%s_%sd_%s_%s_%d_zero_' % (
                            args.ioi, time_type, transform_img_index_avg, multi_class_item,
                            multi_class_item_cp,
                            transform_index_cp,
                            args.mt, ts_operation, mode,
                            args.l) + '.npy'

                        abs_d_list_np_name = abs_d_list_np_dir + '/metric_list_%s_pools%s_cpt_%s-%s-%s-%s_abs_d_%s_%s_%d_zero_' % (
                            args.ioi, time_type, transform_img_index_avg, multi_class_item,
                            multi_class_item_cp,
                            transform_index_cp,
                            ts_operation, mode,
                            args.l) + '.npy'

                        # if os.path.exists(metric_list_np_name):
                        #     print('ex: %s' % (metric_list_np_name))
                        #     continue

                        train_dir = img_dir + '/' + multi_class_item_cp + '/' + transform_index_cp
                        train_dir = train_dir.replace('/', '-')

                        try:
                            # print(save_dir_pre + '/%s/%s_%s_layer%d.npy' % (
                            #     (args.arch + '_' + train_dir + '_mid_res'), args.arch, train_dir, test_layer))
                            t1_name = save_dir_pre + '/%s/%s_%s_layer%d%s.npy' % (
                                (args.arch + '_' + train_dir + '_mid_res'), args.arch, train_dir, test_layer,
                                layer_time)
                            # tensor_array_list[1] = np.load(t1_name)
                        except FileNotFoundError:
                            print(transform_index_cp)
                            continue

                        # for i in range(1, len(tensor_array_list)):
                        #     tensor_array_list[i] = tensor_array_list[i].reshape(tensor_array_list[i].shape[0], -1)

                        # print('%d: ' % count, metric_list_np_name)

                        # if os.path.exists(metric_list_np_name):
                        #     # print('ex: %s' % (metric_list_np_name))
                        #     continue

                        count += 1

                        params.append(
                            (tensor_array_list[0], tensor_array_list[1], metric_list_np_name, abs_d_list_np_name,
                             args.mt,
                             keep_node_pools,
                             keep_node_pools_weight,
                             keep_node_set,
                             t0_name,
                             t1_name,
                             time_type,
                             multi_class_item, count))

                        # cal_jsd_list_between_tensors(tensor_array_list[0], tensor_array_list[1], metric_list_np_name,
                        #                              abs_d_list_np_name, args.mt)
            # if args.ioi == 'o':
            #     p = multiprocessing.Pool(15)
            #     p.map(do_cal_jsd_list_between_tensors_avg, params)
            #     p.close()
            #     p.join()

        # if args.ioi == 'i':
        p = multiprocessing.Pool(15)
        p.map(do_cal_jsd_list_between_tensors_avg, params)
        p.close()
        p.join()

    if args.op == 'cal_pools_jsd_marker_cd_zone_time_zero_inner_all_get_pool_npy':

        # ==================
        # cal the js for zone in zero phase
        # ==================
        # the operation used to transform images, eg. gaussian blur: gb, all channel change: ac
        ts_operation = args.tsop
        img_dir = t + '/transform_images_%s_noise' % ts_operation
        mode = 'zone'
        save_dir_pre = args.tdir + '_layer_npy'
        i_o_index = args.ioi
        time_type = args.tt
        phase = 'zero'

        # ==================
        # cal the js for zone in zero phase
        # ==================
        if args.param == 'time':
            img_dir = t + '/transform_images_t_%s_noise' % ts_operation
            mode = 'time'

        multi_classes = os.listdir(img_dir)
        keep_node_pools_dic = {}
        keep_node_pools_weight_dic = {}
        keep_node_set_dic = {}

        for multi_class_item_cp in multi_classes:
            keep_node_pools = np.load(
                t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item_cp + '/keep_node_pools%s_wad_%s_%s_%s_%s_list.npy' % (
                    time_type, ts_operation, mode, args.l, phase), allow_pickle=True)

            keep_node_pools_weight = np.load(
                t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item_cp + '/keep_node_pools%s_weight_wad_%s_%s_%s_%s_list.npy' % (
                    time_type, ts_operation, mode, args.l, phase), allow_pickle=True)
            deleted_node_set = np.load(
                t + '/' + args.arch + '/deleted_node_npy_dir/deleted_node_%sd_marker_%s_%s_set_%s.npy' % (
                    args.mt, args.dnop, ts_operation, multi_class_item_cp))

            keep_node_set = list(set(list(range(output_size))) - set(deleted_node_set))

            keep_node_pools_dic[multi_class_item_cp] = keep_node_pools
            keep_node_pools_weight_dic[multi_class_item_cp] = keep_node_pools_weight
            keep_node_set_dic[multi_class_item_cp] = keep_node_set

        count = 0

        print('cal_pools_jsd_marker_cd_zone_time_zero_inner_all_get_pool_npy')
        params = []
        for multi_class_item in multi_classes:
            # params = []
            print('===== %s =====' % (multi_class_item))

            layer_time = ''
            if time_type == '2':
                layer_time = '_' + str(int(time_type) - 1) + '_' + multi_class_item
            elif time_type == '3':
                layer_time = '_' + str(int(time_type) - 2) + '_' + multi_class_item + '_' + str(
                    int(time_type) - 1) + '_' + multi_class_item
            elif time_type == '4':
                layer_time = '_' + str(int(time_type) - 3) + '_' + multi_class_item + '_' + str(
                    int(time_type) - 2) + '_' + multi_class_item + '_' + str(
                    int(time_type) - 1) + '_' + multi_class_item

            for test_layer in range(args.l, args.l + 1):

                tensor_array_list = [None, None]

                range_list_inner = os.listdir(img_dir + '/' + multi_class_item)

                iter_list = []

                if args.cp == 'one':
                    iter_list = range_list_inner[compare_index:compare_index + 1]
                elif args.cp == 'all':
                    iter_list = range_list_inner[0:len(range_list_inner)]

                tensor_array_list_zero_list = []

                for transform_img_index_inner in iter_list:

                    train_dir = img_dir + '/' + multi_class_item + '/' + transform_img_index_inner
                    train_dir = train_dir.replace('/', '-')

                    # random_cp_classes = np.load(
                    #     t + '/inner_cp_classes/%s_cp_classes.npy' % (multi_class_item)).tolist()

                    ts_cp_classes = np.load(
                        t + '/ts_cp_classes/%s_ts_cp_classes.npy' % (multi_class_item)).tolist()
                    # ts_cp_classes = []

                    ts_cp_classes.append(multi_class_item)

                    for multi_class_item_cp in ts_cp_classes:

                        if multi_class_item_cp == 'n00000000':
                            continue

                        if multi_class_item_cp != args.ec and args.ec != 'none' or multi_class_item_cp == 'n00000000':
                            continue

                        if time_type != 1:

                            if time_type == '2':
                                layer_time = '_' + str(int(time_type) - 1) + '_' + multi_class_item_cp
                            elif time_type == '3':
                                layer_time = '_' + str(int(time_type) - 2) + '_' + multi_class_item_cp + '_' + str(
                                    int(time_type) - 1) + '_' + multi_class_item_cp
                            elif time_type == '4':
                                layer_time = '_' + str(int(time_type) - 3) + '_' + multi_class_item_cp + '_' + str(
                                    int(time_type) - 2) + '_' + multi_class_item_cp + '_' + str(
                                    int(time_type) - 1) + '_' + multi_class_item_cp

                        t0_name = save_dir_pre + '/%s/%s_%s_layer%d%s.npy' % (
                            (args.arch + '_' + train_dir + '_mid_res'), args.arch, train_dir, test_layer,
                            layer_time)

                        # if os.path.exists(t0_name.split('.')[0] + '_' + time_type + '_' + multi_class_item_cp + '.npy'):
                        #     print('ex: %d - %s' % (
                        #         count, t0_name.split('.')[0] + '_' + time_type + '_' + multi_class_item_cp + '.npy'))
                        #     count += 1
                        #     continue

                        print('cal: %d - %s' % (
                            count, t0_name.split('.')[0] + '_' + time_type + '_' + multi_class_item_cp + '.npy'))

                        # tensor_array_list[0] = np.load(t0_name)

                        keep_node_pools = keep_node_pools_dic[multi_class_item_cp]

                        keep_node_pools_weight = keep_node_pools_weight_dic[multi_class_item_cp]

                        keep_node_set = keep_node_set_dic[multi_class_item_cp]

                        count += 1

                        params.append((t0_name, '', keep_node_pools, keep_node_pools_weight, keep_node_set,
                                       t0_name, time_type, multi_class_item_cp, True, count))

                        # do_get_min_pool_npy((t0_name, '', keep_node_pools, keep_node_pools_weight, keep_node_set,
                        #                t0_name, time_type, multi_class_item_cp, True, count))
        #
        p = multiprocessing.Pool(15)
        p.map(do_get_min_pool_npy, params)
        p.close()
        p.join()

    if args.op == 'cal_jsd_marker_cd_time_one':

        ts_operation = args.tsop

        # img_dir = transform_dir_t
        img_dir = t + '/transform_images_t_%s_noise' % ts_operation
        mode = 'time'
        multi_classes = os.listdir(img_dir)
        phase = 'one'
        params = []
        count = 0
        for multi_class_item in multi_classes:

            if multi_class_item != args.ec and args.ec != 'none':
                continue

            for test_layer in range(args.l, args.l + 1):
                metric_list_np_dir = t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/metric_list_%sd_%s_%s_%d_one_npy' % (
                    args.mt, ts_operation, mode, args.l)
                abs_d_list_np_dir = t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/metric_list_abs_d_%s_%s_%d_one_npy' % (
                    ts_operation, mode, args.l)

                mkdir(metric_list_np_dir)
                mkdir(abs_d_list_np_dir)

                train_dir = img_dir + '/' + multi_class_item + '/0000'
                # train_dir = img_dir + '/' + multi_class_item + '/' + multi_class_item
                train_dir = train_dir.replace('/', '-')

                tensor_array_list = [np.load(
                    save_dir_pre + '/%s/%s_%s_layer%d.npy' % (
                        (args.arch + '_' + train_dir + '_mid_res'), args.arch, train_dir, test_layer)), None]

                print('=====continue cal %sd=====' % args.mt)

                # threads = []

                t_num = len(os.listdir(img_dir + '/' + multi_class_item))

                for seq_index in range(1, t_num):

                    if seq_index < 10:
                        seq_index = '000' + str(seq_index)
                    elif seq_index < 100:
                        seq_index = '00' + str(seq_index)
                    elif seq_index < 1000:
                        seq_index = '0' + str(seq_index)
                    else:
                        seq_index = str(seq_index)

                    count += 1
                    # if count == 4:
                    #     break

                    metric_list_np_name = metric_list_np_dir + '/metric_list_%sd_%s_%s_%d_one_' % (args.mt,
                                                                                                   ts_operation, mode,
                                                                                                   args.l) + seq_index + '.npy'
                    # if os.path.exists(metric_list_np_name):
                    #     print('ex: %s' % metric_list_np_name)
                    #     continue

                    train_dir = img_dir + '/' + multi_class_item + '/' + seq_index
                    train_dir = train_dir.replace('/', '-')

                    tensor_array_list[1] = save_dir_pre + '/%s/%s_%s_layer%d.npy' % (
                        (args.arch + '_' + train_dir + '_mid_res'), args.arch, train_dir, test_layer)

                    # if tensor_array_list[1].shape[0] != tensor_array_list[0].shape[0]:
                    #     break
                    #
                    # for i in range(len(tensor_array_list)):
                    #     tensor_array_list[i] = tensor_array_list[i].reshape(tensor_array_list[i].shape[0], -1)
                    #     # tensor_array_list[i] = tensor_array_normalization(tensor_array_list[i])

                    abs_d_list_np_name = abs_d_list_np_dir + '/metric_list_abs_d_%s_%s_%d_one_' % (
                        ts_operation, mode, args.l) + seq_index + '.npy'

                    print('%d: ' % count, metric_list_np_name)

                    params.append(
                        (tensor_array_list[0], tensor_array_list[1], metric_list_np_name, abs_d_list_np_name, args.mt,
                         count))

                    tensor_array_list[0] = np.load(tensor_array_list[1])

        p = multiprocessing.Pool(15)
        p.map(do_cal_jsd_list_between_tensors, params)
        p.close()
        p.join()

    if args.op == 'analysis_jsd_marker_c_zone_zero2_avg':
        # img_dir = transform_dir
        ts_operation = args.tsop
        img_dir = t + '/transform_images_%s_noise' % ts_operation
        mode = 'zone'
        phase = 'zero'

        if args.param == 'time':
            # img_dir = transform_dir_t
            # img_dir = t + '/transform_images_t_%s_noise' % operation
            mode = 'time'

        for multi_class_item in os.listdir(origin_image_dir):

            if multi_class_item != args.ec and args.ec != 'none' or multi_class_item == 'n00000000':
                continue

            metric_list_np_dir = t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/metric_list_%sd_%s_%s_%d_zero_npy' % (
                args.mt, ts_operation, mode, args.l)
            abs_d_list_np_dir = t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/metric_list_abs_d_%s_%s_%d_zero_npy' % (
                ts_operation, mode, args.l)

            metric_file_name_list = []
            abs_d_file_name_list = []

            if mode == 'zone':
                metric_file_name_list = os.listdir(metric_list_np_dir)
                if args.mt == 'js':
                    abs_d_file_name_list = os.listdir(abs_d_list_np_dir)

            abs_d_file_list = []

            range_list_inner = os.listdir(img_dir + '/' + multi_class_item)

            metric_file_list_list = []
            mkdir(t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/wa_class_npy')

            iter_list = []
            if args.cp == 'one':
                iter_list = range_list_inner[compare_index:compare_index + 1]
            elif args.cp == 'all':
                iter_list = range_list_inner[0:len(range_list_inner)]
                iter_list = ['avg']

            for transform_img_index2 in iter_list:
                # for transform_img_index2 in range_list[compare_index:compare_index + 1]:
                # for transform_img_index2 in range_list[0:len(range_list)]:

                metric_file_list2 = []

                print(transform_img_index2)

                for transform_index_cp in range_list_inner:

                    if transform_index_cp == range_list_inner[compare_index]:
                        continue

                    metric_list_np_name = metric_list_np_dir + '/metric_list_%s-%s_%sd_%s_%s_%d_zero_' % (
                        transform_img_index2, transform_index_cp, args.mt, ts_operation, mode,
                        args.l) + '.npy'

                    metric_file_np = np.load(metric_list_np_name)
                    # print(metric_file_np.shape)
                    metric_file_list2.append(metric_file_np.tolist())
                print(np.array(metric_file_list2).shape)
                metric_file_list_list.append(metric_file_list2)
                np.save(
                    t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/wa_class_npy/%s_file_list_%s-%s_%s_%s_%s.npy' % (
                        args.mt, multi_class_item, transform_img_index2, ts_operation, mode, phase), metric_file_list2)

            print(np.array(metric_file_list_list).shape)
            # metric_file_list = np.mean(np.array(metric_file_list_list), axis=0)
            metric_file_list = np.array(metric_file_list_list)
            np.save(
                t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/all_%s_file_list_%s_%s_%s_%s.npy' % (
                    args.mt, multi_class_item, ts_operation, mode, phase), metric_file_list)

            print(metric_file_list.shape)

            metric_file_list = np.mean(np.array(metric_file_list), axis=0)

            print('wa file: ', metric_file_list.shape)

            np.save(
                t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/%s_file_list_%s_%s_%s_%s.npy' % (
                    args.mt, multi_class_item, ts_operation, mode, phase), metric_file_list)

            if args.mt == 'js':
                for abs_d_file in abs_d_file_name_list:
                    abs_d_file_np = np.load(abs_d_list_np_dir + '/' + abs_d_file)
                    # print(abs_d_file_np.shape)
                    abs_d_file_list.append(abs_d_file_np.reshape(abs_d_file_np.shape[0], ).tolist())
                abs_d_file_list = np.array(abs_d_file_list)
                np.save(
                    t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/abs_d_file_list_%s_%s_%s_%s.npy' % (
                        multi_class_item, ts_operation, mode, phase), abs_d_file_list)

            metric_std_list = []
            metric_mean_list = []
            metric_cv_list = []
            metric_kurt_list = []

            metric_plus_abs_d_std_list = []
            metric_plus_abs_d_mean_list = []
            metric_plus_abs_d_cv_list = []
            metric_plus_abs_d_kurt_list = []

            for i in range(metric_file_list.shape[1]):

                if i % 10000 == 0:
                    print('std cal process: %d' % i)

                mean = np.mean(metric_file_list[:, i]) + 0.0000001
                std = np.std(metric_file_list[:, i]) + 0.0000001
                cv = std / mean
                kurt = np.mean((metric_file_list[:, i] - mean) ** 4) / pow(std * std, 2)
                # kurt = scipy.stats.kurtosis(metric_file_list[:, i])
                metric_std_list.append(std)
                metric_mean_list.append(mean)
                metric_cv_list.append(cv)
                metric_kurt_list.append(kurt)

                if args.mt == 'js':
                    plus_abs_d_np = metric_file_list[:, i] + abs_d_file_list[:, i] * 0.1
                    # plus_abs_d_np = abs_d_file_list[:, i]
                    plus_abs_d_mean = np.mean(plus_abs_d_np) + 0.0000001
                    plus_abs_d_std = np.std(plus_abs_d_np) + 0.0000001
                    plus_abs_d_cv = plus_abs_d_std / plus_abs_d_mean
                    plus_abs_d_kurt = np.mean((plus_abs_d_np - plus_abs_d_mean) ** 4) / pow(
                        plus_abs_d_std * plus_abs_d_std,
                        2)
                    # plus_abs_d_kurt = scipy.stats.kurtosis(plus_abs_d_np)

                    metric_plus_abs_d_std_list.append(plus_abs_d_std)
                    metric_plus_abs_d_mean_list.append(plus_abs_d_mean)
                    metric_plus_abs_d_cv_list.append(plus_abs_d_cv)
                    metric_plus_abs_d_kurt_list.append(plus_abs_d_kurt)

            np.save(
                t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/metric_%sd_std_%s_%s_%d_zero_list.npy' % (
                    args.mt, ts_operation, mode, args.l), np.array(metric_std_list))
            np.save(
                t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/metric_%sd_mean_%s_%s_%d_zero_list.npy' % (
                    args.mt, ts_operation, mode, args.l), np.array(metric_mean_list))
            np.save(
                t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/metric_%sd_cv_%s_%s_%d_zero_list.npy' % (
                    args.mt, ts_operation, mode, args.l), np.array(metric_cv_list))
            np.save(
                t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/metric_%sd_kurt_%s_%s_%d_zero_list.npy' % (
                    args.mt, ts_operation, mode, args.l), np.array(metric_kurt_list))

            if args.mt == 'js':
                np.save(
                    t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/metric_plus_abs_d_std_%s_%s_%d_zero_list.npy' % (
                        ts_operation, mode, args.l), np.array(metric_plus_abs_d_std_list))
                np.save(
                    t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/metric_plus_abs_d_mean_%s_%s_%d_zero_list.npy' % (
                        ts_operation, mode, args.l), np.array(metric_plus_abs_d_mean_list))
                np.save(
                    t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/metric_plus_abs_d_cv_%s_%s_%d_zero_list.npy' % (
                        ts_operation, mode, args.l), np.array(metric_plus_abs_d_cv_list))
                np.save(
                    t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/metric_plus_abs_d_kurt_%s_%s_%d_zero_list.npy' % (
                        ts_operation, mode, args.l), np.array(metric_plus_abs_d_kurt_list))

            # print('===std===')
            # print(np.max(metric_std_list))
            # print(np.min(metric_std_list))
            # print('===std===')
            #
            # print('===mean===')
            # print(np.max(metric_mean_list))
            # print(np.min(metric_mean_list))
            # print('===mean===')

            print('=== cal zone zero %s ===' % multi_class_item)

            print('analysis std')
            order_arr_std = np.array(list(range(len(metric_std_list))))
            quick_sort(metric_std_list, 0, len(order_arr_std) - 1, order_arr_std)
            np.save(
                t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/order_arr_metric_%sd_std_%s_%s_%d_zero_list.npy' % (
                    args.mt, ts_operation, mode, args.l), order_arr_std)

            print('analysis mean')
            order_arr_mean = np.array(list(range(len(metric_mean_list))))
            quick_sort(metric_mean_list, 0, len(order_arr_mean) - 1, order_arr_mean)
            np.save(
                t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/order_arr_metric_%sd_mean_%s_%s_%d_zero_list.npy' % (
                    args.mt, ts_operation, mode, args.l), order_arr_mean)

            print('analysis cv')
            order_arr_cv = np.array(list(range(len(metric_cv_list))))
            quick_sort(metric_cv_list, 0, len(order_arr_cv) - 1, order_arr_cv)
            np.save(
                t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/order_arr_metric_%sd_cv_%s_%s_%d_zero_list.npy' % (
                    args.mt, ts_operation, mode, args.l), order_arr_cv)

            print('analysis kurt')
            order_arr_kurt = np.array(list(range(len(metric_kurt_list))))
            quick_sort(metric_kurt_list, 0, len(order_arr_kurt) - 1, order_arr_kurt)
            np.save(
                t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/order_arr_metric_%sd_kurt_%s_%s_%d_zero_list.npy' % (
                    args.mt, ts_operation, mode, args.l), order_arr_kurt)

            if args.mt == 'js':
                print('analysis plus abs d std')
                order_arr_plus_abs_d_std = np.array(list(range(len(metric_plus_abs_d_std_list))))
                quick_sort(metric_plus_abs_d_std_list, 0, len(order_arr_plus_abs_d_std) - 1, order_arr_plus_abs_d_std)
                np.save(
                    t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/order_arr_metric_plus_abs_d_std_%s_%s_%d_zero_list.npy' % (
                        ts_operation, mode, args.l), order_arr_plus_abs_d_std)

                print('analysis plus abs d mean')
                order_arr_plus_abs_d_mean = np.array(list(range(len(metric_plus_abs_d_mean_list))))
                quick_sort(metric_plus_abs_d_mean_list, 0, len(order_arr_plus_abs_d_mean) - 1,
                           order_arr_plus_abs_d_mean)
                np.save(
                    t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/order_arr_metric_plus_abs_d_mean_%s_%s_%d_zero_list.npy' % (
                        ts_operation, mode, args.l), order_arr_plus_abs_d_mean)

                print('analysis plus abs d cv')
                order_arr_plus_abs_d_cv = np.array(list(range(len(metric_plus_abs_d_cv_list))))
                quick_sort(metric_plus_abs_d_cv_list, 0, len(order_arr_plus_abs_d_cv) - 1, order_arr_plus_abs_d_cv)
                np.save(
                    t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/order_arr_metric_plus_abs_d_cv_%s_%s_%d_zero_list.npy' % (
                        ts_operation, mode, args.l), order_arr_plus_abs_d_cv)

                print('analysis plus abs d kurt')
                order_arr_plus_abs_d_kurt = np.array(list(range(len(metric_plus_abs_d_kurt_list))))
                quick_sort(metric_plus_abs_d_kurt_list, 0, len(order_arr_plus_abs_d_kurt) - 1,
                           order_arr_plus_abs_d_kurt)
                np.save(
                    t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/order_arr_metric_plus_abs_d_kurt_%s_%s_%d_zero_list.npy' % (
                        ts_operation, mode, args.l), order_arr_plus_abs_d_kurt)

    if args.op == 'analysis_jsd_marker_c_zone_zero_other_inner_all_avg':
        # img_dir = transform_dir
        ts_operation = args.tsop
        img_dir = t + '/transform_images_%s_noise' % ts_operation
        mode = 'zone'
        phase = 'zero'

        if args.param == 'time':
            # img_dir = transform_dir_t
            # img_dir = t + '/transform_images_t_%s_noise' % operation
            mode = 'time'

        for multi_class_item_inner in os.listdir(origin_image_dir):

            if multi_class_item_inner != args.ec and args.ec != 'none':
                continue

            range_list_inner = os.listdir(img_dir + '/' + multi_class_item_inner)

            metric_list_np_dir = t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item_inner + '/metric_list_o_%sd_%s_%s_%d_zero_npy' % (
                args.mt, ts_operation, mode, args.l)
            abs_d_list_np_dir = t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item_inner + '/metric_list_o_abs_d_%s_%s_%d_zero_npy' % (
                ts_operation, mode, args.l)

            metric_file_name_list = []
            abs_d_file_name_list = []

            metric_file_list_list = []
            # all_metric_file_list = []

            random_cp_classes = np.load(t + '/inner_cp_classes/%s_cp_classes.npy' % (multi_class_item_inner)).tolist()

            for multi_class_item_cp in random_cp_classes:

                if multi_class_item_inner == multi_class_item_cp:
                    continue
                if multi_class_item_cp == 'n00000000' or multi_class_item_inner == 'n00000000':
                    continue

                print(multi_class_item_inner + '-' + multi_class_item_cp)

                if mode == 'zone':
                    metric_file_name_list = os.listdir(metric_list_np_dir)
                    if args.mt == 'js':
                        abs_d_file_name_list = os.listdir(abs_d_list_np_dir)

                metric_file_list = []
                abs_d_file_list = []

                range_list_cp = os.listdir(img_dir + '/' + multi_class_item_cp)

                iter_list = []

                if args.cp == 'one':
                    iter_list = range_list_inner[compare_index:compare_index + 1]
                elif args.cp == 'all':
                    iter_list = range_list_inner[0:len(range_list_inner)]
                    iter_list = ['avg']

                for transform_index_inner in iter_list:

                    metric_file_index = []

                    # for metric_file in metric_file_name_list:
                    #     if not metric_file.__contains__('-' + multi_class_item_inner + '-' + multi_class_item_cp + '-'):
                    #         continue
                    #     if not metric_file.__contains__('_o_cpt'):
                    #         continue
                    #     if not metric_file.__contains__(transform_index_inner):
                    #         continue

                    for transform_index_cp in range_list_cp:

                        if transform_index_cp == range_list_cp[compare_index]:
                            continue

                        metric_list_np_name = metric_list_np_dir + '/metric_list_o_cpt_%s-%s-%s-%s_%sd_%s_%s_%d_zero_' % (
                            transform_index_inner, multi_class_item_inner, multi_class_item_cp,
                            transform_index_cp, args.mt, ts_operation,
                            mode,
                            args.l) + '.npy'

                        metric_file_np = np.load(metric_list_np_name)
                        metric_file_index.append(metric_file_np.tolist())

                    print('metric_file_index shape: ', np.array(metric_file_index).shape)
                    metric_file_list.append(metric_file_index)

                # metric_file_list = np.mean(np.array(metric_file_list), axis=0)
                metric_file_list_list.append(metric_file_list)
                metric_file_list = np.array(metric_file_list)
                print('metric_file_list shape: ', metric_file_list.shape)
                print(np.array(metric_file_list_list).shape)

                # metric_file_list = np.array(metric_file_list)
                # np.save(
                #     t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item_inner + '/o_%s_file_list_%s_%s_%s_%s.npy' % (
                #         args.mt, multi_class_item_inner, ts_operation, mode, phase), metric_file_list)

            # print(len(metric_file_list_list))
            # print(len(metric_file_list_list[0]))
            print(np.array(metric_file_list_list).shape)
            # np.save(
            #     t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item_inner + '/o_%s_file_list_%s_%s_%s_%s.npy' % (
            #         args.mt, multi_class_item_inner, ts_operation, mode, phase),
            #     metric_file_list_list)
            o_wa_npy = np.mean(np.array(metric_file_list_list), axis=0)
            print(o_wa_npy.shape)
            np.save(
                t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item_inner + '/all_o_%s_file_list_%s_%s_%s_%s.npy' % (
                    args.mt, multi_class_item_inner, ts_operation, mode, phase),
                o_wa_npy)

            o_wa_npy = np.mean(o_wa_npy, axis=0)

            np.save(
                t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item_inner + '/o_%s_file_list_%s_%s_%s_%s.npy' % (
                    args.mt, multi_class_item_inner, ts_operation, mode, phase),
                o_wa_npy)

            # metric_std_list = []
            # metric_mean_list = []
            # metric_cv_list = []
            # metric_kurt_list = []
            #
            # metric_plus_abs_d_std_list = []
            # metric_plus_abs_d_mean_list = []
            # metric_plus_abs_d_cv_list = []
            # metric_plus_abs_d_kurt_list = []
            #
            # for i in range(metric_file_list.shape[1]):
            #
            #     if i % 1000 == 0:
            #         print('std cal process: %d' % i)
            #
            #     mean = np.mean(metric_file_list[:, i]) + 0.0000001
            #     std = np.std(metric_file_list[:, i]) + 0.0000001
            #     cv = std / mean
            #     # kurt = np.mean((metric_file_list[:, i] - mean) ** 4) / pow(std * std, 2)
            #     # kurt = scipy.stats.kurtosis(metric_file_list[:, i])
            #     kurt = random.random()
            #     metric_std_list.append(std)
            #     metric_mean_list.append(mean)
            #     metric_cv_list.append(cv)
            #     metric_kurt_list.append(kurt)
            #
            #     if args.mt == 'js':
            #         plus_abs_d_np = metric_file_list[:, i] + abs_d_file_list[:, i] * 0.1
            #         # plus_abs_d_np = abs_d_file_list[:, i]
            #         plus_abs_d_mean = np.mean(plus_abs_d_np) + 0.0000001
            #         plus_abs_d_std = np.std(plus_abs_d_np) + 0.0000001
            #         plus_abs_d_cv = plus_abs_d_std / plus_abs_d_mean
            #         # plus_abs_d_kurt = np.mean((plus_abs_d_np - plus_abs_d_mean) ** 4) / pow(plus_abs_d_std * plus_abs_d_std,
            #         #                                                                         2)
            #         # plus_abs_d_kurt = scipy.stats.kurtosis(plus_abs_d_np)
            #         plus_abs_d_kurt = random.random()
            #
            #         metric_plus_abs_d_std_list.append(plus_abs_d_std)
            #         metric_plus_abs_d_mean_list.append(plus_abs_d_mean)
            #         metric_plus_abs_d_cv_list.append(plus_abs_d_cv)
            #         metric_plus_abs_d_kurt_list.append(plus_abs_d_kurt)
            #
            # np.save(
            #     t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/o_metric_%sd_std_%s_%s_%d_%s_list.npy' % (
            #         args.mt,
            #         ts_operation, mode, args.l, phase), np.array(metric_std_list))
            # np.save(
            #     t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/o_metric_%sd_mean_%s_%s_%d_%s_list.npy' % (
            #         args.mt,
            #         ts_operation, mode, args.l, phase), np.array(metric_mean_list))
            # np.save(
            #     t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/o_metric_%sd_cv_%s_%s_%d_%s_list.npy' % (
            #         args.mt,
            #         ts_operation, mode, args.l, phase), np.array(metric_cv_list))
            # np.save(
            #     t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/o_metric_%sd_kurt_%s_%s_%d_%s_list.npy' % (
            #         args.mt,
            #         ts_operation, mode, args.l, phase), np.array(metric_kurt_list))
            #
            # if args.mt == 'js':
            #     np.save(
            #         t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/o_metric_plus_abs_d_std_%s_%s_%d_%s_list.npy' % (
            #             ts_operation, mode, args.l, phase), np.array(metric_plus_abs_d_std_list))
            #     np.save(
            #         t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/o_metric_plus_abs_d_mean_%s_%s_%d_%s_list.npy' % (
            #             ts_operation, mode, args.l, phase), np.array(metric_plus_abs_d_mean_list))
            #     np.save(
            #         t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/o_metric_plus_abs_d_cv_%s_%s_%d_%s_list.npy' % (
            #             ts_operation, mode, args.l, phase), np.array(metric_plus_abs_d_cv_list))
            #     np.save(
            #         t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/o_metric_plus_abs_d_kurt_%s_%s_%d_%s_list.npy' % (
            #             ts_operation, mode, args.l, phase), np.array(metric_plus_abs_d_kurt_list))

    if args.op == 'analysis_pools_jsd_marker_c_zone_zero_inner_all_avg':
        # img_dir = transform_dir
        ts_operation = args.tsop
        img_dir = t + '/transform_images_%s_noise' % ts_operation
        mode = 'zone'
        phase = 'zero'
        time_type = args.tt

        count = 0
        params = []
        for multi_class_item in os.listdir(origin_image_dir):

            if multi_class_item != args.ec and args.ec != 'none':
                continue

            print('%d: %s' % (count, multi_class_item))

            count += 1

            # if os.path.exists(
            #         t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/metric_%s_pools%s_%sd_std_%s_%s_%d_zero_list.npy' % (
            #                 args.ioi, time_type, args.mt, ts_operation, mode, args.l)):
            #     continue

            params.append((t, ts_operation, img_dir, mode, phase, time_type, multi_class_item, compare_index, count))

        p = multiprocessing.Pool(6)
        p.map(do_analysis_pools, params)
        p.close()
        p.join()

    if args.op == 'analysis_jsd_marker_c_time_zero_one':
        # img_dir = transform_dir
        ts_operation = args.tsop
        # img_dir = t + '/transform_images_%s_noise' % operation
        mode = 'time'
        phase = args.phase

        for multi_class_item in os.listdir(origin_image_dir):

            if multi_class_item != args.ec and args.ec != 'none':
                continue

            metric_list_np_dir = t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/metric_list_%sd_%s_%s_%d_%s_npy' % (
                args.mt, ts_operation, mode, args.l, phase)
            abs_d_list_np_dir = t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/metric_list_abs_d_%s_%s_%d_%s_npy' % (
                ts_operation, mode, args.l, phase)

            metric_file_name_list = []
            abs_d_file_name_list = []

            if mode == 'zone':
                metric_file_name_list = os.listdir(metric_list_np_dir)
                if args.met == 'js':
                    abs_d_file_name_list = os.listdir(abs_d_list_np_dir)
            elif mode == 'time':

                t_num = len(os.listdir(metric_list_np_dir)) + 1

                for seq_index in range(1, t_num):
                    if seq_index < 10:
                        seq_index = '000' + str(seq_index)
                    elif seq_index < 100:
                        seq_index = '00' + str(seq_index)
                    elif seq_index < 1000:
                        seq_index = '0' + str(seq_index)
                    else:
                        seq_index = str(seq_index)

                    metric_file_name_list.append(
                        'metric_list_%sd_%s_%s_%d_%s_' % (
                            args.mt, ts_operation, mode, args.l, phase) + seq_index + '.npy')
                    abs_d_file_name_list.append(
                        'metric_list_abs_d_%s_%s_%d_%s_' % (ts_operation, mode, args.l, phase) + seq_index + '.npy')

            metric_file_list = []
            abs_d_file_list = []

            for metric_file in metric_file_name_list:
                metric_file_list.append(np.array(np.load(metric_list_np_dir + '/' + metric_file)))

            metric_file_list = np.array(metric_file_list)

            np.save(
                t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/%s_file_list_%s_%s_%s_%s.npy' % (
                    args.mt, multi_class_item, ts_operation, mode, phase), metric_file_list)

            if args.mt == 'js':
                for abs_d_file in abs_d_file_name_list:
                    abs_d_file_list.append(np.array(np.load(abs_d_list_np_dir + '/' + abs_d_file)))
                abs_d_file_list = np.array(abs_d_file_list)
                np.save(
                    t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/abs_d_file_list_%s_%s_%s_%s.npy' % (
                        multi_class_item, ts_operation, mode, phase), abs_d_file_list)

            metric_std_list = []
            metric_mean_list = []
            metric_cv_list = []
            metric_kurt_list = []

            metric_plus_abs_d_std_list = []
            metric_plus_abs_d_mean_list = []
            metric_plus_abs_d_cv_list = []
            metric_plus_abs_d_kurt_list = []

            print(metric_file_list.shape)

            for i in range(metric_file_list.shape[1]):

                if i % 1000 == 0:
                    print('std cal process: %d' % i)

                mean = np.mean(metric_file_list[:, i]) + 0.0000001
                std = np.std(metric_file_list[:, i]) + 0.0000001
                cv = std / mean
                # kurt = np.mean((metric_file_list[:, i] - mean) ** 4) / pow(std * std, 2)
                # kurt = scipy.stats.kurtosis(metric_file_list[:, i])
                kurt = random.random()
                metric_std_list.append(std)
                metric_mean_list.append(mean)
                metric_cv_list.append(cv)
                metric_kurt_list.append(kurt)

                if args.mt == 'js':
                    plus_abs_d_np = metric_file_list[:, i] + abs_d_file_list[:, i] * 0.1
                    # plus_abs_d_np = abs_d_file_list[:, i]
                    plus_abs_d_mean = np.mean(plus_abs_d_np) + 0.0000001
                    plus_abs_d_std = np.std(plus_abs_d_np) + 0.0000001
                    plus_abs_d_cv = plus_abs_d_std / plus_abs_d_mean
                    # plus_abs_d_kurt = np.mean((plus_abs_d_np - plus_abs_d_mean) ** 4) / pow(plus_abs_d_std * plus_abs_d_std,
                    #                                                                         2)
                    # plus_abs_d_kurt = scipy.stats.kurtosis(plus_abs_d_np)
                    plus_abs_d_kurt = random.random()

                    metric_plus_abs_d_std_list.append(plus_abs_d_std)
                    metric_plus_abs_d_mean_list.append(plus_abs_d_mean)
                    metric_plus_abs_d_cv_list.append(plus_abs_d_cv)
                    metric_plus_abs_d_kurt_list.append(plus_abs_d_kurt)

            np.save(
                t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/metric_%sd_std_%s_%s_%d_%s_list.npy' % (
                    args.mt,
                    ts_operation, mode, args.l, phase), np.array(metric_std_list))
            np.save(
                t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/metric_%sd_mean_%s_%s_%d_%s_list.npy' % (
                    args.mt,
                    ts_operation, mode, args.l, phase), np.array(metric_mean_list))
            np.save(
                t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/metric_%sd_cv_%s_%s_%d_%s_list.npy' % (
                    args.mt,
                    ts_operation, mode, args.l, phase), np.array(metric_cv_list))
            np.save(
                t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/metric_%sd_kurt_%s_%s_%d_%s_list.npy' % (
                    args.mt,
                    ts_operation, mode, args.l, phase), np.array(metric_kurt_list))

            if args.mt == 'js':
                np.save(
                    t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/metric_plus_abs_d_std_%s_%s_%d_%s_list.npy' % (
                        ts_operation, mode, args.l, phase), np.array(metric_plus_abs_d_std_list))
                np.save(
                    t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/metric_plus_abs_d_mean_%s_%s_%d_%s_list.npy' % (
                        ts_operation, mode, args.l, phase), np.array(metric_plus_abs_d_mean_list))
                np.save(
                    t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/metric_plus_abs_d_cv_%s_%s_%d_%s_list.npy' % (
                        ts_operation, mode, args.l, phase), np.array(metric_plus_abs_d_cv_list))
                np.save(
                    t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/metric_plus_abs_d_kurt_%s_%s_%d_%s_list.npy' % (
                        ts_operation, mode, args.l, phase), np.array(metric_plus_abs_d_kurt_list))

            # print('===std===')
            # print(np.max(metric_std_list))
            # print(np.min(metric_std_list))
            # print('===std===')
            #
            # print('===mean===')
            # print(np.max(metric_mean_list))
            # print(np.min(metric_mean_list))
            # print('===mean===')

            print('=== cal time %s %s ===' % (phase, multi_class_item))

            print('analysis std')
            order_arr_std = np.array(list(range(len(metric_std_list))))
            # print(metric_std_list)
            quick_sort(metric_std_list, 0, len(order_arr_std) - 1, order_arr_std)
            np.save(
                t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/order_arr_metric_%sd_std_%s_%s_%d_%s_list.npy' % (
                    args.mt,
                    ts_operation, mode, args.l, phase), order_arr_std)

            print('analysis mean')
            order_arr_mean = np.array(list(range(len(metric_mean_list))))
            quick_sort(metric_mean_list, 0, len(order_arr_mean) - 1, order_arr_mean)
            np.save(
                t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/order_arr_metric_%sd_mean_%s_%s_%d_%s_list.npy' % (
                    args.mt,
                    ts_operation, mode, args.l, phase), order_arr_mean)

            print('analysis cv')
            order_arr_cv = np.array(list(range(len(metric_cv_list))))
            quick_sort(metric_cv_list, 0, len(order_arr_cv) - 1, order_arr_cv)
            np.save(
                t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/order_arr_metric_%sd_cv_%s_%s_%d_%s_list.npy' % (
                    args.mt,
                    ts_operation, mode, args.l, phase), order_arr_cv)

            print('analysis kurt')
            order_arr_kurt = np.array(list(range(len(metric_kurt_list))))
            quick_sort(metric_kurt_list, 0, len(order_arr_kurt) - 1, order_arr_kurt)
            np.save(
                t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/order_arr_metric_%sd_kurt_%s_%s_%d_%s_list.npy' % (
                    args.mt,
                    ts_operation, mode, args.l, phase), order_arr_kurt)

            if args.mt == 'js':
                print('analysis plus abs d std')
                order_arr_plus_abs_d_std = np.array(list(range(len(metric_plus_abs_d_std_list))))
                quick_sort(metric_plus_abs_d_std_list, 0, len(order_arr_plus_abs_d_std) - 1, order_arr_plus_abs_d_std)
                np.save(
                    t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/order_arr_metric_plus_abs_d_std_%s_%s_%d_%s_list.npy' % (
                        ts_operation, mode, args.l, phase), order_arr_plus_abs_d_std)

                print('analysis plus abs d mean')
                order_arr_plus_abs_d_mean = np.array(list(range(len(metric_plus_abs_d_mean_list))))
                quick_sort(metric_plus_abs_d_mean_list, 0, len(order_arr_plus_abs_d_mean) - 1,
                           order_arr_plus_abs_d_mean)
                np.save(
                    t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/order_arr_metric_plus_abs_d_mean_%s_%s_%d_%s_list.npy' % (
                        ts_operation, mode, args.l, phase), order_arr_plus_abs_d_mean)

                print('analysis plus abs d cv')
                order_arr_plus_abs_d_cv = np.array(list(range(len(metric_plus_abs_d_cv_list))))
                quick_sort(metric_plus_abs_d_cv_list, 0, len(order_arr_plus_abs_d_cv) - 1, order_arr_plus_abs_d_cv)
                np.save(
                    t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/order_arr_metric_plus_abs_d_cv_%s_%s_%d_%s_list.npy' % (
                        ts_operation, mode, args.l, phase), order_arr_plus_abs_d_cv)

                print('analysis plus abs d kurt')
                order_arr_plus_abs_d_kurt = np.array(list(range(len(metric_plus_abs_d_kurt_list))))
                quick_sort(metric_plus_abs_d_kurt_list, 0, len(order_arr_plus_abs_d_kurt) - 1,
                           order_arr_plus_abs_d_kurt)
                np.save(
                    t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/order_arr_metric_plus_abs_d_kurt_%s_%s_%d_%s_list.npy' % (
                        ts_operation, mode, args.l, phase), order_arr_plus_abs_d_kurt)

    if args.op == 'analysis_deleted_node2':

        print('=====analysis_deleted_node2=====')
        ts_operation = args.tsop

        deleted_node_set_all = []

        model = get_model()

        mkdir(t + '/' + args.arch + '/deleted_node_npy_dir')

        for multi_class_item in os.listdir(t + '/' + args.arch + '/metric_list_npy_dir'):

            if multi_class_item == 'n00000000':
                continue

            deleted_node_marker_a_set = set()
            deleted_node_plus_abs_d_marker_a_set = set()
            deleted_node_marker_a2_set = set()
            deleted_node_plus_abs_d_marker_a2_set = set()
            deleted_node_marker_b_set = set()
            deleted_node_plus_abs_d_marker_b_set = set()
            deleted_node_marker_c_set = set()
            deleted_node_plus_abs_d_marker_c_set = set()
            deleted_node_marker_c2_set = set()
            deleted_node_plus_abs_d_marker_c2_set = set()
            deleted_node_marker_d_set = set()
            deleted_node_plus_abs_d_marker_d_set = set()
            deleted_node_marker_ac_set = set()
            deleted_node_plus_abs_d_marker_ac_set = set()
            deleted_node_plus_abs_d_set = set()
            deleted_node_marker_cov_set = set()
            deleted_node_marker_jsjs_set = set()
            deleted_node_set = set()

            if args.ec != 'none' and args.ec != multi_class_item:
                print(multi_class_item)
                continue

            # ================================
            # marker_a_node
            # ================================

            if args.dnop.__contains__('a'):
                # marker_a_order_arr_js = np.load(
                #     t + '/'+args.arch+'/metric_list_npy_dir/%s/order_arr_metric_marker_a_js_list.npy' % multi_class_item)
                # marker_a_order_arr_plus_abs_d = np.load(
                #     t + '/'+args.arch+'/metric_list_npy_dir/%s/order_arr_metric_marker_a_plus_abs_d_list.npy' % multi_class_item)
                #
                # deleted_node_marker_a_set = set(
                #     marker_a_order_arr_js[args.param4:args.param5].tolist())
                # deleted_node_plus_abs_d_marker_a_set = set(
                #     marker_a_order_arr_plus_abs_d[args.param4:args.param5].tolist())

                # deleted_node_marker_a_set = set(
                #     list(range(len(marker_a_order_arr_js)))) - deleted_node_marker_a_set
                # deleted_node_plus_abs_d_marker_a_set = set(
                #     list(range(len(marker_a_order_arr_js)))) - deleted_node_plus_abs_d_marker_a_set

                # print('=====%d (deleted_node_marker_a_set len)=====' % len(deleted_node_marker_a_set))
                # print('=====%d (deleted_node_plus_abs_d_marker_a_set len)=====' % len(
                #     deleted_node_plus_abs_d_marker_a_set))

                mode = 'zone'
                phase = 'zero'

                marker_a2_order_arr_std = np.load(
                    t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/order_arr_metric_marker_a2_jsd_std_%s_%d_%s_list.npy' % (
                        mode, args.l, phase))
                marker_a2_order_arr_mean = np.load(
                    t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/order_arr_metric_marker_a2_jsd_mean_%s_%d_%s_list.npy' % (
                        mode, args.l, phase))
                marker_a2_order_arr_cv = np.load(
                    t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/order_arr_metric_marker_a2_jsd_cv_%s_%d_%s_list.npy' % (
                        mode, args.l, phase))
                marker_a2_order_arr_kurt = np.load(
                    t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/order_arr_metric_marker_a2_jsd_kurt_%s_%d_%s_list.npy' % (
                        mode, args.l, phase))

                marker_a2_order_arr_plus_abs_d_std = np.load(
                    t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/order_arr_metric_marker_a2_plus_abs_d_std_%s_%d_%s_list.npy' % (
                        mode, args.l, phase))
                marker_a2_order_arr_plus_abs_d_mean = np.load(
                    t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/order_arr_metric_marker_a2_plus_abs_d_mean_%s_%d_%s_list.npy' % (
                        mode, args.l, phase))
                marker_a2_order_arr_plus_abs_d_cv = np.load(
                    t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/order_arr_metric_marker_a2_plus_abs_d_cv_%s_%d_%s_list.npy' % (
                        mode, args.l, phase))
                marker_a2_order_arr_plus_abs_d_kurt = np.load(
                    t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/order_arr_metric_marker_a2_plus_abs_d_kurt_%s_%d_%s_list.npy' % (
                        mode, args.l, phase))

                # print(marker_a2_order_arr_mean)

                marker_a2_order_arr_std_pre = set(marker_a2_order_arr_std[args.param4:args.param5].tolist())
                marker_a2_order_arr_mean_pre = set(marker_a2_order_arr_mean[args.param4:args.param5].tolist())
                marker_a2_order_arr_cv_pre = set(marker_a2_order_arr_cv[args.param4:args.param5].tolist())
                marker_a2_order_arr_kurt_pre = set(marker_a2_order_arr_kurt[args.param4:args.param5].tolist())

                marker_a2_order_arr_plus_abs_d_std_pre = set(
                    marker_a2_order_arr_plus_abs_d_std[args.param4:args.param5].tolist())
                marker_a2_order_arr_plus_abs_d_mean_pre = set(
                    marker_a2_order_arr_plus_abs_d_mean[args.param4:args.param5].tolist())
                marker_a2_order_arr_plus_abs_d_cv_pre = set(
                    marker_a2_order_arr_plus_abs_d_cv[args.param4:args.param5].tolist())
                marker_a2_order_arr_plus_abs_d_kurt_pre = set(
                    marker_a2_order_arr_plus_abs_d_kurt[args.param4:args.param5].tolist())

                deleted_node_marker_a2_set = set(
                    list(range(len(marker_a2_order_arr_mean)))) - marker_a2_order_arr_std_pre.intersection(
                    marker_a2_order_arr_cv_pre)

                deleted_node_plus_abs_d_marker_a2_set = set(
                    list(range(len(marker_a2_order_arr_mean)))) - marker_a2_order_arr_plus_abs_d_std_pre.intersection(
                    marker_a2_order_arr_plus_abs_d_cv_pre)

                deleted_node_marker_a2_set = marker_a2_order_arr_std_pre.intersection(
                    marker_a2_order_arr_cv_pre)

                deleted_node_plus_abs_d_marker_a2_set = marker_a2_order_arr_plus_abs_d_std_pre.intersection(
                    marker_a2_order_arr_plus_abs_d_cv_pre)

                print('=====%d (deleted_node_marker_a_set len)=====' % len(deleted_node_marker_a2_set))
                print('=====%d (deleted_node_plus_abs_d_marker_a_set len)=====' % len(
                    deleted_node_plus_abs_d_marker_a2_set))

            # ================================
            # marker_c_node, marker_d_node
            # ================================

            if args.dnop.__contains__('zzc') or args.dnop.__contains__('d'):
                mode = 'zone'
                phase = 'zero'

                down_bound = args.param2
                up_bound = args.param3

                marker_c_order_arr_std = np.load(
                    t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/order_arr_metric_%sd_std_%s_%s_%d_%s_list.npy' % (
                        args.mt, ts_operation, mode, args.l, phase))
                marker_c_order_arr_mean = np.load(
                    t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/order_arr_metric_%sd_mean_%s_%s_%d_%s_list.npy' % (
                        args.mt, ts_operation, mode, args.l, phase))
                marker_c_order_arr_cv = np.load(
                    t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/order_arr_metric_%sd_cv_%s_%s_%d_%s_list.npy' % (
                        args.mt, ts_operation, mode, args.l, phase))
                marker_c_order_arr_kurt = np.load(
                    t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/order_arr_metric_%sd_kurt_%s_%s_%d_%s_list.npy' % (
                        args.mt, ts_operation, mode, args.l, phase))

                marker_c_order_arr_std_pre = set(marker_c_order_arr_std[down_bound:int(up_bound / 1)].tolist())
                marker_c_order_arr_mean_pre = set(marker_c_order_arr_mean[down_bound:int(up_bound / 1)].tolist())
                marker_c_order_arr_cv_pre = set(marker_c_order_arr_cv[down_bound:up_bound].tolist())
                marker_c_order_arr_kurt_pre = set(marker_c_order_arr_kurt[down_bound:up_bound].tolist())

                if args.mt == 'js':
                    marker_c_order_arr_plus_abs_d_std = np.load(
                        t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/order_arr_metric_plus_abs_d_std_%s_%s_%d_%s_list.npy' % (
                            ts_operation, mode, args.l, phase))
                    marker_c_order_arr_plus_abs_d_mean = np.load(
                        t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/order_arr_metric_plus_abs_d_mean_%s_%s_%d_%s_list.npy' % (
                            ts_operation, mode, args.l, phase))
                    marker_c_order_arr_plus_abs_d_cv = np.load(
                        t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/order_arr_metric_plus_abs_d_cv_%s_%s_%d_%s_list.npy' % (
                            ts_operation, mode, args.l, phase))
                    marker_c_order_arr_plus_abs_d_kurt = np.load(
                        t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/order_arr_metric_plus_abs_d_kurt_%s_%s_%d_%s_list.npy' % (
                            ts_operation, mode, args.l, phase))

                    # metric_plus_abs_d_cv_list = np.load(
                    #     t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/metric_plus_abs_d_cv_%s_%s_%d_zero_list.npy' % (
                    #         ts_operation, mode, args.l))
                    # metric_plus_abs_d_std_list = np.load(
                    #     t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/metric_plus_abs_d_std_%s_%s_%d_zero_list.npy' % (
                    #         ts_operation, mode, args.l))
                    # metric_plus_abs_d_mean_list = np.load(
                    #     t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/metric_plus_abs_d_mean_%s_%s_%d_zero_list.npy' % (
                    #         ts_operation, mode, args.l))
                    #
                    # # print(np.max(metric_plus_abs_d_cv_list))
                    # # print(np.max(metric_plus_abs_d_std_list))
                    # # print(np.min(metric_plus_abs_d_mean_list))
                    #
                    # zero_node_list = []
                    # for i in range(len(metric_plus_abs_d_mean_list)):
                    #     if metric_plus_abs_d_mean_list[i] == 0:
                    #         zero_node_list.append(i)

                    marker_c_order_arr_plus_abs_d_std_pre = set(
                        marker_c_order_arr_plus_abs_d_std[down_bound:int(up_bound / 1)].tolist())
                    marker_c_order_arr_plus_abs_d_mean_pre = set(
                        marker_c_order_arr_plus_abs_d_mean[down_bound:int(up_bound / 1)].tolist())
                    marker_c_order_arr_plus_abs_d_cv_pre = set(
                        marker_c_order_arr_plus_abs_d_cv[down_bound:up_bound].tolist())
                    marker_c_order_arr_plus_abs_d_kurt_pre = set(
                        marker_c_order_arr_plus_abs_d_kurt[down_bound:up_bound].tolist())

                    deleted_node_plus_abs_d_marker_c_set = set(
                        list(range(len(marker_c_order_arr_mean)))) - (
                                                               marker_c_order_arr_plus_abs_d_std_pre)

                # print(marker_c_order_arr_mean_pre - marker_c_order_arr_plus_abs_d_mean_pre)

                # deleted_node_marker_c_set = set(
                #     list(range(len(marker_c_order_arr_mean)))) - marker_c_order_arr_std_pre.intersection(
                #     marker_c_order_arr_cv_pre)
                #
                # deleted_node_plus_abs_d_marker_c_set = set(
                #     list(range(len(marker_c_order_arr_mean)))) - marker_c_order_arr_plus_abs_d_std_pre.intersection(
                #     marker_c_order_arr_plus_abs_d_cv_pre)

                # deleted_node_marker_c_set = set(
                #     list(range(len(marker_c_order_arr_mean)))) - marker_c_order_arr_std_pre

                # deleted_node_plus_abs_d_marker_c_set = set(
                #     list(range(len(marker_c_order_arr_mean)))) - marker_c_order_arr_plus_abs_d_kurt_pre

                # deleted_node_plus_abs_d_marker_c_set = set(np.load('node_30_1.npy'))

                deleted_node_marker_c_set = set(
                    list(range(len(marker_c_order_arr_mean)))) - (
                                                marker_c_order_arr_std_pre)

                # ================================
                # marker_d_node_zero
                # ================================

                if args.dnop.__contains__('d'):
                    mode = 'time'
                    phase = 'zero'
                    #
                    # marker_d_un_regular = np.load(
                    #     t + '/'+args.arch+'/metric_list_npy_dir/' + multi_class_item + '/un_regular_metric_js_%s_%d_%s_list.npy' % (
                    #         mode, args.l, phase))
                    # marker_d_un_regular_plus_abs_d = np.load(
                    #     t + '/'+args.arch+'/metric_list_npy_dir/' + multi_class_item + '/un_regular_metric_plus_abs_d_%s_%d_%s_list.npy' % (
                    #         mode, args.l, phase))
                    #

                    confidence_bound = args.cb
                    p_value_bound = args.pb
                    coint_node_plus_abs_d_list = np.load(
                        t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/coint_metric_plus_abs_d_%s_%s_%d_%s_c%d_%.2f_list.npy' % (
                            ts_operation, mode, args.l, phase, confidence_bound, p_value_bound))
                    coint_node_js_list = np.load(
                        t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/coint_metric_js_%s_%s_%d_%s_c%d_%.2f_list.npy' % (
                            ts_operation, mode, args.l, phase, confidence_bound, p_value_bound))
                    # not_coint_node_js_list = np.load(
                    #     t + '/'+args.arch+'/metric_list_npy_dir/' + multi_class_item + '/not_coint_metric_js_%s_%d_%s_list.npy' % (
                    #         mode, args.l, phase))

                    # coint_node_js_list = coint_node_js_list.reshape(1, -1)

                    # deleted_node_marker_d_set = set(coint_node_js_list)
                    # deleted_node_plus_abs_d_marker_d_set = set(coint_node_plus_abs_d_list)
                    #
                    deleted_node_marker_d_set = set(coint_node_js_list).intersection(set(
                        marker_c_order_arr_plus_abs_d_cv[down_bound:int(up_bound / 3)]))
                    deleted_node_plus_abs_d_marker_d_set = set(coint_node_plus_abs_d_list).intersection(set(
                        marker_c_order_arr_plus_abs_d_cv[down_bound:int(up_bound / 3)]))

                    deleted_node_marker_d_set = set(coint_node_js_list).intersection(set(
                        marker_c_order_arr_plus_abs_d_cv[down_bound:int(up_bound / 4)]).intersection(set(
                        marker_c_order_arr_plus_abs_d_std[down_bound:int(up_bound / 4)])))
                    deleted_node_plus_abs_d_marker_d_set = set(coint_node_plus_abs_d_list).intersection(set(
                        marker_c_order_arr_plus_abs_d_cv[down_bound:int(up_bound / 4)]).intersection(set(
                        marker_c_order_arr_plus_abs_d_std[down_bound:int(up_bound / 4)])))

                    deleted_node_marker_d_set = set(
                        marker_c_order_arr_plus_abs_d_cv[down_bound:int(up_bound / 4)])
                    deleted_node_plus_abs_d_marker_d_set = set(
                        marker_c_order_arr_plus_abs_d_cv[down_bound:int(up_bound / 4)])

                    print(len(deleted_node_marker_d_set))
                    print(len(deleted_node_plus_abs_d_marker_d_set))

            if args.dnop.__contains__('toc'):

                mode = 'time'
                phase = args.phase

                marker_c2_order_arr_std = np.load(
                    t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/order_arr_metric_%sd_std_%s_%s_%d_%s_list.npy' % (
                        args.mt, ts_operation, mode, args.l, phase))
                marker_c2_order_arr_mean = np.load(
                    t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/order_arr_metric_%sd_mean_%s_%s_%d_%s_list.npy' % (
                        args.mt, ts_operation, mode, args.l, phase))
                marker_c2_order_arr_cv = np.load(
                    t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/order_arr_metric_%sd_cv_%s_%s_%d_%s_list.npy' % (
                        args.mt, ts_operation, mode, args.l, phase))
                marker_c2_order_arr_kurt = np.load(
                    t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/order_arr_metric_%sd_kurt_%s_%s_%d_%s_list.npy' % (
                        args.mt, ts_operation, mode, args.l, phase))

                metric_std_list = np.load(
                    t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/metric_%sd_std_%s_%s_%d_%s_list.npy' % (
                        args.mt, ts_operation, mode, args.l, phase))

                # print(metric_std_list[marker_c2_order_arr_std[args.param6]])
                # print(metric_std_list[marker_c2_order_arr_std[args.param6 + 1]])

                down_bound = args.param6
                up_bound = args.param7

                metric_std_list_order = []
                for i in range(0, len(metric_std_list)):
                    metric_std_list_order.append(metric_std_list[marker_c2_order_arr_std[i]])

                std_median = np.median(metric_std_list_order)

                median_index = 0
                for i in range(0, len(metric_std_list)):
                    if metric_std_list_order[i] > std_median:
                        print('median_index: ', i)
                        median_index = i
                        break

                median_std = np.std(metric_std_list_order[0:median_index])
                median_mean = np.mean(metric_std_list_order[0:median_index])

                metric_std_list_del_big = []

                for i in range(0, len(metric_std_list)):
                    if metric_std_list_order[i] < median_mean + 5 * median_std:
                        metric_std_list_del_big.append(metric_std_list[marker_c2_order_arr_std[i]])
                    else:
                        print('normal_bound: ', i)
                        up_bound = i
                        break

                # std_mean = np.mean(metric_std_list_del_big)
                # std_std = np.std(metric_std_list_del_big)
                # std_rate = 0
                #
                # for i in range(len(marker_c2_order_arr_std)):
                #     if metric_std_list[marker_c2_order_arr_std[i]] > std_mean - std_rate * std_std:
                #         print('up_bound: ', i)
                #         up_bound = i
                #         break

                # up_bound = int(median_index)

                marker_c2_order_arr_std_pre = set(marker_c2_order_arr_std[down_bound:up_bound].tolist())
                marker_c2_order_arr_mean_pre = set(marker_c2_order_arr_mean[down_bound:up_bound].tolist())
                marker_c2_order_arr_cv_pre = set(marker_c2_order_arr_cv[down_bound:up_bound].tolist())
                marker_c2_order_arr_kurt_pre = set(marker_c2_order_arr_kurt[down_bound:up_bound].tolist())

                if args.mt == 'js':
                    other_mt = 'wa'

                    marker_c2_order_arr_plus_abs_d_std = np.load(
                        t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/order_arr_metric_plus_abs_d_std_%s_%s_%d_%s_list.npy' % (
                            ts_operation, mode, args.l, phase))
                    marker_c2_order_arr_plus_abs_d_mean = np.load(
                        t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/order_arr_metric_plus_abs_d_mean_%s_%s_%d_%s_list.npy' % (
                            ts_operation, mode, args.l, phase))
                    marker_c2_order_arr_plus_abs_d_cv = np.load(
                        t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/order_arr_metric_plus_abs_d_cv_%s_%s_%d_%s_list.npy' % (
                            ts_operation, mode, args.l, phase))
                    marker_c2_order_arr_plus_abs_d_kurt = np.load(
                        t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/order_arr_metric_plus_abs_d_kurt_%s_%s_%d_%s_list.npy' % (
                            ts_operation, mode, args.l, phase))

                    marker_c2_order_arr_plus_abs_d_std_pre = set(
                        marker_c2_order_arr_plus_abs_d_std[down_bound:up_bound].tolist())
                    marker_c2_order_arr_plus_abs_d_mean_pre = set(
                        marker_c2_order_arr_plus_abs_d_mean[down_bound:up_bound].tolist())
                    marker_c2_order_arr_plus_abs_d_cv_pre = set(
                        marker_c2_order_arr_plus_abs_d_cv[down_bound:up_bound].tolist())
                    marker_c2_order_arr_plus_abs_d_kurt_pre = set(
                        marker_c2_order_arr_plus_abs_d_kurt[down_bound:up_bound].tolist())

                    deleted_node_plus_abs_d_marker_c2_set = set(
                        list(range(len(marker_c2_order_arr_mean)))) - marker_c2_order_arr_plus_abs_d_std_pre

                # metric_plus_abs_d_cv_list = np.load(
                #     t + '/'+args.arch+'/metric_list_npy_dir/%s' % multi_class_item + '/metric_plus_abs_d_cv_%s_%s_%d_%s_list.npy' % (
                #         ts_operation, mode, args.l, phase))
                # metric_plus_abs_d_std_list = np.load(
                #     t + '/'+args.arch+'/metric_list_npy_dir/%s' % multi_class_item + '/metric_plus_abs_d_std_%s_%s_%d_%s_list.npy' % (
                #         ts_operation, mode, args.l, phase))
                # metric_plus_abs_d_mean_list = np.load(
                #     t + '/'+args.arch+'/metric_list_npy_dir/%s' % multi_class_item + '/metric_plus_abs_d_mean_%s_%s_%d_%s_list.npy' % (
                #         ts_operation, mode, args.l, phase))
                #
                # # print(np.max(metric_plus_abs_d_cv_list))
                # # print(np.max(metric_plus_abs_d_std_list))
                # # print(np.min(metric_plus_abs_d_mean_list))
                #
                # print(metric_plus_abs_d_cv_list[marker_c2_order_arr_plus_abs_d_cv[args.param7]])
                # print(metric_plus_abs_d_cv_list[marker_c2_order_arr_plus_abs_d_cv[args.param7 - 1]])
                #
                # zero_node_list = []
                # for i in range(len(metric_plus_abs_d_mean_list)):
                #     if metric_plus_abs_d_mean_list[i] == 0:
                #         zero_node_list.append(i)

                # print(marker_c2_order_arr_mean_pre - marker_c2_order_arr_plus_abs_d_mean_pre)

                # deleted_node_marker_c2_set = set(
                #     list(range(len(marker_c2_order_arr_mean)))) - marker_c2_order_arr_std_pre.intersection(
                #     marker_c2_order_arr_cv_pre)
                #
                # deleted_node_plus_abs_d_marker_c2_set = set(
                #     list(range(len(marker_c2_order_arr_mean)))) - marker_c2_order_arr_plus_abs_d_std_pre.intersection(
                #     marker_c2_order_arr_plus_abs_d_cv_pre)

                deleted_node_marker_c2_set = set(
                    list(range(len(marker_c2_order_arr_mean)))) - marker_c2_order_arr_std_pre

                # deleted_node_marker_c2_set = marker_c2_order_arr_std_pre
                #
                # deleted_node_plus_abs_d_marker_c2_set = marker_c2_order_arr_plus_abs_d_std_pre
                #
                # deleted_node_plus_abs_d_marker_c2_set = set(np.load('node_30_1.npy'))

                # deleted_node_marker_c2_set = set(
                #     list(range(len(marker_c2_order_arr_mean)))) - marker_c2_order_arr_cv_pre.intersection(
                #     marker_c2_order_arr_std_pre)

                # deleted_node_plus_abs_d_marker_c2_set = set(
                #     list(range(len(marker_c2_order_arr_mean)))) - marker_c2_order_arr_plus_abs_d_cv_pre.intersection(
                #     marker_c2_order_arr_plus_abs_d_std_pre)

            # ================================
            # marker_b_node
            # ================================

            if args.dnop.__contains__('b'):
                mode = 'zone'
                phase = 'zero'

                marker_b_order_arr_std = np.load(
                    t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/order_arr_metric_marker_b_jsd_std_%s_%d_%s_list.npy' % (
                        mode, args.l, phase))
                marker_b_order_arr_mean = np.load(
                    t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/order_arr_metric_marker_b_jsd_mean_%s_%d_%s_list.npy' % (
                        mode, args.l, phase))
                marker_b_order_arr_cv = np.load(
                    t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/order_arr_metric_marker_b_jsd_cv_%s_%d_%s_list.npy' % (
                        mode, args.l, phase))
                marker_b_order_arr_kurt = np.load(
                    t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/order_arr_metric_marker_b_jsd_kurt_%s_%d_%s_list.npy' % (
                        mode, args.l, phase))

                # marker_b_order_arr_plus_abs_d_std = np.load(
                #     t + '/'+args.arch+'/metric_list_npy_dir/' + multi_class_item + '/order_arr_metric_marker_b_plus_abs_d_std_%s_%d_%s_list.npy' % (
                #         mode, args.l, phase))
                # marker_b_order_arr_plus_abs_d_mean = np.load(
                #     t + '/'+args.arch+'/metric_list_npy_dir/' + multi_class_item + '/order_arr_metric_marker_b_plus_abs_d_mean_%s_%d_%s_list.npy' % (
                #         mode, args.l, phase))
                # marker_b_order_arr_plus_abs_d_cv = np.load(
                #     t + '/'+args.arch+'/metric_list_npy_dir/' + multi_class_item + '/order_arr_metric_marker_b_plus_abs_d_cv_%s_%d_%s_list.npy' % (
                #         mode, args.l, phase))
                # marker_b_order_arr_plus_abs_d_kurt = np.load(
                #     t + '/'+args.arch+'/metric_list_npy_dir/' + multi_class_item + '/order_arr_metric_marker_b_plus_abs_d_kurt_%s_%d_%s_list.npy' % (
                #         mode, args.l, phase))

                # print(marker_b_order_arr_mean)

                # marker_b_order_arr_std_pre = set(marker_b_order_arr_std[args.param2:args.param3].tolist())
                # marker_b_order_arr_mean_pre = set(marker_b_order_arr_mean[args.param2:args.param3].tolist())
                # marker_b_order_arr_cv_pre = set(marker_b_order_arr_cv[args.param2:args.param3].tolist())
                # marker_b_order_arr_kurt_pre = set(marker_b_order_arr_kurt[args.param2:args.param3].tolist())
                marker_b_order_arr_std_pre = set(marker_b_order_arr_std[0:8000].tolist())
                marker_b_order_arr_mean_pre = set(marker_b_order_arr_mean[0:8000].tolist())
                marker_b_order_arr_cv_pre = set(marker_b_order_arr_cv[0:8000].tolist())
                marker_b_order_arr_kurt_pre = set(marker_b_order_arr_kurt[0:8000].tolist())

                # marker_b_order_arr_plus_abs_d_std_pre = set(
                #     marker_b_order_arr_plus_abs_d_std[args.param2:args.param3].tolist())
                # marker_b_order_arr_plus_abs_d_mean_pre = set(
                #     marker_b_order_arr_plus_abs_d_mean[args.param2:args.param3].tolist())
                # marker_b_order_arr_plus_abs_d_cv_pre = set(
                #     marker_b_order_arr_plus_abs_d_cv[args.param2:args.param3].tolist())
                # marker_b_order_arr_plus_abs_d_kurt_pre = set(
                #     marker_b_order_arr_plus_abs_d_kurt[args.param2:args.param3].tolist())

                deleted_node_marker_b_set = set(
                    list(range(len(marker_b_order_arr_mean)))) - marker_b_order_arr_std_pre.intersection(
                    marker_b_order_arr_cv_pre)

                # deleted_node_plus_abs_d_marker_b_set = set(
                #     list(range(len(marker_b_order_arr_mean)))) - marker_b_order_arr_plus_abs_d_std_pre.intersection(
                #     marker_b_order_arr_plus_abs_d_cv_pre)

                print('=====%d (deleted_node_marker_b_set len)=====' % len(deleted_node_marker_b_set))
                # print('=====%d (deleted_node_plus_abs_d_marker_b_set len)=====' % len(
                #     deleted_node_plus_abs_d_marker_b_set))

            # ================================
            # marker_toc_cov_node
            # ================================

            if args.dnop.__contains__('cov'):

                cov_i_list = np.load(t + '/' + args.arch + '/cov_npy_dir/cov_%sd_marker_toc_%s_set_%s.npy' % (
                    args.mt, ts_operation, multi_class_item))

                toc_deleted_node_set = np.load(
                    t + '/' + args.arch + '/deleted_node_npy_dir/deleted_node_%sd_marker_toc_%s_set_%s.npy' % (
                        args.mt, ts_operation, multi_class_item))

                deleted_node_marker_cov_set_pre = []

                print(len(cov_i_list))

                for i in range(len(cov_i_list)):
                    if cov_i_list[i][0][1] > 0:
                        deleted_node_marker_cov_set_pre.append(toc_deleted_node_set[i])

                deleted_node_marker_cov_set = set(list(range(output_size))) - set(deleted_node_marker_cov_set_pre)
                print(len(deleted_node_marker_cov_set))

            # ================================
            # cal acc
            # ================================

            if args.dnop.__contains__('jsjs'):
                mode = 'zone'

                down_bound = args.param2
                up_bound = args.param3

                order_arr_jsjs = np.load(
                    t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/order_arr_metric_%sd_jsjs_%s_%s_%d_zero_list.npy' % (
                        args.mt, ts_operation, mode, args.l))

                order_arr_jsjs_pre = set(order_arr_jsjs[down_bound:int(up_bound / 1)].tolist())

                deleted_node_marker_jsjs_set = set(
                    list(range(len(order_arr_jsjs)))) - (order_arr_jsjs_pre)

                pass

            deleted_node_marker_types = args.dnop.split('+')
            for deleted_node_marker_type in deleted_node_marker_types:
                if deleted_node_marker_type == 'a2':
                    deleted_node_set = deleted_node_set.union(deleted_node_marker_a2_set)
                    deleted_node_plus_abs_d_set = deleted_node_plus_abs_d_set.union(
                        deleted_node_plus_abs_d_marker_a2_set)
                elif deleted_node_marker_type == 'zzc':
                    deleted_node_set = deleted_node_set.union(deleted_node_marker_c_set)
                    deleted_node_plus_abs_d_set = deleted_node_plus_abs_d_set.union(
                        deleted_node_plus_abs_d_marker_c_set)
                elif deleted_node_marker_type == 'toc':
                    deleted_node_set = deleted_node_set.union(deleted_node_marker_c2_set)
                    deleted_node_plus_abs_d_set = deleted_node_plus_abs_d_set.union(
                        deleted_node_plus_abs_d_marker_c2_set)
                elif deleted_node_marker_type == 'cov':
                    deleted_node_set = deleted_node_set.union(deleted_node_marker_cov_set)
                    deleted_node_plus_abs_d_set = deleted_node_plus_abs_d_set.union(
                        deleted_node_marker_cov_set)
                elif deleted_node_marker_type == 'd':
                    deleted_node_set = deleted_node_set.union(deleted_node_marker_d_set)
                    deleted_node_plus_abs_d_set = deleted_node_plus_abs_d_set.union(
                        deleted_node_plus_abs_d_marker_d_set)
                elif deleted_node_marker_type == 'b':
                    deleted_node_set = deleted_node_set.union(deleted_node_marker_a2_set)
                    deleted_node_plus_abs_d_set = deleted_node_plus_abs_d_set.union(
                        deleted_node_plus_abs_d_marker_a2_set)
                elif deleted_node_marker_type == 'jsjs':
                    deleted_node_set = deleted_node_set.union(deleted_node_marker_jsjs_set)
                    deleted_node_plus_abs_d_set = deleted_node_plus_abs_d_set.union(
                        deleted_node_marker_jsjs_set)

            deleted_node_set = list(deleted_node_set)
            deleted_node_plus_abs_d_set = list(deleted_node_plus_abs_d_set)

            # deleted_node_set = list(deleted_node_set.union(
            #     set(np.load(t + '/'+args.arch+'/deleted_node_npy_dir/deleted_node_js_marker_%s_%s_set_%s.npy' % (
            #         args.dnop, 'sf', multi_class_item)))))
            # deleted_node_plus_abs_d_set = list(deleted_node_plus_abs_d_set.union(
            #     set(np.load(t + '/'+args.arch+'/deleted_node_npy_dir/deleted_node_plus_abs_d_marker_%s_%s_set_%s.npy' % (
            #         args.dnop, 'sf', multi_class_item)))))

            # deleted_node_set_all.extend(deleted_node_plus_abs_d_set)

            np.save(
                t + '/' + args.arch + '/deleted_node_npy_dir/deleted_node_%sd_marker_%s_%s_set_%s.npy' % (
                    args.mt, args.dnop, ts_operation, multi_class_item), np.array(deleted_node_set))

            if args.mt == 'js':
                np.save(
                    t + '/' + args.arch + '/deleted_node_npy_dir/deleted_node_plus_abs_d_marker_%s_%s_set_%s.npy' % (
                        args.dnop, ts_operation, multi_class_item), np.array(deleted_node_plus_abs_d_set))

            print('==========')

            print('===%s===' % multi_class_item)

            print('=====%d (deleted_node_set len)=====' % len(deleted_node_set))
            print('=====%d (deleted_node_plus_abs_d_set len)=====' % len(deleted_node_plus_abs_d_set))

            class_dic = imagenet_class_index_dic()

            o_acc_num_list = []
            o_total_num_list = []

            f_acc_num_list = []
            f_total_num_list = []

            oo_acc_num_list = []
            oo_total_num_list = []

            of_acc_num_list = []
            of_total_num_list = []

            ofn = 0
            ofp = 0
            otp = 0
            otn = 0

            fn = 0
            fp = 0
            tp = 0
            tn = 0

            other_final_output_list = []

            for other_class in os.listdir(origin_image_dir):

                if other_class == multi_class_item or other_class == 'n00000000':
                    continue

                other_image_dir_class_dir = single_val_dir + '/' + other_class

                # print('===%s===' % other_class[0])

                origin_acc, o_acc_item, origin_top_k_acc, o_top_k_item_num, o_item_num, final_acc, f_acc_item, final_top_k_acc, f_top_k_item_num, f_item_num = get_deleted_node_accuracy(
                    other_image_dir_class_dir, class_dic[multi_class_item], deleted_node_set, arch, args.l,
                    model,
                    xrate)

                # other_final_output_list.append(get_deleted_node_output(
                #     other_image_dir_class_dir, class_dic[multi_class_item], deleted_node_set, arch, args.l,
                #     model, xrate).detach().cpu().numpy())

                # print('=== ori: %.2f-%.2f ===' % (origin_acc, origin_top_k_acc))
                # print('=== %sd: %.2f-%.2f ===' % (args.mt, final_acc, final_top_k_acc))
                oo_acc_num_list.append(o_acc_item)
                oo_total_num_list.append(o_item_num)
                of_acc_num_list.append(f_acc_item)
                of_total_num_list.append(f_item_num)

            # ofp = np.sum(oo_acc_num_list)
            # otn = np.sum(oo_total_num_list) - ofp
            # fp = np.sum(of_acc_num_list)
            # tn = np.sum(of_total_num_list) - fp

            # print(oo_acc_num_list)
            # print(oo_total_num_list)
            # print(of_total_num_list)

            print('=== ori: %.2f ===' % (np.sum(oo_acc_num_list) / np.sum(oo_total_num_list)))
            print('=== %sd: %.2f ===' % (args.mt, np.sum(of_acc_num_list) / np.sum(of_total_num_list)))

            origin_image_dir_class_dir = single_val_dir + '/' + multi_class_item
            origin_acc, o_acc_item, origin_top_k_acc, o_top_k_item_num, o_item_num, final_acc, f_acc_item, final_top_k_acc, f_top_k_item_num, f_item_num = get_deleted_node_accuracy(
                origin_image_dir_class_dir, class_dic[multi_class_item], deleted_node_set, arch, args.l,
                model,
                xrate)

            print('=== ori: %.2f-%.2f ===' % (origin_acc, origin_top_k_acc))
            print('=== %sd: %.2f-%.2f ===' % (args.mt, final_acc, final_top_k_acc))

            # o_acc_num_list.append(o_acc_item)
            # o_total_num_list.append(o_item_num)
            # f_acc_num_list.append(f_acc_item)
            # f_total_num_list.append(f_item_num)
            #
            # # otp = np.sum(o_acc_num_list)
            # # ofn = np.sum(o_total_num_list) - otp
            # # tp = np.sum(f_acc_num_list)
            # # fn = np.sum(f_total_num_list) - tp
            # #
            # # oacc = (otp + otn) / (otp + ofp + ofn + otn)
            # # otpr = otp / (otp + ofn)
            # # ofpr = ofp / (ofp + otn)
            # # otnr = otn / (ofp + otn)
            # # ober = 1 / 2 * (ofpr + ofn / (ofn + otp))
            # #
            # # acc = (tp + tn) / (tp + fp + fn + tn)
            # # tpr = tp / (tp + fn)
            # # fpr = fp / (fp + tn)
            # # tnr = tn / (fp + tn)
            # # ber = 1 / 2 * (fpr + fn / (fn + tp))
            # #
            # # print(oacc, otpr, ofpr, otnr, ober)
            # # print(acc, tpr, fpr, tnr, ber)
            #
            # if args.mt == 'js':
            #     plus_abs_d_origin_acc, plus_abs_d_o_acc_item, plus_abs_d_origin_top_k_acc, plus_abs_d_o_top_k_item_num, plus_abs_d_o_item_num, plus_abs_d_final_acc, plus_abs_d_f_acc_item, plus_abs_d_final_top_k_acc, plus_abs_d_f_top_k_item_num, plus_abs_d_f_item_num = get_deleted_node_accuracy(
            #         origin_image_dir_class_dir, class_dic[multi_class_item], deleted_node_plus_abs_d_set, arch, args.l,
            #         model,
            #         xrate)
            #     print('=== pad: %.2f-%.2f ===' % (plus_abs_d_final_acc, plus_abs_d_final_top_k_acc))

            print('================')

            file = r'acc_%s_%s_%s_%d_%.2f.txt' % (op, arch, args.tdir, args.l, xrate)
            with open(file, 'a+') as f:
                f.write('%s %d %d %d %.4f %.4f %.4f %.4f %s %s %s\n' % (
                    multi_class_item, args.param6, args.param7, len(deleted_node_set),
                    origin_acc, origin_top_k_acc, final_acc, final_top_k_acc, args.dnop, args.mt, args.tsop))

            final_output = get_deleted_node_output(
                origin_image_dir_class_dir, class_dic[multi_class_item], deleted_node_set, arch, args.l,
                model, xrate).detach().cpu().numpy()

            class_index_int = int(class_dic[multi_class_item])
            y = [1 for i in range(500)]
            y.extend([2 for i in range(50)])
            y = np.array(y)
            scores = []
            for ofo in other_final_output_list:
                for sample in ofo:
                    scores.append(sample[class_index_int])
            for sample in final_output:
                scores.append(sample[class_index_int])

            scores = np.array(scores)

            # print(y.shape)
            # print(scores.shape)

            fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
            # print(fpr, '\n', tpr, '\n', thresholds)
            print('auc: ', metrics.auc(fpr, tpr))

            print(np.sum(np.array(o_acc_num_list)) / np.sum(np.array(o_total_num_list)))
            print(np.sum(np.array(f_acc_num_list)) / np.sum(np.array(f_total_num_list)))

        # print(len(deleted_node_set_all))
        #
        # class_dic = imagenet_class_index_dic()
        #
        # o_acc_num_list = []
        # o_total_num_list = []
        #
        # f_acc_num_list = []
        # f_total_num_list = []
        #
        # counter_index = Counter(deleted_node_set_all)
        # prepare_nodes_set = counter_index.most_common()
        #
        # deleted_node_set_all_final = []
        #
        # for item in prepare_nodes_set:
        #     # print(item[1])
        #     if item[1] > 20:
        #         deleted_node_set_all_final.append(item[0])
        #
        # print('=== deleted_node_set_all_final len is %s ====' % len(deleted_node_set_all_final))
        #
        # for origin_image_dir_class in random.sample(os.listdir(single_val_dir), 10):
        #     if args.ec != 'none' and args.ec != origin_image_dir_class:
        #         continue
        #
        #     origin_image_dir_class_dir = single_val_dir + '/' + origin_image_dir_class
        #     print(origin_image_dir_class_dir)
        #     origin_acc, o_acc_item, origin_top_k_acc, o_top_k_item_num, o_item_num, final_acc, f_acc_item, final_top_k_acc, f_top_k_item_num, f_item_num = get_deleted_node_accuracy(
        #         origin_image_dir_class_dir, class_dic[origin_image_dir_class], deleted_node_set_all_final, arch, args.l,
        #         model,
        #         xrate)
        #
        #     print('=== ori: %.2f-%.2f ===' % (origin_acc, origin_top_k_acc))
        #     print('=== jsd: %.2f-%.2f ===' % (final_acc, final_top_k_acc))
        #
        #     o_acc_num_list.append(o_acc_item)
        #     o_total_num_list.append(o_item_num)
        #     f_acc_num_list.append(f_acc_item)
        #     f_total_num_list.append(f_item_num)
        #
        # print(np.sum(np.array(o_acc_num_list)) / np.sum(np.array(o_total_num_list)))
        # print(np.sum(np.array(f_acc_num_list)) / np.sum(np.array(f_total_num_list)))

    if args.op == 'test_accuracy':

        # random_class = ['n02917067', 'n02802426', 'n02483362', 'n02487347', 'n02494079', 'n02486410', 'n13054560',
        #                 'n12998815', 'n02835271', 'n03792782', 'n04482393']

        # random_class = ['n01728572', 'n01728920', 'n01729322', 'n01729977', 'n01734418', 'n01735189', 'n01737021',
        #                 'n01739381', 'n01740131', 'n01742172', 'n01744401', 'n01748264', 'n01749939', 'n01751748',
        #                 'n01753488', 'n01755581', 'n01756291', 'n02276258', 'n02277742', 'n02279972', 'n02280649',
        #                 'n02281406', 'n02281787', 'n02123045', 'n02123159', 'n02123394', 'n02123597', 'n02124075',
        #                 'n02125311', 'n02127052', 'n02128385', 'n02128757', 'n02128925', 'n02085620', 'n02085782',
        #                 'n02085936', 'n02086079', 'n02086240', 'n02086646', 'n02086910', 'n02087046', 'n02087394',
        #                 'n02088094']

        # random_class = ['n03982430', 'n03876231', 'n03874599', 'n03775546', 'n03527444', 'n02799071', 'n02877765',
        #                 'n03937543', 'n04447861', 'n12985857', 'n12144580']

        random_class = os.listdir(single_val_dir)

        # random_class = random.sample(os.listdir(single_val_dir), 10)

        model = get_model()
        class_dic = imagenet_class_index_dic()

        all_acc_item = 0
        all_val_item = 0

        for multi_class_item in random_class:
            img_dir = single_val_dir + '/' + multi_class_item

            model.eval()

            class_index = class_dic[multi_class_item]
            # print(class_index)

            img_tensor = trans_tensor_from_image(img_dir, arch)
            img_tensor_output = model(img_tensor)

            # print(
            #     [np.argmax(img_tensor_output[i].cpu().detach().numpy()) for i in range(np.shape(img_tensor_output)[0])])

            # print()
            # img_tensor_mid1 = img_tensor
            # img_tensor_mid2 = img_tensor
            # for i in range(1, 9):
            #     img_tensor_mid1 = cal_middle_output(model, arch, img_tensor_mid1, i)
            #     img_tensor_mid2 = cal_middle_output(model, arch, img_tensor_mid2, i)
            #     print('===layer_%d==' % i)
            #     if i <6:
            #         print(img_tensor_mid1[0][0][0])
            #         print(img_tensor_mid2[0][0][0])
            #     else:
            #         print(img_tensor_mid1[100][100])
            #         print(img_tensor_mid2[100][100])
            #
            # img_tensor_output = model(img_tensor)
            # img_tensor_output2 = model(img_tensor)
            #
            # print('===output===')
            # print(img_tensor_output.topk(5, 1, True, True))
            # print(img_tensor_output2.topk(5, 1, True, True))

            origin_acc, o_acc_item, o_item_num, o_topk_acc, o_topk_item = get_accuracy_from_output(img_tensor_output,
                                                                                                   class_index)

            all_acc_item += o_acc_item
            all_val_item += o_item_num

            print('%s: %.4f-%.4f' % (multi_class_item, origin_acc, o_topk_acc))

        print('%s: %.4f-%.4f' % ('all', all_acc_item, all_val_item))

    if args.op == 'print_model':
        # print model
        print(get_model())
        pass

    if args.op == 'cal_zzc_cov':

        ts_operation = args.tsop
        mode = 'zone'
        metric = 'std'
        time_type = args.tt
        phase = 'zero'
        point_times = 4
        if args.arch == 'resnet_50':
            point_times = 4

        for multi_class_item in os.listdir(origin_image_dir):

            print('==== %s ====' % (multi_class_item))

            if multi_class_item != args.ec and args.ec != 'none' or multi_class_item == 'n00000000':
                continue

            # if os.path.exists(
            #         t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/log_io%s_%s_file_list_cof_%s_%s_%s_%s.npy' % (
            #                 time_type, args.mt, multi_class_item, ts_operation, 'zone', 'zero')):
            #     continue

            deleted_node_set = None
            keep_node_set = None
            o_metric_file_list = None
            metric_file_list = None
            std_npy = None

            deleted_node_set = np.load(
                t + '/' + args.arch + '/deleted_node_npy_dir/deleted_node_%sd_marker_%s_%s_set_%s.npy' % (
                    args.mt, args.dnop, ts_operation, multi_class_item))

            keep_node_set = list(set(list(range(output_size))) - set(deleted_node_set))

            print(len(keep_node_set))

            if time_type == '1':

                all_o_metric_file_list = np.load(
                    t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/all_o_%s_file_list_%s_%s_%s_%s.npy' % (
                        args.mt, multi_class_item, ts_operation, 'zone', 'zero'))

                all_metric_file_list = np.load(
                    t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/all_%s_file_list_%s_%s_%s_%s.npy' % (
                        args.mt, multi_class_item, ts_operation, 'zone', 'zero'))

                std_npy = np.load(
                    t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/metric_wad_%s_%s_zone_%d_zero_list.npy' % (
                        metric, ts_operation, args.l))
            else:

                # keep_node_set = np.load(
                #     t + '/' + args.arch + '/pools_keep_node_npy_dir/keep_node_pools%s_%sd_marker_%s_set_%s.npy' % (
                #         str(int(time_type) - 1), args.mt, ts_operation, multi_class_item))

                all_o_metric_file_list = np.load(
                    t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/all_o_pools%s_%s_file_list_%s_%s_%s_%s.npy' % (
                        str(int(time_type) - 1), args.mt, multi_class_item, ts_operation, mode, phase))

                all_metric_file_list = np.load(
                    t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/all_i_pools%s_%s_file_list_%s_%s_%s_%s.npy' % (
                        str(int(time_type) - 1), args.mt, multi_class_item, ts_operation, mode, phase))

                std_npy = np.load(
                    t + '/' + args.arch + '/metric_list_npy_dir/%s' % multi_class_item + '/metric_i_pools%s_%sd_std_%s_%s_%d_zero_list.npy' % (
                        str(int(time_type) - 1), args.mt, ts_operation, mode, args.l))

            # metric_file_list[metric_file_list == 0] = 1
            io_er_list = []
            log_io_er_list = []

            for inner_index in range(all_o_metric_file_list.shape[0]):

                o_metric_file_list = all_o_metric_file_list[inner_index]
                metric_file_list = all_metric_file_list[inner_index]

                print(o_metric_file_list.shape)

                o_metric_file_list_del_index = []
                metric_file_list_del_index = []
                io_metric_file_list_del_index = []
                log_io_metric_file_list_del_index = []

                # metric_file_list[metric_file_list == 0] = 1

                for i in keep_node_set:
                    # if i == 5842 or i == 3482:
                    #     print(i)
                    #     print(o_metric_file_list[:, i])
                    #     print(metric_file_list[:, i])
                    #
                    # else:
                    #     continue

                    constant_plus = 1

                    o_metric_file_list_del_index.append((o_metric_file_list[:, i]))
                    metric_file_list_del_index.append((metric_file_list[:, i]))
                    io_metric_file_list_del_index.append(
                        (o_metric_file_list[:, i] + constant_plus) / (np.mean(metric_file_list[:, i] + constant_plus)))
                    log_io_metric_file_list_del_index.append(
                        np.log(
                            ((o_metric_file_list[:, i] + constant_plus) / (
                                np.mean(metric_file_list[:, i] + constant_plus)))) /
                        std_npy[
                            i])
                # io_metric_file_list_del.append(
                #     (o_metric_file_list[:, i] + 10e-7) / (metric_file_list[:, i] + 10e-7))
                # log_io_metric_file_list_del.append(
                #     np.log(((o_metric_file_list[:, i] + 10e-7) / (metric_file_list[:, i] + 10e-7))) / std_npy[
                #         i])
                # log_io_metric_file_list_del.append(
                #     np.log(((o_metric_file_list[:, i] + 1) / (metric_file_list[:, i] + 1))))

                print('np.array(io_metric_file_list_del): ', np.array(io_metric_file_list_del_index).shape)
                # io_er_list_inner = np.mean(io_metric_file_list_del, axis=1)
                # log_io_er_list_inner = np.mean(log_io_metric_file_list_del, axis=1)
                # io_er_list.append(io_er_list_inner)
                # log_io_er_list.append(log_io_er_list_inner)
                io_er_list.append(io_metric_file_list_del_index)
                log_io_er_list.append(log_io_metric_file_list_del_index)

            print('np.array(io_er_list).shape: ', np.array(io_er_list).shape)

            io_metric_file_list_del = np.mean(np.array(io_er_list), axis=0)
            log_io_metric_file_list_del = np.mean(np.array(log_io_er_list), axis=0)

            print(log_io_metric_file_list_del.shape)

            io_er_list = np.mean(np.array(io_metric_file_list_del), axis=1)
            log_io_er_list = np.mean(np.array(io_metric_file_list_del), axis=1)

            # print(metric_file_list[:,9034])
            # print(o_metric_file_list[:, 9034])

            # print(log_io_er_list[-1])

            # print(er_list)

            np.save(
                t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/io%s_%s_file_list_%s_%s_%s_%s.npy' % (
                    time_type, args.mt, multi_class_item, ts_operation, 'zone', 'zero'), io_metric_file_list_del)
            np.save(
                t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/log_io%s_%s_file_list_%s_%s_%s_%s.npy' % (
                    time_type, args.mt, multi_class_item, ts_operation, 'zone', 'zero'), log_io_metric_file_list_del)

            np.save(
                t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/io%s_%s_er_list_%s_%s_%s_%s.npy' % (
                    time_type, args.mt, multi_class_item, ts_operation, 'zone', 'zero'), io_er_list)

            np.save(
                t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/log_io%s_%s_er_list_%s_%s_%s_%s.npy' % (
                    time_type, args.mt, multi_class_item, ts_operation, 'zone', 'zero'), log_io_er_list)

            r_i_list_topk_value, r_i_list_topk_index = torch.from_numpy(io_er_list).topk(
                point_times * int(np.sqrt(len(keep_node_set))))

            r_i_list_topk_index = r_i_list_topk_index.numpy()
            keep_node_set = np.array(keep_node_set)

            mkdir(t + '/' + args.arch + '/best_node_pools_npy_dir')
            np.save(t + '/' + args.arch + '/best_node_pools_npy_dir/best_node_pools%s_%sd_marker_%s_%s_set_%s.npy' % (
                args.tt, args.mt, args.dnop, ts_operation, multi_class_item), keep_node_set[r_i_list_topk_index])

            if args.isall != 'all':
                log_io_metric_file_list_del = log_io_metric_file_list_del[r_i_list_topk_index]
                io_metric_file_list_del = io_metric_file_list_del[r_i_list_topk_index]

            log_metric_cov = np.cov(log_io_metric_file_list_del)
            metric_cov = np.cov(io_metric_file_list_del)
            log_metric_cof = np.corrcoef(log_io_metric_file_list_del)
            metric_cof = np.corrcoef(io_metric_file_list_del)

            print(metric_cov.shape)
            # print(keep_node_set[r_i_list_topk_index])

            np.save(
                t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/io%s_%s_file_list_cov_%s_%s_%s_%s.npy' % (
                    time_type, args.mt, multi_class_item, ts_operation, 'zone', 'zero'), metric_cov)
            np.save(
                t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/log_io%s_%s_file_list_cov_%s_%s_%s_%s.npy' % (
                    time_type, args.mt, multi_class_item, ts_operation, 'zone', 'zero'), log_metric_cov)
            np.save(
                t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/io%s_%s_file_list_cof_%s_%s_%s_%s.npy' % (
                    time_type, args.mt, multi_class_item, ts_operation, 'zone', 'zero'), metric_cof)
            np.save(
                t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/log_io%s_%s_file_list_cof_%s_%s_%s_%s.npy' % (
                    time_type, args.mt, multi_class_item, ts_operation, 'zone', 'zero'), log_metric_cof)

    if args.op == 'cal_zzc_cov2':

        ts_operation = args.tsop
        mode = 'zone'
        metric = 'std'
        time_type = args.tt
        phase = 'zero'
        point_times = 4
        if args.arch == 'resnet_50':
            point_times = 1

        params = []

        count = 0

        for multi_class_item in os.listdir(origin_image_dir):

            print('==== %s ====' % (multi_class_item))

            if multi_class_item != args.ec and args.ec != 'none' or multi_class_item == 'n00000000':
                continue

            count += 1

            # if os.path.exists(
            #         t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/log_io%s_%s_file_list_cof_%s_%s_%s_%s.npy' % (
            #                 time_type, args.mt, multi_class_item, ts_operation, 'zone', 'zero')):
            #     continue

            params.append((t, ts_operation, mode, metric, time_type, phase, point_times, multi_class_item, count))

        p = multiprocessing.Pool(15)
        p.map(do_cal_cov, params)
        p.close()
        p.join()

    if args.op == 'analysis_zzc_cov':

        ts_operation = args.tsop
        mode = 'zone'
        phase = 'zero'
        time_type = args.tt
        point_times = 4
        related_point_number = 7
        if args.arch == 'resnet_50':
            point_times = 1
            related_point_number = 5

        for multi_class_item in os.listdir(origin_image_dir):

            if multi_class_item != args.ec and args.ec != 'none' or multi_class_item == 'n00000000':
                continue

            io_metric_file_list = np.load(
                t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/io%s_%s_file_list_%s_%s_%s_%s.npy' % (
                    time_type, args.mt, multi_class_item, ts_operation, 'zone', 'zero'))
            log_io_metric_file_list = np.load(
                t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/log_io%s_%s_file_list_%s_%s_%s_%s.npy' % (
                    time_type, args.mt, multi_class_item, ts_operation, 'zone', 'zero'))

            # metric_cov = np.load(
            #     t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/io%s_%s_file_list_cov_%s_%s_%s_%s.npy' % (
            #         time_type, args.mt, multi_class_item, ts_operation, 'zone', 'zero'))
            # log_metric_cov = np.load(
            #     t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/log_io%s_%s_file_list_cov_%s_%s_%s_%s.npy' % (
            #         time_type, args.mt, multi_class_item, ts_operation, 'zone', 'zero'))
            # metric_cof = np.load(
            #     t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/io%s_%s_file_list_cof_%s_%s_%s_%s.npy' % (
            #         time_type, args.mt, multi_class_item, ts_operation, 'zone', 'zero'))
            log_metric_cof = np.load(
                t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/log_io%s_%s_file_list_cof_%s_%s_%s_%s.npy' % (
                    time_type, args.mt, multi_class_item, ts_operation, 'zone', 'zero'))

            er_list = np.load(
                t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/io%s_%s_er_list_%s_%s_%s_%s.npy' % (
                    time_type, args.mt, multi_class_item, ts_operation, 'zone', 'zero')).reshape(-1)

            log_er_list = np.load(
                t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/log_io%s_%s_er_list_%s_%s_%s_%s.npy' % (
                    time_type, args.mt, multi_class_item, ts_operation, 'zone', 'zero')).reshape(-1)

            print('=== %s ===' % (multi_class_item))

            deleted_node_set = np.load(
                t + '/' + args.arch + '/deleted_node_npy_dir/deleted_node_%sd_marker_%s_%s_set_%s.npy' % (
                    args.mt, args.dnop, ts_operation, multi_class_item))

            keep_node_set = list(set(list(range(output_size))) - set(deleted_node_set))
            keep_node_set = np.array(keep_node_set)

            # if time_type != '1':
            #     keep_node_set = np.load(
            #         t + '/' + args.arch + '/pools_keep_node_npy_dir/keep_node_pools%s_%sd_marker_%s_set_%s.npy' % (
            #             str(int(time_type) - 1), args.mt, ts_operation, multi_class_item))

            print('keep node set len: %d' % (len(keep_node_set)))

            r_i_list_topk_value, r_i_list_topk_index = torch.from_numpy(er_list).topk(
                point_times * int(np.sqrt(len(keep_node_set))))

            r_i_list_topk_index = r_i_list_topk_index.numpy()

            mkdir(t + '/' + args.arch + '/best_node_pools_npy_dir')
            np.save(t + '/' + args.arch + '/best_node_pools_npy_dir/best_node_pools%s_%sd_marker_%s_%s_set_%s.npy' % (
                args.tt, args.mt, args.dnop, ts_operation, multi_class_item), keep_node_set[r_i_list_topk_index])
            # mkdir(t + '/' + args.arch + '/keep_node_npy_dir')
            # np.save(
            #     t + '/' + args.arch + '/keep_node_npy_dir' + '/keep_node_file_list_%s_%s_%s_%s_%s.npy' % (
            #         args.mt, multi_class_item, ts_operation, 'zone', 'zero'), keep_node_set[r_i_list_topk_index])

            # metric = 'std'
            # std_npy = np.load(
            #     t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/metric_wad_%s_%s_time_5_one_list.npy' % (
            #         metric,ts_operation))

            keep_node_pools = []
            keep_node_pools_weight = []
            no_center_node_list = []

            best_node_index = []
            best_node_pools = []
            best_node_pools_weight = []

            det_0_count = 0

            # print(er_list[r_i_list_topk_index])
            # print(keep_node_set[r_i_list_topk_index])
            # for j in range(0):

            if not args.isall == 'all':
                keep_node_set = keep_node_set[r_i_list_topk_index]
                io_metric_file_list = io_metric_file_list[r_i_list_topk_index]
                log_er_list = r_i_list_topk_value
                log_io_metric_file_list = log_io_metric_file_list[r_i_list_topk_index]

            for j in range(len(keep_node_set)):

                # if keep_node_set[j] != 11:
                #     continue

                # if j not in r_i_list_topk_index:
                #     continue

                indexes = []

                if args.isall == 'all':

                    # htopk_v, htopk_index = torch.from_numpy(log_metric_cof[j, r_i_list_topk_index]).topk(10)
                    ltopk_v, htopk_index = torch.from_numpy(log_metric_cof[j, r_i_list_topk_index]).topk(
                        related_point_number,
                        largest=False)
                    ltopk_v, ltopk_index = torch.from_numpy(np.abs(log_metric_cof[j, r_i_list_topk_index])).topk(
                        related_point_number,
                        largest=False)

                    for index in r_i_list_topk_index[ltopk_index.numpy()]:
                        # print(er_list[index])
                        if not test_array_all_the_same(io_metric_file_list[index, :]):
                            # if er_list[index] > er_list[j] and not test_array_all_the_same(io_metric_file_list[index, :]):
                            indexes.append(index)
                    for index in r_i_list_topk_index[htopk_index.numpy()]:
                        if not test_array_all_the_same(io_metric_file_list[index, :]):
                            # if er_list[index] > er_list[j] and not test_array_all_the_same(io_metric_file_list[index, :]):
                            indexes.append(index)
                else:

                    # print(log_metric_cof.shape)

                    ltopk_v, htopk_index = torch.from_numpy(log_metric_cof[j, :]).topk(related_point_number,
                                                                                       largest=False)
                    ltopk_v, ltopk_index = torch.from_numpy(np.abs(log_metric_cof[j, :])).topk(related_point_number,
                                                                                               largest=False)

                    # print(io_metric_file_list.shape)

                    for index in ltopk_index.numpy():
                        # print(er_list[index])
                        if not test_array_all_the_same(io_metric_file_list[index, :]):
                            # if er_list[index] > er_list[j] and not test_array_all_the_same(io_metric_file_list[index, :]):
                            indexes.append(index)
                    for index in htopk_index.numpy():
                        if not test_array_all_the_same(io_metric_file_list[index, :]):
                            # if er_list[index] > er_list[j] and not test_array_all_the_same(io_metric_file_list[index, :]):
                            indexes.append(index)
                    # print('log_er:', r_i_list_topk_value[indexes])
                # print(io_metric_file_list.shape)
                # print(er_list[j])

                # print()
                if not test_array_all_the_same(io_metric_file_list[j, :]):
                    indexes.append(j)
                else:
                    print('no_center')
                    no_center_node_list.append(j)
                # print('index:', indexes)
                # print('indexs: ', keep_node_set[indexes])
                indexes = list(set(indexes))
                # indexes.append(j)

                indexes_big = []

                indexes = np.array(indexes)

                # if indexes.shape[0] <= 1:
                #     if indexes.shape[0] == 0:
                #         keep_node_pools.append(np.array([j]))
                #         keep_node_pools_weight.append(np.array([1]))
                #     else:
                #         keep_node_pools.append(keep_node_set[indexes])
                #         keep_node_pools_weight.append(np.array([1]))
                #     det_0_count += 1
                #     print('lt 1')
                #     continue

                market_er_list = log_er_list[indexes].reshape(-1, 1)
                if args.isall == 'best':
                    market_er_list = market_er_list.numpy()
                # print('max_marker: ',np.max(market_er_list))

                ee = np.ones(market_er_list.shape).reshape(-1, 1)

                pools_wa_list = log_io_metric_file_list[indexes]
                pools_wa_cov = np.cov(pools_wa_list)
                # print(pools_wa_list)

                wa_cov_inv = np.linalg.inv(pools_wa_cov)

                # print(market_er_list.shape)

                a = np.dot(np.dot(market_er_list.transpose(), wa_cov_inv), market_er_list)[0][0] + 0.0000001
                b = np.dot(np.dot(market_er_list.transpose(), wa_cov_inv), ee)[0][0] + 0.0000001
                c = np.dot(np.dot(ee.transpose(), wa_cov_inv), ee)[0][0] + 0.0000001
                b2 = np.dot(np.dot(ee.transpose(), wa_cov_inv), market_er_list)[0][0] + 0.0000001

                # print('a: ', a)
                print('b: ', b, ' = ', b2)
                # print('c: ', c)
                # print('b2: ', b2)

                # miu_p = (a * a) / (b * b2)
                miu_p = np.sqrt(a * a / (b * b2))

                d = a * c - b * b2 + 0.0000001
                a_inv = np.ones((2, 2))
                a_inv[0][0] = c
                a_inv[0][1] = -b
                a_inv[1][0] = -b2
                a_inv[1][1] = a
                # print(a_inv.shape)
                a_inv *= (1 / d)

                m1 = np.dot(wa_cov_inv, np.concatenate((market_er_list, ee), axis=1))
                m2 = np.dot(m1, a_inv)
                m3 = np.dot(m2, np.concatenate(([miu_p], [1]), axis=0))

                weight = m3
                # weight[weight < 0] = 0
                # print('indexes: ', keep_node_set[indexes])
                # print(weight)
                # if keep_node_set[j] == 288:
                #     sys.exit(0)
                # print('miu_p_w: ', np.dot(weight.transpose(), market_er_list))
                # weight = weight / np.sum(weight)

                # weight = tensor_array_normalization(m3)
                # print(np.sum(weight))
                # print('j: ', keep_node_set[j])
                # print('er:', er_list[indexes])

                # print('weight: ', weight)
                print('miu_p: ', miu_p)
                # if keep_node_set[j] == 8073:
                #     break
                # weight = weight / np.sum(weight)
                # print(np.sum(weight))
                # print(tensor_array_normalization(m3))

                # print(tensor_array_normalization(m3)/np.sum(m3))
                # print(np.sum(weight))

                keep_node_pools.append((keep_node_set[j], keep_node_set[indexes]))
                keep_node_pools_weight.append(weight)

                if j in r_i_list_topk_index:
                    best_node_index.append(keep_node_set[j])
                    best_node_pools.append((keep_node_set[j], keep_node_set[indexes]))
                    best_node_pools_weight.append(weight)

                # print('shape indexes: ', indexes.shape)
                # print('shape weights: ', weight.shape)
                # if j==2:
                #     print('shape indexes: ',keep_node_pools[j].shape)
                #     print('shape weights: ',weight.shape)
                #     print(j)
                # break

                # print('======================')
                # break

            # print('det_0_count: ', det_0_count)
            # print(keep_node_set)
            print('keep_node_pools_weight: ', len(keep_node_pools_weight))

            np.save(
                t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/keep_node_pools%s_wad_%s_%s_%s_%s_list.npy' % (
                    time_type, ts_operation, mode, args.l, phase),
                np.array(keep_node_pools))

            np.save(
                t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/no_center_node_pools%s_wad_%s_%s_%s_%s_list.npy' % (
                    time_type, ts_operation, mode, args.l, phase),
                np.array(no_center_node_list))

            np.save(
                t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/keep_node_pools%s_weight_wad_%s_%s_%s_%s_list.npy' % (
                    time_type, ts_operation, mode, args.l, phase),
                np.array(keep_node_pools_weight))

            np.save(
                t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/best_node_index%s_wad_%s_%s_%s_%s_list.npy' % (
                    time_type, ts_operation, mode, args.l, phase),
                np.array(best_node_index))

            np.save(
                t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/best_node_pools%s_wad_%s_%s_%s_%s_list.npy' % (
                    time_type, ts_operation, mode, args.l, phase),
                np.array(best_node_pools))

            np.save(
                t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/best_node_pools%s_weight_wad_%s_%s_%s_%s_list.npy' % (
                    time_type, ts_operation, mode, args.l, phase),
                np.array(best_node_pools_weight))

            # oo_acc_num_list = []
            # oo_total_num_list = []
            #
            # of_acc_num_list = []
            # of_total_num_list = []
            #
            # for other_class in os.listdir(t + '/' + args.arch + '/metric_list_npy_dir'):
            #
            #     if other_class != 'n12985857':
            #         continue
            #
            #     other_image_dir_class_dir = single_val_dir + '/' + other_class
            #
            #     # ============
            #     # auc
            #     # ============
            #
            #     origin_acc, o_acc_item, origin_top_k_acc, o_top_k_item_num, o_item_num, final_acc, f_acc_item, final_top_k_acc, f_top_k_item_num, f_item_num = get_min_pool_accuracy(
            #         other_image_dir_class_dir, class_dic[multi_class_item], deleted_node_set, keep_node_pools,
            #         keep_node_pools_weight, arch,
            #         args.l,
            #         model,
            #         xrate, return_output=False)
            #
            #     oo_acc_num_list.append(o_acc_item)
            #     oo_total_num_list.append(o_item_num)
            #     of_acc_num_list.append(f_acc_item)
            #     of_total_num_list.append(f_item_num)
            #
            # ofp = np.sum(oo_acc_num_list)
            # otn = np.sum(oo_total_num_list) - ofp
            # fp = np.sum(of_acc_num_list)
            # tn = np.sum(of_total_num_list) - fp
            #
            # print('=== ori: %.2f ===' % (np.sum(oo_acc_num_list) / np.sum(oo_total_num_list)))
            # print('=== %sd: %.2f ===' % (args.mt, np.sum(of_acc_num_list) / np.sum(of_total_num_list)))
            #
            # origin_image_dir_class_dir = single_val_dir + '/' + multi_class_item
            #
            # origin_acc, o_acc_item, origin_top_k_acc, o_top_k_item_num, o_item_num, final_acc, f_acc_item, final_top_k_acc, f_top_k_item_num, f_item_num = get_min_pool_accuracy(
            #     origin_image_dir_class_dir, class_dic[multi_class_item], deleted_node_set, keep_node_pools,
            #     keep_node_pools_weight, arch,
            #     args.l,
            #     model,
            #     xrate, return_output=False)
            #
            # print('=== ori: %.2f-%.2f ===' % (origin_acc, origin_top_k_acc))
            # print('=== %sd: %.2f-%.2f ===' % (args.mt, final_acc, final_top_k_acc))

            # for i in keep_node_set:
            #     if metric_cov[j][i] > 0:
            #         j_pos_list.append(i)
            #
            # print(len(j_pos_list))

            # break

    if args.op == 'analysis_zzc_cov2':

        ts_operation = args.tsop
        mode = 'zone'
        phase = 'zero'
        time_type = args.tt
        point_times = 4
        related_point_number = 5
        if args.arch == 'resnet_50':
            point_times = 1
            related_point_number = 5

        params = []
        count = 0
        for multi_class_item in os.listdir(origin_image_dir):

            if multi_class_item != args.ec and args.ec != 'none' or multi_class_item == 'n00000000':
                continue

            count += 1

            # if os.path.exists(
            #         t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/keep_node_pools%s_weight_wad_%s_%s_5_%s_list.npy' % (
            #                 time_type, ts_operation, mode, phase)):
            #     continue

            params.append(
                (t, ts_operation, mode, phase, time_type, point_times, related_point_number, multi_class_item, count))

        p = multiprocessing.Pool(1)
        p.map(do_analysis_cov, params)
        p.close()
        p.join()

    if args.op == 'cal_io_big_node':

        ts_operation = args.tsop
        mode = 'zone'
        class_dic = imagenet_class_index_dic()
        model = get_model()
        time_type = args.tt
        phase = 'zero'
        img_dir = t + '/transform_images_%s_noise' % ts_operation

        mkdir(t + '/' + args.arch + '/pools_keep_node_npy_dir/')

        for multi_class_item in os.listdir(origin_image_dir):

            if multi_class_item != args.ec and args.ec != 'none' or multi_class_item == 'n00000000':
                continue

            print('class: ', multi_class_item)

            o_wa_npy = np.load(
                t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/%s_pools%s_%s_file_list_%s_%s_%s_%s.npy' % (
                    'o', time_type, args.mt, multi_class_item, ts_operation, mode, phase))
            i_wa_npy = np.load(
                t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/%s_pools%s_%s_file_list_%s_%s_%s_%s.npy' % (
                    'i', time_type, args.mt, multi_class_item, ts_operation, mode, phase))

            deleted_node_set = np.load(
                t + '/' + args.arch + '/deleted_node_npy_dir/deleted_node_%sd_marker_%s_%s_set_%s.npy' % (
                    args.mt, args.dnop, ts_operation, multi_class_item))
            keep_node_set = list(set(list(range(output_size))) - set(deleted_node_set))
            # if time_type != '1':
            #     keep_node_set = np.load(
            #         t + '/' + args.arch + '/pools_keep_node_npy_dir/keep_node_pools%s_%sd_marker_%s_set_%s.npy' % (
            #             str(int(time_type) - 1), args.mt, ts_operation, multi_class_item))

            print('keep: ', len(keep_node_set))

            # print(i_wa_npy.shape)
            # print(o_wa_npy.shape)
            new_keep_node_set = []

            for index in keep_node_set:
                # print(index)
                if np.mean(i_wa_npy[:, index]) < np.mean(o_wa_npy[:, index]):
                    new_keep_node_set.append(index)

            new_keep_node_set = np.array(new_keep_node_set)
            print('obi: ', new_keep_node_set.shape)

            np.save(t + '/' + args.arch + '/pools_keep_node_npy_dir/keep_node_pools%s_%sd_marker_%s_set_%s.npy' % (
                time_type, args.mt, ts_operation, multi_class_item), new_keep_node_set)

            # iter_list = []
            # range_list_inner = os.listdir(img_dir + '/' + multi_class_item)
            # if args.cp == 'all':
            #     iter_list = range_list_inner[0:len(range_list_inner)]
            #
            # for transform_index_cp in iter_list:
            #
            #     if transform_index_cp == range_list_inner[compare_index]:
            #         print('cmp: ', range_list_inner[compare_index])
            #         continue
            #
            #     print(transform_index_cp)
            #     new_keep_node_set = []
            #     i_wa_npy = np.load(
            #         t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item + '/wa_class_npy/%s%s_file_list_%s-%s_%s_%s_%s.npy' % (
            #             args.mt, time_type, multi_class_item, transform_index_cp, ts_operation, mode, phase))
            #     # print(i_wa_npy.shape)
            #     for index in keep_node_set:
            #         # print(index)
            #         if np.mean(i_wa_npy[:, index]) < np.mean(o_wa_npy[:, index]):
            #             new_keep_node_set.append(index)
            #     new_keep_node_set = np.array(new_keep_node_set)
            #     print(new_keep_node_set.shape)

    if args.op == 'get_o_validate_mid_res':

        ts_operation = args.tsop
        img_dir = t + '/transform_images_%s_noise' % ts_operation
        mode = 'zone'
        save_dir_pre = 'validate' + '_layer_npy'
        i_o_index = args.ioi
        time_type = args.tt
        phase = 'zero'
        model = get_model().cuda()

        params = []
        for multi_class_item in os.listdir(origin_image_dir):
            validate_class_dir = single_val_dir + '/' + multi_class_item
            get_layer_cuda(validate_class_dir, args.arch, args.l, save_dir_pre, model)

    if args.op == 'get_o_train_mid_res':

        ts_operation = args.tsop
        img_dir = t + '/transform_images_%s_noise' % ts_operation
        mode = 'zone'
        save_dir_pre = args.tdir + '_o_train_layer_npy'
        i_o_index = args.ioi
        time_type = args.tt
        phase = 'zero'
        vector_dir = t + '/vector_dir/' + time_type + '/all_train_img'

        model = get_model().cuda()

        for multi_class_item in os.listdir(origin_image_dir):
            train_class_dir = origin_image_dir + '/' + multi_class_item
            get_layer_cuda(train_class_dir, args.arch, args.l, save_dir_pre, model)

    if args.op == 'partition_o_train_vector':

        # ==================
        # cal the js for zone in zero phase
        # ==================
        # the operation used to transform images, eg. gaussian blur: gb, all channel change: ac
        ts_operation = args.tsop
        img_dir = t + '/transform_images_%s_noise' % ts_operation
        mode = 'zone'
        save_dir_pre = args.tdir + '_o_train_layer_npy'
        i_o_index = args.ioi
        time_type = args.tt
        phase = 'zero'
        vector_dir = t + '/vector_dir/' + time_type + '/o_train'

        # ==================
        # cal the js for zone in zero phase
        # ==================
        if args.param == 'time':
            img_dir = t + '/transform_images_t_%s_noise' % ts_operation
            mode = 'time'

        multi_classes = os.listdir(origin_image_dir)

        for multi_class_item in multi_classes:

            print('----' + multi_class_item + '----')

            class_vector_dir = vector_dir + '/' + multi_class_item
            mkdir(class_vector_dir)

            if multi_class_item != args.ec and args.ec != 'none' or multi_class_item == 'n00000000':
                continue

            t0_name = save_dir_pre + '/%s_imagenet_2012-%s-origin_images-%s_mid_res/%s_imagenet_2012-%s-origin_images-%s_layer%s.npy' % (
                args.arch, args.tdir, multi_class_item, args.arch, args.tdir, multi_class_item, args.l)

            if args.arch == 'resnet50':
                t0_name = save_dir_pre + '/resnet50_imagenet_2012-%s-origin_images-%s_mid_res/resnet50_imagenet_2012-%s-origin_images-%s_layer%d.npy' % (
                    args.tdir, multi_class_item, args.tdir, multi_class_item, args.l)

            # t0_name = save_dir_pre + '/alexnet_imagenet_2012-single_train-%s_mid_res/alexnet_imagenet_2012-single_train-%s_layer5.npy' % (
            #     multi_class_item, multi_class_item)
            tensor_array = np.load(t0_name)

            tensor_array = tensor_array.reshape(tensor_array.shape[0], -1)

            for img_index in range(tensor_array.shape[0]):
                np.save(
                    class_vector_dir + '/%s-%s-%s.npy' % (multi_class_item, img_index, time_type),
                    tensor_array[img_index])

    if args.op == 'partition_o_validate_vector':

        # ==================
        # cal the js for zone in zero phase
        # ==================
        # the operation used to transform images, eg. gaussian blur: gb, all channel change: ac
        ts_operation = args.tsop
        img_dir = t + '/transform_images_%s_noise' % ts_operation
        mode = 'zone'
        save_dir_pre = 'validate' + '_layer_npy'
        i_o_index = args.ioi
        time_type = args.tt
        phase = 'zero'
        vector_dir = t + '/vector_dir/' + time_type + '/o_val'

        # ==================
        # cal the js for zone in zero phase
        # ==================

        multi_classes = os.listdir(origin_image_dir)

        for multi_class_item in multi_classes:

            print('----' + multi_class_item + '----')

            class_vector_dir = vector_dir + '/' + multi_class_item
            mkdir(class_vector_dir)

            if multi_class_item != args.ec and args.ec != 'none' or multi_class_item == 'n00000000':
                continue

            t0_name = save_dir_pre + '/alexnet_imagenet_2012-single_val-%s_mid_res/alexnet_imagenet_2012-single_val-%s_layer5.npy' % (
                multi_class_item, multi_class_item)

            if args.arch == 'resnet50':
                t0_name = save_dir_pre + '/resnet50_imagenet_2012-single_val-%s_mid_res/resnet50_imagenet_2012-single_val-%s_layer%d.npy' % (
                    multi_class_item, multi_class_item, args.l)

            tensor_array = np.load(t0_name)

            tensor_array = tensor_array.reshape(tensor_array.shape[0], -1)

            for img_index in range(tensor_array.shape[0]):
                np.save(
                    class_vector_dir + '/%s-%s-%s.npy' % (multi_class_item, img_index, time_type),
                    tensor_array[img_index])

    if args.op == 'partition_train_comb_index_vector':

        # ==================
        # cal the js for zone in zero phase
        # ==================
        # the operation used to transform images, eg. gaussian blur: gb, all channel change: ac
        ts_operation = args.tsop
        img_dir = t + '/transform_images_%s_noise' % ts_operation
        mode = 'zone'
        i_o_index = args.ioi
        time_type = args.tt
        phase = 'zero'
        save_dir_pre = args.tdir + '_o_train_layer_npy'
        vector_dir = t + '/vector_dir/' + time_type + '/train'
        comb_vector_dir = t + '/vector_dir/' + time_type + '/comb_train'

        if args.tv == 'val':
            save_dir_pre = 'validate' + '_layer_npy'
            vector_dir = t + '/vector_dir/' + time_type + '/val'
            comb_vector_dir = t + '/vector_dir/' + time_type + '/comb_val'

        # ==================
        # cal the js for zone in zero phase
        # ==================
        if args.param == 'time':
            img_dir = t + '/transform_images_t_%s_noise' % ts_operation
            mode = 'time'

        multi_classes = os.listdir(img_dir)

        keep_node_pools_dic = {}
        keep_node_pools_weight_dic = {}
        keep_node_set_dic = {}

        for time_index in range(1, int(time_type) + 1):

            for multi_class_item_cp in multi_classes:
                keep_node_pools = np.load(
                    t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item_cp + '/keep_node_pools%s_wad_%s_%s_%s_%s_list.npy' % (
                        time_index, ts_operation, mode, args.l, phase), allow_pickle=True)

                keep_node_pools_weight = np.load(
                    t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item_cp + '/keep_node_pools%s_weight_wad_%s_%s_%s_%s_list.npy' % (
                        time_index, ts_operation, mode, args.l, phase), allow_pickle=True)
                deleted_node_set = np.load(
                    t + '/' + args.arch + '/deleted_node_npy_dir/deleted_node_%sd_marker_%s_%s_set_%s.npy' % (
                        args.mt, args.dnop, ts_operation, multi_class_item_cp))

                keep_node_set = list(set(list(range(output_size))) - set(deleted_node_set))

                if multi_class_item_cp not in keep_node_pools_dic.keys():
                    keep_node_pools_dic[multi_class_item_cp] = []
                    keep_node_pools_weight_dic[multi_class_item_cp] = []
                keep_node_pools_dic[multi_class_item_cp].append(keep_node_pools)
                keep_node_pools_weight_dic[multi_class_item_cp].append(keep_node_pools_weight)
                keep_node_set_dic[multi_class_item_cp] = keep_node_set

        count = 0

        for multi_class_item in multi_classes:

            params = []

            print('----' + multi_class_item + '----')

            class_vector_dir = vector_dir + '/' + multi_class_item
            mkdir(class_vector_dir)

            class_comb_vector_dir = comb_vector_dir + '/' + multi_class_item
            mkdir(class_comb_vector_dir)

            if multi_class_item != args.ec and args.ec != 'none' or multi_class_item == 'n00000000':
                continue

            # if args.tv == 'train':
            #     t0_name = save_dir_pre + '/%s_imagenet_2012-%s-origin_images-%s_mid_res/%s_imagenet_2012-%s-origin_images-%s_layer%d.npy' % (
            #         args.arch, args.tdir, multi_class_item, args.arch, args.tdir, multi_class_item, args.l)
            #     # t0_name = save_dir_pre + '/alexnet_imagenet_2012-single_train-%s_mid_res/alexnet_imagenet_2012-single_train-%s_layer5.npy' % (
            #     #     multi_class_item, multi_class_item)
            # else:
            #     t0_name = save_dir_pre + '/%s_imagenet_2012-single_val-%s_mid_res/%s_imagenet_2012-single_val-%s_layer%d.npy' % (
            #         args.arch, multi_class_item, args.arch, multi_class_item, args.l)
            #     # t0_name = save_dir_pre + '/alexnet_imagenet_2012-single_train-%s_mid_res/alexnet_imagenet_2012-single_train-%s_layer5.npy' % (
            #     #     multi_class_item, multi_class_item)

            # random_cp_classes = np.load(
            #     t + '/inner_cp_classes/%s_cp_classes.npy' % (multi_class_item)).tolist()

            # tensor_array = np.load(t0_name)
            # tensor_array = tensor_array.reshape(tensor_array.shape[0], -1)

            for multi_class_item2 in multi_classes:

                # print(multi_class_item2)
                img_index = 0
                # print(tensor_array.shape)

                for tensor_array_index in os.listdir(
                        t + '/vector_dir/' + time_type + '/o_%s/' % (args.tv) + multi_class_item):
                    tensor_array_item = np.load(t + '/vector_dir/' + time_type + '/o_%s/' % (
                        args.tv) + multi_class_item + '/' + tensor_array_index)

                    tensor_array_item = tensor_array_item.reshape(1, tensor_array_item.shape[0])

                    mkdir(class_comb_vector_dir + '/' + multi_class_item2)

                    save_path = class_comb_vector_dir + '/' + multi_class_item2 + '/%s' % (
                        tensor_array_index)

                    count += 1
                    # if os.path.exists(save_path):
                    #     print('%s - exists: %s' % (count, save_path))
                    #     continue

                    params.append((time_type, keep_node_pools_dic[multi_class_item2],
                                   keep_node_pools_weight_dic[multi_class_item2],
                                   keep_node_set_dic[multi_class_item2], tensor_array_item, save_path, count))

                    # img_index += 1

            p = multiprocessing.Pool(15)
            p.map(do_get_min_pool_npy2, params)
            p.close()
            p.join()

    if args.op == 'partition_train_vector_by_all_sim':

        ts_operation = args.tsop
        img_dir = t + '/transform_images_%s_noise' % ts_operation
        time_type = args.tt
        comb_vector_dir = t + '/vector_dir/' + time_type + '/comb_%s' % (args.tv)
        cos_vector_dir = t + '/vector_dir/' + time_type + '/weight_%s' % (args.tv)

        multi_classes = os.listdir(origin_image_dir)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print(torch.cuda.current_device())

        os.environ["CUDA_VISIBLE_DEVICES"] = '0'

        model = torch.load('%s_best_model.pth' % (args.tdir)).to(device).eval()

        for multi_class_item in multi_classes:

            print('----' + multi_class_item + '----')

            class_vector_dir = cos_vector_dir + '/' + multi_class_item

            mkdir(class_vector_dir)

            class_comb_vector_dir = comb_vector_dir + '/' + multi_class_item

            if multi_class_item != args.ec and args.ec != 'none' or multi_class_item == 'n00000000':
                continue

            img_index = 0

            for tensor_array_index in os.listdir(
                    t + '/vector_dir/' + time_type + '/o_%s/' % (args.tv) + multi_class_item):

                tensor_array_item_list = []
                # random_cp_classes = np.load(
                #     t + '/inner_cp_classes/%s_cp_classes.npy' % (multi_class_item_inner)).tolist()

                for multi_class_item2 in multi_classes:
                    # print(multi_class_item2)

                    tensor_array_item = np.load(
                        class_comb_vector_dir + '/' + multi_class_item2 + '/' + tensor_array_index)
                    # print(tensor_array_item.shape)
                    # tensor_array_item = tensor_array_item.reshape(1, tensor_array_item.shape[0])
                    tensor_array_item_list.append(tensor_array_item)

                o_train_vector = np.load(
                    t + '/vector_dir/' + time_type + '/o_%s/' % (args.tv) + multi_class_item + '/%s' % (
                        tensor_array_index)).reshape(1, -1)

                o_train_vector = torch.tensor(o_train_vector, dtype=torch.float32)
                input_tensor = o_train_vector.to(device)
                weight = model(input_tensor)
                weight = torch.nn.functional.softmax(weight, dim=1).data.cpu().numpy()
                # print('weight: ', weight)

                tensor_array_item_list = np.array(tensor_array_item_list)
                tensor_array_item_weight_sum = np.zeros(tensor_array_item_list[0].shape)

                for weight_index in range(len(weight[0])):
                    tensor_array_item_weight_sum += weight[0][weight_index] * tensor_array_item_list[weight_index]

                # print(ss_max_class)
                tensor_array_item_weight_sum = tensor_array_item_weight_sum.reshape(-1, )

                # np.save(class_vector_dir + '/%s' % (tensor_array_index),
                #         np.concatenate((tensor_array_item_weight_sum, o_train_vector.numpy().reshape(-1, ))))

                np.save(class_vector_dir + '/%s' % (tensor_array_index), tensor_array_item_weight_sum)
                #
                img_index += 1

    if args.op == 'partition_train_vector_by_all_sim2':

        ts_operation = args.tsop
        img_dir = t + '/transform_images_%s_noise' % ts_operation
        time_type = args.tt
        comb_vector_dir = t + '/vector_dir/' + time_type + '/comb_%s' % (args.tv)
        cos_vector_dir = t + '/vector_dir/' + time_type + '/weight_%s' % (args.tv)
        weight_vector_dir = t + '/vector_dir/' + time_type + '/weight_weight_%s' % (args.tv)

        multi_classes = os.listdir(img_dir)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # print(torch.cuda.current_device())

        os.environ["CUDA_VISIBLE_DEVICES"] = '0'

        model = torch.load('%s_best_model.pth' % (args.tdir)).to(device).eval()
        # model = get_model().cuda()

        for multi_class_item in multi_classes:

            class_vector_dir = weight_vector_dir + '/' + multi_class_item

            mkdir(class_vector_dir)

            if multi_class_item != args.ec and args.ec != 'none' or multi_class_item == 'n00000000':
                continue
            print('----' + multi_class_item + '----')

            for tensor_array_index in os.listdir(
                    t + '/vector_dir/' + time_type + '/o_%s/' % (args.tv) + multi_class_item):

                o_train_vector = np.load(t + '/vector_dir/' + time_type + '/o_%s/' % (
                    args.tv) + multi_class_item + '/' + tensor_array_index).reshape(1, -1)

                o_train_vector = torch.tensor(o_train_vector, dtype=torch.float32)
                input_tensor = o_train_vector.to(device)
                weight = model(input_tensor)
                weight = torch.nn.functional.softmax(weight).data.cpu().numpy()
                # print('weight: ', weight[0].shape)

                weight_top_v, weight_top_i = torch.from_numpy(weight).topk(10)
                weight_top_v = weight_top_v.numpy().tolist()
                weight_top_i = weight_top_i.numpy().tolist()

                top_weight_tuple_list = []
                # print(weight_top_i)
                for weight_top_index in range(len(weight_top_v[0])):
                    # print(weight_top_i[weight_top_index])
                    top_weight_tuple_list.append(
                        (int(weight_top_i[0][weight_top_index]), weight_top_v[0][weight_top_index]))

                # print(top_weight_tuple_list)

                np.save(class_vector_dir + '/top_weight_%s' % (tensor_array_index), np.array(top_weight_tuple_list))

    if args.op == 'partition_train_comb_index_vector2':

        # ==================
        # cal the js for zone in zero phase
        # ==================
        # the operation used to transform images, eg. gaussian blur: gb, all channel change: ac
        ts_operation = args.tsop
        img_dir = t + '/transform_images_%s_noise' % ts_operation
        mode = 'zone'
        i_o_index = args.ioi
        time_type = args.tt
        phase = 'zero'
        save_dir_pre = args.tdir + '_o_train_layer_npy'
        vector_dir = t + '/vector_dir/' + time_type + '/train'
        comb_vector_dir = t + '/vector_dir/' + time_type + '/comb_train'
        weight_vector_dir = t + '/vector_dir/' + time_type + '/weight_weight_%s' % (args.tv)

        if args.tv == 'val':
            save_dir_pre = 'validate' + '_layer_npy'
            vector_dir = t + '/vector_dir/' + time_type + '/val'
            comb_vector_dir = t + '/vector_dir/' + time_type + '/comb_val'

        # ==================
        # cal the js for zone in zero phase
        # ==================
        if args.param == 'time':
            img_dir = t + '/transform_images_t_%s_noise' % ts_operation
            mode = 'time'

        multi_classes = os.listdir(img_dir)

        keep_node_pools_dic = {}
        keep_node_pools_weight_dic = {}
        keep_node_set_dic = {}

        for time_index in range(1, int(time_type) + 1):

            for multi_class_item_cp in multi_classes:
                keep_node_pools = np.load(
                    t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item_cp + '/keep_node_pools%s_wad_%s_%s_%s_%s_list.npy' % (
                        time_index, ts_operation, mode, args.l, phase), allow_pickle=True)

                keep_node_pools_weight = np.load(
                    t + '/' + args.arch + '/metric_list_npy_dir/' + multi_class_item_cp + '/keep_node_pools%s_weight_wad_%s_%s_%s_%s_list.npy' % (
                        time_index, ts_operation, mode, args.l, phase), allow_pickle=True)
                deleted_node_set = np.load(
                    t + '/' + args.arch + '/deleted_node_npy_dir/deleted_node_%sd_marker_%s_%s_set_%s.npy' % (
                        args.mt, args.dnop, ts_operation, multi_class_item_cp))

                keep_node_set = list(set(list(range(output_size))) - set(deleted_node_set))

                if multi_class_item_cp not in keep_node_pools_dic.keys():
                    keep_node_pools_dic[multi_class_item_cp] = []
                    keep_node_pools_weight_dic[multi_class_item_cp] = []
                keep_node_pools_dic[multi_class_item_cp].append(keep_node_pools)
                keep_node_pools_weight_dic[multi_class_item_cp].append(keep_node_pools_weight)
                keep_node_set_dic[multi_class_item_cp] = keep_node_set

        count = 0

        for multi_class_item in multi_classes:
            params = []
            class_vector_dir = vector_dir + '/' + multi_class_item
            mkdir(class_vector_dir)

            class_comb_vector_dir = comb_vector_dir + '/' + multi_class_item
            mkdir(class_comb_vector_dir)

            if multi_class_item != args.ec and args.ec != 'none' or multi_class_item == 'n00000000':
                continue

            print('----' + multi_class_item + '----')

            if args.tv == 'train':
                t0_name = save_dir_pre + '/%s_imagenet_2012-%s-origin_images-%s_mid_res/%s_imagenet_2012-%s-origin_images-%s_layer%d.npy' % (
                    args.arch, args.tdir, multi_class_item, args.arch, args.tdir, multi_class_item, args.l)
                # t0_name = save_dir_pre + '/alexnet_imagenet_2012-single_train-%s_mid_res/alexnet_imagenet_2012-single_train-%s_layer5.npy' % (
                #     multi_class_item, multi_class_item)
            else:
                t0_name = save_dir_pre + '/%s_imagenet_2012-single_val-%s_mid_res/%s_imagenet_2012-single_val-%s_layer%d.npy' % (
                    args.arch, multi_class_item, args.arch, multi_class_item, args.l)
                # t0_name = save_dir_pre + '/alexnet_imagenet_2012-single_train-%s_mid_res/alexnet_imagenet_2012-single_train-%s_layer5.npy' % (
                #     multi_class_item, multi_class_item)

            # random_cp_classes = np.load(
            #     t + '/inner_cp_classes/%s_cp_classes.npy' % (multi_class_item)).tolist()

            tensor_array = np.load(t0_name)
            tensor_array = tensor_array.reshape(tensor_array.shape[0], -1)

            for tensor_array_index in os.listdir(
                    t + '/vector_dir/' + time_type + '/o_%s/' % (args.tv) + multi_class_item):

                top_weight = np.load(
                    weight_vector_dir + '/' + multi_class_item + '/top_weight_%s' % (tensor_array_index)).tolist()

                img_index = int(tensor_array_index.split('-')[-2])

                multi_class_item2_index = 0
                # print(top_weight[:, 0])
                # print(top_weight)
                top_weight = np.array(sorted(top_weight, key=lambda s: s[0]))
                # print(top_weight)
                # print(top_weight[:][1])

                for multi_class_item2 in multi_classes:

                    # print(multi_class_item2)

                    if not multi_class_item2_index in top_weight[:, 0]:
                        multi_class_item2_index += 1
                        continue
                    else:
                        multi_class_item2_index += 1

                    # tensor_array2 = np.load(t0_name)

                    tensor_array_item = tensor_array[img_index]
                    tensor_array_item = tensor_array_item.reshape(1, tensor_array_item.shape[0])

                    mkdir(class_comb_vector_dir + '/' + multi_class_item2)

                    save_path = class_comb_vector_dir + '/' + multi_class_item2 + '/%s-%s-%s.npy' % (
                        multi_class_item, img_index, time_type)

                    count += 1
                    # if os.path.exists(save_path):
                    #     # print('%s - exists: %s' % (count, save_path))
                    #     continue

                    params.append((time_type, keep_node_pools_dic[multi_class_item2],
                                   keep_node_pools_weight_dic[multi_class_item2],
                                   keep_node_set_dic[multi_class_item2], tensor_array_item, save_path, count))

            p = multiprocessing.Pool(15)
            p.map(do_get_min_pool_npy2, params)
            p.close()
            p.join()

    if args.op == 'partition_train_vector_by_all_sim3':

        ts_operation = args.tsop
        img_dir = t + '/transform_images_%s_noise' % ts_operation
        time_type = args.tt
        o_vecotr_dir = t + '/vector_dir/' + time_type + '/o_%s' % (args.tv)
        comb_vector_dir = t + '/vector_dir/' + time_type + '/comb_%s' % (args.tv)
        cos_vector_dir = t + '/vector_dir/' + time_type + '/weight_%s' % (args.tv)
        weight_vector_dir = t + '/vector_dir/' + time_type + '/weight_weight_%s' % (args.tv)

        multi_classes = os.listdir(img_dir)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print(torch.cuda.current_device())

        os.environ["CUDA_VISIBLE_DEVICES"] = '0'

        # model = torch.load('%s_best_model.pth' % (args.tdir)).to(device).eval()

        for multi_class_item in multi_classes:

            class_vector_dir = cos_vector_dir + '/' + multi_class_item

            mkdir(class_vector_dir)

            class_comb_vector_dir = comb_vector_dir + '/' + multi_class_item

            if multi_class_item != args.ec and args.ec != 'none' or multi_class_item == 'n00000000':
                continue

            print('----' + multi_class_item + '----')

            img_index = 0

            # print(len(os.listdir(o_vecotr_dir + '/' + multi_class_item)))

            for tensor_array_index in os.listdir(o_vecotr_dir + '/' + multi_class_item):

                # if os.path.exists(class_vector_dir + '/%s' % (tensor_array_index)):
                #     print(class_vector_dir + '/%s.npy' % (tensor_array_index))
                #     continue

                tensor_array_item_list = []
                weight = []
                # random_cp_classes = np.load(
                #     t + '/inner_cp_classes/%s_cp_classes.npy' % (multi_class_item_inner)).tolist()

                top_weight = np.load(
                    weight_vector_dir + '/' + multi_class_item + '/top_weight_%s' % (tensor_array_index)).tolist()

                multi_class_item2_index = 0
                # top_weight_list = top_weight.tolist()
                top_weight = np.array(sorted(top_weight, key=lambda s: s[0]))

                # print('weight: ', top_weight)

                for multi_class_item2 in multi_classes:
                    # print(multi_class_item2)

                    if not multi_class_item2_index in top_weight[:, 0]:
                        multi_class_item2_index += 1
                        continue
                    else:
                        multi_class_item2_index += 1

                    tensor_array_item = np.load(
                        class_comb_vector_dir + '/' + multi_class_item2 + '/' + tensor_array_index, allow_pickle=True)

                    # print(tensor_array_item.shape)
                    # tensor_array_item = tensor_array_item.reshape(1, tensor_array_item.shape[0])
                    tensor_array_item_list.append(tensor_array_item)
                    weight.append(top_weight[:, 1][top_weight[:, 0].tolist().index(multi_class_item2_index - 1)])

                o_train_vector = np.load(
                    t + '/vector_dir/' + time_type + '/o_%s/' % (args.tv) + multi_class_item + '/%s' % (
                        tensor_array_index)).reshape(1, -1)

                tensor_array_item_list = np.array(tensor_array_item_list)
                tensor_array_item_weight_sum = np.zeros(tensor_array_item_list[0].shape)

                # print(weight.shape)
                for weight_index in range(len(weight)):
                    tensor_array_item_weight_sum += weight[weight_index] * tensor_array_item_list[weight_index]

                # print(ss_max_class)
                tensor_array_item_weight_sum = tensor_array_item_weight_sum.reshape(-1, )

                np.save(class_vector_dir + '/%s' % (tensor_array_index),
                        tensor_array_item_weight_sum + o_train_vector.reshape(-1, ))
                #
                img_index += 1

    if args.op == 'get_all_transform_images':

        ts_operation = args.tsop
        img_dir = t + '/transform_images_%s_noise' % ts_operation
        mode = 'zone'
        save_dir_pre = args.tdir + '_o_train_layer_npy'
        i_o_index = args.ioi
        time_type = args.tt
        phase = 'zero'
        vector_dir = t + '/vector_dir/' + time_type + '/all_train_img'

        for multi_class_item in os.listdir(img_dir):

            class_vector_dir = vector_dir + '/' + multi_class_item
            mkdir(class_vector_dir)

            for multi_class_item_index in os.listdir(img_dir + '/' + multi_class_item):

                for multi_class_item_index_img in os.listdir(
                        img_dir + '/' + multi_class_item + '/' + multi_class_item_index + '/' + multi_class_item_index):
                    shutil.copy(
                        img_dir + '/' + multi_class_item + '/' + multi_class_item_index + '/' + multi_class_item_index + '/' + multi_class_item_index_img,
                        class_vector_dir + '/' + multi_class_item_index + '-' + multi_class_item_index_img.split('.')[
                            0] + '.jpg')

    if args.op == 'transform_cp_classes':

        ts_cp_classes_dic = {}

        mkdir(t + '/ts_cp_classes')

        for multi_class_item in os.listdir(origin_image_dir):
            cp_classes = np.load(t + '/inner_cp_classes/%s_cp_classes.npy' % (multi_class_item))
            for item in cp_classes:
                # if multi_class_item ==
                if ts_cp_classes_dic.keys().__contains__(item):
                    ts_cp_classes_dic[item].append(multi_class_item)
                else:
                    ts_cp_classes_dic[item] = []
                    ts_cp_classes_dic[item].append(multi_class_item)

        for multi_class_item in os.listdir(origin_image_dir):
            np.save(t + '/ts_cp_classes/%s_ts_cp_classes.npy' % (multi_class_item), ts_cp_classes_dic[multi_class_item])

    if args.op == 'get_cp_classes':

        cp_classes_dic = {}

        mkdir(t + '/ts_cp_classes')

        for multi_class_item in os.listdir(origin_image_dir):
            cp_classes_dir = 'imagenet_2012/%s/%s/metric_list_npy_dir/%s/metric_list_o_wad_sc_zone_%d_zero_npy' % (
                args.tdir, args.arch, multi_class_item, args.l)
            for item in os.listdir(cp_classes_dir):
                cp_class_item = item.split('-')[2]
                if multi_class_item in cp_classes_dic:
                    cp_classes_dic[multi_class_item].add(cp_class_item)
                else:
                    cp_classes_dic[multi_class_item] = set()
                    cp_classes_dic[multi_class_item].add(cp_class_item)

        for multi_class_item in os.listdir(origin_image_dir):
            np.save(t + '/inner_cp_classes/%s_cp_classes.npy' % (multi_class_item),
                    np.array(list(cp_classes_dic[multi_class_item])))

    if args.op == 'get_origin_classes':

        multi_classes = os.listdir(origin_image_dir)
        multi_classes.sort()

        with open(t + '/origin_classes.txt', 'w+') as f:
            for multi_class_item in multi_classes:
                f.write(multi_class_item + '\n')
        # f.flush()
        f.close()

        np.save(t + '/%s_origin_classes.npy' % (args.tdir), np.array(multi_classes))

    if args.op == 'partition_train_vector_by_multi_layer':

        ts_operation = args.tsop
        img_dir = t + '/transform_images_%s_noise' % ts_operation
        time_type = args.tt
        o_vector_dir = t + '/vector_dir/' + time_type + '/o_%s' % (args.tv)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(torch.cuda.current_device())
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        model = torch.load('%s_best_model.pth' % (args.tdir)).to(device).eval()

        multi_classes = os.listdir(img_dir)

        for multi_class_item in multi_classes:

            print('----' + multi_class_item + '----')

            for img_index in os.listdir(o_vector_dir + '/' + multi_class_item):
                o_train_vector = np.load(
                    o_vector_dir + '/' + multi_class_item + '/' + img_index).reshape(1, -1)

                o_train_vector = torch.tensor(o_train_vector, dtype=torch.float32)
                input_tensor = o_train_vector.to(device)
                weight = model(input_tensor)
                weight = torch.nn.functional.softmax(weight, dim=1).data.cpu().numpy()
                # print(weight)
                # print(np.sum(weight))
                o_train_vector = o_train_vector.numpy().reshape(-1, )

                # o_train_vector = (o_train_vector - np.mean(
                #     o_train_vector)) / np.std(o_train_vector)

                img_index_split = img_index.split('-')

                for time_type_index in range(1, int(time_type) + 1):
                    comb_vector_dir = t + '/vector_dir/' + str(time_type_index) + '/comb_%s' % (args.tv)
                    weight_vector_dir = t + '/vector_dir/' + str(time_type_index) + '/weight_%s' % (args.tv)
                    mkdir(weight_vector_dir + '/' + multi_class_item)

                    tensor_array_item_list = []

                    for multi_class_item2 in multi_classes:
                        img_index_tt = '%s-%s-%s.npy' % (multi_class_item2, img_index_split[1], str(time_type_index))
                        tensor_array_item = np.load(
                            comb_vector_dir + '/' + multi_class_item2 + '/' + multi_class_item2 + '/' + img_index_tt)
                        # print(tensor_array_item.shape)
                        # tensor_array_item = tensor_array_item.reshape(1, tensor_array_item.shape[0])
                        tensor_array_item_list.append(tensor_array_item)

                    tensor_array_item_list = np.array(tensor_array_item_list)
                    tensor_array_item_weight_sum = np.zeros(tensor_array_item_list[0].shape)

                    for weight_index in range(len(weight[0])):
                        tensor_array_item_weight_sum += weight[0][weight_index] * tensor_array_item_list[weight_index]

                    tensor_array_item_weight_sum = tensor_array_item_weight_sum.reshape(-1, )

                    # tensor_array_item_weight_sum = (tensor_array_item_weight_sum - np.mean(
                    #     tensor_array_item_weight_sum)) / np.std(tensor_array_item_weight_sum)

                    # o_train_vector += tensor_array_item_weight_sum
                    # np.save(weight_vector_dir + '/%s' % (img_index), )
                    np.save(weight_vector_dir + '/' + multi_class_item + '/%s-%s-%s.npy' % (
                        multi_class_item, img_index_split[1], str(time_type_index)),
                            o_train_vector+tensor_array_item_weight_sum)




main()
