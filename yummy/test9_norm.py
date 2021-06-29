import numpy as np
from collections import OrderedDict
import sys
import pymongo
import pickle

import tensorflow as tf
from PIL import Image
from bson.binary import Binary
import torch

model_type = 'alexnet'
m_image_dir = 'cat_trans'
batch_number = 4
layer_number = 8


def cal_psnr(target, ref, max_value):
    rmse = 0

    diff = target - ref
    diff = diff.flatten('C')
    # print('mean diff is ', np.mean(diff))
    rmse += np.math.sqrt(np.mean(diff ** 2.))
    # print('rmse is ', rmse)
    # print('max value is ', max_value)

    return 20 * np.math.log10(max_value / rmse)


def cal_ssim(target, ref, max_value):
    k1 = 0.01
    k2 = 0.03
    L = max_value
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2

    mu1 = target.mean()
    mu2 = ref.mean()
    std1 = np.sqrt(target.var())
    std2 = np.sqrt(target.var())
    target_minus_mu1 = np.subtract(target, mu1).flatten('C')
    ref_minus_mu2 = np.subtract(ref, mu2).flatten('C')
    xmm_ymm = target_minus_mu1.dot(ref_minus_mu2)
    target_size = target.shape
    shape_product = 1
    for d in target_size:
        shape_product *= d
    # print(target_size)
    cov_xy = xmm_ymm.sum() / (shape_product - 1)

    l = (2 * mu1 * mu2 + C1) / (pow(mu2, 2) + pow(mu2, 2) + C1)
    c = (2 * std1 * std2 + C2) / (pow(std1, 2) + pow(std2, 2) + C2)
    s = (cov_xy + C3) / (std2 * std1 + C3)

    return l * c * s


for layer_index in range(-1, layer_number):

    # print('cal avg, cur layer is %d, batch is %d' % (layer_index, 0))

    max_i = -256
    min_i = 256

    batch_array = np.load('%s-%s-layer%d_batch%d.npy' % (model_type, m_image_dir, layer_index, 0))
    img_count = 0

    cur_max_i = np.max(batch_array)
    cur_min_i = np.min(batch_array)

    img_sum = np.zeros(batch_array[0].shape)
    img_count += len(batch_array)
    for image_id in range(len(batch_array)):
        # cur_img = batch_array[image_id]
        img_sum = np.add(img_sum, batch_array[image_id])

    for batch_index in range(1, batch_number):

        # print('cal avg, cur layer is %d, batch is %d' % (layer_index, batch_index))

        batch_array = np.load('%s-%s-layer%d_batch%d.npy' % (model_type, m_image_dir, layer_index, batch_index))

        img_count += len(batch_array)
        cur_max_i = np.max(batch_array)
        cur_min_i = np.min(batch_array)
        if cur_max_i > max_i:
            max_i = cur_max_i
        if cur_min_i < min_i:
            min_i = cur_min_i

        for image_id in range(len(batch_array)):
            img_sum = np.add(img_sum, batch_array[image_id])

    # interval = max_i - min_i
    # img_sum = img_sum - min_i
    interval = max_i
    img_avg = img_sum / img_count

    # if layer_index == 7:
    #     print(img_avg)

    psnr_list = []

    # print('max is ', max_i)
    # print('min is ', min_i)

    for batch_index in range(0, batch_number):

        # print('cal psnr, cur layer is %d, batch is %d' % (layer_index, batch_index))

        batch_array = np.load('%s-%s-layer%d_batch%d.npy' % (model_type, m_image_dir, layer_index, batch_index))
        batch_array = (batch_array - min_i) / (max_i - min_i)

        for image_id in range(len(batch_array)):
            psnr = 0
            if layer_index < 5:
                # psnr = cal_psnr((batch_array[image_id]), (img_avg - min_i) / (max_i - min_i), 1)
                # psnr = cal_ssim((batch_array[image_id]), (img_avg - min_i) / (max_i - min_i), 1)

                # for i in range(0, len(batch_array[image_id])):
                    psnr += tf.image.ssim(tf.convert_to_tensor(batch_array[image_id]), tf.convert_to_tensor(((img_avg - min_i) / (max_i - min_i))), 1)
                # psnr /= len(batch_array[image_id])
            elif layer_index >= 7:

                # print('-------')
                # psnr = cal_psnr(batch_array[image_id][0:2] / max_i, img_avg[0:2] / max_i, 1)
                # psnr = cal_ssim(batch_array[image_id][0:2] / max_i, img_avg[0:2] / max_i, 1)
                for i in range(0, len(batch_array[image_id])):
                    psnr = tf.image.ssim(batch_array[image_id][0:2] / max_i, img_avg[0:2] / max_i, 1)
                # print(batch_array[image_id][0:2])
                # print(img_avg[0:2])
                # print(psnr)

            # psnr = tf.image.psnr(batch_array[image_id] - min_i, img_avg, interval)
            psnr_list.append(psnr)

    psnr_sum = 0
    for p in psnr_list:
        psnr_sum = psnr_sum + p

    psnr_avg = psnr_sum / img_count

    # print(psnr_avg)

    print('--- LAYER %d avg SSIM is %f ---' % (layer_index, psnr_avg))
