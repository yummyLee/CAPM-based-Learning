import skimage
import torch
import torchvision.models as models
import numpy as np
import torch.nn as nn
import pymongo
import argparse
import torch.utils.data
import torchvision.datasets as datasets
import tensorflow as tf
from torchvision.transforms import transforms
from PIL import Image
from scipy.signal import convolve2d
from SSIM_PIL import compare_ssim
from PIL import Image
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from sklearn.neighbors.kde import KernelDensity
import scipy.stats

MONGODB_HOST = '127.0.0.1'
# 端口号，默认27017
MONGODB_PORT = 27017
# 设置数据库名称
MONGODB_DBNAME = 'alexnet'
# 存放本数据的表名称
MONGODB_COLNAME = 'one_dog'

parser = argparse.ArgumentParser()
parser.add_argument('-b', type=int, default=256, help='batch size')
parser.add_argument('-dir', type=str, help='image to be operated')
parser.add_argument('-arch', default='alexnet', help='type of net')
parser.add_argument('-op', type=str, help='operation')
parser.add_argument('-g', type=str, default='psnr', help='grades type')
parser.add_argument('-f', type=int, default=-1, help='data to next layer')
parser.add_argument('-sl', type=int, default=1000, help='len of generated series')
parser.add_argument('-ts', type=str, default='none', help='trans before input')

args = parser.parse_args()

save_image_dir = args.dir + '_pic'


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


def tensor_array_normalization(wait_norm_tensors):
    max_i = np.max(wait_norm_tensors)
    min_i = np.min(wait_norm_tensors)

    return (wait_norm_tensors - min_i) / (max_i - min_i)


def cal_psnr(targ, ref, max_value):
    rmse = 0

    diff = targ - ref
    diff = diff.flatten('C')
    rmse += np.math.sqrt(np.mean(diff ** 2.))

    return 20 * np.math.log10(max_value / rmse), -1


def cal_z_psnr(targ, ref, max_value):
    targ = targ.flatten('C')
    ref = ref.flatten('C')

    diff = []

    for m_index in range(targ.shape[0]):
        # if targ[m_index] > pow(10, -10) or ref[m_index] > pow(10, -10):
        if ref[m_index] > pow(10, -10):
            diff.append(targ[m_index] - ref[m_index])

    m_mse = np.math.sqrt(np.mean(np.array(diff) ** 2.))

    # print(np.array(diff).shape[0])
    # print(targ.shape[0])

    return 20 * np.math.log10(max_value / m_mse), 1 - (np.array(diff).shape[0] / targ.shape[0])


def cal_ssim(targ, ref, max_value, win_size):
    targ = targ.reshape(1, targ.shape[0], targ.shape[1])
    ref = ref.reshape(1, ref.shape[0], ref.shape[1])
    targ3 = targ.reshape([1, targ.shape[0], targ.shape[1], targ.shape[2]])
    ref3 = ref.reshape([1, ref.shape[0], ref.shape[1], ref.shape[2]])
    targ3 = targ3.astype(np.float32)
    ref3 = ref3.astype(np.float32)
    ssim_module = SSIM(win_size=win_size, win_sigma=1.5, data_range=max_value, size_average=False, channel=1)
    ssim_val = ssim_module(torch.from_numpy(targ3), torch.from_numpy(ref3))
    return ssim_val, -1


def cal_skssim(targ, ref, max_value, win_size):
    # targ = targ.reshape(1, targ.shape[0], targ.shape[1])
    # ref = ref.reshape(1, ref.shape[0], ref.shape[1])
    # targ3 = targ.reshape([1, targ.shape[0], targ.shape[1], targ.shape[2]])
    # ref3 = ref.reshape([1, ref.shape[0], ref.shape[1], ref.shape[2]])
    # targ3 = targ3.astype(np.float32)
    # ref3 = ref3.astype(np.float32)
    # ssim_module = SSIM(win_size=win_size, win_sigma=1.5, data_range=max_value, size_average=False, channel=1)
    ssim_val = skimage.measure.compare_ssim(targ, ref, win_size=win_size, data_range=max_value, multichannel=False)
    return ssim_val, -1


def cal_z_ssim(targ, ref, max_value):
    k1 = 0.01
    k2 = 0.03
    L = max_value
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2

    targ2 = []
    ref2 = []

    for m_index in range(targ.shape[0]):
        if targ[m_index] > pow(10, -30) or ref[m_index] > pow(10, -30):
            targ2.append(targ[m_index])
            ref2.append(ref[m_index])

    targ2 = np.from_array(targ2)
    ref2 = np.from_array(ref2)

    mu1 = targ2.mean()
    mu2 = ref2.mean()
    std1 = np.sqrt(targ2.var())
    std2 = np.sqrt(targ2.var())

    target_minus_mu1 = np.subtract(targ2, mu1).flatten('C')
    ref_minus_mu2 = np.subtract(ref2, mu2).flatten('C')
    xmm_ymm = target_minus_mu1.dot(ref_minus_mu2)
    target_size = targ.shape[0]
    target2_size = targ2.shape[0]
    # print(target_size)
    cov_xy = xmm_ymm.sum() / (target2_size - 1)

    m_l = (2 * mu1 * mu2 + C1) / (pow(mu2, 2) + pow(mu2, 2) + C1)
    c = (2 * std1 * std2 + C2) / (pow(std1, 2) + pow(std2, 2) + C2)
    s = (cov_xy + C3) / (std2 * std1 + C3)

    return m_l * c * s, 1 - (target2_size / target_size)


def cal_kld(targ, ref, max_value):
    targ = targ.reshape(-1, 1)
    ref = ref.reshape(-1, 1)
    t_kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(targ)
    r_kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(ref)

    sample_size = 10000

    t_samples = t_kde.sample(sample_size)
    r_samples = r_kde.sample(sample_size)

    count_size = 100

    t_count = np.zeros([count_size + 2, 1])
    r_count = np.zeros([count_size + 2, 1])
    for m_i in range(sample_size):

        if t_samples[m_i] <= 0:
            t_count[0] += 1
        elif t_samples[m_i] >= 1:
            t_count[count_size + 1] += 1
        else:
            t_count[np.math.floor(t_samples[m_i] * count_size)] += 1

    for m_i in range(sample_size):

        if r_samples[m_i] <= 0:
            r_count[0] += 1
        elif r_samples[m_i] >= 1:
            r_count[count_size + 1] += 1
        else:
            r_count[np.math.floor(r_samples[m_i] * count_size)] += 1

    t_count = t_count / sample_size
    r_count = r_count / sample_size

    # print(t_count)
    # print(r_count)

    KL = scipy.stats.entropy(t_count, r_count)
    # JS = js_divergence(t_count, r_count)
    #
    # print(KL)
    # print(JS)

    return KL, -1


def js_divergence(p, q):
    M = (p + q) / 2
    return 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)


def cal_jsd(targ, ref, max_value):
    targ = targ.reshape(-1, 1)
    ref = ref.reshape(-1, 1)
    t_kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(targ)
    r_kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(ref)

    sample_size = 10000

    t_samples = t_kde.sample(sample_size)
    r_samples = r_kde.sample(sample_size)

    count_size = 100

    t_count = np.zeros([count_size + 2, 1])
    r_count = np.zeros([count_size + 2, 1])
    for m_i in range(sample_size):

        if t_samples[m_i] <= 0:
            t_count[0] += 1
        elif t_samples[m_i] >= 1:
            t_count[count_size + 1] += 1
        else:
            t_count[np.math.floor(t_samples[m_i] * count_size)] += 1

    for m_i in range(sample_size):

        if r_samples[m_i] <= 0:
            r_count[0] += 1
        elif r_samples[m_i] >= 1:
            r_count[count_size + 1] += 1
        else:
            r_count[np.math.floor(r_samples[m_i] * count_size)] += 1

    t_count = t_count / sample_size
    r_count = r_count / sample_size

    # print(t_count)
    # print(r_count)

    # KL = scipy.stats.entropy(t_count, r_count)
    JS = js_divergence(t_count, r_count)
    #
    # print(KL)
    # print(JS)

    return JS, -1


def cal_layer_grades(grade_type, wait_to_cal_array, array_normal=True, max_value=1):
    for image_tensor in range(len(wait_to_cal_array)):

        if array_normal:
            wait_to_cal_array = tensor_array_normalization(wait_to_cal_array)

        img_sum = np.zeros(wait_to_cal_array[0].shape)

        for image_id in range(len(wait_to_cal_array)):
            img_sum = np.add(img_sum, wait_to_cal_array[image_id])

        img_avg = img_sum / len(wait_to_cal_array)

        m_grade_list = []
        m_zero_rate_list = []

        for image_id in range(len(wait_to_cal_array)):

            if grade_type == 'psnr':
                psnr, m_zero_rate = cal_psnr((wait_to_cal_array[image_id]), img_avg, max_value)
                # print(psnr)
                m_grade_list.append(psnr)
                m_zero_rate_list.append(m_zero_rate)

            if grade_type == 'ssim':
                win_size = 5
                wait_to_cal = wait_to_cal_array[image_id]
                # print(img_avg.shape)
                # print(len(img_avg.shape))

                if len(img_avg.shape) == 1:
                    wait_to_cal = wait_to_cal.reshape(1, img_avg.shape[0])
                    img_avg = img_avg.reshape(1, img_avg.shape[0])
                    m_ssim, m_zero_rate = cal_ssim(wait_to_cal, img_avg, max_value, 1)
                else:
                    if img_avg.shape[0] == 1:
                        wait_to_cal = wait_to_cal.reshape(1, img_avg.shape[1])
                        img_avg = img_avg.reshape(1, img_avg.shape[1])
                        m_ssim, m_zero_rate = cal_ssim(wait_to_cal, img_avg, max_value, 1)
                    elif img_avg.shape[1] == 1:
                        wait_to_cal = wait_to_cal.reshape(1, img_avg.shape[0])
                        img_avg = img_avg.reshape(1, img_avg.shape[0])
                        m_ssim, m_zero_rate = cal_ssim(wait_to_cal, img_avg, max_value, 1)
                    else:
                        wait_to_cal_reshape = wait_to_cal.reshape(img_avg.shape[1], -1)
                        img_avg_reshape = img_avg.reshape(img_avg.shape[1], -1)
                        m_ssim, m_zero_rate = cal_ssim(wait_to_cal_reshape, img_avg_reshape, max_value, win_size)
                m_grade_list.append(m_ssim)
                m_zero_rate_list.append(m_zero_rate)

            if grade_type == 'skssim':
                win_size = 5
                wait_to_cal = wait_to_cal_array[image_id]
                # print(img_avg.shape)
                # print(len(img_avg.shape))

                if len(img_avg.shape) == 1:
                    wait_to_cal = wait_to_cal.reshape(1, img_avg.shape[0])
                    img_avg = img_avg.reshape(1, img_avg.shape[0])
                    m_ssim, m_zero_rate = cal_skssim(wait_to_cal, img_avg, max_value, 1)
                else:
                    if img_avg.shape[0] == 1:
                        wait_to_cal = wait_to_cal.reshape(1, img_avg.shape[1])
                        img_avg = img_avg.reshape(1, img_avg.shape[1])
                        m_ssim, m_zero_rate = cal_skssim(wait_to_cal, img_avg, max_value, 1)
                    elif img_avg.shape[1] == 1:
                        wait_to_cal = wait_to_cal.reshape(1, img_avg.shape[0])
                        img_avg = img_avg.reshape(1, img_avg.shape[0])
                        m_ssim, m_zero_rate = cal_skssim(wait_to_cal, img_avg, max_value, 1)
                    else:
                        wait_to_cal_reshape = wait_to_cal.reshape(img_avg.shape[1], -1)
                        img_avg_reshape = img_avg.reshape(img_avg.shape[1], -1)
                        m_ssim, m_zero_rate = cal_skssim(wait_to_cal_reshape, img_avg_reshape, max_value, win_size)
                m_grade_list.append(m_ssim)
                m_zero_rate_list.append(m_zero_rate)

            if grade_type == 'z_psnr':
                t_ssim, m_zero_rate = cal_z_psnr((wait_to_cal_array[image_id]), img_avg, max_value)
                # print('ssim is', t_ssim)
                m_grade_list.append(t_ssim)
                m_zero_rate_list.append(m_zero_rate)
                # print(m_zero_rate)

            if grade_type == 'z_ssim':
                t_ssim, m_zero_rate = cal_z_ssim((wait_to_cal_array[image_id]), img_avg, max_value)
                # print('ssim is', t_ssim)
                m_grade_list.append(t_ssim)
                m_zero_rate_list.append(m_zero_rate)

            if grade_type == 'kld':
                m_kld, m_zero_rate = cal_kld((wait_to_cal_array[image_id]), img_avg, max_value)
                # print('ssim is', t_ssim)
                m_grade_list.append(m_kld)
                m_zero_rate_list.append(m_zero_rate)

            if grade_type == 'jsd':
                m_jsd, m_zero_rate = cal_jsd((wait_to_cal_array[image_id]), img_avg, max_value)
                # print('ssim is', t_ssim)
                m_grade_list.append(m_jsd)
                m_zero_rate_list.append(m_zero_rate)

        # grade_sum = 0
        # for g in m_grade_list:
        #     grade_sum = grade_sum + g
        # m_grade_avg = grade_sum / len(wait_to_cal_array)

        m_grade_var = np.array(m_grade_list).std()
        m_grade_avg = np.array(m_grade_list).mean()
        m_grade_min = np.array(m_grade_list).min()
        m_grade_max = np.array(m_grade_list).max()
        m_zero_rate_avg = np.array(m_zero_rate_list).mean()

        # print(m_zero_rate_avg)

        return m_grade_avg, m_grade_var, m_grade_min, m_grade_max, m_grade_list, m_zero_rate_avg


def cal_middle_output(m_model, arch, input_tensor, layer_index):
    output = None
    if arch == 'alexnet':
        model_f = m_model.features
        # model_a = m_model.avgpool
        model_c = m_model.classifier

        if layer_index == 1:
            output = model_f[2](model_f[1](model_f[0](input_tensor.cpu()).cpu()).cpu()).cpu()
        elif layer_index == 2:
            output = model_f[5](model_f[4](model_f[3](input_tensor.cpu()).cpu()).cpu()).cpu()
        elif layer_index == 3:
            output = model_f[7](model_f[6](input_tensor.cpu()).cpu()).cpu()
        elif layer_index == 4:
            output = model_f[9](model_f[8](input_tensor.cpu()).cpu()).cpu()
        elif layer_index == 5:
            output = model_f[12](model_f[11](model_f[10](input_tensor.cpu()).cpu()).cpu()).cpu()
        elif layer_index == 6:
            input_flat = input_tensor.cpu().view(-1, 9216)
            output = model_c[2](model_c[1](model_c[0](input_flat).cpu()).cpu())
        elif layer_index == 7:
            output = model_c[5](model_c[4](model_c[3](input_tensor.cpu()).cpu()).cpu()).cpu()
        elif layer_index == 8:
            output = model_c[6](input_tensor.cpu()).cpu()

    if arch == 'vgg16':
        model_f = m_model.features
        model_c = m_model.classifier

        if layer_index == 1:
            output = model_f[1](model_f[0](input_tensor.cpu()).cpu()).cpu()
        elif layer_index == 2:
            output = model_f[3](model_f[2](input_tensor.cpu()).cpu()).cpu()
        elif layer_index == 3:
            output = model_f[6](model_f[5](model_f[4](input_tensor.cpu()).cpu()).cpu()).cpu()
        elif layer_index == 4:
            output = model_f[8](model_f[7](input_tensor.cpu()).cpu()).cpu()
        elif layer_index == 5:
            output = model_f[11](model_f[10](model_f[9](input_tensor.cpu()).cpu()).cpu()).cpu()
        elif layer_index == 6:
            output = model_f[13](model_f[12](input_tensor.cpu()).cpu()).cpu()
        elif layer_index == 7:
            output = model_f[15](model_f[14](input_tensor.cpu()).cpu()).cpu()
        elif layer_index == 8:
            output = model_f[18](model_f[17](model_f[16](input_tensor.cpu()).cpu()).cpu()).cpu()
        elif layer_index == 9:
            output = model_f[20](model_f[19](input_tensor.cpu()).cpu()).cpu()
        elif layer_index == 10:
            output = model_f[22](model_f[21](input_tensor.cpu()).cpu()).cpu()
        elif layer_index == 11:
            output = model_f[25](model_f[24](model_f[23](input_tensor.cpu()).cpu()).cpu()).cpu()
        elif layer_index == 12:
            output = model_f[27](model_f[26](input_tensor.cpu()).cpu()).cpu()
        elif layer_index == 13:
            output = model_f[29](model_f[28](input_tensor.cpu()).cpu()).cpu()
        elif layer_index == 14:
            output = model_f[30](input_tensor.cpu())
        elif layer_index == 15:
            input_flat = input_tensor.cpu().view(-1, 25088)
            output = model_c[1](model_c[0](input_flat).cpu()).cpu()
        elif layer_index == 16:
            output = model_c[4](model_c[3](model_c[2](input_tensor.cpu()).cpu())).cpu()
        elif layer_index == 17:
            output = model_c[6](model_c[5](input_tensor).cpu()).cpu()

    return output


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


if args.op == 'get_layer':

    print('---- OP IS GET_LAYER ----')

    model = models.__dict__[args.arch](pretrained=True)
    print(model)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.dir, transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            normalize
        ])),
        batch_size=args.b,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    outputs_multi_layer = []

    mkdir((args.arch + '_' + args.dir + '_mid_res'))

    outputs = None

    for i, (image_input, target) in enumerate(transform_data_loader):
        image_input, target = image_input.to(device), target.to(device)
        inputs = image_input.cpu()
        if i == 0:
            outputs = inputs
        else:
            outputs = torch.cat((outputs, inputs))
    np.save('%s/%s_%s_layer%d.npy' % ((args.arch + '_' + args.dir + '_mid_res'), args.arch, args.dir, 0),
            outputs.numpy())
    print('%s/%s_%s_layer%d.npy saved' % ((args.arch + '_' + args.dir + '_mid_res'), args.arch, args.dir, 0))

    num_of_layer = 0

    if args.arch == 'alexnet':
        num_of_layer = 8
        print('---- ARCH IS ALEXNET ----')

    if args.arch == 'vgg16':
        num_of_layer = 17
        print('---- ARCH IS VGG16 ----')

    for layer_id in range(1, num_of_layer + 1):

        if args.ts == 'normal':
            outputs = tensor_array_normalization(outputs)
        elif args.ts == 'bn':
            outputs = nn.BatchNorm2d(torch.from_array(outputs), momentum=0.99).numpy()

        outputs = cal_middle_output(model, args.arch, outputs, layer_id)
        np.save('%s/%s_%s_layer%d.npy' % ((args.arch + '_' + args.dir + '_mid_res'), args.arch, args.dir, layer_id),
                outputs.detach().cpu().numpy())
        print('%s/%s_%s_layer%d.npy saved' % (
            (args.arch + '_' + args.dir + '_mid_res'), args.arch, args.dir, layer_id))

if args.op == 'get_mid_layer':
    print('---- OP IS GET_MID_LAYER ----')

    model = models.__dict__[args.arch](pretrained=True)
    print(model)

    tensor_array = np.load(
        '%s/%s_%s_layer%d.npy' % ((args.arch + '_' + args.dir + '_mid_res'), args.arch, args.dir, args.f))
    outputs = cal_middle_output(model, args.arch, torch.from_numpy(tensor_array), args.f + 1)
    np.save('%s/%s_%s_layer%d.npy' % ((args.arch + '_' + args.dir + '_mid_res'), args.arch, args.dir, args.f + 1),
            outputs.detach().cpu().numpy())
    print('%s/%s_%s_layer%d.npy saved' % (
        (args.arch + '_' + args.dir + '_mid_res'), args.arch, args.dir, args.f + 1))

if args.op == 'get_grades':

    print('---- OP IS GET_GRADES ----')

    num_of_layer = 0

    if args.arch == 'alexnet':
        num_of_layer = 8
        print('---- ARCH IS ALEXNET ----')

    if args.arch == 'vgg16':
        num_of_layer = 17
        print('---- ARCH IS VGG16 ----')

    for layer_id in range(0, num_of_layer + 1):
        tensor_array = np.load(
            '%s/%s_%s_layer%d.npy' % ((args.arch + '_' + args.dir + '_mid_res'), args.arch, args.dir, layer_id))
        grade_avg, grade_std, grade_min, grade_max, grade_list, zero_rate = cal_layer_grades(args.g, tensor_array)
        print_layer_grades_info(layer_id, args.g, grade_avg, grade_std, grade_min, grade_max, zero_rate, args.ts)

if args.op == 'r_get_grades':

    print('---- OP IS R_GET_GRADES ----')

    model = models.__dict__[args.arch](pretrained=True)

    start_f_id = 0
    end_f_id = 0

    if args.arch == 'alexnet':
        end_f_id = 8
        print('---- ARCH IS ALEXNET ----')
    elif args.arch == 'vgg16':
        end_f_id = 17
        print('---- ARCH IS VGG16 ----')

    if args.f != -1:
        start_f_id = args.f
        end_f_id = args.f + 1

    for layer_id in range(start_f_id, end_f_id):

        tensor_array = np.load(
            '%s/%s_%s_layer%d.npy' % ((args.arch + '_' + args.dir + '_mid_res'), args.arch, args.dir, layer_id))

        if args.ts == 'normal':
            # print('---- TS IS NORMAL ----')
            tensor_array = tensor_array_normalization(tensor_array)
        elif args.ts == 'bn':
            # print('---- TS IS BN ----')
            # print(len(tensor_array[0].shape))
            # print(tensor_array[0].shape)

            if len(tensor_array[0].shape) == 1:
                wait_to_nor_array = tensor_array.reshape([tensor_array.shape[0], 1, tensor_array[0].shape[0]])
                m_trans = nn.BatchNorm1d(wait_to_nor_array[0].shape[0], momentum=0.99, affine=False)
                tensor_array = m_trans(torch.from_numpy(wait_to_nor_array)).detach().cpu().numpy()
            else:
                if tensor_array[0].shape[0] == 1:
                    wait_to_nor_array = tensor_array.reshape(
                        [tensor_array.shape[0], 1, tensor_array[0].shape[1]])
                    m_trans = nn.BatchNorm1d(wait_to_nor_array[0].shape[0], momentum=0.99, affine=False)
                    tensor_array = m_trans(torch.from_numpy(wait_to_nor_array)).detach().cpu().numpy()
                elif tensor_array[0].shape[1] == 1:
                    wait_to_nor_array = tensor_array.reshape(
                        [tensor_array.shape[0], 1, tensor_array[0].shape[0]])
                    m_trans = nn.BatchNorm1d(wait_to_nor_array[0].shape[0], momentum=0.99, affine=False)
                    tensor_array = m_trans(torch.from_numpy(wait_to_nor_array)).detach().cpu().numpy()
                else:
                    wait_to_nor_array = tensor_array
                    m_trans = nn.BatchNorm2d(wait_to_nor_array[0].shape[0], momentum=0.99, affine=False)
                    tensor_array = m_trans(torch.from_numpy(wait_to_nor_array)).detach().cpu().numpy()

            # print(wait_to_nor_array[0].shape)

        # print('tensor_array.shape is ', tensor_array.shape)
        outputs = cal_middle_output(model, args.arch, torch.from_numpy(tensor_array), layer_id + 1)
        grade_avg, grade_std, grade_min, grade_max, grade_list, zero_rate = cal_layer_grades(args.g,
                                                                                             outputs.detach().cpu().numpy())
        print_layer_grades_info(layer_id + 1, args.g, grade_avg, grade_std, grade_min, grade_max, zero_rate,
                                args.ts)

if args.op == 'generate_series':

    if args.arch == 'alexnet':
        model = models.__dict__[args.arch](pretrained=True)

        for layer_id in range(0, 8):
            tensor_array = np.load(
                '%s/%s_%s_layer%d.npy' % ((args.arch + '_' + args.dir + '_mid_res'), args.arch, args.dir, layer_id))
            tensor_array = generate_first_order_difference(tensor_array, args.sl)

            if layer_id == 0:
                grade_avg, grade_std, grade_min, grade_max, grade_list, zero_rate = cal_layer_grades(args.g,
                                                                                                     tensor_array)
                print_layer_grades_info(0, args.g, grade_avg, grade_std, grade_min, grade_max, zero_rate, args.ts)

            if args.ts == 'normal':
                tensor_array = tensor_array_normalization(tensor_array)

            outputs = cal_middle_output(model, args.arch, torch.from_numpy(tensor_array), layer_id + 1)
            grade_avg, grade_std, grade_min, grade_max, grade_list, zero_rate = cal_layer_grades(args.g,
                                                                                                 outputs.detach().cpu().numpy())
            print_layer_grades_info(layer_id + 1, args.g, grade_avg, grade_std, grade_min, grade_max, zero_rate,
                                    args.ts)

if args.op == 'from_mid_to_end':

    if args.arch == 'alexnet':

        model = models.__dict__[args.arch](pretrained=True)

        for layer_id in range(0, 8):
            tensor_array = np.load(
                '%s/%s_%s_layer%d.npy' % ((args.arch + '_' + args.dir + '_mid_res'), args.arch, args.dir, layer_id))

            if layer_id == 0:
                grade_avg, grade_std, grade_min, grade_max, grade_list, zero_rate = cal_layer_grades(args.g,
                                                                                                     tensor_array)
                print_layer_grades_info(0, args.g, grade_avg, grade_std, grade_min, grade_max, zero_rate, args.ts)

            if args.ts == 'normal' and layer_id != 0:
                tensor_array = tensor_array_normalization(tensor_array)

            outputs = torch.from_numpy(tensor_array)

            for layer_id_2 in range(layer_id, 8):
                outputs = cal_middle_output(model, args.arch, outputs, layer_id_2 + 1)
                if layer_id_2 == 7:
                    grade_avg, grade_std, grade_min, grade_max, grade_list, zero_rate = cal_layer_grades(args.g,
                                                                                                         outputs.detach().cpu().numpy(),
                                                                                                         False, 255)
                    print_layer_grades_info(layer_id_2 + 1, args.g, grade_avg, grade_std, grade_min, grade_max,
                                            zero_rate,
                                            args.ts)
