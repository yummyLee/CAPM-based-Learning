import random

import cv2
import scipy.ndimage
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import math
# import xlrd
# from xlrd import open_workbook
# from xlutils.copy import copy
# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# from statsmodels.tsa.stattools import coint
# import scipy.misc as misc
import scipy.ndimage as ndimage
# from keras.preprocessing import image

from scipy.spatial.distance import cosine


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")

    is_exists = os.path.exists(path)

    if not is_exists:
        try:
            os.makedirs(path)
        except FileExistsError:
            return True
        # print(path + ' success')
        return True
    else:
        # print(path + ' existed')
        return False


def translateit(image, offset, mode, isseg=False):
    order = 0 if isseg else 5
    return scipy.ndimage.interpolation.shift(image, (int(offset[0]), int(offset[1]), 0), order=order, mode=mode)


def scaleit(image, factor, mode, isseg=False):
    order = 3 if isseg else 0

    height, width, depth = image.shape

    zheight = int(np.round(factor * height))

    zwidth = int(np.round(factor * width))

    zdepth = depth

    if factor < 1.0:

        newimg = np.zeros_like(image)

        row = (height - zheight) // 2

        col = (width - zwidth) // 2

        layer = (depth - zdepth) // 2

        newimg[row:row + zheight, col:col + zwidth, layer:layer + zdepth] = scipy.ndimage.interpolation.zoom(image, (
            float(factor), float(factor), 1.0), order=order, mode=mode)[0:zheight, 0:zwidth, 0:zdepth]

        return newimg

    elif factor > 1.0:

        row = (zheight - height) // 2

        col = (zwidth - width) // 2

        layer = (zdepth - depth) // 2

        newimg = scipy.ndimage.interpolation.zoom(image[row:row + zheight, col:col + zwidth, layer:layer + zdepth],
                                                  (float(factor), float(factor), 1.0), order=order, mode=mode)

        extrah = (newimg.shape[0] - height) // 2

        extraw = (newimg.shape[1] - width) // 2

        extrad = (newimg.shape[2] - depth) // 2

        newimg = newimg[extrah:extrah + height, extraw:extraw + width, extrad:extrad + depth]

        return newimg

    else:

        return image


def rotateit(image, theta, mode, isseg=False):
    order = 0 if isseg else 5

    return scipy.ndimage.rotate(image, float(theta), reshape=False, order=order, mode=mode)


def intensifyit(image, factor):
    return image * float(factor)


def flipit(image, axes):
    if axes[0]:
        image = np.fliplr(image)

    if axes[1]:
        image = np.flipud(image)

    return image


def g_blur(m_img, rate):
    # print((rate + 1) * 3 + 0.001)
    img = cv2.GaussianBlur(m_img, (13, 13), rate)
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    return img, 'RGB'


def change_channel(m_img, channel_id, rate1):
    m_img2 = np.zeros_like(m_img)

    try:
        chs = Image.fromarray(m_img).split()
        for i in range(3):
            #     m_img2[..., i] = np.asarray(chs[i]) * rate1
            if i != channel_id:
                m_img2[..., i] = np.asarray(chs[i])
            else:
                m_img2[..., channel_id] = np.asarray(chs[channel_id]) * rate1
        return m_img2, 'RGB'
    except ValueError:
        m_img2 = m_img * rate1
        return m_img2, 'F'


def change_channel_2(m_img, channel_id, rate1, rate2, rate3):
    try:
        chs = Image.fromarray(np.uint8(m_img)).split()
        if channel_id == 0:
            m_img[..., channel_id] = np.asarray(chs[channel_id]) * rate1
        elif channel_id == 1:
            m_img[..., channel_id] = np.asarray(chs[channel_id]) * rate2
        elif channel_id == 2:
            m_img[..., channel_id] = np.asarray(chs[channel_id]) * rate3
        return m_img
    except ValueError:
        m_img = m_img * rate1
        return m_img


def change_channel_test():
    img_path = 'imagenet_2012/transform_314_2/imagenet_m/n04536866/n04536866/n04536866_238.JPEG'
    img1 = Image.open(img_path)

    img1 = np.asarray(img1)

    print('m_img2 shape: ', img1.shape)


# change_channel_test()


def change_one_channel(m_img, channel_id, rate):
    chs = Image.fromarray(m_img).split()
    m_img2 = np.zeros_like(m_img)
    for i in range(3):
        if i == channel_id:
            m_img2[..., channel_id] = np.asarray(chs[channel_id]) * rate
        else:
            m_img2[..., i] = np.asarray(chs[i])
    return m_img2


def sp_noise(image, prob):
    """
    添加椒盐噪声
    prob:噪声比例
    """
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def gasuss_noise(image, mean=0, var=0.001):
    """
        添加高斯噪声
        mean : 均值
        var : 方差
    """
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    # cv.imshow("gasuss", out)
    return out


def img_transform(m_img1, m_img2):
    modes = ['wrap', 'nearest', 'reflect']

    r1 = random.randint(0, 50)
    r2 = random.randint(0, 50)
    r3 = random.randint(0, 2)
    m_img1 = translateit(m_img1, [r1, r2], mode=modes[r3])
    m_img2 = translateit(m_img2, [r1, r2], mode=modes[r3])

    r4 = random.randint(0, 360)
    r5 = random.randint(0, 2)
    m_img1 = rotateit(m_img1, r4, mode=modes[r5])
    m_img2 = rotateit(m_img2, r4, mode=modes[r5])

    if random.choice([True, False]):
        r6 = random.choice([True, False])
        r7 = random.choice([True, False])
        m_img1 = flipit(m_img1, [r6, r7])
        m_img2 = flipit(m_img2, [r6, r7])

    if random.random() > 0.4:
        r8 = random.randint(0, 3)
        r91 = random.uniform(0.3, 1.2)
        r92 = random.uniform(0.3, 1.2)
        r93 = random.uniform(0.3, 1.2)
        m_img1 = change_channel_2(m_img1, r8, r91, r92, r93)
        m_img2 = change_channel_2(m_img2, r8, r91, r92, r93)

    if random.random() > 0.4:
        r10 = random.uniform(1, 1.3)
        r11 = random.randint(0, 2)
        m_img1 = scaleit(m_img1, r10, mode=modes[r11])
        m_img2 = scaleit(m_img2, r10, mode=modes[r11])

    if random.random() > 0.4:
        r12 = random.uniform(0.005, 0.009)
        m_img1 = sp_noise(m_img1, r12)
        m_img2 = sp_noise(m_img2, r12)

    # if random.random() > 0.7:
    #     r13 = random.uniform(0, 0.001)
    #     m_img1 = gasuss_noise(m_img1, 0, r13)
    #     m_img2 = gasuss_noise(m_img2, 0, r13)

    return m_img1, m_img2


def gen_img_transform_param():
    m_rs = [random.randint(0, 80), random.randint(0, 80), random.randint(0, 2), random.randint(0, 360),
            random.randint(0, 2), random.choice([True, False]), random.choice([True, False]),
            random.choice([True, False]), random.random(), random.randint(0, 3), random.uniform(0.4, 1),
            random.uniform(0.4, 1), random.uniform(0.4, 1), random.random(), random.uniform(1, 1.3),
            random.randint(0, 2), random.random(), random.uniform(0.001, 0.007)]

    return m_rs


def img_transform_m(rs, m_img):
    modes = ['wrap', 'nearest', 'reflect', 'constant']
    #
    # m_img = translateit(m_img, [rs[0], rs[1]], mode=modes[rs[2]])
    # # print('1', type(m_img))
    m_img = rotateit(m_img, rs[3], mode=modes[rs[4]])
    # # print('2', type(m_img))
    # if rs[5]:
    #     m_img = flipit(m_img, [rs[6], rs[7]])
    #     # print('3', type(m_img))
    if rs[8] > 0.4:
        m_img = change_channel_2(m_img, rs[9], rs[10], rs[11], rs[12])
    #     # print('4', type(m_img))
    if rs[13] > 0.4:
        m_img = scaleit(m_img, rs[14], mode=modes[rs[15]])
        # print('5', type(m_img))
    # if rs[16] > 0.4:
    #     m_img = sp_noise(m_img, rs[17])

    return m_img


# def get_data_from_xls(file_path, sheet, op, index1, index2=0):
#     # file_path = r'F:/test.xlsx'
#     # 文件路径的中文转码，如果路径非中文可以跳过
#     # file_path = file_path.decode('utf-8')
#     # 获取数据
#     data = xlrd.open_workbook(file_path)
#     # 获取sheet 此处有图注释（见图1）
#     table = data.sheet_by_name(sheet)
#
#     # 获取总行数
#     nrows = table.nrows
#
#     # 获取总列数
#     ncols = table.ncols
#
#     # 获取一行的数值，例如第5行
#     # rowvalue = table.row_values(281)
#     #
#     # # 获取一列的数值，例如第6列
#
#     result_data = None
#
#     if op == 'col':
#         result_data = table.col_values(index1, 1, 361)
#     #
#     # # 获取一个单元格的数值，例如第5行第6列
#     # cell_value = table.cell(5, 6).value
#
#     # print(rowvalue)
#     # print(type(result_data))
#     return result_data


# plt.imshow(img)


def gen_sin(num):
    angle_list = np.arange(0, math.pi, math.pi / num)
    angle_list_sin = []
    for angle in angle_list:
        angle_list_sin.append(math.sin(angle))

    return angle_list_sin


def gen_sin2():
    step = 7
    angle_list = np.arange(-math.pi / 2, step * 6 * math.pi - math.pi / 2, step * 6 * math.pi / 1080)
    angle_list_sin = []
    for angle in angle_list:
        angle_list_sin.append(math.sin(angle))

    return angle_list_sin


def gen_linear():
    x = np.arange(1080)
    delta = np.random.uniform(0, 0.33, size=(1080,))
    y = 1 / 1080 * x + delta
    # return delta
    return y


def gen_linear_no_noise():
    x = np.arange(1080)
    y = 1 / 1080 * x
    # return delta
    return y


def gen_gb_seq():
    if os.path.exists('gb_seq.npy'):
        return np.load('gb_seq.npy')

    y = (np.array(gen_sin2()) + 1) / 2 + gen_linear()
    # y = (np.array(gen_sin2()) + 1) / 2
    y = y * 2.5 - min(y)

    np.save('gb_seq.npy', y)

    return y


def gen_gb_seq_no_noise():
    x = range(1080)
    # y = gen_linear()
    y = (np.array(gen_sin2()) + 1) / 2 + gen_linear_no_noise()
    # y = (np.array(gen_sin2()) + 1) / 2
    y = y * 2.5 - min(y)
    return y


def gen_gb_seq_no_linear_no_noise():
    x = range(1080)
    # y = gen_linear()
    y = (np.array(gen_sin2()) + 1) / 2
    # y = (np.array(gen_sin2()) + 1) / 2
    y = y * 2.5 - min(y)
    return y


def gen_gb_seq_no_linear():
    x = range(1080)
    # y = gen_linear()
    delta = np.random.uniform(0, 0.33, size=(1080,))
    y = (np.array(gen_sin2()) + 1) / 2 + delta
    # y = (np.array(gen_sin2()) + 1) / 2
    y = y * 2.5 - min(y)
    return y


def gen_ac_seq():
    if os.path.exists('ac_seq.npy'):
        return np.load('ac_seq.npy')

    x = range(1080)
    # y = gen_linear()
    y = (np.array(gen_sin2()) + 1) / 2 + gen_linear()
    y = (y - min(y)) / (max(y) - min(y))

    np.save('ac_seq.npy', y)

    return y


def gen_ro_seq():
    if os.path.exists('ro_seq.npy'):
        return np.load('ro_seq.npy')

    x = range(1080)
    # y = gen_linear()
    y = (np.array(gen_sin2()) + 1) / 2 + gen_linear()
    y = (y - min(y)) / (max(y) - min(y)) * 360

    np.save('ro_seq.npy', y)

    return y


def gen_sin_pic(operation):
    num = 360

    angle_list_sin = gen_sin(num)

    img1 = Image.open('imagenet-cat1/cat/imagenet-cat1.jpg')
    img2 = Image.open('imagenet-dog1/dog/imagenet-dog1.jpg')
    img3 = Image.open('imagenet-dog2/dog/imagenet-dog2.jpg')
    img4 = Image.open('imagenet-cat2/cat/imagenet-cat2.jpg')
    img1 = np.asarray(img1)
    img2 = np.asarray(img2)
    img3 = np.asarray(img3)
    img4 = np.asarray(img4)
    imgs = [img1, img2, img3, img4]

    mkdir('imagenet-cat1/imagenet-cat1-%s_sin/1/' % operation)
    mkdir('imagenet-dog1/imagenet-dog1-%s_sin/0/' % operation)
    mkdir('imagenet-cat2/imagenet-cat2-%s_sin/1/' % operation)
    mkdir('imagenet-dog2/imagenet-dog2-%s_sin/0/' % operation)

    for i in range(0, num):
        m_imgs = []

        for j in range(0, 4):
            if operation == 'rotate':
                m_imgs.append(rotateit(imgs[j], theta=angle_list_sin[i] * 360, mode='wrap'))
            elif operation == 'channel_change':
                m_imgs.append(change_one_channel(imgs[j], 0, angle_list_sin[i]))
            elif operation == 'all_channel_change':
                m_imgs.append(change_channel(imgs[j], angle_list_sin[i]))

            m_imgs[j] = Image.fromarray(m_imgs[j])

        if i < 10:
            i = '00' + str(i)
        elif i < 100:
            i = '0' + str(i)

        m_imgs[0].save('imagenet-cat1/imagenet-cat1-%s_sin/1/cat1-%s.jpg' % (operation, i))
        m_imgs[1].save('imagenet-dog1/imagenet-dog1-%s_sin/0/dog1-%s.jpg' % (operation, i))
        m_imgs[2].save('imagenet-dog2/imagenet-dog2-%s_sin/0/dog2-%s.jpg' % (operation, i))
        m_imgs[3].save('imagenet-cat2/imagenet-cat2-%s_sin/1/cat2-%s.jpg' % (operation, i))


# gen_sin_pic('all_channel_change')
# gen_sin(360)

# get_data_from_xls('alexnet_imagenet-cat1-all_channel_change_sin_layer8.xlsx', 'page_1', 'col', 282)


# print(len(sin_list))
# print(len(data_list))


def tensor_array_normalization(wait_norm_tensors):
    # print('---- TENSOR ARRAY NORMALIZTION ----')

    max_i = np.max(wait_norm_tensors)
    min_i = np.min(wait_norm_tensors)

    return (wait_norm_tensors - min_i) / (max_i - min_i)


# def jud_coint():
#     for i in range(280, 1002):
#         sin_list = gen_sin(360)
#         sin_list = np.array(sin_list)
#         # print(sin_list)
#         data_list = get_data_from_xls('alexnet_imagenet-cat1-all_channel_change_sin_layer8.xlsx', 'page_1', 'col', i)
#         print(data_list)
#         data_list = tensor_array_normalization(data_list)
#
#         print('%d - cos_sim: ' % i, cosine(data_list, sin_list))


# jud_coint()

def rotate2(m_img, angle):
    return ndimage.rotate(m_img, angle=angle, mode='nearest'), 'RGB'


def transform(img_path, save_path, operation, t_num):
    mkdir(save_path)

    # print('save')

    if operation == 'sf' or operation == 'sc':
        img1 = Image.open(img_path)
        img1_shape = np.asarray(img1).shape
        h = img1_shape[0]
        w = img1_shape[1]

        frame_points = []

        if operation == 'sf':
            window_h = h / 1.33
            window_w = w / 1.33

            # print(window_w)
            # print(w - window_h / 2)

            center_bound_points = [[window_h / 2, window_w / 2], [window_h / 2, w - window_w / 2],
                                   [h - window_h / 2, w - window_w / 2], [h - window_h / 2, window_w / 2]]

            center_points = []

            # t_num = 40

            for i in np.arange(center_bound_points[0][1], center_bound_points[1][1],
                               (center_bound_points[1][1] - center_bound_points[0][1]) / t_num):
                center_points.append([center_bound_points[0][0], i])

            for i in np.arange(center_bound_points[1][0], center_bound_points[2][0],
                               (center_bound_points[2][0] - center_bound_points[1][0]) / t_num):
                center_points.append([i, center_bound_points[1][1]])

            for i in np.arange(center_bound_points[2][1], center_bound_points[3][1],
                               (center_bound_points[3][1] - center_bound_points[2][1]) / t_num):
                center_points.append([center_bound_points[2][0], i])

            for i in np.arange(center_bound_points[3][0], center_bound_points[0][0],
                               (center_bound_points[0][0] - center_bound_points[3][0]) / t_num):
                center_points.append([i, center_bound_points[3][1]])

            for i in center_points:
                frame_points.append(
                    (
                        int(i[1] - window_w / 2), int(i[0] - window_h / 2), int(i[1] + window_w / 2),
                        int(i[0] + window_h / 2)))

        if operation == 'sc':
            center_points = [h / 2, w / 2]
            frame_half_wh = [0.6 * h / 2, 0.6 * w / 2]

            for rate in np.arange(1, 1 / 0.6, (1 / 0.6 - 1) / t_num):
                frame_point = (int(center_points[1] - frame_half_wh[1] * rate),
                               int(center_points[0] - frame_half_wh[0] * rate),
                               int(center_points[1] + frame_half_wh[1] * rate),
                               int(center_points[0] + frame_half_wh[0] * rate))
                if frame_point[0] < 0 or frame_point[1] < 0 or frame_point[2] > w or frame_point[3] > h:
                    break

                frame_points.append(frame_point)

        for i in range(len(frame_points)):
            m_imgs = img1.crop(frame_points[i])
            save_path_split = save_path.split('/')

            if i < 10:
                i = '000' + str(i)
            elif i < 100:
                i = '00' + str(i)
            elif i < 1000:
                i = '0' + str(i)
            else:
                i = str(i)

            mkdir(save_path_split[0] + '/' + save_path_split[1] + '/transform_images_t_%s_noise/' % operation +
                  save_path_split[-2].split('_')[0] + '/%s/%s' % (i, i))

            m_imgs.save(save_path + '/%s.jpg' % i)

            m_imgs.save(
                save_path_split[0] + '/' + save_path_split[1] + '/transform_images_t_%s_noise/' % operation +
                save_path_split[-2].split('_')[0] + '/%s/%s/%s_%s.jpg' % (i, i, save_path_split[-1], i))

        return 'RGB'

    angle_list = gen_gb_seq()
    angle_list_sin = gen_ac_seq()
    angle_list_ro = gen_ro_seq()

    # print('save')

    for i in range(0, t_num):
        m_imgs = None
        mode = None

        # print('save')

        if operation == 'ac':
            img1 = Image.open(img_path)
            img1 = np.asarray(img1)

            # if len(img1.shape) < 3:
            #     return 'L'

            m_imgs, mode = change_channel(img1, 0, angle_list_sin[i])
        elif operation == 'gb':
            img1 = cv2.imread(img_path)
            m_imgs, mode = g_blur(img1, angle_list[i])

        elif operation == 'ro':
            img1 = cv2.imread(img_path)
            m_imgs, mode = rotate2(img1, angle_list_ro[i])

            # if mode == 'RGB':
        m_imgs = Image.fromarray(m_imgs)
        if mode != 'RGB':
            m_imgs = m_imgs.convert('RGB')

        if i < 10:
            i = '000' + str(i)
        elif i < 100:
            i = '00' + str(i)
        elif i < 1000:
            i = '0' + str(i)
        else:
            i = str(i)

        save_path_split = save_path.split('/')

        # mkdir(save_path_split[0] + '/' + save_path_split[1] + '/transform_images_t_gb_noise/' +
        #       save_path_split[-2].split('_')[0] + '/%s/%s' % (i, i))
        #
        # m_imgs.save(save_path + '/%s.jpg' % i)
        #
        # m_imgs.save(
        #     save_path_split[0] + '/' + save_path_split[1] + '/transform_images_t_gb_noise/' +
        #     save_path_split[-2].split('_')[0] + '/%s/%s/%s_%s.jpg' % (i, i, save_path_split[-1], i))

        m_imgs.save(save_path + '/%s.jpg' % i, mode=mode)

        # print(1111)

        mkdir(save_path_split[0] + '/' + save_path_split[1] + '/transform_images_t_%s_noise/' % operation +
              save_path_split[-2].split('_')[0] + '/%s/%s' % (i, i))
        m_imgs.save(
            save_path_split[0] + '/' + save_path_split[1] + '/transform_images_t_%s_noise/' % operation +
            save_path_split[-2].split('_')[0] + '/%s/%s/%s_%s.jpg' % (i, i, save_path_split[-1], i))

    return 'RGB'


def transform_random_one_image2(image_file, class_name, save_path):
    img1 = Image.open(image_file)
    img1 = np.asarray(img1)
    data_gen_args = dict(featurewise_center=True,
                         featurewise_std_normalization=False,
                         rotation_range=360,
                         width_shift_range=[0.025, 0.025],
                         height_shift_range=[0.025, 0.025],
                         zoom_range=[0.92, 1.33],
                         horizontal_flip=True,
                         shear_range=0.2,
                         vertical_flip=True,
                         channel_shift_range=85,
                         fill_mode='wrap')

    data_gen = image.ImageDataGenerator(**data_gen_args)

    for i in range(100):
        transform_parameters = data_gen.get_random_transform(img1.shape)
        transform_image = data_gen.apply_transform(img1, transform_parameters)
        rs = gen_img_transform_param()
        print(i)
        m_imgs = img_transform_m(rs, transform_image)
        m_imgs = Image.fromarray(np.uint8(m_imgs))
        m_imgs.save(save_path + '/%s_%d.jpg' % (class_name, i))


def random_transform_one_img(img_path_name, class_name, save_path):
    img1 = Image.open(img_path_name)
    img1 = np.asarray(img1)
    # if len(img1.shape) < 3:
    #     return
    for i in range(0, 100):
        print('===' + save_path + '/%s_%d.jpg' % (class_name, i) + '===')
        rs = gen_img_transform_param()
        m_imgs = img_transform_m(rs, img1)
        m_imgs = Image.fromarray(m_imgs)
        m_imgs.save(save_path + '/%s_%d.jpg' % (class_name, i))


operation = 'gb'

# transform('ILSVRC2012_val_00009796.JPEG', 'test_black', operation, 100)

# random_transform_one_img('n03982430_2153.JPEG', 'test_black')

x = range(1080)
y = gen_gb_seq()
# y = gen_gb_seq_no_linear_no_noise()
# y = gen_gb_seq_no_noise()
#
# # x1 = np.arange(1080)
# # y1 = np.zeros_like(x1)
# # for i in x1:
# #     y1[i] = 6 + i / 50 + 6 * math.sin(7 * math.pi * i / 180)
plt.plot(x, y, color="black")
plt.xlabel("波动幅度", fontsize=21)
plt.ylabel("时间", fontsize=21)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.legend(prop={'size': 21})
# plt.plot(x1, y1)
plt.show()

# transform_random_one_image2('n03982430_2153.JPEG', 'nnnn', 'test_black')
