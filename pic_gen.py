import os
import random

import matplotlib.pyplot as plt
from PIL import Image
from keras.preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import numpy as np


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


def transform_random_one_image(image_file, class_name, save_path):
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
        print(i)
        m_imgs = Image.fromarray(np.uint8(transform_image))
        m_imgs.save(save_path + '/%s_%d.jpg' % (class_name, i))


transform_random_one_image('n03982430_2153.JPEG', 'nnnn', 'test_black')


def transform_random_image(image_file, save_path):
    print('=====transform_random_image=====')

    class_img_num = len(os.listdir(image_file))

    mkdir(save_path)

    data_gen_args = dict(featurewise_center=True,
                         featurewise_std_normalization=False,
                         # rotation_range=[360],
                         width_shift_range=[0.025, 0.025],
                         height_shift_range=[0.025, 0.025],
                         zoom_range=[0.92, 1.33],
                         horizontal_flip=True,
                         shear_range=0.2,
                         vertical_flip=True,
                         channel_shift_range=85,
                         fill_mode='wrap')
    fill_mode = ['wrap', 'nearest', 'reflect']

    for i in range(3):

        data_gen_args['fill_mode'] = fill_mode[i]

        datagen = image.ImageDataGenerator(**data_gen_args)

        gen_data = datagen.flow_from_directory(image_file,
                                               batch_size=20,
                                               shuffle=False,
                                               save_to_dir=save_path,
                                               save_prefix='gen',
                                               target_size=(227, 227))

        for ii in range(class_img_num):
            gen_data.next()

        # data_gen = ImageDataGenerator(**data_gen_args)
        #
        # gen = data_gen.flow_from_directory(image_file, target_size=(224, 224), batch_size=2, save_to_dir=save_path,
        #                                    save_prefix='xx', save_format='jpg')
        #
        # for i in range(3):
        #     gen.next()

# img_file_path = 'transform_random_test_dir'
# img_save_path = 'n04251144_10020_random_transform'
# transform_random_image(img_file_path, img_save_path)
