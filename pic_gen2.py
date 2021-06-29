import os
import random
import numpy as np

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def sp_noise(image):
    """
    添加椒盐噪声
    prob:噪声比例
    """
    prob = 0.2
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                image[i][j] = 0
            elif rdn > thres:
                image[i][j] = 255

    return image


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
                     fill_mode='wrap',
                     preprocessing_function=sp_noise)

datagen = ImageDataGenerator(**data_gen_args)

for file_name in os.listdir('transform_random_test_dir/0'):
    img = load_img('transform_random_test_dir/0/' + file_name)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    i = 1
    j = 1
    for batch in datagen.flow(x,
                              batch_size=32,
                              save_to_dir='transform_random_images',
                              save_prefix='car',
                              save_format='png'):
        i += 1
        if i > 15:
            break
