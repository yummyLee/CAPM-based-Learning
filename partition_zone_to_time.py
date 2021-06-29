import os
import shutil


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


transform_dir = 'imagenet_2012/transform_s0609_2230'
zone_dir = transform_dir + '/transform_images_ro_noise'
time_dir = transform_dir + '/transform_images_t_ro_noise'
origin_dir = transform_dir + '/origin_images'

mkdir(zone_dir)

for multi_class_item in os.listdir(transform_dir):
    class_dir = 'imagenet_2012/transform_s0609_2230/' + multi_class_item
    shutil.move(class_dir, zone_dir)

mkdir(time_dir)

new_range_list = []
for seq_index in range(200):
    if seq_index < 10:
        seq_index = '000' + str(seq_index)
    elif seq_index < 100:
        seq_index = '00' + str(seq_index)
    elif seq_index < 1000:
        seq_index = '0' + str(seq_index)
    else:
        seq_index = str(seq_index)
    new_range_list.append(seq_index)

for multi_class_item in os.listdir(zone_dir):

    mkdir(origin_dir + '/' + multi_class_item)

    image_items = os.listdir(zone_dir + '/' + multi_class_item)

    image_items_dir = zone_dir + '/' + multi_class_item

    for image_item in image_items:
        mkdir(image_items_dir + '/' + '/' + image_item + '/' + image_item)
        for i in os.listdir(image_items_dir + '/' + image_item):
            if i == image_item:
                continue
            os.rename(image_items_dir + '/' + image_item + '/' + i,
                      image_items_dir + '/' + image_item + '/' + image_item + '/' + i)

    count = 0

    for image_item in image_items:

        if image_item != 'moto9':
            continue

        file_list = os.listdir(image_items_dir + '/' + image_item + '/' + image_item)

        origin_pre = 'moto09' + '_g'

        for i in range(0, 200):
            os.rename(
                image_items_dir + '/' + image_item + '/' + image_item + '/' + origin_pre +
                new_range_list[i] + '.jpg',
                image_items_dir + '/' + image_item + '/' + image_item + '/' + image_item + '_' +
                new_range_list[i] + '.jpg')

    for i in range(190):
        mkdir(time_dir + '/' + multi_class_item + '/' + new_range_list[i] + '/' + new_range_list[i])
        for image_item in image_items:
            shutil.copy(
                image_items_dir + '/' + image_item + '/' + image_item + '/' + image_item + '_' +
                new_range_list[i] + '.jpg',
                time_dir + '/' + multi_class_item + '/' + new_range_list[i] + '/' + new_range_list[
                    i] + '/' + image_item + '_' +
                new_range_list[i] + '.jpg')

# for i in range(2, 20):
#     mkdir('D:/n02690373/air%d' % i)
