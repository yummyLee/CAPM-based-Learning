import argparse
import os
import random
import shutil
import time
import warnings
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import Image
import numpy as np
from collections import OrderedDict
import sys
import pymongo
import pickle
from bson.binary import Binary
import argparse

import calculate2

MONGODB_HOST = '127.0.0.1'
# 端口号，默认27017
MONGODB_PORT = 27017
# 设置数据库名称
MONGODB_DBNAME = 'alexnet'
# 存放本数据的表名称
MONGODB_COLNAME = 'one_dog'

model_pth_name = 'cat_dog_alexnet_model_best.pth'
model_type = 'alexnet'

parser = argparse.ArgumentParser()
parser.add_argument('-op', default='test', type=str, help='operation')
parser.add_argument('-dir', default='', type=str, help='image dir for the operation')
parser.add_argument('-b', default=256, type=int, help='batch size')

args = parser.parse_args()


def mkdir(path):
    import os
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        # print(path + ' success')
        return True
    else:
        # print(path + ' existed')
        return False


# alexmodel = models.alexnet()
#
# state_dict = torch.load(model_pth_name)['state_dict']
#
# # print(state_dict)
#
# new_state_dict = OrderedDict()
#
# for k, v in state_dict.items():
#     k = k.replace('module.', '')
#     new_state_dict[k] = v
#
# alexmodel.load_state_dict(new_state_dict)

alexmodel = models.vgg16(pretrained=True)

# print(alexmodel)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

m_batch_size = args.b
m_image_dir = '/home/yaming/yummy/imagenet_2012/single_val/n03876231'

# if len(sys.argv) > 1:
#     if sys.argv[1] == 'test':
#         if len(sys.argv) > 2:
#             m_image_dir = sys.argv[2]
#         else:
#             exit()
#     elif sys.argv[1] == 'get_layer':
#         if len(sys.argv) > 2:
#             m_image_dir = sys.argv[2]
#         if len(sys.argv) > 3:
#             MONGODB_COLNAME = sys.argv[3]
#     else:
#         exit()

client = pymongo.MongoClient(host=MONGODB_HOST, port=MONGODB_PORT)
db = client[MONGODB_DBNAME]

m_save_image_dir = m_image_dir + '_pic'

test_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(m_image_dir, transforms.Compose([  # [1]
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])),
    batch_size=m_batch_size, shuffle=False,
    num_workers=2, pin_memory=True)

# test_loader_un_normalize = torch.utils.data.DataLoader(
#     datasets.ImageFolder(m_image_dir, transforms.Compose([
#         transforms.Resize((227, 227)),
#         transforms.ToTensor()
#     ])),
#     batch_size=m_batch_size, shuffle=False,
#     num_workers=2, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss().cuda(None)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        # print(_)
        pred = pred.t()
        # print(pred)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def validate(val_loader, model, criterion, device):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    target = int(calculate2.imagenet_class_index_dic()['n03876231'])

    print(type(target))

    with torch.no_grad():
        end = time.time()
        for i, (input, _) in enumerate(val_loader):

            target = torch.Tensor([target for i in range(np.shape(input)[0])])

            input = input.cpu()
            target = target.cpu()

            # compute output
            output = model(input)

            # loss = criterion(output, target)
            # print(target)

            # print('target size is ', target.size())
            # print('output size is ', output.size())

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            # losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))


def m_validate(val_loader, model, criterion, device):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cpu()
            target = target.cpu()

            # compute output
            output = model(input)

            loss = criterion(output, target)

            # print('target size is ', target.size())
            # print('output size is ', output.size())

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))


if args.op == 'test':
    validate(test_loader, alexmodel, criterion, None)

# if args.op == 'get_layer':
#
#     for i, (image_input, target) in enumerate(test_loader_un_normalize):
#         # inputs, labels = iter(train_loader).next()
#
#         image_input, target = image_input.to(device), target.to(device)
#         inputs = image_input.cpu()
#
#         np.save('%s-%s-layer-1_batch%d' % (model_type, m_image_dir, i), inputs.numpy())
#
#         # for image_id in range(len(inputs)):
#         #     for zero_channel in range(3):
#         #         print(
#         #             'current source image is %d, layer is %d, channel is %d' % (image_id + i * m_batch_size + 1,
#         #                                                                         -1, zero_channel + 1))
#         #         # print(inputs.numpy()[image_id][zero_channel])
#         #
#         #         data = dict()
#         #         data['image_id'] = image_id + i * m_batch_size + 1
#         #         data['layer_id'] = -1
#         #         data['channel_id'] = zero_channel + 1
#         #         data['numpy_matrix'] = Binary(pickle.dumps(inputs.numpy()[image_id][zero_channel]))
#         #
#         #         db.one_dog.insert_one(data)
#
#     batch_index = 0
#
#     for i, (image_input, target) in enumerate(test_loader):
#         # inputs, labels = iter(train_loader).next()
#
#         image_input, target = image_input.to(device), target.to(device)
#
#         inputs = image_input.cpu()
#
#         np.save('%s-%s-layer0_batch%d' % (model_type, m_image_dir, i), inputs.numpy())
#
#         # for image_id in range(len(inputs)):
#         #
#         #     mkdir('./%s/x%d' % (m_save_image_dir, i * m_batch_size + image_id + 1))
#         #
#         #     Image.fromarray(((inputs.numpy()[image_id][0] * 0.229 + 0.485) * 255).astype('uint8'), mode='L').save(
#         #         './%s/x%d/1.jpeg' % (m_save_image_dir, i * m_batch_size + image_id + 1))
#         #     Image.fromarray(((inputs.numpy()[image_id][1] * 0.224 + 0.456) * 255).astype('uint8'), mode='L').save(
#         #         './%s/x%d/0.jpeg' % (m_save_image_dir, i * m_batch_size + image_id + 1))
#         #     Image.fromarray(((inputs.numpy()[image_id][2] * 0.229 + 0.485) * 255).astype('uint8'), mode='L').save(
#         #         './%s/x%d/2.jpeg' % (m_save_image_dir, i * m_batch_size + image_id + 1))
#         #
#         #     rgb_array = np.zeros((227, 227, 3), 'uint8')
#         #     rgb_array[..., 0] = (inputs.numpy()[image_id][0] * 0.229 + 0.485) * 255
#         #     rgb_array[..., 1] = (inputs.numpy()[image_id][1] * 0.224 + 0.456) * 255
#         #     rgb_array[..., 2] = (inputs.numpy()[image_id][2] * 0.225 + 0.406) * 255
#         #     Image.fromarray(rgb_array, mode='RGB').save('./%s/x%d/rgb.jpeg' % (m_save_image_dir,
#         #                                                                        i * m_batch_size + image_id + 1))
#         #
#         #     for zero_channel in range(3):
#         #         print(
#         #             'current image is %d, layer is %d, channel is %d' % (image_id + i * m_batch_size + 1,
#         #                                                                  0, zero_channel + 1))
#         #         # print(inputs.numpy()[image_id][zero_channel])
#         #
#         #         # save to database
#         #
#         #         data = dict()
#         #         data['image_id'] = image_id + i * m_batch_size + 1
#         #         data['layer_id'] = 0
#         #         data['channel_id'] = zero_channel + 1
#         #         data['numpy_matrix'] = Binary(pickle.dumps(inputs.numpy()[image_id][zero_channel]))
#         #
#         #         db.one_dog.insert_one(data)
#
#         model_f = alexmodel.features
#         model_a = alexmodel.avgpool
#         model_c = alexmodel.classifier
#
#         x = []
#
#         x.append(model_f[2](model_f[1](model_f[0](image_input.cpu()).cpu()).cpu()).cpu())
#         x.append(model_f[5](model_f[4](model_f[3](x[0]).cpu()).cpu()).cpu())
#         x.append(model_f[7](model_f[6](x[1]).cpu()).cpu())
#         x.append(model_f[9](model_f[8](x[2]).cpu()).cpu())
#         x.append(model_f[12](model_f[11](model_f[10](x[3]).cpu()).cpu()).cpu())
#
#         # print(x[4].shape)
#         x4 = x[4].view(-1, 9216)
#
#         x.append(model_c[2](model_c[1](model_c[0](x4).cpu()).cpu()))
#         x.append(model_c[5](model_c[4](model_c[3](x[5]).cpu()).cpu()).cpu())
#         x.append(model_c[6]((x[6])).cpu())
#
#         # print('x-length', len(x))
#         for layer_id in range(len(x)):
#
#             print('current layer is %d, size is ', (layer_id, x[layer_id].size()))
#
#             for image_id in range(len(x[layer_id])):
#                 print('current image channel length', x[layer_id][image_id].size())
#
#                 p = x[layer_id].detach().cpu().numpy()
#                 np.save('%s-%s-layer%d_batch%d' % (model_type, m_image_dir, layer_id, i), p)
#
#                 # if layer_id < 5:
#                 #
#                 #     for ii in range(len(x[layer_id][image_id])):
#                 #         # print('sava pic%d.' % (i + 1))
#                 #         pic = p[image_id][ii]
#                 #         # print('The pic is:\n', pic)
#                 #
#                 #         # Image.fromarray(((pic * 0.226 + 0.449) * 255).astype('uint8'), mode='L').save(
#                 #         #     './%s/x%d/layer%d-%d.jpeg' % (m_save_image_dir, (image_id + i * m_batch_size + 1),
#                 #         #                                   layer_id + 1, (ii + 1)))
#                 #         print(
#                 #             'current image is %d, layer is %d, channel is %d' % (image_id + i * m_batch_size + 1,
#                 #                                                                  layer_id + 1, ii + 1))
#                 #         # print('current pic size is', p[image_id][ii].size())
#                 #         data = dict()
#                 #         data['image_id'] = image_id + i * m_batch_size + 1
#                 #         data['layer_id'] = layer_id + 1
#                 #         data['channel_id'] = ii + 1
#                 #         data['numpy_matrix'] = Binary(pickle.dumps(pic))
#                 #
#                 #         db.one_dog.insert_one(data)
#                 #
#                 # else:
#                 #
#                 #     print(
#                 #         'current image is %d, layer is %d, channel is %d' % (image_id + i * m_batch_size + 1,
#                 #                                                              layer_id + 1, 1))
#                 #     data = dict()
#                 #     data['image_id'] = image_id + i * m_batch_size + 1
#                 #     data['layer_id'] = layer_id + 1
#                 #     data['channel_id'] = 1
#                 #     data['numpy_matrix'] = Binary(pickle.dumps(p[image_id]))
#                 #
#                 #     db.one_dog.insert_one(data)
#                 #     # print(p.shape)
