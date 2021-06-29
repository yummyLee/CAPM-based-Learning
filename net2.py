# !/usr/bin/python
# coding: utf8
# @Time    : 2018-08-04 19:23
# @Author  : Liam
# @Email   : luyu.real@qq.com
# @Software: PyCharm
#                        .::::.
#                      .::::::::.
#                     :::::::::::
#                  ..:::::::::::'
#               '::::::::::::'
#                 .::::::::::
#            '::::::::::::::..
#                 ..::::::::::::.
#               ``::::::::::::::::
#                ::::``:::::::::'        .:::.
#               ::::'   ':::::'       .::::::::.
#             .::::'      ::::     .:::::::'::::.
#            .:::'       :::::  .:::::::::' ':::::.
#           .::'        :::::.:::::::::'      ':::::.
#          .::'         ::::::::::::::'         ``::::.
#      ...:::           ::::::::::::'              ``::.
#     ```` ':.          ':::::::::'                  ::::..
#                        '.:::::'                    ':'````..
#                     美女保佑 永无BUG
import argparse
import os
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import net

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

args = parser.parse_args()

f = 'imagenet_2012/train'
v = 'imagenet_2012/val'
t = 'imagenet_2012/' + args.tdir
r = 8

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

# 定义一些超参数
batch_size = 64
learning_rate = 0.005
num_epoches = 200

# 数据预处理。transforms.ToTensor()将图片转换成PyTorch中处理的对象Tensor,并且进行标准化（数据在0~1之间）
# transforms.Normalize()做归一化。它进行了减均值，再除以标准差。两个参数分别是均值和标准差
# transforms.Compose()函数则是将各种预处理的操作组合到了一起
data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

# 数据集的下载器
# train_dataset = datasets.MNIST(
#     root='./data', train=True, transform=data_tf, download=True)
# test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

ts_operation = args.tsop
img_dir = t + '/transform_images_%s_noise' % ts_operation
mode = 'zone'
i_o_index = args.ioi
time_type = args.tt
phase = 'zero'
vector_dir = t + '/vector_dir/' + time_type + '/train'
multi_classes = os.listdir(img_dir)

data = []
label = []

for multi_class_item_index in range(len(multi_classes)):
    multi_class_item = multi_classes[multi_class_item_index]
    class_vector_dir = vector_dir + '/' + multi_class_item

    range_list_inner = os.listdir(img_dir + '/' + multi_class_item)
    iter_list = range_list_inner[0:len(range_list_inner)]

    for transform_img_index_inner in iter_list:
        # print(transform_img_index_inner)
        data.append(np.load(
            class_vector_dir + '/%s-%s-%s.npy' % (multi_class_item, transform_img_index_inner, time_type)))
        label.append(multi_class_item_index)

data = np.array(data)
label = np.array(label)

data = torch.tensor(data)
label = torch.tensor(label)

train_loader = DataLoader(torch.utils.data.TensorDataset(data, label), batch_size=batch_size, shuffle=True)

# 选择模型
model = net.simpleNet(1801, 9216, 1024, 8)
# model = net.Activation_Net(28 * 28, 300, 100, 10)
# model = net.Batch_Net(28 * 28, 300, 100, 10)
if torch.cuda.is_available():
    model = model.cuda()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

print(train_loader)


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


# 训练模型

for epoch in range(num_epoches):
    for data in train_loader:
        # print(data)
        img, label = data
        img = img.view(img.size(0), -1)
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        out = model(img)
        loss = criterion(out, label)
        print_loss = loss.data.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))

# # 模型评估
# model.eval()
# eval_loss = 0
# eval_acc = 0
# for data in test_loader:
#     img, label = data
#     img = img.view(img.size(0), -1)
#     if torch.cuda.is_available():
#         img = img.cuda()
#         label = label.cuda()
#
#     out = model(img)
#     loss = criterion(out, label)
#     eval_loss += loss.data.item() * label.size(0)
#     _, pred = torch.max(out, 1)
#     num_correct = (pred == label).sum()
#     eval_acc += num_correct.item()
# print('Test Loss: {:.6f}, Acc: {:.6f}'.format(
#     eval_loss / (len(test_dataset)),
#     eval_acc / (len(test_dataset))
# ))
