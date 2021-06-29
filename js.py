import numpy as np
import scipy.spatial
import torch
import scipy.stats
import scipy.spatial.distance


def norma(m):
    # 归一化,0-1之外为异常值
    m_mean = np.mean(m)
    m_std = np.std(m)
    m_max = m_mean + 2 * m_std
    m_min = m_mean - 2 * m_std
    m_rescale = (m - m_min) / (m_max - m_min)
    return m_rescale


def js_h(m1, m2, save_path):
    """输入200*256*6*6 计算js散度"""
    m1 = torch.from_numpy(m1)
    m2 = torch.from_numpy(m2)
    m1 = m1.view(200, -1).detach().t().cpu()  # 按维度处理
    m2 = m2.view(200, -1).detach().t().cpu()  # 按维度处理
    m_feature = m1.size()[0]
    print('feature=' + str(m_feature))
    JS = []
    # 归一化
    # m1_rescale = norma(m1.numpy())
    # m2_rescale = norma(m2.numpy())
    m1_rescale, m2_rescale = two_tensor_array_normalization(m1.numpy(), m2.numpy())
    print('game on')
    for j in range(0, m_feature):
        # 分箱
        bins = np.arange(0.00, 1.01, 0.1)
        hist_1, bin_edges_1 = np.histogram(m1_rescale[j], bins)
        hist_2, bin_edges_2 = np.histogram(m2_rescale[j], bins)
        js = scipy.spatial.distance.jensenshannon(hist_1 + 0.0001, hist_2 + 0.0001)
        JS.append(js * js)

    np.save(save_path, np.array(JS))
    return JS


def two_tensor_array_normalization(wait_norm_tensors1, wait_norm_tensors2, y_min=0.0, y_max=1.0):
    # print('---- TENSOR ARRAY NORMALIZTION ----')

    std1 = np.std(wait_norm_tensors1)
    avg1 = np.average(wait_norm_tensors1)
    max_o1 = np.max(wait_norm_tensors1)
    min_o1 = np.min(wait_norm_tensors1)
    max_i1 = min(avg1 + 2 * std1, max_o1)
    min_i1 = max(avg1 - 2 * std1, min_o1)
    # print(wait_norm_tensors1.shape)
    wait_norm_tensors1[wait_norm_tensors1 > max_i1] = -1
    wait_norm_tensors1[wait_norm_tensors1 < min_i1] = -1
    # print(wait_norm_tensors1.shape)

    std2 = np.std(wait_norm_tensors2)
    avg2 = np.average(wait_norm_tensors2)
    max_o2 = np.max(wait_norm_tensors2)
    min_o2 = np.min(wait_norm_tensors2)
    max_i2 = min(avg2 + 2 * std2, max_o2)
    min_i2 = max(avg2 - 2 * std2, min_o2)
    # print(wait_norm_tensors2.shape)
    wait_norm_tensors2[wait_norm_tensors2 > max_i1] = -1
    wait_norm_tensors2[wait_norm_tensors2 < min_i2] = -1
    # print(wait_norm_tensors2.shape)

    max_bound = max(max_i1, max_i2)
    min_bound = max(min(min_i1, min_i2), 0)

    return (y_max - y_min) * (wait_norm_tensors1 - min_bound) / (max_bound - min_bound) + y_min, (y_max - y_min) * (
            wait_norm_tensors2 - min_bound) / (max_bound - min_bound) + y_min


def js_divergence(p, q):
    M = (p + q) / 2
    return 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)


def cal_jsd2(targ, ref, max_value=1):
    targ = targ.reshape(-1, 1)
    ref = ref.reshape(-1, 1)

    # np.save('seq1.npy', targ)
    # np.save('seq2.npy', ref)

    img_one_length = min(targ.shape[0], ref.shape[0])
    # print(img_one_length)

    # count_size = 100
    #
    # t_count = np.zeros([count_size + 2, 1])
    # r_count = np.zeros([count_size + 2, 1])
    # for m_i in range(img_one_length):
    #     if targ[m_i] <= 0:
    #         t_count[0] += 1
    #     elif targ[m_i] >= 1:
    #         t_count[count_size + 1] += 1
    #     else:
    #         t_count[np.math.floor(targ[m_i] * count_size)] += 1
    #
    #     if ref[m_i] <= 0:
    #         r_count[0] += 1
    #     elif ref[m_i] >= 1:
    #         r_count[count_size + 1] += 1
    #     else:
    #         r_count[np.math.floor(ref[m_i] * count_size)] += 1

    bin_edges = np.arange(0.00, 1.01, 0.1)
    t_count = np.histogram(targ, bin_edges)[0] + 0.0001
    r_count = np.histogram(ref, bin_edges)[0] + 0.0001

    # print('t_count', np.sum(t_count))

    t_count = t_count / (np.sum(t_count))
    r_count = r_count / (np.sum(r_count))

    # t_count = t_count / (img_one_length)
    # r_count = r_count / (img_one_length)

    # print(sum(t_count))

    JS = js_divergence(t_count, r_count)
    # JS = scipy.spatial.distance.jensenshannon(t_count, r_count)
    # JS = JS * JS

    return JS, -1


def cal_jsd_list_between_tensors(tensor_a, tensor_b, save_path):
    print('=====started=====' + save_path)

    tensor_a = tensor_a.reshape(tensor_a.shape[0], -1)
    tensor_b = tensor_b.reshape(tensor_b.shape[0], -1)

    tensor_a, tensor_b = two_tensor_array_normalization(tensor_a, tensor_b)

    metric_list = []
    for i in range(min(tensor_a.shape[1], tensor_b.shape[1])):
        # if i % 5000 == 0:
        #     print('%s current cal process is %d' % (save_path, i))

        metric, minus_one = cal_jsd2(tensor_a[:, i], tensor_b[:, i], 1)

        metric_list.append(np.asarray(metric))

        # break

    metric_list_np = np.array(metric_list)
    np.save(save_path, metric_list_np)

    print('=====completed=====' + save_path)


def preprocess(sequence):
    seq = sequence[~np.isnan(sequence)]
    mean, std = np.mean(seq), np.std(seq)
    low = np.max(mean - 2 * std, 0)
    high = mean + 2 * std
    seq[seq < low] = low
    seq[seq > high] = high
    return seq


def binning(sequence, Bin_number, low, high):
    p = np.zeros(Bin_number)
    stride = (high - low) / Bin_number
    # low==high,两序列为同一常数（0）
    if stride == 0:
        # 给予相同概率密度估计
        p[0] = 1
        return p
    seq = sequence.reshape((-1,))
    seqlen = len(seq)
    for i in range(seqlen):
        binum = (int)((seq[i] - low) / stride)
        if binum < 0:
            p[0] += 1
        elif binum > Bin_number - 1:
            p[Bin_number - 1] += 1
        else:
            p[binum] += 1
    return p / seqlen


def cal_jsd(data_1, data_2, Bin_number, save_path):
    shape = np.shape(data_1)[1]
    js = np.zeros(shape)  # (1,9216)
    for i in range(shape):
        seq_1 = preprocess(data_1[:, i])
        seq_2 = preprocess(data_2[:, i])
        low = min(np.min(seq_1), np.min(seq_2))  # 分箱下界
        high = max(np.max(seq_1), np.max(seq_2))  # 分箱上界
        p = binning(seq_1, Bin_number, low, high)
        q = binning(seq_2, Bin_number, low, high)
        js[i] = scipy.spatial.distance.jensenshannon(p, q, base=None)
        js[i] = js[i] * js[i]
        np.save(save_path, np.array(js))
    return js  # (1,9216)


def pre_handle(point):
    """
    方法是为了删除异常点，选择删除两边5%的点
    :param point: 要删除点的数组
    :return: 删除节点之后的数组
    """

    n = 0
    miu = point.mean()
    sigma = point.std()
    if sigma == 0:
        return point

    m_max = miu + 2 * sigma
    m_min = miu - 2 * sigma

    point = (point - m_min) / (m_max - m_min)

    return point


# 进行分箱
# def fenxiang(sequence, Bin_number=10, max_=1, min_=0, del_zero=False):
#     """
#     将数据进行分箱，是为了将离散的数据进行概率化。
#     有两种选择： 直接数字分箱或者将其拟合数据再分箱

#     :param del_zero:
#     :param sequence: 数据结果
#     :param Bin_number: 箱的个数。默认值100
#     :return: 返回分享结果，即为概率的结果数组。
#     """
#     sequence = sequence.reshape((-1,))

#     sequence = (sequence - min_) / (max_ - min_)

#     p = np.ones(Bin_number, )

#     bins = []
#     for low in range(0, 100, int(100 / Bin_number)):
#         bins.append((low / 100, (low + 100 / Bin_number) / 100))
#     #     print(bins)

#     for j in range(np.shape(sequence)[0]):
#         for i in range(0, len(bins)):
#             if sequence[j] == 0 and del_zero:
#                 break
#             if bins[i][0] <= sequence[j] < bins[i][1]:
#                 p[i] += 1
#     for i in range(Bin_number):
#         p[i] = p[i] / (np.shape(sequence)[0] + Bin_number)
#     return p


# 计算js散度
def JS_divergence(point_1, point_2, del_zero=False, num_bins=100):
    """
    计算js散度的函数
    :param num_bins:
    :param del_zero:
    :param point_1: point表示要计算js散度的两个值，一般都是相同长度的概率数组。
    :param point_2:
    :return: 返回js散度
    """
    global js

    try:
        point_1 = point_1.reshape(-1, )
        point_2 = point_2.reshape(-1, )
        # x_1 = pre_handle(point_1)
        # x_2 = pre_handle(point_2)
        x_1, x_2 = two_tensor_array_normalization(point_1, point_2)

        #         min_ = max(min(min(x_1), min(x_2)), 0)
        #         max_ = min(max(max(x_1), max(x_2)), 1)

        #         p = fenxiang(x_1, 10, max_, min_, del_zero)
        #         q = fenxiang(x_2, 10, max_, min_, del_zero)
        bins = np.arange(0.00, 1.01, 0.1)
        hist_1, bin_edges_1 = np.histogram(x_1, bins)
        hist_2, bin_edges_2 = np.histogram(x_2, bins)
        js = scipy.spatial.distance.jensenshannon(hist_1 + 0.0001, hist_2 + 0.0001)

        # max0 = max(np.max(x_1), np.max(x_2))
        # min0 = min(np.min(x_1), np.min(x_2))
        # bins = np.linspace(min0 - 1e-4, max0 - 1e-4, num=num_bins)
        # PDF1 = pd.cut(x_1, bins).value_counts() / len(x_1)
        # PDF2 = pd.cut(x_2, bins).value_counts() / len(x_2)
        # p = PDF1.values
        # q = PDF2.values
        #         M = (p + q) / 2
        # #         js = 0.5 * entropy(p, M) + 0.5 * entropy(q, M)
        #         js = scipy.spatial.distance.jensenshannon(p, q)
        js = js * js

    except Exception as e:
        print(e)
    return js


def cal_jsd_dhw(a, b, save_path):
    test_1 = []
    for i in range(9216):
        # print(i)
        test_1.append(JS_divergence(a[:, i], b[:, i]))

    np.save(save_path, np.array(test_1))
    return test_1


t1 = torch.load('ch5_change-001.pt').numpy()
t2 = torch.load('ch5_change-0020.pt').numpy()
t1 = t1.reshape(200, -1)
t2 = t2.reshape(200, -1)

# print('------')
# js_h(t1, t2, 'test_jsd_t1_11_2212.npy')
# print('------')
# cal_jsd_list_between_tensors(t1, t2, 'test_jsd_t1_11_2238.npy')
# print('------')
# cal_jsd(t1, t2, 10, 'test_jsd_t1_11_2243.npy')
# print('------')
cal_jsd_dhw(t1, t2, 'test_jsd_t1_11_2307.npy')

# tensor_test_norm1 = t1[0]
# tensor_test_norm2 = t2[0]
# two_tensor_array_normalization(tensor_test_norm1, tensor_test_norm2)
