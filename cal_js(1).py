## 首先是预处理方法，为了删除左右异常值
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
        x_1 = pre_handle(point_1)
        x_2 = pre_handle(point_2)

        #         min_ = max(min(min(x_1), min(x_2)), 0)
        #         max_ = min(max(max(x_1), max(x_2)), 1)

        #         p = fenxiang(x_1, 10, max_, min_, del_zero)
        #         q = fenxiang(x_2, 10, max_, min_, del_zero)

        bins = [n / 10 for n in range(0, 11, 1)]  # 箱子边界，0-1之内
        hist_1, bin_edges_1 = np.histogram(x_1, bins)
        hist_2, bin_edges_2 = np.histogram(x_2, bins)
        js = scipy.spatial.distance.jensenshannon(hist_1 + 1, hist_2 + 1)

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
