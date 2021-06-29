'''
@Author  :    HY
@Software:   PyCharm
@File    :   mutilprocess.py
@Time    :   2019/9/18 23:45
@Desc    :
'''
import time
import multiprocessing
from tqdm import tqdm
from scipy.spatial.distance import cdist
import numpy as np


def doSomething(a, d):
    print('a = ', a)
    print('d = ', d)
    return 1


def do(param):
    return doSomething(param[0], param[1])


if __name__ == '__main__':
    datas = []
    for i in range(0, 70):
        a = np.random.random((2,))
        datas.append(a)
    # t1 = time.time()
    # for e in tqdm(datas):
    #     doSomething(e, datas)
    # t2 = time.time()
    # print('t2-t1:%4f' % (t2 - t1))
    param = []
    for ele in range(70):
        t = (ele, ele)
        print(t)
        param.append(t)
    print('*' * 10)

    p = multiprocessing.Pool(4)
    # b = p.map(doSomething, param)

    t1 = time.time()
    p.map(do, param)
    p.close()
    p.join()
    t2 = time.time()
    print('t2-t1:%4f' % (t2 - t1))
