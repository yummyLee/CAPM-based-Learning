import numpy as np
import sys


def func(d_max, d_dic):
    count = 1
    # print(d_dic)
    for d in range(d_max + 1):
        d_num = d_dic[d]
        # print(d, d_num)
        if d_num == 1:
            d_count = np.power(2, d)
            # print('dcount: ', d_count)
        else:
            d_count = np.math.factorial(2 * d_dic[d - 1]) / (
                    np.math.factorial(d_num) * np.math.factorial(2 * d_dic[d - 1] - d_num))

        count = count * d_count

    return int(count % (np.power(10, 9) + 7))


# d_max = 2
# d_dic = {1: 1, 0: 1, 2: 2}

# print(func(d_max, d_dic))

if __name__ == "__main__":
    n = int(sys.stdin.readline().strip())
    ds = sys.stdin.readline().strip().split(' ')
    d_dic = {}
    d_max = 0
    for d in ds:
        d = int(d)
        if d > d_max:
            d_max = d
        if not d in d_dic.keys():
            d_dic[d] = 1
        else:
            d_dic[d] += 1

    print(func(d_max, d_dic))
