import sys


def cal(str1, str2):
    len1 = len(str1)
    len2 = len(str2)

    t_str1 = str1
    t_str2 = str2

    change = False

    if len1 < len2:
        t_str2 = str1
        t_str1 = str2
        change = True
    else:
        for i in range(len1):
            if str1[i] == str2[i]:
                continue
            elif str1[i] < str2[i]:
                t_str1 = str2
                t_str2 = str1
                change = True
                break

    min_len = min(len1, len2)
    max_len = max(len1, len2)

    res_str = []

    borrow = 0

    # 102
    #  23
    #   9

    for i in range(1, min_len + 1):

        if int(t_str1[-i]) + borrow < 0:
            int(t_str1[-i]) + 10 + borrow
            borrow = -1

        res = int(t_str1[-i]) - int(t_str2[-i])
        borrow = 0
        if res < 0:
            res_str.append(int(t_str1[-i]) + borrow + 10 - int(t_str2[-i]))
            borrow = -1
        else:
            res_str.append(res)

    if len1 == len2:
        if borrow == -1:
            res_str.append('-')
    else:
        if int(t_str1[-(min_len + 1)]) + borrow < 0:

            pass
        else:
            res_str.append(int(t_str1[-min_len + 1]) + borrow)
            if max_len > min_len:
                for i in range(min_len + 1, max_len + 1):
                    res_str.append(t_str1[-i])

    if change:
        if res_str[-1] == '-':
            res_str[-1] = '+'
        else:
            res_str.append('-')

    return res_str


# 11
# 23

print(cal('23', '98'))
