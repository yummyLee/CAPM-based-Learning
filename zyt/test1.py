import numpy as np
import sys


def judge(a):
    if a > 10:
        if a % 10 == 7 and int(str(a)[-2]) % 2 == 1:
            return True
    return False


def func2(arr, start_index, depth):
    h = arr.shape[0]
    w = arr.shape[1]
    team_list = []

    for i in np.arange(0, w, 1):
        start_index += 1
        if judge(start_index):
            team_list.append([depth, i + depth])
    for i in np.arange(1, h, 1):
        start_index += 1
        if judge(start_index):
            # team_list.append(1)
            team_list.append([i + depth, w - 1 + depth])
    if h - 2 >= 0:
        for i in np.arange(w - 2, -1, -1):
            start_index += 1
            if judge(start_index):
                # print(2)
                team_list.append([h - 1 + depth, i + depth])
    if w - 2 >= 0:
        for i in np.arange(h - 2, 0, -1):
            start_index += 1
            if judge(start_index):
                # print(3)
                team_list.append([i + depth, 0 + depth])

    if h - 1 > 0 and w - 1 > 0:
        team_list.extend(func2(arr[1:h - 1, 1:w - 1], start_index, depth + 1))

    return team_list


def func(m, n):
    arr = np.zeros((m, n))
    team_list = func2(arr, 0, 0)
    print(team_list)


if __name__ == "__main__":
    mn = sys.stdin.readline().strip().split(' ')
    m = int(mn[0])
    n = int(mn[1])
    if m < 10 or m > 1000 or n < 10 or n > 1000:
        print([])
    else:
        func(m, n)
