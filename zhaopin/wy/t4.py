# import numpy as np


def dfs(p_map, boys, boy_index, girls_set):
    if len(girls_set) == 0 or boy_index > len(boys) - 1:
        return 0
    boy = boys[boy_index]
    res_max = 0
    if boy in pp_map.keys():
        for girl in girls_set:
            if girl in pp_map[boy].keys():
                girls_set.remove(girl)
                res = 1 + dfs(p_map, boys, boy_index + 1, girls_set)
                girls_set.add(girl)
                if res > res_max:
                    res_max = res

    res = dfs(p_map, boys, boy_index + 1, girls_set)
    if res > res_max:
        res_max = res
    return res_max


if __name__ == '__main__':
    boys_str = input().split(' ')
    girls_str = input().split(' ')

    boys = []
    girls = []

    for i in range(len(boys_str)):
        boys.append(int(boys_str[i]))

    for i in range(len(girls_str)):
        girls.append(int(girls_str[i]))

    n = int(input())

    boy_max = max(boys)
    girl_max = max(girls)

    max_index = max(boy_max, girl_max) + 1

    pp_map = {}

    for line in range(n):
        i, j = map(int, input().split(' '))
        if i in pp_map.keys():
            pp_map[i][j] = 1
        else:
            pp_map[i] = {}
            pp_map[i][j] = 1

    # print(np.array(pp_map))

    girls_set = set(girls)
    boy_index = 0
    boy = boys[boy_index]

    res_max = 0
    if boy in pp_map.keys():
        for girl in girls_set:
            # print(boy, girl)
            if girl in pp_map[boy].keys():
                girls_set.remove(girl)
                res = 1 + dfs(pp_map, boys, boy_index + 1, girls_set)
                # print(res)
                if res > res_max:
                    res_max = res
                girls_set.add(girl)
                res = 0

    res = dfs(pp_map, boys, boy_index + 1, girls_set)
    if res > res_max:
        res_max = res

    print(res_max)
