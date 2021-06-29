import sys
import numpy as np


def cal(matrix, visited, i, j, is_set_one, cur_size):
    cur_size += 1
    # print(i, j, cur_size, is_set_one)

    if i - 1 >= 0 and visited[i - 1][j] == 0:
        if matrix[i - 1][j] == 1:
            visited[i - 1][j] = 1
            cal(matrix, visited, i - 1, j, is_set_one, cur_size)
        elif is_set_one[0] == 0:
            visited[i - 1][j] = 1
            is_set_one[0] = 1
            cal(matrix, visited, i - 1, j, is_set_one, cur_size)

    if i + 1 < len(matrix) and visited[i + 1][j] == 0:
        if matrix[i + 1][j] == 1:
            visited[i + 1][j] = 1
            cal(matrix, visited, i + 1, j, is_set_one, cur_size)
        elif is_set_one[0] == 0:
            visited[i + 1][j] = 1
            is_set_one[0] = 1
            cal(matrix, visited, i + 1, j, is_set_one, cur_size)

    if j - 1 >= 0 and visited[i][j - 1] == 0:
        if matrix[i][j - 1] == 1:
            visited[i][j - 1] = 1
            cal(matrix, visited, i, j - 1, is_set_one, cur_size)
        elif is_set_one[0] == 0:
            is_set_one[0] = 1
            visited[i][j - 1] = 1
            cal(matrix, visited, i, j - 1, is_set_one, cur_size)

    if j + 1 < len(matrix[0]) and visited[i][j + 1] == 0:
        if matrix[i][j + 1] == 1:
            visited[i][j + 1] = 1
            cal(matrix, visited, i, j + 1, is_set_one, cur_size)
        elif is_set_one[0] == 0:
            visited[i][j + 1] = 1
            is_set_one[0] = 1
            cal(matrix, visited, i, j + 1, is_set_one, cur_size)

    return cur_size


if __name__ == "__main__":
    ns = sys.stdin.readline().strip().split(' ')
    n = int(ns[0])
    m = int(ns[1])
    matrix = []
    visited = [0] * n
    visited2 = [0] * n
    for i in range(n):
        visited[i] = [0] * m
        visited2[i] = [0] * m
        ele = list(map(int, sys.stdin.readline().strip().split(' ')))
        matrix.append(ele)

    # visited[0][0] = 1

    # cur_size = cal(matrix, visited, 0, 0, is_set_one, 0)
    # print(np.sum(np.array(visited)))

    size_max = 0
    for i in range(n):
        for j in range(m):
            visited = [0] * n
            visited2 = [0] * n
            is_set_one = [0]
            for o in range(n):
                visited[o] = [0] * m
            visited[i][j] = 1
            if matrix[i][j] == 1:
                cal(matrix, visited, i, j, is_set_one, 0)
                cur_size = 0
                for o in range(n):
                    for p in range(m):
                        if visited[o][p] == 1:
                            cur_size += 1
                # print(cur_size)
                if cur_size > size_max:
                    size_max = cur_size

    print(size_max - 1)
