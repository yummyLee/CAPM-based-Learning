import numpy as np


def cal(seats, i, j, cur_i, cur_j, visited, k, cur_k, path):
    print(cur_i, cur_j, cur_k, path)

    if cur_k == k:
        return

    if seats[cur_i][cur_j] == 1:
        return

    if seats[cur_i][cur_j] == 0:

        visited[cur_i][cur_j] = 1
        path.append((cur_i, cur_j))
        # cur_k += 1

        if cur_i + 1 < i and visited[cur_i + 1][cur_j] == 0 and cur_k < k:
            # path.append((cur_i + 1, j))
            cal(seats, i, j, cur_i + 1, cur_j, visited, k, cur_k + 1, path)
        if cur_i - 1 >= 0 and visited[cur_i - 1][cur_j] == 0 and cur_k < k:
            # path.append((cur_i + 1, j))
            cal(seats, i, j, cur_i - 1, cur_j, visited, k, cur_k + 1, path)
        if cur_j + 1 < j and visited[cur_i][cur_j + 1] == 0 and cur_k < k:
            # path.append((cur_i + 1, j))
            cal(seats, i, j, cur_i, cur_j + 1, visited, k, cur_k + 1, path)
        if cur_j - 1 >= 0 and visited[cur_i][cur_j - 1] == 0 and cur_k < k:
            # path.append((cur_i + 1, j))
            cal(seats, i, j, cur_i, cur_j - 1, visited, k, cur_k + 1, path)


def cal2():
    height = 3
    width = 3

    # seats = [[0] * width for _ in range(height)]
    seats = [[1, 0, 1], [0, 0, 0], [1, 0, 1]]

    print(np.array(seats))

    k = 3

    found = False

    for i in range(height):
        for j in range(width):
            if seats[i][j] == 0:
                # print(i,j)
                visited = [[0] * width for _ in range(height)]
                path = []
                cal(seats, height, width, i, j, visited, k, 0, path)
                if len(path) == k:
                    found = True
                    print(path)
                    break
        if found:
            break


cal2()
