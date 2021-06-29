def dfs(matrix, nn, visited, index):
    # print(index)
    # print(visited)
    for i in range(nn + 1):
        if matrix[index][i] == 1 and not visited[i]:
            visited[i] = True
            dfs(matrix, nn, visited, i)


def detect(matrix, nn, start_index):
    visited = [False for _ in range(nn + 1)]
    # print('start ', start_index)
    # print(visited)
    visited[start_index] = True
    for i in range(nn + 1):
        if matrix[start_index][i] == 1 and not visited[i]:
            visited[i] = True
            dfs(matrix, nn, visited, i)

    res = 'Yes'
    for i in range(1, nn + 1):
        if not visited[i]:
            res = 'No'
    return res


if __name__ == '__main__':

    t = int(input())

    for i in range(t):
        n, m, k = map(int, input().split(' '))
        matrix = [[0] * (n + 1) for _ in range((n + 1))]
        start_index = -1
        for j in range(m):
            index1, index2, cost = map(int, input().split(' '))
            if cost <= k:
                matrix[index1][index2] = 1
                matrix[index2][index1] = 1
                start_index = index1

        end_end = '\n'
        if i == t - 1:
            end_end = ''
        print(detect(matrix, n, start_index), end=end_end)

# 1
# 3 3 500
# 1 2 400
# 1 3 500
# 2 3 600