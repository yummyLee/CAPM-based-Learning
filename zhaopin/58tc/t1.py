import numpy


def dfs(data, visited, i, j, m, n):
    # print(i, j)

    count = 1
    if i + 1 < m and data[i + 1][j] == 1 and not visited[i + 1][j]:
        visited[i + 1][j] = True
        count += dfs(data, visited, i + 1, j, m, n)
    if i - 1 >= 0 and data[i - 1][j] == 1 and not visited[i - 1][j]:
        visited[i - 1][j] = True
        count += dfs(data, visited, i - 1, j, m, n)
    if j + 1 < n and data[i][j + 1] == 1 and not visited[i][j + 1]:
        visited[i][j + 1] = True
        count += dfs(data, visited, i, j + 1, m, n)
    if j - 1 >= 0 and data[i][j - 1] == 1 and not visited[i][j - 1]:
        visited[i][j - 1] = True
        count += dfs(data, visited, i, j - 1, m, n)

    return count


def getMaxArea(data):
    m = len(data)
    n = len(data[0])

    visited = [[False] * n for _ in range(m)]

    # print(numpy.array(visited))

    max_count = 0
    for i in range(m):
        for j in range(n):
            if data[i][j] == 1 and not visited[i][j]:
                visited[i][j] = True
                count = dfs(data, visited, i, j, m, n)
                if count > max_count:
                    max_count = count

    return max_count


data = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [1, 1, 1, 0],[0, 0, 1, 0], [0, 0, 1, 0], [0, 1, 0, 0], [1, 1, 1, 0]]

print(numpy.array(data))

print(getMaxArea(data))
