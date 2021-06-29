def dijkstra(matrix, s, e, n):
    distance = [n * 50 for _ in range(n)]
    distance[s] = 0
    visited = [0 for _ in range(n)]

    j = s

    for k in range(n):

        visited[j] = 1
        next_j = -1
        min_value = 50 * n

        for i in range(n):
            w = matrix[i][j]
            if w != n * 50 and visited[i] == 0:
                if distance[j] + w < distance[i]:
                    distance[i] = distance[j] + w
                    if distance[i] < min_value:
                        next_j = i
                        min_value = distance[v]

        j = next_j

    return distance[e]


if __name__ == '__main__':
    n, m = map(int, input().split(' '))

    n += 1

    matrix = [[n * 50] * n for _ in range(n)]

    for i in range(m):
        u, v, time = map(int, input().split(' '))
        matrix[u][v] = matrix[v][u] = time

    last_line = input().split(' ')

    s = int(last_line[0])
    e = int(last_line[1])
    start = last_line[2]

    cost_time = dijkstra(matrix, s, e, n)

    date = start.split('.')
    month = int(date[0])
    date2 = date[1].split('/')
    day = int(date2[0])
    clock = int(date2[1])

    clock += cost_time

    cost_day = int(clock / 24)

    clock = clock % 24

    day += cost_day

    month_day = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    if day > month_day[month]:
        month += 1

    print('%d.%d/%d' % (month, day, clock), end='')

# 4 4
# 1 2 25
# 1 3 18
# 2 4 28
# 3 4 22
# 1 4 7.9/8
