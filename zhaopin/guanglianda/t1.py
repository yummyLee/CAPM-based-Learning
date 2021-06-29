if __name__ in '__main__':
    n, m, x = map(int, input().split(' '))
    ais = input().split(' ')
    a = []
    for i in range(n):
        a.append(int(ais[i]))

    a.sort()
    queue = []
    head = 0

    index = 0

    # print(a)

    while a[index] <= a[index + 1] and m > 0:
        a[index] += x
        m -= 1
    queue.append(a[index])

    index = 1
    while index < n and m > 0:
        # print(m)
        # print(queue)
        if a[index] <= queue[head]:
            while a[index] <= queue[head] and m > 0:
                a[index] += x
                m -= 1
            queue.append(a[index])
            index += 1
        else:
            while queue[head] <= queue[head + 1] and m > 0:
                queue[head] += x
                m -= 1
            queue.append(queue[head])
            head += 1

    print(a)
    print(min(queue[head], a[index - 1]))
