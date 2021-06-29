if __name__ == '__main__':
    n, m, k = map(int, input().split())
    aa = input().split(' ')
    aai = []

    for i in range(len(aa)):
        aai.append(int(aa[i]))

    index = 0
    start = index
    end = index
    res = []
    while index < n:

        # print(index, aai[index])

        if aai[index] >= k:
            index += 1
            if index == n:
                end = index - 1
                res.append((start, end))
            continue
        else:
            if index - 1 >= start:
                end = index - 1
                res.append((start, end))
                # print(start, end)
            while index < n and aai[index] < k:
                index += 1
                start = index

    # print(res)

    sum_res = 0

    for i in range(len(res)):
        # print(res[i][1] - res[i][0] + 1)
        if res[i][1] - res[i][0] + 1 >= m:
            sum_res += (res[i][1] - res[i][0] + 2 - m)
            # print(sum_res)
        # print('----')

    print(sum_res, end='')
