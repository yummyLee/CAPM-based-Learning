def cal(n, m):
    if n == 0:
        return 1
    elif n < 0:
        return 0

    res = 0
    for i in range(1, m + 1):
        res += cal(n - i, m) % (10e9 + 3)

    return res


if __name__ == '__main__':
    n, m = map(int, input().split(' '))
    res = cal(n, m)
    res = int(res)
    print(res, end='')
