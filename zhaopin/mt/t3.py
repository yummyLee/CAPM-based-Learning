def cal(n, k, d, is_d):
    if n < 0:
        return 0
    elif n == 0:
        return 1

    if not is_d:
        if n < d:
            return 0
        elif n == d:
            return 1
    res = 0

    for i in range(1, k + 1):
        if i >= d:
            is_d = True
        res += cal(n - i, k, d, is_d)

    return res % 998244353


if __name__ == '__main__':
    n, k, d = map(int, input().split(' '))
    print(cal(n, k, d, False), end='')
