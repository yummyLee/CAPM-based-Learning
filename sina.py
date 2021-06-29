def get_factor(num):
    res = []
    f = 2
    if num == f:
        res.append(str(f))
    else:
        while num >= f:
            # print(num)
            rest = num % f
            if rest == 0:
                res.append(str(f))
                num = int(num / f)
            else:
                f += 1

    print(' '.join(res))


if __name__ == '__main__':
    num = int(input())
    get_factor(num)
