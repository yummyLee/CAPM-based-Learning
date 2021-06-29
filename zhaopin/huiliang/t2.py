import collections

if __name__ == '__main__':
    str_1 = input()

    count = collections.Counter(str_1)
    str_len = len(str_1)
    res = str_len

    index = 0
    mean = int(str_len / 4)

    for i, s in enumerate(str_1):
        count[s] -= 1
        while index < str_len and all(mean >= count[x] for x in 'ABCD'):
            res = min(res, i - index + 1)
            count[str_1[index]] += 1
            index += 1

    print(res, end='')
