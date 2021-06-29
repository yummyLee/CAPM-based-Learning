if __name__ == '__main__':

    t = int(input())

    for i in range(t):
        a, b, n = map(int, input().split(' '))
        cis = input().split(' ')
        c = []
        for ci in cis:
            c.append(int(ci))

        start = 0
        end = 0
        j = 0
        res = 'No'
        count = 0
        while j < len(c):
            # print(j)
            while j < len(c) and start < len(c) and c[j] - c[start] < a:
                count += 1
                j += 1

            # print(start, j, count)
            if count >= b:
                # print('Yes')
                res = 'Yes'
                break

            while j < len(c) and start < len(c) and c[j] - c[start] >= a:
                start += 1
                count -= 1

            # print('--', start, j, count)

        end = '\n'
        if i == t - 1:
            end = ''
        print(res, end=end)

# 1
# 5 4 1
# 1 2 3 6 8 9 10 11
