import sys

if __name__ == '__main__':

    mkn = sys.stdin.readline().strip('').replace('\n', '').split(' ')
    m = int(mkn[0])
    k = int(mkn[1])
    n = int(mkn[2])

    a = []
    b = []
    for i in range(m):
        nums = sys.stdin.readline().strip().replace('\n', '').split(' ')
        aa = []
        for num in nums:
            aa.append(int(num))
        a.append(aa)

    for i in range(k):
        nums = sys.stdin.readline().strip().replace('\n', '').split(' ')
        bb = []
        for num in nums:
            bb.append(int(num))
        b.append(bb)

    c = [0] * m
    for i in range(m):
        c[i] = [0] * m

    # print(a)
    # print(b)
    # print(c)
    for i in range(m):
        for j in range(k):
            for p in range(n):
                c[i][p] += a[i][j] * b[j][p]

    for i in range(m):
        for j in range(n):
            end = '\n'
            if j != n - 1:
                end = ' '
            print(c[i][j], end=end)
        # print()
