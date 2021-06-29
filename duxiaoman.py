if __name__ == '__main__':
    n, q = map(int, input().split(' '))
    ais = input().split(' ')
    adic = {}
    adic[1] = 0
    for i in range(2, n + 1):
        adic[i] = int(ais[i - 2])

    # print(adic)

    for i in range(q):
        x, k = map(int, input().split(' '))
        count = 0
        anc = x
        while count < k:
            if anc == 0:
                break
            anc = adic[anc]
            count += 1

        print(anc)

# 5 3
# 1 2 3 4
# 5 1
# 1 1
# 5 6