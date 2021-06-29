if __name__ == '__main__':
    n, m = map(int, input().split(' '))

    matrixs = []
    for i in range(n):
        matrixs.append(input())

    res = -1

    for i in range(1, n):

        up = i - 1
        down = i
        error = 0
        while up >= 0 and down <= n - 1:

            # print(i, up, down)
            if matrixs[up] == matrixs[down]:
                up -= 1
                down += 1
            else:
                error = 1
                break

        if up == -1 and error == 0:
            res = i
            break

    for i in range(res):
        print(' '.join(matrixs[i].split(' ')))
