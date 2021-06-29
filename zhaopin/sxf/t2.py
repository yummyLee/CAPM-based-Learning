if __name__ == '__main__':

    t = int(input())

    for i in range(t):
        n = int(input())
        cis = input().split(' ')
        c = []

        count = 0

        one_index = 0

        for j in range(n):
            c.append(int(cis[j]))
            if j + 1 == c[j]:
                count += 1
            if c[j] == 1:
                one_index = j

        print('count: ', count)

        if one_index != n - 1:
            c[one_index] = c[-1]
            c[-1] = 1
            count += 1
            if one_index + 1 == c[one_index]:
                count += 1
            else:
                count -= 1

        print(count)
