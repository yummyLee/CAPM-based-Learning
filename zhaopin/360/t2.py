if __name__ == '__main__':
    n, m = map(int, input().split(' '))
    s = []
    x = []
    set_possible = {}
    not_rest = {}

    for i in range(1, n + 1):
        set_possible[i] = 1
        not_rest[i] = -1

    last_one = -1
    first_one = -1
    not_rest_count = 0
    for i in range(m):
        a, b = map(int, input().split(' '))

        if b == 0:
            last_one = a
            if not_rest[a] == -1:
                set_possible[a] = 0
                not_rest[a] = -1
                not_rest_count -= 1
            else:
                set_possible[a] = 0
        else:
            if first_one == -1:
                first_one = a
            not_rest[a] = 1
            not_rest_count += 1

    if not_rest_count == 0:
        set_possible[last_one] = 1
    if not_rest_count == m:
        set_possible[first_one] = 1

    list_possible = []

    for i in range(1, n + 1):
        if set_possible[i] == 1:
            list_possible.append(i)

    list_possible = map(str, list_possible)

    print(' '.join(list_possible))
