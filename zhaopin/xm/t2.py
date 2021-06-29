import sys

if __name__ == '__main__':
    n = sys.stdin.readline().replace('\n', '').strip()
    n = int(n)

    aas = sys.stdin.readline().replace('\n', '').strip().split(' ')
    a = []
    for aa in aas:
        a.append(int(aa))

    if n == 1:
        print(1)
    else:
        a.sort(reverse=True)

        cur_index = 0

        for i in range(1, n):
            if a[i] < a[cur_index]:
                cur_index += 1

        print(len(a) - cur_index)
