import sys

if __name__ == "__main__":
    ns = sys.stdin.readline().strip().split(' ')
    n = int(ns[0])
    m = int(ns[1])
    ms = []
    for i in range(m):
        ms.append(int(sys.stdin.readline().strip()))

    ns = list(range(1, n + 1))

    result_list = []

    for n in ns:
        if n in result_list:
            break
        for m in ms:
            if n % m == 0:
                result_list.append(n)
                break

    print(len(set(result_list)))
