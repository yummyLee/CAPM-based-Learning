import sys


def cal(all_path, ks, limit_time, rest_time, already_run, need_rest, finish_time):
    # print(all_path, ks, limit_time, rest_time, already_run, need_rest, fish_time)

    if already_run >= all_path:
        finish_time.append(limit_time - rest_time)

    if rest_time == 0 or sum(ks[limit_time - rest_time:limit_time]) < all_path - already_run:
        return

    if need_rest == 1:
        cal(all_path, ks, limit_time, rest_time, already_run, 3, finish_time)

    elif need_rest > 1:
        cal(all_path, ks, limit_time, rest_time - 1, already_run, 3, finish_time)

        cal(all_path, ks, limit_time, rest_time - 1, already_run + ks[limit_time - rest_time], need_rest - 1, finish_time)


if __name__ == '__main__':
    m = sys.stdin.readline().strip('').replace('\n', '')
    m = int(m)
    n = sys.stdin.readline().strip('').replace('\n', '')
    n = int(n)

    ks = []
    for i in range(n):
        ks.append(int(sys.stdin.readline().strip('').replace('\n', '')))

    all_path = m
    already_run = 0
    need_rest = 3
    finish_time = []

    cal(m, ks, n, n, 0, 3, finish_time)

    if len(finish_time) == 0:
        print(-1)
    else:
        print(min(finish_time))
