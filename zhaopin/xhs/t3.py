import sys


def cal(all_path, l, t, cur_path, aa, danger_time, next_danger, danger_times):
    # print(all_path, l, t, cur_path, aa, danger_time, danger_times)

    if cur_path >= all_path:
        danger_times.append(danger_time)
        return
    if str(cur_path) in aa:
        danger_time += 1

    min_path = l

    for i in range(next_danger, len(aa)):
        if int(aa[i]) > int(aa[next_danger]):
            next_danger = i
            break

    while int(next_danger) - cur_path > l:
        cur_path += min_path

    for today_path in range(l, t + 1):
        cal(all_path, l, t, cur_path + today_path, aa, danger_time, next_danger, danger_times)


# print(all_path, ks, limit_time, rest_time, already_run, need_rest, fish_time)


if __name__ == '__main__':

    x = sys.stdin.readline().strip('').replace('\n', '')
    x = int(x)
    ltn = sys.stdin.readline().strip('').replace('\n', '').split(' ')

    ll = int(ltn[0])
    tt = int(ltn[1])
    nn = int(ltn[2])

    aa = sys.stdin.readline().strip('').replace('\n', '').split(' ')

    danger_times = []

    aa.sort()

    cal(x, ll, tt, 0, aa, 0, 0, danger_times)

    print(min(danger_times))
