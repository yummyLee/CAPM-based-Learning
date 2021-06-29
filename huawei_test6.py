import sys


def transform(arr):
    for i in range(len(arr)):
        arr[i] = bin(arr[i]).replace('0b', '')
        arr[i] = str(arr[i])
        if len(arr[i]) < 32:
            arr[i] = ('0' * (32 - len(arr[i]))) + arr[i]
        # print(arr[i])

    result_str_post = ''
    for i in range(len(arr)):
        for j in range(0, len(arr[i]), 2):
            result_str_post += arr[i][j + 1]
            result_str_post += arr[i][j]
            j += 1

    result_str = result_str_post[-2] + result_str_post[-1]
    result_str += result_str_post[0:len(result_str_post) - 2]

    # print(result_str_post)
    # print(result_str)

    result_arr = []
    for i in range(len(arr)):
        # print('result_str', result_str[32 * i:(i + 1) * 32])
        result_arr.append(int(result_str[32 * i:(i + 1) * 32], 2))

    # print(result_arr)
    return result_arr


# test_arr = [1, 2]
#
# transform(test_arr)

if __name__ == "__main__":
    ds = sys.stdin.readline().strip().split(' ')
    test_arr = []
    for d in ds:
        test_arr.append(int(d))

    # print(test_arr)

    test_result = transform(test_arr)
    for i in range(len(test_result)):
        if i == len(test_result) - 1:
            print(test_result[i], end='')
        else:
            print(test_result[i], end=' ')
