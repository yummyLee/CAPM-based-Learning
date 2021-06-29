import sys

if __name__ == '__main__':
    input_str = sys.stdin.readline().strip()

    int_arr = []

    result_str = ''
    # for char in input_str:
    #     int_arr.append(int(char))

    for i in range(1, len(input_str) + 1):
        int_arr.append(int(input_str[-i]))

    first_big_index = -1
    first_small_index = -1

    # print(int_arr)

    for i in range(len(int_arr)):
        for j in range(i + 1, len(int_arr)):
            # print(int_arr[i], int_arr[j])
            if i < len(int_arr) - 1 and int_arr[i] < int_arr[j]:
                first_big_index = j
                break
        if first_big_index != -1:
            break

    for i in range(len(int_arr)):
        if i < len(int_arr) - 1 and int_arr[i] < int_arr[first_big_index]:
            first_small_index = i
            break

    for i in range(first_small_index + 1, len(int_arr)):
        if int_arr[i] == int_arr[first_small_index]:
            first_small_index = i
            continue
        else:
            break

    # print(first_big_index, int_arr[first_big_index])
    # print(first_small_index, int_arr[first_small_index])

    if first_big_index != -1 and first_small_index != -1:
        temp = int_arr[first_big_index]
        int_arr[first_big_index] = int_arr[first_small_index]
        int_arr[first_small_index] = temp
    elif first_big_index == -1:
        result_str = '0'

    for i in range(1, len(input_str) + 1):
        result_str += str(int_arr[-i])

    # print(result_str)

    if result_str.startswith('0') or result_str == input_str:
        print('0')
    else:
        print(result_str)
