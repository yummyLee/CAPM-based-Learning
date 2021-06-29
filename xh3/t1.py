if __name__ == '__main__':
    input_str = input()

    index = 0
    str_len = len(input_str)

    res_str = ''

    while index < str_len:
        count = 0
        pre = input_str[index]
        while index < str_len and input_str[index] == pre:
            index += 1
            count += 1

        if count > 1:
            res_str += (pre + str(count))
        else:
            res_str += pre

    print(res_str)
