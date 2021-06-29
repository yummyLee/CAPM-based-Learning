str = input()

str_len = len(str)

error_count = 0

i = 0
j = str_len - 1

del_index = -1
res = 0

while i <= int(str_len) / 2:

    # print(i, j)
    if str[i] == str[j]:
        i += 1
        j -= 1
        continue
    else:
        if error_count == 1:
            res += 1
            break
        error_count += 1
        del_index = i
        i += 1
        if str[i] != str[j]:
            res += 1
            break

if res == 0:
    res_str = ''
    for i in range(str_len):
        if i != del_index:
            res_str += str[i]
    print(res_str)
else:
    i = 0
    j = str_len - 1
    error_count = 0

    while j >= int(str_len) / 2:

        # print(i, j)
        if str[i] == str[j]:
            i += 1
            j -= 1
            continue
        else:
            if error_count == 1:
                res += 1
                break
            error_count += 1
            del_index = j
            j -= 1
            if str[i] != str[j]:
                res += 1
                break
    if res == 1:
        res_str = ''
        for i in range(str_len):
            if i != del_index:
                res_str += str[i]
        print(res_str)
    else:
        print('false')
