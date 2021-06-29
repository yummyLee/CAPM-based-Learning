def cal(str):
    s = []

    i = 0

    match_count = 0

    while i < len(str):
        if str[i] == '{':
            left_count = 0
            while str[i] == '{':
                left_count += 1
                i += 1
            s.append(('{', left_count))
        elif str[i] == '}':
            right_count = 0
            while str[i] == '}':
                right_count += 1
                i += 1
            if s[-1][0] == '{' and s[-1][1] == right_count:
                s.pop()
                match_count += 1
        else:
            i += 1

    return match_count

if __name__ == '__main__':
    res = cal('aaa{a{{{{bbb}}}}ccc}ddd')
    print(res)
