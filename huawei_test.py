# coding=utf-8
# 本题为考试单行多行输入输出规范示例，无需提交，不计分。
import sys

operator = ['NOT', 'AND', 'OR']

for line in sys.stdin:
    a = line.split()
    a_len = len(a)
    is_error = True
    index = 0
    next_not = False
    next_not_count = 0
    if a[0] == 'NOT' and len(a) > 1:
        next_not_count += 1
        next_not = True
    for i in range(0, len(a)):
        index = (i + next_not_count) % 2
        # print(i, '====', index)
        if index == 0:
            if a[i] in operator:
                if a[i] == 'NOT' and next_not:
                    if i == len(a) - 1:
                        is_error = False
                        break
                    next_not = False
                    next_not_count += 1
                    # print('4444')
                    continue
                else:
                    is_error = False
                    # print('222')
                    break
            if not a[i].islower():
                is_error = False
                # print('111')
                break
        else:
            if a[i] not in operator:
                is_error = False
                # print('333')
                break
            if a[i] == 'AND' or a[i] == 'OR':
                next_not = True
                continue
            if a[i] == 'NOT':
                if i != 0 and a[i - 1] not in ['AND', 'NOT']:
                    is_error = False
                next_not = False
    if is_error:
        print(1)
    else:
        print(0)
