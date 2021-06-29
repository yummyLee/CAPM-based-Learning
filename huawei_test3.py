# coding=utf-8
# 本题为考试单行多行输入输出规范示例，无需提交，不计分。
import sys

head_file_list = []
head_file_include_list = []
wait_to_check = ''
checked_set = set()
error_include_list = []


def check_loop(wait_to_check_file, contact_str, include_list):
    for include_file in include_list:
        if include_file == wait_to_check_file:
            error_include_list.append(contact_str)
            return
        else:
            if checked_set not in checked_set:
                checked_set.add(include_file)
                if include_file in head_file_list:
                    check_loop(wait_to_check_file, contact_str + ' ' + include_file,
                               head_file_include_list[head_file_list.index(include_file)])
            else:
                return


for line in sys.stdin:
    a = line.strip().split(':')
    if a[0] != 'search head file':
        head_file_list.append(a[0])
        head_file_include_list.append(a[1].split(' '))
    else:
        wait_to_check = a[1]
        for i in head_file_list:
            # print('====', i)
            if i == wait_to_check:
                continue
            checked_set.add(i)
            if i in head_file_list:
                check_loop(wait_to_check, wait_to_check + ' ' + i, head_file_include_list[head_file_list.index(i)])
            checked_set.clear()

        if len(error_include_list) > 0:
            print('Bad coding -- loop include as bellow:')
            for error_include_str in error_include_list:
                print(error_include_str)
        else:
            print('none loop include %s' % wait_to_check)

# a.h:b.h
# b.h:c.h
# c.h:a.h
# search head file:a.h
