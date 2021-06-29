# coding=utf-8
# 本题为考试多行输入输出规范示例，无需提交，不计分。
import sys

if __name__ == "__main__":
    # 读取第一行的n
    n = int(sys.stdin.readline().strip())
    ans = 0
    social_num_list = []
    for i in range(n):
        # 读取每一行
        line = sys.stdin.readline().strip()
        social_num_list.append(line)

    if n == 1:
        print('%s %d' % (social_num_list[0], 1))

    social_num_list = sorted(social_num_list)
    # print(ans)
    social_num_list_index = []
    social_num_list_count = []

    social_num_list.append('None')
    count = 1
    for i in range(1, len(social_num_list)):
        # print(social_num_list[i])
        if social_num_list[i] == social_num_list[i - 1]:
            count += 1
        else:
            social_num_list_count.append(count)
            count = 1
            social_num_list_index.append(i - 1)

    for i in range(len(social_num_list_index)):
        index = social_num_list_index[i]
        print('%s %d' % (social_num_list[index], social_num_list_count[i]))
