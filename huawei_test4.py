"""
goodgoogood
goodgoodgood
"""


def recog_repeat(str):
    pre = str[0]
    for i in range(1, len(str)):
        if str[i] == pre and str[0:i] == str[i:i + i]:
            return str[0:i]


def count_repeat(str, sub_str):
    return len(str) / len(sub_str)


test_str = 'goodgoogoodgoo'
test_sub_str = recog_repeat(test_str)
print('%s %d\n' % (test_sub_str, count_repeat(test_str, test_sub_str)))
