import sys

if __name__ == '__main__':
    input_str = sys.stdin.readline().strip()
    is_fist_alpha = True

    result_str = ''

    is_next_alpha_upper = False

    for char in input_str:

        # print('c: ', char)
        # print(result_str)

        if char.isalpha():

            if is_fist_alpha:
                result_str += char.lower()
                is_fist_alpha = False
                is_next_alpha_upper = False
            elif is_next_alpha_upper:
                result_str += char.upper()
                is_next_alpha_upper = False
            else:
                result_str += char.lower()
            continue
        elif char.isalnum():

            result_str += char
            continue
        else:
            # print('u')
            is_next_alpha_upper = True
            continue

    print(result_str)
