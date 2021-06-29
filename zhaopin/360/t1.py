import sys

if __name__ == '__main__':

    while True:
        try:
            pw = input()
            res = 'Irregular password'
            if len(pw) >= 8:
                upper = 0
                lower = 0
                s_char = 0
                num = 0
                for i in range(len(pw)):
                    # print(pw[i])
                    if pw[i].isdigit():
                        num = 1
                        # continue
                    elif pw[i].isupper():
                        upper = 1
                        # continue
                    elif pw[i].islower():
                        lower = 1
                        # continue
                    else:
                        s_char = 1

                    # print(upper, lower, s_char, num)
                    if upper + lower + s_char + num == 4:
                        res = 'Ok'
                        break

            print(res)

        except:
            break
