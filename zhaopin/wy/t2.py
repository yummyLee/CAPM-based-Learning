import numpy as np

if __name__ == '__main__':
    str = input()
    str_len = len(str)

    dp = [[False] * str_len for _ in range(str_len)]
    counter = 0

    for i in range(1, str_len):
        dp[i][i] = True
        if str[i - 1] == str[i]:
            dp[i - 1][i] = True
            counter += 1

    # print(np.array(dp))

    for j in range(1, str_len):
        for i in range(j - 1):
            if str[i] == str[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                # print(i, j)
                counter += 1

    print(counter)
