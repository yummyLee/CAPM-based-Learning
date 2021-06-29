import sys

if __name__ == "__main__":
    # 读取第一行的n
    n = int(sys.stdin.readline().strip())
    nums = sys.stdin.readline().strip().split(' ')

    max_count = 0

    if n < 3:
        print(0)

    else:
        count = 0
        i = 1
        while i < n:

            # print(count)

            if nums[i] > nums[i - 1]:
                if count == 0:
                    count += 1
                count += 1
            if nums[i] == nums[i - 1]:
                count = 0
            if nums[i] < nums[i - 1]:
                if count >= 2:
                    count += 1
                    if max_count < count:
                        max_count = count
                while i < n - 1:
                    i += 1
                    if nums[i] < nums[i - 1]:
                        count += 1
                        # print(count, 'count')
                        if max_count < count:
                            max_count = count
                    else:
                        i -= 1
                        count = 0
                        if max_count < count:
                            max_count = count
                        break
            i += 1

        print(max_count)
