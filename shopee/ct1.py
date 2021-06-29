import sys

if __name__ == "__main__":

    for line in sys.stdin:
        nums = line.strip().split(';')
        nums_ten = []
        for num in nums:
            index = 1
            num_ten = 0
            for num_len in range(1, len(num) + 1):
                num_ten += int(num[-num_len]) * index
                index *= 2
            nums_ten.append(num_ten)

        res = nums_ten[0] * nums_ten[1]
        res_two = []
        while True:
            h = int(res / 2)
            rest = res % 2
            res_two.append(str(rest))
            if h == 0:
                break
            res = h

        # print(res_two)
        res_two.reverse()
        print(''.join(res_two))
