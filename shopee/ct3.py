import math
import sys


def hcy(x, y):
    if x > y:
        s = y
    else:
        s = x
    hcf = 1
    for i in range(1, int(s) + 1):
        if (x % i) == 0 and (y % i) == 0:
            hcf = i

    return hcf


def lcm(x, y):
    if x > y:
        g = x
    else:
        g = y

    lcm = 0
    while True:
        if g % x == 0 and g % y == 0:
            lcm = g
            break
        g += 1
    return lcm


def func(nums):
    fm = pow(10, len(nums) - 2)
    fz = int(float(nums) * fm)
    hcf = hcy(fm, fz)
    # print(fz, fm)
    # print(hcf)
    # print('%d/%d' % (fz / hcf, fm / hcf))
    return int(fz / hcf), int(fm / hcf)


def func2(nums, nums2):
    print('---',nums)
    fm = pow(10, len(nums) - 2)
    print('----',fm)
    fz = int(float(nums) * fm)
    print('----',fz)
    fm = fm - pow(10, len(nums2) - 2)
    hcf = hcy(fm, fz)
    print('---', fz, fm)
    # print(hcf)
    # print('%d/%d' % (fz / hcf, fm / hcf))
    return int(fz / hcf), int(fm / hcf)


if __name__ == "__main__":

    for line in sys.stdin:
        nums = line.strip()
        if '(' not in nums:
            fm = pow(10, len(nums) - 2)
            fz = math.ceil(float(nums) * fm)
            hcf = hcy(fm, fz)
            # print(fz, fm)
            # print(hcf)
            print('%d/%d' % (fz / hcf, fm / hcf))
        else:
            bracket = []
            bracket_zero = []
            count = 0
            for num in nums:
                if num == '(':
                    break
                bracket.append(num)
                count += 1
                if count > 2:
                    bracket_zero.append('0')
                else:
                    bracket_zero.append(num)

            for index in range(count + 1, len(nums)):
                if nums[index] == ')':
                    break
                bracket_zero.append(nums[index])

            bracket = ''.join(bracket)
            bracket_zero = ''.join(bracket_zero)
            print(bracket)
            print(bracket_zero)

            fz, fm = func(bracket)
            print(fz, fm)
            fzr, fmr = func2(bracket_zero, bracket)
            print(fzr, fmr)

            lcm = lcm(fm, fmr)

            fz = fz * (lcm / fm)
            fzr = fzr * (lcm / fmr)

            fzf = fz + fzr
            fmf = lcm

            hcff = hcy(fzf, fmf)

            fzf = fzf / hcff
            fmf = fmf / hcff

            print('%d/%d' % (fzf, fmf))
