class Solution:
    def findMedianSortedArrays(self, nums1, nums2):
        # write code here
        l1 = len(nums1)
        l2 = len(nums2)
        mid = int((l1 + l2) / 2)

        index1 = 0
        index2 = 0

        is_odd = 1
        if (l1 + l2) % 2 == 0:
            is_odd = 0

        mid_index = -1
        mid_num = 0
        mid_num_pre = 0

        while index1 < l1 and index2 < l2:

            if nums1[index1] <= nums2[index2]:
                mid_num_pre = mid_num
                mid_num = nums1[index1]
                index1 += 1
            else:

                mid_num_pre = mid_num
                mid_num = nums2[index2]
                index2 += 1
            mid_index += 1

            # print(index1, index2, mid_index, mid,mid_num)

            if mid_index == mid:
                if not is_odd:
                    return (mid_num_pre + mid_num) / 2
                else:
                    return float(mid_num)

        un_index = 0
        un_nums = None

        print(index1, index2, mid_index, mid, mid_num)

        if index1 < l1:
            un_index = index1
            un_nums = nums1
        if index2 < l2:
            un_index = index2
            un_nums = nums2

        while un_index < len(un_nums):
            mid_num_pre = mid_num
            mid_num = un_nums[un_index]
            un_index += 1
            mid_index += 1
            if mid_index == mid:
                if not is_odd:
                    return (mid_num_pre + mid_num) / 2
                else:
                    return mid_num


# print(Solution.findMedianSortedArrays(Solution(), [1, 3], [3, 3, 3]))
