class Solution:
    def sorted_two_list(self, Input1, Input2):
        # write code here
        index1 = 0
        index2 = 0
        index = 0
        res = []
        while index1 < len(Input1) and index2 < len(Input2):
            if Input1[index1] <= Input2[index2]:
                res.append(Input1[index1])
                index1 += 1
            else:
                res.append(Input2[index2])
                index2 += 1

        while index1 < len(Input1):
            res.append(Input1[index1])
            index1 += 1

        while index2 < len(Input2):
            res.append(Input1[index2])
            index2 += 1

        return res
