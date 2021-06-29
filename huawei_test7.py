import sys
import time


class Stack:
    def __init__(self):
        self.items = []

    def is_empty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[-1]

    def size(self):
        return len(self.items)


def largest_rec(heights):
    stack = Stack()
    result = 0
    for item in range(len(heights)):
        value = 0
        if item < len(heights):
            value = heights[item]
        if stack.is_empty() or value > heights[stack.peek()]:
            stack.push(item)
            # time.sleep(1)
        else:
            tmp = stack.pop()
            tmp2 = -1
            if stack.is_empty():
                tmp2 = item
            else:
                tmp2 = item - stack.peek() - 1

            result = max(result, heights[tmp] * tmp2)

    return result


# test_arr = [1, 2]
#
# transform(test_arr)

if __name__ == "__main__":
    ds = sys.stdin.readline().strip().split('],[')
    w = ds[0].replace('[', '').split(',')
    h = ds[1].replace(']', '').split(',')
    hh = []
    for i in range(len(w)):
        wi = int(w[i])
        if wi == 1:
            hh.append(int(h[i]))
        else:
            for j in range(wi):
                hh.append(int(h[i]))
    # print(hh)
    print(largest_rec(hh))
