# import sys
# import numpy as np
#
# if __name__ == '__main__':
#
#     t = int(sys.stdin.readline().strip().replace('\n', ''))
#     for i in range(t):
#         wh = sys.stdin.readline().strip().replace('\n', '').split(' ')
#         w = int(wh[0])
#         h = int(wh[1])
#
#         screen = []
#         people = []
#         for hh in range(h):
#             line = sys.stdin.readline().strip().replace('\n', '')
#             screen_line = []
#             for c in line:
#                 screen_line.append(c)
#             screen.append(screen_line)
#
#         pq = sys.stdin.readline().strip().replace('\n', '').split(' ')
#         p = int(pq[0])
#         q = int(pq[1])
#
#         for hh in range(q):
#             line = sys.stdin.readline().strip().replace('\n', '')
#             people_line = []
#             for c in line:
#                 people_line.append(c)
#             people.append(people_line)
#
#         ijab = sys.stdin.readline().strip().replace('\n', '').split(' ')
#         i = int(ijab[0])
#         j = int(ijab[1])
#         a = int(ijab[2])
#         b = int(ijab[3])
#
#         big_screen = [0] * (h + 2 * q)
#         for hq in range(h + 2 * q):
#             big_screen[hq] = [0] * (w + 2 * p)
#
#         print(i, j)
#         i += p
#         j += q
#         i -= 1
#         j -= 1
#
#         for hh in range(h):
#             for ww in range(w):
#                 ele = screen[hh][ww]
#                 big_screen[hh + q][ww + p] = ele
#
#         print(np.array(people).shape)
#         print(np.array(big_screen).shape)
#         # print(np.array(people).shape)
#         print(np.array(people))
#         print(np.array(big_screen))
#
#         q_start = i
#         p_start = j
#
#         if not q_start
#
#         for qq in range(q):
#             for pp in range(p):
#                 ele = people[qq][pp]
#
#                 big_screen[i + qq][j + pp] = ele
#
#         test_screen_empty()
