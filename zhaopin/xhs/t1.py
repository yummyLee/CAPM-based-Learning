import math
import sys

if __name__ == '__main__':
    ops = sys.stdin.readline().strip('').replace('\n', '').split(' ')

    scores = []

    for op in ops:

        if op == 'T':
            if len(scores) < 1:
                continue
            score = scores[-1] * 3
            scores.append(score)

        elif op == 'C':

            if len(scores) == 0:
                continue
            scores.pop()

        elif op == '-':

            if len(scores) < 2:
                continue
            score = abs(scores[-1] - scores[-2])
            scores.append(score)

        elif op == '+':
            if len(scores) < 2:
                continue
            score = scores[-1] + scores[-2]
            scores.append(score)

        elif op.isdigit() or (op.startswith('-') and op[1:len(op)].isdigit()):

            score = int(op)
            scores.append(score)

        # print(scores)

    print(sum(scores))
