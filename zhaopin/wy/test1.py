import sys

if __name__ == '__main__':

    while True:

        nm = sys.stdin.readline().strip().replace('\n', '').split(' ')

        if len(nm) < 2:
            break

        n = int(nm[0])
        m = int(nm[1])

        ns = sys.stdin.readline().strip().replace('\n', '').split(' ')

        stacks = []

        for i in range(n):
            stacks.append([])

        # print(stacks)
        kiops = []
        for j in range(m):

            ki = int(sys.stdin.readline())

            kiop = []

            for i in range(ki):
                operations = sys.stdin.readline().strip().replace('\n', '').split(' ')
                kiop.append(operations)

            kiops.append(kiop)

        for i in range(m):

            hands = [0, 0]

            price_sum = 0

            for operations in kiops[i]:

                # print('op', i, ': ', operations)
                op = operations[1]

                # print('op', j, '-', i, ': ', operations)

                hand = -1
                if operations[0].startswith('r'):
                    hand = 1
                else:
                    hand = 0

                if op.startswith('t'):

                    goods_num = int(operations[2])
                    if len(stacks[goods_num - 1]) == 0:
                        hands[hand] = int(ns[goods_num - 1])
                    else:
                        hands[hand] = stacks[goods_num - 1][-1]

                elif op.startswith('k'):

                    price_sum += hands[hand]
                    # print('price_sum: ', price_sum)
                    hands[hand] = 0

                elif op.startswith('r'):

                    goods_num = int(operations[2])
                    stacks[goods_num - 1].append(hands[hand])
                    hands[hand] = 0

            # print(hands)
            # print(price_sum)
            print(int(price_sum + hands[0] + hands[1]))
