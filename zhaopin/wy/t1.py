if __name__ == '__main__':
    m, n = map(int, input().split(' '))

    tree = {}
    p_tree = {}

    for i in range(n):
        p, d, c = input().split(' ')
        if p not in tree:
            tree[p] = {}
            tree[p][d] = c
        else:
            tree[p][d] = c

        p_tree[c] = p

    res_list = set()

    for i in p_tree.keys():
        if i not in tree.keys():
            p = p_tree[i]
            if 'right' in tree[p].keys() and 'left' in tree[p].keys():
                left = tree[p]['left']
                right = tree[p]['right']
                if left not in tree.keys() and right not in tree.keys():
                    res_list.add(p)

    print(len(res_list))
