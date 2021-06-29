class Tree:
    def __init__(self, val, left, right):
        self.val = val
        self.left = left
        self.right = right


def dfs(tree1, tree2):
    res = 0

    # print(tree1.val, tree2.val)

    if tree1 == None:
        if tree2 == None:
            return res
        else:
            res += 1
            return res

    if tree1.val == tree2.val:
        res += dfs(tree1.left, tree2.left)
        res += dfs(tree1.right, tree2.right)
        return res
    else:
        res += 1
        return res


tree1 = Tree(1, None, None)
tree1.left = Tree(2, None, None)
tree1.right = Tree(None, None, None)

tree2 = Tree(1, None, None)
tree2.left = Tree(2, None, None)
tree2.right = Tree(3, None, None)

print(dfs(tree1, tree2))
