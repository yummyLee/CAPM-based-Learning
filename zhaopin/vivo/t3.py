def compileSeq(input):
    # write code here
    input = input.replace('"', '').split(',')
    index = []
    res = []

    is_compile = {}
    depend = {}

    for i in range(len(input)):
        if input[i] == '-1':
            res.append(str(i))
            depend[i] = -1
            is_compile[i] = True
        else:
            index.append(i)
            is_compile[i] = False
            depend[i] = int(input[i])

    # print(res)
    # print(depend)
    # print(is_compile)
    # print(index)

    for i in index:

        if is_compile[i]:
            continue

        depend_index = depend[i]
        s = [i]
        while depend_index != -1 and is_compile[depend_index] == False:
            s.append(depend_index)
            is_compile[depend_index] = True
            depend_index = depend[depend_index]

        # print(i, '---', s)
        for j in range(len(s)):
            res.append(str(s[-j - 1]))

    return "\"" + ",".join(res) + "\""


print(compileSeq('1,2,-1,-1'))
