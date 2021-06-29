def transform(k, str):
    new_str = ''
    for i in range(len(str)):
        if str[i] == '-':
            new_str = str[0:i - 1]
            break

    index = 0
    count = 0
    temp_str = ''
    for i in range(len(new_str) + 1, len(str)):

        if str[i] == '-':
            continue
        else:
            count += 1
            temp_str += str[i]
            if str[i].isalpha():
                if str[i].isupper():
                    index += 1
                else:
                    index -= 1
            if count == k or (i == len(str) - 1 and count < k):
                if index > 0:
                    temp_str = temp_str.upper()
                elif index < 0:
                    temp_str = temp_str.lower()
                new_str = new_str + '-' + temp_str
                temp_str = ''
                index = 0
                count = 0

    return new_str


print(transform(3, '12abc-abCABc-4aBB'))
