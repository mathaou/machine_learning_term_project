arr = ['4', '1', '1', '1', '3', '2', '3', '1', '2', '1', '7\n']
def encode_int(arr):
    ret = []
    temp = list(map(lambda x: "{0}{1}".format('{0:04b}'.format(int(arr[x])), '{0:04b}'.format(int(arr[x+1]))), range(0, len(arr) - 1, 2)))
    [ret.extend(list(x)) for x in temp]
    return ret

print(encode_int(arr))