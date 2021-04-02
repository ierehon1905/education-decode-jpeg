# Что в расчетах используется
from_arr = [2, 0, 3, 0, 0, 0, 0, 0,
            0, 1, 2, 0, 0, 0, 0, 0,
            0, -1, -1, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0]

# Что в файле в сыром виде
to_arr = [2, 0, 0, 0, 1, 3, 0, 2,
       -1, 1, 0, 0, -1, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 
       0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0]

def from_zigzag_order(arr):
    res = []
    level = 0
    pos = [0,0]
    size = int(len(arr)**0.5)
#     print(f"{size=}")
    for i in range(len(arr)):
        index = pos[0] + pos[1]*size
#         print(f"{index=}")
        res.append(arr[index])
        if (level % 2 == 0 and pos[1] == 0) or (level % 2 == 1 and pos[1]+1 == size):
            pos[0] += 1
            level += 1
        elif (level % 2 == 1 and pos[0] == 0) or (level % 2 == 0 and pos[0]+1 == size):
            pos[1] += 1
            level += 1
        elif level % 2 == 0:
            pos[0] += 1
            pos[1] -= 1
        elif level % 2 == 1:
            pos[0] -= 1
            pos[1] += 1
        else:
            print("Weird")
    return res

def to_zigzag_order(arr):
    level = 0
    pos = [0,0]
    size = int(len(arr)**0.5)
    res = [[0 for j in range(size)] for i in range(size)]
#     print(f"{size=}")
    for i in range(len(arr)):
        index = pos[0] + pos[1]*size
        val = arr[i]
        res[pos[1]][pos[0]] = val
#         print(f"{index=}")
#         res.append(arr[index])
        if (level % 2 == 0 and pos[1] == 0) or (level % 2 == 1 and pos[1]+1 == size):
            pos[0] += 1
            level += 1
        elif (level % 2 == 1 and pos[0] == 0) or (level % 2 == 0 and pos[0]+1 == size):
            pos[1] += 1
            level += 1
        elif level % 2 == 0:
            pos[0] += 1
            pos[1] -= 1
        elif level % 2 == 1:
            pos[0] -= 1
            pos[1] += 1
        else:
            print("Weird")
    return res

to_zig = to_zigzag_order(to_arr)
from_zig = from_zigzag_order(from_arr)

print("To zigzag (Что в расчетах используется)")
for l in to_zig:
    print(l)
print("From zigzag (Что в файле в сыром виде)")
for i in range(8):
    for j in range(8):
        print(from_zig[j + i*8], end=", ")
    print('')