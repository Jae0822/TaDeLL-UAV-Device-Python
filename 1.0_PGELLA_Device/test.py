import copy
import numpy as np

# X = [0, 1, 2, 3, 4]
# Y = copy.deepcopy(X)
# s = 0
# while X:
#     for x in X:
#         # xx = Y[X[x]]
#         if s == 1:
#             X.remove(x)
#
# print(X)

# x = []
# for i in range(0, 5):
#     x.append([])
#     print(x)
# print(x)

l = [np.array([[1],[2]]), np.array([[2], [3]]), np.array([[4],[4]])]
x = 0
for i in range(0, len(l)):
    x = x + l[i][0][0]
x = x/(len(l))
print(x)