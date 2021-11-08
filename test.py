import numpy as np

a = np.array([[1,2,3], [4,5,6]])
b = np.array([[11,12,13], [14,15,16]])

c = np.concatenate((a,b), axis = 1)
print(c)