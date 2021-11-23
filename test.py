import numpy as np

a = np.array([[1,2,3], [4,5,6]])
b = np.array([])

c = np.append(b,a, axis = 1)
print(c)