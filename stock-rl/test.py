import numpy as np

X = np.array([1, 2, 3])
y = X.copy()

y[2] = 4
print(y)
print(X)