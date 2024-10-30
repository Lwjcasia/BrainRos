import numpy as np
from scipy.io import loadmat
import matplotlib
import matplotlib.pyplot as plt


def get_U_Matrix(weights):
    weights = np.transpose(weights, (1, 2, 0))
    X, Y, D = weights.shape
    um = np.nan * np.zeros((X, Y, 8))  # 8邻域

    ii = [0, -1, -1, -1, 0, 1, 1, 1]
    jj = [-1, -1, 0, 1, 1, 1, 0, -1]

    for x in range(X):
        for y in range(Y):
            w_2 = weights[x, y]

            for k, (i, j) in enumerate(zip(ii, jj)):
                if (x + i >= 0 and x + i < X and y + j >= 0 and y + j < Y):
                    w_1 = weights[x + i, y + j]
                    um[x, y, k] = np.linalg.norm(w_1 - w_2)

    um = np.nansum(um, axis=2)
    return um / um.max()


W = loadmat('./results/SOM/1_7999.mat')
weights = W['data']
UM = get_U_Matrix(weights)

dx, dy = 1, 1
x = np.arange(1, 127, dx)
y = np.arange(1, 127, dy)
X, Y = np.meshgrid(x, y)

extent = np.min(x), np.max(x), np.min(y), np.max(y)

plt.imshow(UM, cmap="Oranges", extent=extent, alpha=1)
plt.colorbar()
plt.title(label="U-matrix")
plt.show()


