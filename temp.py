from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
import matplotlib.pyplot as plt


X = np.arange(0, 1, 0.01)
Y = np.arange(1, 2, 0.001)

XX, YY = np.meshgrid(X, Y)

ZZ = np.tensordot(Y , X , axes=0)
ax = plt.figure().add_subplot(projection='3d')
ax.plot_surface(XX, YY, ZZ)
plt.show()

print(ZZ)
