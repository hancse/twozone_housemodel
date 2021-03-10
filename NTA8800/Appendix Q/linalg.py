# https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

"""
N_POINTS = 10
TARGET_X_SLOPE = 2
TARGET_y_SLOPE = 3
TARGET_OFFSET  = 5
EXTENTS = 5
NOISE = 5

# create random data
xs = [np.random.uniform(2*EXTENTS)-EXTENTS for i in range(N_POINTS)]
ys = [np.random.uniform(2*EXTENTS)-EXTENTS for i in range(N_POINTS)]
zs = []

for i in range(N_POINTS):
    zs.append(xs[i]*TARGET_X_SLOPE + \
              ys[i]*TARGET_y_SLOPE + \
              TARGET_OFFSET + np.random.normal(scale=NOISE))
"""
N_POINTS = 3
xs = np.array([7, 7, -7])
ys = np.array([35, 55, 35])
zs = np.array([4.0, 3.0, 2.5])

# plot raw data
plt.figure()
ax = plt.subplot(111, projection='3d')
# ax.set_xlim3d((10, -10))
# ax.set_ylim3d((25, 55))
ax.scatter(xs, ys, zs, color='r')


# do fit
tmp_A = []
tmp_b = []
for i in range(len(xs)):
    tmp_A.append([1, xs[i], ys[i]])
    tmp_b.append(zs[i])
b = np.matrix(tmp_b).T
A = np.matrix(tmp_A)
# matrix package depecated, use np.array
# .I (inversion) replaced by linalg.inv

# Manual solution
fit = (A.T * A).I * A.T * b
errors = b - A * fit
residual = np.linalg.norm(errors)

# Or use Scipy
# from scipy.linalg import lstsq
# fit, residual, rnk, s = lstsq(A, b)

print("solution:")
print("z = %f + %f x + %f y" % (fit[0], fit[1], fit[2]))
print("errors:")
print(errors)
print("residual:")
print(residual)

# plot plane
xlim = ax.get_xlim()
ylim = ax.get_ylim()
X,Y = np.meshgrid(np.arange(xlim[0], xlim[1], 2.0),
                  np.arange(ylim[0], ylim[1], 2.0))
Z = np.zeros(X.shape)
for r in range(X.shape[0]):
    for c in range(X.shape[1]):
        Z[r,c] = fit[1] * X[r,c] + fit[2] * Y[r,c] + fit[0]
ax.plot_wireframe(X,Y,Z, color='grey')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
angle = 45
ax.view_init(0, angle)
plt.show()
