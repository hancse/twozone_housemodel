# https://stackoverflow.com/questions/53698635/how-to-define-a-plane-with-3-points-and-plot-it-in-3d

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

points = [[0.65612, 0.53440, 0.24175],
           [0.62279, 0.51946, 0.25744],
           [0.61216, 0.53959, 0.26394]]

points = [[7, 35, 4.0],
           [7, 55, 3.0],
           [-7, 35,2.5]]

p0, p1, p2 = points
x0, y0, z0 = p0
x1, y1, z1 = p1
x2, y2, z2 = p2

ux, uy, uz = u = [x1-x0, y1-y0, z1-z0]
vx, vy, vz = v = [x2-x0, y2-y0, z2-z0]

u_cross_v = [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx]

point  = np.array(p0)
normal = np.array(u_cross_v)
print("normal:", normal)
d = -point.dot(normal)
print("d:", d)
print(1./ normal[2])
print("z = %f * xx + %f * yy + %f" % (-normal[0]/normal[2], -normal[1]/ normal[2], -d/normal[2]))

xx, yy = np.meshgrid(np.arange(-10, 20, 2.0), np.arange(20, 50, 2.0))

z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

# plot the surface
plt3d = plt.figure().gca(projection='3d')
plt3d.plot_surface(xx, yy, z)
angle = 135
plt3d.view_init(20, angle)
plt.show()