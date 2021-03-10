
import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_fix_mins_maxs(mins, maxs):
    deltas = (maxs - mins) / 12.
    mins = mins + deltas / 4.
    maxs = maxs - deltas / 4.

    return [mins, maxs]


minmax = get_fix_mins_maxs(-0.08, 0.08)

# Generate Example Data
x = [0.04, 0, -0.04]
y = [0.04, 0, -0.04]
z = [0.04, 0, -0.04]

# Start plotting environment
fig = plt.figure()
ax = Axes3D(fig)

# Plot 3 lines positioned against the axes "walls"
ax.plot(x, y, -0.08, zdir='z', c='r')
ax.plot(x, z, 0.08, zdir='y', c='g')
ax.plot(y, z, -0.08, zdir='x', c='b')

# Label each axis
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# Set each axis limits
ax.set_xlim(minmax)
ax.set_ylim(minmax)
ax.set_zlim(minmax)

# Equally stretch all axes
# ax.set_aspect("equal")

# Set plot size for saving to disk
plt.gcf().set_size_inches(11.7, 8.3)

# Save figure in .eps and .png format
# plt.savefig('test.eps', format='eps')
# plt.savefig('test.png', format='png', dpi=300)

plt.show()