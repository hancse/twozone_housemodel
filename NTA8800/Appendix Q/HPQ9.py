
# https://www.kdnuggets.com/2016/11/linear-regression-least-squares-matrix-multiplication-concise-technical-overview.html
# https://stackoverflow.com/questions/53698635/how-to-define-a-plane-with-3-points-and-plot-it-in-3d
# https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points

import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def calc_WP_general(X_val, Y_val, Z_val, order=1):
    """general linear compressor or heat pump model as polynomial in Tevap and Tcond
       with optional cross terms

    Args:
        X_val: (ndarray): first regressor e.g. Tevap
        Y_val: (ndarray): second regressor e.g. Tcond
        Z_val: (ndarray): model target array e.g. COP or Power
        order: (int):     choose from 1, 2 or 3

    Returns:
       (ndarray):    coeficient array

    The model is: Z_val = p[0] + p[1]*X_val + p[2]*Y_val                              # first order, NTA8800, Appendix Q
                        \+ p[3]*X_val**2 + p[4]*Y_val**2 + p[5]*X_val*Y*val           # 2nd order, TRNSYS 401
                        \+ p[6] * X_val ** 3 + p[7] * Y_val ** 3
                        \+ p[8]*X_val*Y_val**2 + p[9]* Y_val*X_val**2                 # 3rd order, AHRI standard 540
    """
    # Setup matrices
    Npoints = np.shape(X_val)[0]
    y = np.zeros((Npoints, 1))
    y[:, 0] = Z_val

    # The model is: Z_val = p[0] + p[1]*X_val + p[2]*Y_val
    #                     + p[3]*X_val**2 + p[4]*Y_val**2 + p[5]*X_val*Y*val
    #                     + p[6]*X_val*Y_val**2 + p[7]* Y_val*X_val**2 + p[8] * X_val ** 3 + p[9] * Y_val ** 3

    if order == 1 and Npoints > 2:
        X = np.array([np.ones(Npoints), X_val, Y_val]).T
    elif order == 2 and Npoints > 5:
        X = np.array([np.ones(Npoints), X_val, Y_val,
                      np.square(X_val), np.square(Y_val), np.multiply(X_val, Y_val)]).T
    elif order == 3 and Npoints > 9:
        X = np.array([np.ones(Npoints), X_val, Y_val,
                      np.square(X_val), np.square(Y_val),
                      np.multiply(X_val, Y_val),
                      np.power(X_val, 3), np.power(Y_val, 3),
                      np.multiply(X_val, np.square(Y_val)), np.multiply(Y_val, np.square(X_val))]).T
    else:
        print("Order not given or Npoints too small")
        return None
    # Solve for projection matrix
    par = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    residuals = y - X.dot(par)
    SSE = np.linalg.norm(residuals)

    print("solution: z = %f + %f x + %f y" % (par[0], par[1], par[2]))
    print("residuals:")
    print(residuals)
    print("SSE: %f" % SSE)

    return par

def calc_plane(X_val, Y_val, Z_val):
    # Setup matrices
    Npoints = np.shape(X_val)[0]

    # The model is: Z_val = p[0] + p[1]*X_val + p[2]*Y_val
    Npar = 3
    X = np.zeros((Npoints, Npar))
    y = np.zeros((Npoints, 1))
    #X[:, 0] = np.ones(Npoints)
    #X[:, 1] = X_val
    #X[:, 2] = Y_val
    X = np.array([np.ones(Npoints), X_val, Y_val]).T
    y[:, 0] = Z_val

    # Solve for projection matrix
    par = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    residuals = y - X.dot(par)
    SSE = np.linalg.norm(residuals)

    print("solution: z = %f + %f x + %f y" % (par[0], par[1], par[2]))
    print("residuals:")
    print(residuals)
    print("SSE: %f" % SSE)

    return par

def plot_lines(x_val, ytop35, ytop45, ytop55, ybottom35, ybottom45, ybottom55):
    fig, ax = plt.subplots(2, 1)
    index = np.where(ytop35 > 1)
    ax[0].plot(x_val[index], ytop35[index], label='35$\degree C$')
    index = np.where(ytop45 > 1)
    ax[0].plot(x_val[index], ytop45[index], label='45$\degree C$')
    index = np.where(ytop55 > 1)
    ax[0].plot(x_val[index], ytop55[index], label='55$\degree C$')
    ax[0].legend()
    ax[0].set_xlabel('$T_{outside}$')
    ax[0].set_ylabel('COP')
    ax[0].axhline(y=1.0, color='r')
    index = np.where(ybottom35 > 0)
    ax[1].plot(x_val[index], ybottom35[index], label='35$\degree C$')
    index = np.where(ybottom45 > 0)
    ax[1].plot(x_val[index], ybottom45[index], label='45$\degree C$')
    index = np.where(ybottom55 > 0)
    ax[1].plot(x_val[index], ybottom55[index], label='55$\degree C$')
    ax[1].legend()
    ax[1].set_xlabel('$T_{outside}$')
    ax[1].set_ylabel('$P_{max}$')
    ax[1].axhline(y=0.0, color='r')
    ax[0].set_title('COP and $P_{max}$ according to NTA8800 Appendix Q9')
    plt.tight_layout()
    plt.show()


def plot_plane(x_val, y_val, z_val, par, zstring, zmin, zmax):
    # plot raw data
    fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111, projection="3d")
    ax = Axes3D(fig)
    angle = 45
    ax.view_init(45, angle)


    # ax.scatter(x_val, y_val, z_val, s=100, color='r')
    ax.plot3D(x_val, y_val, z_val, 'o-', markersize=10, color='r', zorder=2)
    mmx = get_fix_mins_maxs(-10, 10)
    ax.set_xlim3d(mmx)
    mmy = get_fix_mins_maxs(20, 60)
    ax.set_ylim3d(mmy)
    mmz = get_fix_mins_maxs(zmin, zmax)
    ax.set_zlim3d(mmz)

    plt.title('Bilinear fit to {0} according to NTA8800 Appendix Q'.format(zstring))
    ax.set_xlabel('Tin')
    ax.set_ylabel('Tout')
    ax.set_zlabel(zstring)

    # plot plane
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Find regression line
    xx, yy = np.meshgrid(np.arange(xlim[0], xlim[1], 2.0),
                         np.arange(ylim[0], ylim[1], 2.0))
    zz = np.array(par[0] + par[1] * xx + par[2] * yy)

    # zz = np.zeros(xx.shape)
    # for row in range(xx.shape[0]):
      #  for col in range(xx.shape[1]):
       #     zz[row, col] = par[0] + par[1] * xx[row, col] + par[2] * yy[row, col]

    # plot the surface
    # ax.plot_wireframe(xx, yy, zz, color='lime', alpha=0.5)
    ax.plot_surface(xx, yy, zz, color='lime', alpha=0.5)
    CS1 = ax.contour(xx, yy, zz, zdir = 'z', cmap=cm.winter)

    plt.show()


def get_fix_mins_maxs(mins, maxs):
    deltas = (maxs - mins) / 12.
    mins = mins + deltas / 4.
    maxs = maxs - deltas / 4.

    return [mins, maxs]


if __name__ == "__main__":
    print("Heat Pump Model (L/W) according to NTA8800:2020, Appendix Q9")
    Tin = np.array([7, 7, -7])
    Tout = np.array([35, 55, 35])
    COP = np.array([4.0, 3.0, 2.5])
    # coef = calc_plane(Tin, Tout, COP)
    coef = calc_WP_general(Tin, Tout, COP, order=1)
    plot_plane(Tin, Tout, COP, coef, 'COP', 1.0, 5.0)

    P_max = np.array([6.0, 2.0, 3.0])
    # P_coef = calc_plane(Tin, Tout, P_max)
    P_coef = calc_WP_general(Tin, Tout, P_max, order=1)
    plot_plane(Tin, Tout, P_max, P_coef, 'Power', 0.0, 10.0)

    Tin_space = np.linspace(-20, 20, 41, endpoint=True)
    COP_35 = coef[0] + coef[1]*Tin_space +coef[2]*35.0
    COP_45 = coef[0] + coef[1]*Tin_space +coef[2]*45.0
    COP_55 = coef[0] + coef[1]*Tin_space +coef[2]*55.0

    P_35 = P_coef[0] + P_coef[1]*Tin_space + P_coef[2]*35.0
    P_45 = P_coef[0] + P_coef[1]*Tin_space + P_coef[2]*45.0
    P_55 = P_coef[0] + P_coef[1]*Tin_space + P_coef[2]*55.0

    plot_lines(Tin_space, COP_35, COP_45, COP_55, P_35, P_45, P_55)



