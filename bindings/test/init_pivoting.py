import time
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import math
from slycot import sb02md
from pyc3 import (
    LCS,
    C3Options,
    ImprovedC3,
    ImprovedC3CostMatrices,
)

# pydrake.math import DiscreteAlgebraicRiccatiEquation


def init_pivoting(x_curr:np.array, N:int) -> LCS:

    x4 = x_curr[4]
    x6 = x_curr[6]
    x8 = x_curr[8]

    x5 = x_curr[6]
    x7 = x_curr[8]

    mu1 = 0.1
    mu2 = 9.81
    mu3 = 1.0
    g = 9.81
    dt = 0.01
    h = 1
    w = 1
    mm = 1
    rt = (-1) * np.sqrt(h * h + w * w)
    sinn = math.sin(x4)
    coss = math.cos(x4)
    z = w* sinn - h* coss


    Ainit = np.array([
        [1,  dt, 0,  0,  0,  0,  0,  0,  0,  0],
        [0,   1, 0,  0,  0,  0,  0,  0,  0,  0],

        [0,   0, 1,  dt, 0,  0,  0,  0,  0,  0],
        [0,   0, 0,   1, 0,  0,  0,  0,  0,  0],

        [0,   0, 0,   0, 1,  dt, 0,  0,  0,  0],
        [0,   0, 0,   0, 0,   1, 0,  0,  0,  0],

        [0,   0, 0,   0, 0,   0, 1,  dt, 0,  0],
        [0,   0, 0,   0, 0,   0, 0,   1, 0,  0],

        [0,   0, 0,   0, 0,   0, 0,   0, 1,  dt],
        [0,   0, 0,   0, 0,   0, 0,   0, 0,   1],
    ])

    Binit = np.array([
        [0,  0,    -dt*dt*coss,    dt*dt*sinn],
        [0,  0,     -dt*coss,      dt*sinn],
        [0,  0,     dt*dt*sinn,   dt*dt*coss],
        [0,  0,     dt*sinn,      dt*coss],
        [0,  0,   -dt*dt*x6,     dt*dt*x8],
        [0,  0,    -dt*x6,       dt*x8],
        [dt*dt,  0,    0,           0],
        [dt,     0,    0,           0],
        [0,    dt*dt,  0,           0],
        [0,     dt,    0,           0],
    ])

    Einit = np.array([
        [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 ],
        [  0,  0,  0,  0,  0,  0,  0, -1,  0,  0 ],
        [  0,  0,  0,  0,  0,  0,  0,  1,  0,  0 ],
        [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 ],
        [  0,  0,  0,  0,  0,  0,  0,  0,  0,  -1],
        [  0,  0,  0,  0,  0,  0,  0,  0,  0,  1 ],
        [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 ],
        [  0,  -1,  0,  0, 0,  -rt, 0,  0,  0,  0 ],
        [  0,  1,  0,  0,  0,  rt,  0,  0,  0,  0 ],
        [  0,  0,  1,  dt,  -h * sinn + w * coss+z, z*dt,  0,  0,  0,  0 ],
    ])

    Finit = np.array([
        [0,      -1,       -1,       0,    0,    0,    0,     0,     0,     0],
        [1,       dt,     -dt,       0,    0,    0,    0,     0,     0,     0],
        [1,      -dt,      dt,       0,    0,    0,    0,     0,     0,     0],
        [0,       0,        0,       0,   -1,   -1,    0,     0,     0,     0],
        [0,       0,        0,       1,    dt,   -dt,   0,     0,     0,     0],
        [0,       0,        0,       1,   -dt,    dt,   0,     0,     0,     0],
        [0,       0,        0,       0,    0,    0,    0,    -1,    -1,   mu3],
        [
            0,
            -(dt * sinn - rt * dt * h),
            -(-dt * sinn + rt * dt * h),
            0,
            -(-dt * coss + rt * dt * w),
            -(dt * coss - dt * rt * w),
            -(-1),
            -(dt - rt * dt * coss * w - rt * dt * sinn * h),
            -(-dt + rt * dt * coss * w + rt * dt * sinn * h),
            -(sinn * w * rt * dt - h * coss * rt * dt)
        ],
        [
            0,
            -(-dt * sinn + rt * dt * h),
            -(dt * sinn - rt * dt * h),
            0,
            -(dt * coss - rt * dt * w),
            -(-dt * coss + rt * dt * w),
            -(-1),
            -(-dt + rt * dt * coss * w + rt * dt * sinn * h),
            -(dt - rt * dt * coss * w - rt * dt * sinn * h),
            -(-sinn * w * rt * dt + h * coss * rt * dt)
        ],
        [
            0,
            dt * dt * coss - z * dt * dt * h,
            -dt * dt * coss + z * dt * dt * h,
            0,
            dt * dt * sinn + z * dt * dt * w,
            -dt * dt * sinn - dt * dt * w * z,
            0,
            -z * dt * dt * coss * w - z * dt * dt * sinn * h,
            z * dt * dt * coss * w + z * dt * dt * sinn * h,
            dt * dt + z * dt * dt * sinn * w - z * h * coss * dt * dt
        ]
    ])


    Dinit = np.array([
        [  0,    dt*dt*sinn,         -dt*dt*sinn,     0,      -coss*dt*dt,       dt*dt*coss,      0,         dt*dt,         -dt*dt,           0 ],
        [  0,    dt*sinn,            -dt*sinn,        0,      -coss*dt,          dt*coss,         0,         dt,            -dt,               0 ],
        [  0,    dt*dt*coss,         -dt*dt*coss,     0,       dt*dt*sinn,       -dt*dt*sinn,     0,         0,              0,             dt*dt ],
        [  0,    dt*coss,            -dt*coss,        0,       dt*dt*sinn,       -dt*sinn,        0,         0,              0,              dt ],
        [  0,   -dt*dt*h,            dt*dt*h,         0,       dt*dt*w,          -dt*dt*w,        0,     -dt*dt*coss*w - dt*dt*sinn*h,  dt*dt*coss*w + dt*dt*sinn*h,  dt*dt*sinn*w - h*coss*dt*dt ],
        [  0,   -dt*h,               dt*h,            0,       dt*w,             -dt*w,           0,     -dt*coss*w - dt*sinn*h,      dt*coss*w + dt*sinn*h,      dt*sinn*w - h*coss*dt ],
        [  0,   -dt*dt,              dt*dt,           0,         0,                0,             0,         0,               0,                0 ],
        [  0,   -dt,                 dt,             0,         0,                0,             0,         0,               0,                0 ],
        [  0,    0,                   0,             0,         -dt*dt,             dt*dt,         0,      0,               0,                0 ],
        [  0,    0,                   0,            0,        -dt,              dt,             0,         0,               0,                0 ],
    ])

    c = np.array([
        0,
        -h*g,
        h*g,
        0,
        -h*g,
        h*g,
        0,
        0,
        0,
        0
    ])

    d = np.array([
        0,
        0,
        -dt*dt*mm*g,
        -dt*mm*g,
        0,
        0,
        0,
        0,
        0,
        0
    ])


    Hinit = np.array([
        [0,               0,                  mu1,                             0],
        [-dt,             0,                  0,                               0],
        [ dt,             0,                  0,                               0],
        [  0,             0,                  0,                              mu2],
        [  0,           -dt,                  0,                               0],
        [  0,            dt,                  0,                               0],
        [  0,             0,                  0,                               0],
        [  0,             0,              -(-dt * coss - rt * dt * x5),     -(dt * sinn + dt * rt * x7)],
        [  0,             0,  -(dt * coss + rt * dt * x5),  -(-dt * sinn - dt * rt * x7)],
        [  0,             0,  dt*dt * sinn - dt*dt * x5 * z,   dt*dt * coss + dt*dt * x7 * z ]
    ])

    d_col = d.reshape(10, 1)   # now shape is (10, 1)
    c_col = c.reshape(10, 1)   # now shape is (10, 1)

    return LCS(Ainit, Binit, Dinit, d_col, Einit, Finit, Hinit, c_col,N, dt)


def make_pivoting_cost(lcs):


    N = lcs.N()
    n = lcs.num_states()
    m = lcs.num_lambdas()
    k = lcs.num_inputs()

    R = [0.01 * np.eye(k) for _ in range(N)]

    Qinit = np.eye(n)
    Qinit[4,4] = 100
    Qinit[2,2] = 100
    Qinit[0,0] = 100
    Qinit[6,6] = 50
    Qinit[8,8] = 50
    Qinit[5,5] = 11
    Qinit[3,3] = 9
    Qinit[1,1] = 11

    Q = [Qinit for _ in range(N+1)]

    #X0 = dare_slycot(lcs.A()[0], lcs.B()[0], Q[0], R[0])
    #Q.append(X0)

    #Q.append(DiscreteAlgebraicRiccatiEquation(lcs.A()[0], lcs.B()[0], Q[0], R[0]))

    Ginit = np.zeros((n + 2 * m + k, n + 2 * m + k))
    Ginit[n + m + k : n + m + k + m, n + m + k : n + m + k + m] = np.eye(m)
    Ginit[n : n + m, n : n + m] = np.eye(m)
    G = [Ginit for _ in range(N)]


    U = np.zeros((n + 2 * m + k, n + 2 * m + k))
    U[n : n + m, n : n + m] = np.eye(m)
    U[n + m + k : n + m + k + m, n + m + k : n + m + k + m] = 10000 * np.eye(m)
    U = [U for _ in range(N)]


    return ImprovedC3CostMatrices(Q, R, G, U)




