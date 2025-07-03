import time
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import math
from adjustText import adjust_text
#from init_pivoting import init_pivoting, make_pivoting_cost

from pyc3 import (
    LCS,
    C3Options,
    ImprovedC3,
    ImprovedC3CostMatrices,
    CostMatrices,
    C3MIQP
)
from matplotlib.patches import Patch

def init_pivoting(x_curr:np.array, N:int) -> LCS:

    x4 = x_curr[4]
    x6 = x_curr[6]
    x8 = x_curr[8]
    #
    x5 = x_curr[6]
    x7 = x_curr[8]

    # x4 = 0
    # x6 = 0
    # x8 = 0
    #
    # x5 = 0
    # x7 = 0

    mu1 = 0.5
    mu2 = 0.5
    mu3 = 0.5
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
        [  0,    dt*coss,            -dt*coss,        0,       dt*sinn,       -dt*sinn,        0,         0,              0,              dt ],
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

    Rinit = 0.01*np.eye(k)
    Rinit[0,0] = 1
    Rinit[1,1] = 1

    R = [Rinit for _ in range(N)]

    Qinit = np.eye(n)
    Qinit[4,4] = 1000
    Qinit[2,2] = 1000
    Qinit[0,0] = 1000
    Qinit[6,6] = 0
    Qinit[8,8] = 0
    Qinit[5,5] = 11
    Qinit[3,3] = 9
    Qinit[1,1] = 11

    Q = [Qinit for _ in range(N+1)]

    #X0 = dare_slycot(lcs.A()[0], lcs.B()[0], Q[0], R[0])
    #Q.append(X0)

    #Q.append(DiscreteAlgebraicRiccatiEquation(lcs.A()[0], lcs.B()[0], Q[0], R[0]))

    Ginit = np.zeros((n + 2 * m + k, n + 2 * m + k))
    Ginit[n + m + k : n + m + k + m, n + m + k : n + m + k + m] = 1*np.eye(m)
    Ginit[n : n + m, n : n + m] = 2*np.eye(m)
    G = [Ginit for _ in range(N)]


    U = np.zeros((n + 2 * m + k, n + 2 * m + k))
    U[n : n + m, n : n + m] = np.eye(m) #u2
    U[n + m + k : n + m + k + m, n + m + k : n + m + k + m] = 5000 * np.eye(m)  #u1
    U = [U for _ in range(N)]


    return ImprovedC3CostMatrices(Q, R, G, U)


def drawgraph_t_pred_lambda_gamma(N_,AMDD_iter,contact_index,figure):
    n_x = 10
    n_lambda = 10
    n_u = 4
    iter = 100


    #qp_lambda = np.zeros((iter,n_lambda))
    #delta_lambda = np.zeros((iter,n_lambda))

    qp_sol = np.asarray(np.load("qp_data.npy"))
    delta_sol = np.asarray(np.load("delta_data.npy"))
    time_x = np.asarray(np.load("dt.npy"))
    print(qp_sol.shape)

    qp_lambda = qp_sol[:,:,n_x:n_x+n_lambda]
    delta_lambda = delta_sol[:,:,n_x:n_x+n_lambda]

    f1_lambda_qp = qp_lambda[:,:,0:3]
    f2_lambda_qp = qp_lambda[:,:,3:6]
    ground_lambda_qp = qp_lambda[:,:,6:4]

    f1_lambda_delta = delta_lambda[:,:,0:3]
    f2_lambda_delta  = delta_lambda[:,:,3:6]
    ground_lambda_delta  = delta_lambda[:,:,6:4]

    n_pred = np.arange(1,N_+1,1)

    print(f1_lambda_delta.shape)

    f_lambda_qp_draw = np.zeros((len(time_x),len(n_pred),3))

    f_lambda_delta_draw = np.zeros((len(time_x),len(n_pred),3))



    qp_gamma = qp_sol[:,:,n_x+n_lambda+n_u:n_x+n_lambda+n_u+n_lambda]
    delta_gamma = delta_sol[:,:,n_x+n_lambda+n_u:n_x+n_lambda+n_u+n_lambda]

    f1_gamma_qp = qp_gamma[:,:,0:3]
    f2_gamma_qp = qp_gamma[:,:,3:6]
    ground_gamma_qp = qp_gamma[:,:,6:4]

    f1_gamma_delta = delta_gamma[:,:,0:3]
    f2_gamma_delta  = delta_gamma[:,:,3:6]
    ground_gamma_delta  = delta_gamma[:,:,6:4]

    f_gamma_qp_draw = np.zeros((len(time_x),len(n_pred),3))
    f_gamma_delta_draw = np.zeros((len(time_x),len(n_pred),3))

    if (figure == 1):
        f_lambda_qp_draw[1:] = f1_lambda_qp
        f_lambda_delta_draw[1:] = f1_lambda_delta

        f_gamma_qp_draw[1:] = f1_gamma_qp
        f_gamma_delta_draw[1:] = f1_gamma_delta

    else:
        f_lambda_qp_draw[1:] = f2_lambda_qp
        f_lambda_delta_draw[1:] = f2_lambda_delta

        f_gamma_qp_draw[1:] = f2_gamma_qp
        f_gamma_delta_draw[1:] = f2_gamma_delta

    T, A = np.meshgrid(time_x, n_pred, indexing="ij")

    fig, ax = plt.subplots(2, 2, figsize=(12, 5),
                             subplot_kw={"projection": "3d"})


    Z_lambda_qp = f_lambda_qp_draw[:, :, contact_index]
    Z_lambda_delta = f_lambda_delta_draw[:, :, contact_index]

    Z_gamma_qp = f_gamma_qp_draw[:, :, contact_index]
    Z_gamma_delta = f_gamma_delta_draw[:, :, contact_index]

    # first surface → blue wireframe
    wire1 = ax[0,0].plot_wireframe(T, A, Z_lambda_qp,
                              rstride=3, cstride=3,
                              linewidth=0.8,
                              color="tab:blue",      # any colour name works
                              alpha=0.9)             # transparency is allowed

    # second surface → orange wireframe
    wire2 = ax[0,0].plot_wireframe(T, A, Z_lambda_delta,
                              rstride=3, cstride=3,
                              linewidth=0.8,
                              color="tab:orange",
                              alpha=0.9)

    # ─── labels & legend (wireframes don’t auto-label) ─────────────────
    ax[0,0].set_xlabel("time [s]")
    ax[0,0].set_ylabel("Predict step")
    ax[0,0].set_zlabel(r"$\lambda$ value")

    from matplotlib.lines import Line2D
    ax[0,0].legend(handles=[
        Line2D([0], [0], color="tab:blue",   lw=2, label=r"$qp}$"),
        Line2D([0], [0], color="tab:orange", lw=2, label=r"$delta$")
    ])

    wire1 = ax[0,1].plot_wireframe(T, A, Z_lambda_qp-Z_lambda_delta,
                                 rstride=3, cstride=3,
                                 linewidth=0.8,
                                 color="tab:red",      # any colour name works
                                 alpha=0.9)             # transparency is allowed

    ax[0,1].set_xlabel("time [s]")
    ax[0,1].set_ylabel("Predict step")
    ax[0,1].set_zlabel(r"$\lambda$ difference value")



    # first surface → blue wireframe
    wire1 = ax[1,0].plot_wireframe(T, A, Z_gamma_qp,
                                   rstride=3, cstride=3,
                                   linewidth=0.8,
                                   color="tab:blue",      # any colour name works
                                   alpha=0.9)             # transparency is allowed

    # second surface → orange wireframe
    wire2 = ax[1,0].plot_wireframe(T, A, Z_gamma_delta,
                                   rstride=3, cstride=3,
                                   linewidth=0.8,
                                   color="tab:orange",
                                   alpha=0.9)

    # ─── labels & legend (wireframes don’t auto-label) ─────────────────
    ax[1,0].set_xlabel("time [s]")
    ax[1,0].set_ylabel("Predict step")
    ax[1,0].set_zlabel(r"$\gamma$ value")

    from matplotlib.lines import Line2D
    ax[0,0].legend(handles=[
        Line2D([0], [0], color="tab:blue",   lw=2, label=r"$qp}$"),
        Line2D([0], [0], color="tab:orange", lw=2, label=r"$delta$")
    ])

    wire1 = ax[1,1].plot_wireframe(T, A, Z_gamma_qp-Z_gamma_delta,
                                   rstride=3, cstride=3,
                                   linewidth=0.8,
                                   color="tab:red",      # any colour name works
                                   alpha=0.9)             # transparency is allowed

    ax[1,1].set_xlabel("time [s]")
    ax[1,1].set_ylabel("Predict step")
    ax[1,1].set_zlabel(r"$\gamma$ difference value")




    plt.tight_layout()
    plt.show()

def drawgraph_t_pred_gamma(N_,AMDD_iter,contact_index):
    n_x = 10
    n_lambda = 10
    n_u = 4
    iter = 100


    #qp_lambda = np.zeros((iter,n_lambda))
    #delta_lambda = np.zeros((iter,n_lambda))

    qp_sol = np.asarray(np.load("qp_data.npy"))
    delta_sol = np.asarray(np.load("delta_data.npy"))
    time_x = np.asarray(np.load("dt.npy"))
    print(qp_sol.shape)

    qp_lambda = qp_sol[:,:,n_x+n_lambda+n_u:n_x+n_lambda+n_u+n_lambda]
    delta_lambda = delta_sol[:,:,n_x+n_lambda+n_u:n_x+n_lambda+n_u+n_lambda]

    f1_lambda_qp = qp_lambda[:,:,0:3]
    f2_lambda_qp = qp_lambda[:,:,3:6]


    f1_lambda_delta = delta_lambda[:,:,0:3]
    f2_lambda_delta  = delta_lambda[:,:,3:6]


    n_pred = np.arange(1,N_+1,1)

    print(f1_lambda_delta.shape)

    f1_lambda_qp_draw = np.zeros((len(time_x),len(n_pred),3))
    f1_lambda_qp_draw[1:] = f2_lambda_qp

    f1_lambda_delta_draw = np.zeros((len(time_x),len(n_pred),3))
    f1_lambda_delta_draw[1:] = f2_lambda_delta


    T, A = np.meshgrid(time_x, n_pred, indexing="ij")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5),
                           subplot_kw={"projection": "3d"})

    k = 2
    Z_qp = f1_lambda_qp_draw[:, :, contact_index]
    Z_delta = f1_lambda_delta_draw[:, :, contact_index]

    # first surface → blue wireframe
    wire1 = ax[0].plot_wireframe(T, A, Z_qp,
                                 rstride=3, cstride=3,
                                 linewidth=0.8,
                                 color="tab:blue",      # any colour name works
                                 alpha=0.9)             # transparency is allowed

    # second surface → orange wireframe
    wire2 = ax[0].plot_wireframe(T, A, Z_delta,
                                 rstride=3, cstride=3,
                                 linewidth=0.8,
                                 color="tab:orange",
                                 alpha=0.9)

    # ─── labels & legend (wireframes don’t auto-label) ─────────────────
    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("Predict step")
    ax[0].set_zlabel(r"$\lambda$ value")

    from matplotlib.lines import Line2D
    ax[0].legend(handles=[
        Line2D([0], [0], color="tab:blue",   lw=2, label=r"$qp}$"),
        Line2D([0], [0], color="tab:orange", lw=2, label=r"$delta$")
    ])

    wire1 = ax[1].plot_wireframe(T, A, Z_qp-Z_delta,
                                 rstride=3, cstride=3,
                                 linewidth=0.8,
                                 color="tab:blue",      # any colour name works
                                 alpha=0.9)             # transparency is allowed

    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("Predict step")
    ax[1].set_zlabel(r"$\lambda$ difference value")


    plt.tight_layout()
    plt.show()

    #print(qp_lambda)
    #print(delta_lambda)


def drawgraph_t_pred_state(N_,state_index):
    n_x = 10
    n_lambda = 10
    n_u = 4
    iter = 100

    n_pred = np.arange(1,N_+1,1)
    #qp_lambda = np.zeros((iter,n_lambda))
    #delta_lambda = np.zeros((iter,n_lambda))

    qp_sol = np.asarray(np.load("qp_data.npy"))

    time_x = np.asarray(np.load("dt.npy"))

    x_ = qp_sol[:,:,:n_x]

    x_draw = np.zeros((len(time_x),len(n_pred),n_x))
    x_draw[1:] = x_

    x_state = x_draw[:,:,state_index]

    T, A = np.meshgrid(time_x, n_pred, indexing="ij")

    fig, ax = plt.subplots(1, 1, figsize=(12, 5),
                           subplot_kw={"projection": "3d"})

    wire1 = ax.plot_wireframe(T, A, x_state,
                                 rstride=3, cstride=3,
                                 linewidth=0.8,
                                 color="tab:blue",      # any colour name works
                                 alpha=0.9)             # transparency is allowed


    # ─── labels & legend (wireframes don’t auto-label) ─────────────────
    ax.set_xlabel("time [s]")
    ax.set_ylabel("Predict step")
    ax.set_zlabel(r"$\lambda$ value")

    plt.tight_layout()
    plt.show()


def drawgraph_t_pred_ground_lambda_gamma(N_,AMDD_iter,contact_index):
    n_x = 10
    n_lambda = 10
    n_u = 4
    iter = 100

    qp_sol = np.asarray(np.load("qp_data.npy"))
    delta_sol = np.asarray(np.load("delta_data.npy"))
    time_x = np.asarray(np.load("dt.npy"))

    qp_lambda = qp_sol[:,:,n_x:n_x+n_lambda]
    delta_lambda = delta_sol[:,:,n_x:n_x+n_lambda]

    ground_lambda_qp = qp_lambda[:,:,6:10]

    ground_lambda_delta  = delta_lambda[:,:,6:10]

    n_pred = np.arange(1,N_+1,1)


    f_lambda_qp_draw = np.zeros((len(time_x),len(n_pred),4))

    f_lambda_delta_draw = np.zeros((len(time_x),len(n_pred),4))



    qp_gamma = qp_sol[:,:,n_x+n_lambda+n_u:n_x+n_lambda+n_u+n_lambda]
    delta_gamma = delta_sol[:,:,n_x+n_lambda+n_u:n_x+n_lambda+n_u+n_lambda]

    ground_gamma_qp = qp_gamma[:,:,6:10]

    ground_gamma_delta  = delta_gamma[:,:,6:10]

    f_gamma_qp_draw = np.zeros((len(time_x),len(n_pred),4))
    f_gamma_delta_draw = np.zeros((len(time_x),len(n_pred),4))

    f_lambda_qp_draw[1:] = ground_lambda_qp
    f_lambda_delta_draw[1:] = ground_lambda_delta

    f_gamma_qp_draw[1:] = ground_gamma_qp
    f_gamma_delta_draw[1:] = ground_gamma_delta


    T, A = np.meshgrid(time_x, n_pred, indexing="ij")

    fig, ax = plt.subplots(2, 2, figsize=(12, 5),
                           subplot_kw={"projection": "3d"})


    Z_lambda_qp = f_lambda_qp_draw[:, :, contact_index]
    Z_lambda_delta = f_lambda_delta_draw[:, :, contact_index]

    Z_gamma_qp = f_gamma_qp_draw[:, :, contact_index]
    Z_gamma_delta = f_gamma_delta_draw[:, :, contact_index]

    # first surface → blue wireframe
    wire1 = ax[0,0].plot_wireframe(T, A, Z_lambda_qp,
                                   rstride=3, cstride=3,
                                   linewidth=0.8,
                                   color="tab:blue",      # any colour name works
                                   alpha=0.9)             # transparency is allowed

    # second surface → orange wireframe
    wire2 = ax[0,0].plot_wireframe(T, A, Z_lambda_delta,
                                   rstride=3, cstride=3,
                                   linewidth=0.8,
                                   color="tab:orange",
                                   alpha=0.9)

    # ─── labels & legend (wireframes don’t auto-label) ─────────────────
    ax[0,0].set_xlabel("time [s]")
    ax[0,0].set_ylabel("Predict step")
    ax[0,0].set_zlabel(r"$\lambda$ value")

    from matplotlib.lines import Line2D
    ax[0,0].legend(handles=[
        Line2D([0], [0], color="tab:blue",   lw=2, label=r"$qp}$"),
        Line2D([0], [0], color="tab:orange", lw=2, label=r"$delta$")
    ])

    wire1 = ax[0,1].plot_wireframe(T, A, Z_lambda_qp-Z_lambda_delta,
                                   rstride=3, cstride=3,
                                   linewidth=0.8,
                                   color="tab:red",      # any colour name works
                                   alpha=0.9)             # transparency is allowed

    ax[0,1].set_xlabel("time [s]")
    ax[0,1].set_ylabel("Predict step")
    ax[0,1].set_zlabel(r"$\lambda$ difference value")



    # first surface → blue wireframe
    wire1 = ax[1,0].plot_wireframe(T, A, Z_gamma_qp,
                                   rstride=3, cstride=3,
                                   linewidth=0.8,
                                   color="tab:blue",      # any colour name works
                                   alpha=0.9)             # transparency is allowed

    # second surface → orange wireframe
    wire2 = ax[1,0].plot_wireframe(T, A, Z_gamma_delta,
                                   rstride=3, cstride=3,
                                   linewidth=0.8,
                                   color="tab:orange",
                                   alpha=0.9)

    # ─── labels & legend (wireframes don’t auto-label) ─────────────────
    ax[1,0].set_xlabel("time [s]")
    ax[1,0].set_ylabel("Predict step")
    ax[1,0].set_zlabel(r"$\gamma$ value")

    from matplotlib.lines import Line2D
    ax[0,0].legend(handles=[
        Line2D([0], [0], color="tab:blue",   lw=2, label=r"$qp}$"),
        Line2D([0], [0], color="tab:orange", lw=2, label=r"$delta$")
    ])

    wire1 = ax[1,1].plot_wireframe(T, A, Z_gamma_qp-Z_gamma_delta,
                                   rstride=3, cstride=3,
                                   linewidth=0.8,
                                   color="tab:red",      # any colour name works
                                   alpha=0.9)             # transparency is allowed

    ax[1,1].set_xlabel("time [s]")
    ax[1,1].set_ylabel("Predict step")
    ax[1,1].set_zlabel(r"$\gamma$ difference value")


    plt.tight_layout()
    plt.show()


def drawgraph_N_ADMM_lambda_gamma(time_frame,AMDD_iter,contact_index):
    n_x = 10
    n_lambda = 10
    n_u = 4

    qp_debug = np.asarray(np.load("qp_debug.npy"))[time_frame]

    delta_debug = np.asarray(np.load("delta_debug.npy"))[time_frame]

    contact_force_qp = qp_debug[:,:,n_x+contact_index:n_x+contact_index+1]
    contact_force_delta = delta_debug[:,:,n_x+contact_index:n_x+contact_index+1]
    print(np.shape(contact_force_qp))

    AMDD_iter_array = np.arange(0,ADMM_iter+1,1)
    n_pred = np.arange(0,N_+1,1)

    T, A = np.meshgrid(AMDD_iter_array, n_pred, indexing="ij")


    contact_force_qp_draw = np.zeros((len(AMDD_iter_array),len(n_pred),1))
    contact_force_delta_draw = np.zeros((len(AMDD_iter_array),len(n_pred),1))

    contact_force_qp_draw[1:,1:,:] = contact_force_qp
    contact_force_delta_draw[1:,1:,:] = contact_force_delta

    qp_lamba = contact_force_qp_draw[:,:,0]
    delta_lambda = contact_force_delta_draw[:,:,0]



    contact_gamma_qp = qp_debug[:,:,n_x+n_lambda+n_u+contact_index:n_x+n_lambda+n_u+contact_index+1]
    contact_gamma_delta = delta_debug[:,:,n_x+n_lambda+n_u+contact_index:n_x+n_lambda+n_u+contact_index+1]

    contact_gamma_qp_draw = np.zeros((len(AMDD_iter_array),len(n_pred),1))
    contact_gamma_delta_draw = np.zeros((len(AMDD_iter_array),len(n_pred),1))

    contact_gamma_qp_draw[1:,1:,:] = contact_gamma_qp
    contact_gamma_delta_draw[1:,1:,:] = contact_gamma_delta

    qp_gamma = contact_gamma_qp_draw[:,:,0]
    delta_gamma = contact_gamma_delta_draw[:,:,0]




    fig, ax = plt.subplots(2, 2, figsize=(12, 5),
                           subplot_kw={"projection": "3d"})

    wire1 = ax[0,0].plot_wireframe(T, A, qp_lamba,
                                   rstride=3, cstride=3,
                                   linewidth=0.8,
                                   color="tab:blue",      # any colour name works
                                   alpha=0.9)             # transparency is allowed

    # second surface → orange wireframe
    wire2 = ax[0,1].plot_wireframe(T, A, delta_lambda,
                                   rstride=3, cstride=3,
                                   linewidth=0.8,
                                   color="tab:orange",
                                   alpha=0.9)
    from matplotlib.lines import Line2D
    ax[0,0].legend(handles=[
        Line2D([0], [0], color="tab:blue",   lw=2, label=r"$qp}$"),
    ])

    ax[0,1].legend(handles=[
        Line2D([0], [0], color="tab:orange",   lw=2, label=r"$delta}$"),
    ])

    ax[0,0].set_xlabel("ADMM Iteration")
    ax[0,0].set_ylabel("Predict step")
    ax[0,0].set_zlabel(r"$\lambda$ value")

    ax[0,1].set_xlabel("ADMM Iteration")
    ax[0,1].set_ylabel("Predict step")
    ax[0,1].set_zlabel(r"$\lambda$ value")


    wire1 = ax[1,0].plot_wireframe(T, A, qp_gamma,
                                   rstride=3, cstride=3,
                                   linewidth=0.8,
                                   color="tab:blue",      # any colour name works
                                   alpha=0.9)             # transparency is allowed

    # second surface → orange wireframe
    wire2 = ax[1,1].plot_wireframe(T, A, delta_gamma,
                                   rstride=3, cstride=3,
                                   linewidth=0.8,
                                   color="tab:orange",
                                   alpha=0.9)
    ax[1,0].legend(handles=[
        Line2D([0], [0], color="tab:blue",   lw=2, label=r"$qp}$"),
    ])

    ax[1,1].legend(handles=[
        Line2D([0], [0], color="tab:orange",   lw=2, label=r"$delta}$"),
    ])

    ax[1,0].set_xlabel("ADMM Iteration")
    ax[1,0].set_ylabel("Predict step")
    ax[1,0].set_zlabel(r"$\gamma$ value")

    ax[1,1].set_xlabel("ADMM Iteration")
    ax[1,1].set_ylabel("Predict step")
    ax[1,1].set_zlabel(r"$\gamma$ value")


    plt.tight_layout()
    plt.show()



def drawgraph_att_atADMM_iter(time_frame,N,ADMM_iter,state_index):
    n_x = 10
    n_lambda = 10
    n_u = 4

    qp_debug = np.asarray(np.load("qp_debug.npy"))[time_frame][ADMM_iter]

    print(f"qp_debug,{qp_debug}")
    state_draw = qp_debug[:,state_index]

    print(f"state_draw,{state_draw}")
    n_pred = np.arange(1,N_+1,1)

    plt.plot(n_pred,state_draw)
    plt.show()


def drawgraph_atN_ADMM_lambda_gamma(time_frame,N,ADMM_iter,contact_index,finger):
    n_x = 10
    n_lambda = 10
    n_u = 4

    qp_debug = np.asarray(np.load("qp_debug.npy"))[time_frame]

    delta_debug = np.asarray(np.load("delta_debug.npy"))[time_frame]

    qp_final_step = np.asarray(np.load("qp_data.npy"))[time_frame][N]

    output_force = np.asarray(np.load("output_force.npy"))[time_frame]

    output_distance = np.asarray(np.load("output_distance.npy"))[time_frame]


    f1_output_force = output_force[0:3]
    f2_output_force = output_force[3:6]

    f1_output_distance = output_distance[0:3]
    f2_output_distance = output_distance[3:6]

#lambda
    contact_force_qp = qp_debug[:,:,n_x:n_x+n_lambda]
    contact_force_delta = delta_debug[:,:,n_x:n_x+n_lambda]
    contact_force_qp_final = qp_final_step[n_x:n_x+n_lambda]
    #print(contact_force_qp_final)


    f1_contact_force_qp = contact_force_qp[:,:,0:3]
    f1_contact_force_delta = contact_force_delta[:,:,0:3]
    f1_contact_force_final = contact_force_qp_final[0:3]
    f1_contact_force_qp_draw = f1_contact_force_qp[:,N,contact_index]
    f1_contact_force_delta_draw = f1_contact_force_delta[:,N,contact_index]
    f1_contact_force_final_draw = f1_contact_force_final[contact_index]



    f2_contact_force_qp = contact_force_qp[:,:,3:6]
    f2_contact_force_delta = contact_force_delta[:,:,3:6]
    f2_contact_force_final = contact_force_qp_final[3:6]
    f2_contact_force_qp_draw = f2_contact_force_qp[:,N,contact_index]
    f2_contact_force_delta_draw = f2_contact_force_delta[:,N,contact_index]
    f2_contact_force_final_draw = f2_contact_force_final[contact_index]


    AMDD_iter_array = np.arange(0,ADMM_iter+1,1)
    n_pred = np.arange(0,N_+1,1)

    if (finger == 1):
        contact_force_qp_draw = f1_contact_force_qp_draw
        contact_force_delta_draw  = f1_contact_force_delta_draw
        contact_force_final = f1_contact_force_final_draw

        output_force_draw = f1_output_force[contact_index]
    else:
        contact_force_qp_draw = f2_contact_force_qp_draw
        contact_force_delta_draw  = f2_contact_force_delta_draw
        contact_force_final = f2_contact_force_final_draw

        output_force_draw = f2_output_force[contact_index]

#gamma
    contact_gamma_qp = qp_debug[:,:,n_x+n_lambda+n_u:n_x+n_lambda+n_u+n_lambda]
    contact_gamma_delta = delta_debug[:,:,n_x+n_lambda+n_u:n_x+n_lambda+n_u+n_lambda]
    contact_gamma_qp_final = qp_final_step[n_x+n_lambda+n_u:n_x+n_lambda+n_u+n_lambda]


    f1_contact_gamma_qp = contact_gamma_qp[:,:,0:3]
    f1_contact_gamma_delta = contact_gamma_delta[:,:,0:3]
    f1_contact_gamma_final = contact_gamma_qp_final[0:3]

    f1_contact_gamma_qp_draw = f1_contact_gamma_qp[:,N,contact_index]
    f1_contact_gamma_delta_draw = f1_contact_gamma_delta[:,N,contact_index]
    f1_contact_gamma_final_draw = f1_contact_gamma_final[contact_index]


    f2_contact_gamma_qp = contact_gamma_qp[:,:,3:6]
    f2_contact_gamma_delta = contact_gamma_delta[:,:,3:6]
    f2_contact_gamma_final = contact_gamma_qp_final[3:6]

    f2_contact_gamma_qp_draw = f2_contact_gamma_qp[:,N,contact_index]
    f2_contact_gamma_delta_draw = f2_contact_gamma_delta[:,N,contact_index]
    f2_contact_gamma_final_draw = f2_contact_gamma_final[contact_index]


    if (finger == 1):
        contact_gamma_qp_draw = f1_contact_gamma_qp_draw
        contact_gamma_delta_draw  = f1_contact_gamma_delta_draw
        contact_gamma_final = f1_contact_gamma_final_draw

        output_distance_draw = f1_output_distance[contact_index]
    else:
        contact_gamma_qp_draw = f2_contact_gamma_qp_draw
        contact_gamma_delta_draw  = f2_contact_gamma_delta_draw
        contact_gamma_final = f2_contact_gamma_final_draw

        output_distance_draw = f2_output_distance[contact_index]



    contact_pair_qp_draw = np.zeros((ADMM_iter,2))
    contact_pair_delta_draw = np.zeros((ADMM_iter,2))

    for i in range(ADMM_iter):
        contact_pair_qp_draw[i] = np.array([contact_force_qp_draw[i],contact_gamma_qp_draw[i]])
        contact_pair_delta_draw[i] = np.array([contact_force_delta_draw[i],contact_gamma_delta_draw[i]])


    #plt.scatter(contact_force_qp_draw,contact_gamma_qp_draw)
    texts = []
    for i in range(ADMM_iter):

        plt.scatter(contact_force_qp_draw[i],contact_gamma_qp_draw[i],color="green")
        txt = plt.text(contact_force_qp_draw[i],contact_gamma_qp_draw[i],i,color="green")
        texts.append(txt)

        if (contact_force_delta_draw[i] == 0):

            if (contact_gamma_delta_draw[i]<= 1e-8):
                plt.scatter(contact_force_delta_draw[i],contact_force_delta_draw[i],color="red")
                txt = plt.text(contact_force_delta_draw[i],contact_force_delta_draw[i],i,color = "red")
                texts.append(txt)
            else:
                #plt.axhline(contact_gamma_delta_draw[i],color = "red",linestyle="--",linewidth = 1)
                plt.axvline(0,color = "red",linestyle="--",linewidth = 1)
                txt = plt.text(0,contact_gamma_qp_draw[i],i,color = "red")
                texts.append(txt)
        else:
            #plt.axvline(contact_force_delta_draw[i],color = "blue",linestyle="--",linewidth = 1,label=i)
            plt.axhline(0,color = "blue",linestyle="--",linewidth = 1)
            txt = plt.text(contact_force_qp_draw[i],0,i,color = "blue")
            texts.append(txt)

        print(f"ADMM iter: {i}, projection step: {contact_force_delta_draw[i],contact_gamma_delta_draw[i]}")

    plt.scatter(contact_force_final,contact_gamma_final,color="black")
    txt = plt.text(contact_force_final,contact_gamma_final,"final")
    texts.append(txt)
    print(f"final: {contact_force_final,contact_gamma_final}")

    if (N == 0):
        plt.scatter(output_force_draw,output_distance_draw,color="brown")
        txt = plt.text(output_force_draw,output_distance_draw,"Output",color = "brown")
        texts.append(txt)
        print(f"output: {output_force_draw,output_distance_draw}")


    #plt.legend(title='Lines')
    plt.xlabel(r"$\lambda$ value")
    plt.ylabel(r"$\gamma$ value")

    plt.title(f"Time frame: {time_frame}, Predict step: {N+1}, contact_index: {contact_index}, finger: {finger}")

    adjust_text(texts,
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
                expand_text=(1.2, 1.2),
                expand_points=(1.2, 1.2))

    plt.show()



def drawgraph_ground_atN_ADMM_lambda_gamma(time_frame,N,ADMM_iter,contact_index):
    n_x = 10
    n_lambda = 10
    n_u = 4

    qp_debug = np.asarray(np.load("qp_debug.npy"))[time_frame]

    delta_debug = np.asarray(np.load("delta_debug.npy"))[time_frame]

    qp_final_step = np.asarray(np.load("qp_data.npy"))[time_frame][N]

    output_force = np.asarray(np.load("output_force.npy"))[time_frame]

    output_distance = np.asarray(np.load("output_distance.npy"))[time_frame]


    f1_output_force = output_force[6:10]


    f1_output_distance = output_distance[6:10]


    #lambda
    contact_force_qp = qp_debug[:,:,n_x:n_x+n_lambda]
    contact_force_delta = delta_debug[:,:,n_x:n_x+n_lambda]
    contact_force_qp_final = qp_final_step[n_x:n_x+n_lambda]
    #print(contact_force_qp_final)


    f1_contact_force_qp = contact_force_qp[:,:,6:10]
    f1_contact_force_delta = contact_force_delta[:,:,6:10]
    f1_contact_force_final = contact_force_qp_final[6:10]
    f1_contact_force_qp_draw = f1_contact_force_qp[:,N,contact_index]
    f1_contact_force_delta_draw = f1_contact_force_delta[:,N,contact_index]
    f1_contact_force_final_draw = f1_contact_force_final[contact_index]



    AMDD_iter_array = np.arange(0,ADMM_iter+1,1)
    n_pred = np.arange(0,N_+1,1)


    contact_force_qp_draw = f1_contact_force_qp_draw
    contact_force_delta_draw  = f1_contact_force_delta_draw
    contact_force_final = f1_contact_force_final_draw

    output_force_draw = f1_output_force[contact_index]


    #gamma
    contact_gamma_qp = qp_debug[:,:,n_x+n_lambda+n_u:n_x+n_lambda+n_u+n_lambda]
    contact_gamma_delta = delta_debug[:,:,n_x+n_lambda+n_u:n_x+n_lambda+n_u+n_lambda]
    contact_gamma_qp_final = qp_final_step[n_x+n_lambda+n_u:n_x+n_lambda+n_u+n_lambda]


    f1_contact_gamma_qp = contact_gamma_qp[:,:,6:10]
    f1_contact_gamma_delta = contact_gamma_delta[:,:,6:10]
    f1_contact_gamma_final = contact_gamma_qp_final[6:10]

    f1_contact_gamma_qp_draw = f1_contact_gamma_qp[:,N,contact_index]
    f1_contact_gamma_delta_draw = f1_contact_gamma_delta[:,N,contact_index]
    f1_contact_gamma_final_draw = f1_contact_gamma_final[contact_index]




    contact_gamma_qp_draw = f1_contact_gamma_qp_draw
    contact_gamma_delta_draw  = f1_contact_gamma_delta_draw
    contact_gamma_final = f1_contact_gamma_final_draw

    output_distance_draw = f1_output_distance[contact_index]




    contact_pair_qp_draw = np.zeros((ADMM_iter,2))
    contact_pair_delta_draw = np.zeros((ADMM_iter,2))

    for i in range(ADMM_iter):
        contact_pair_qp_draw[i] = np.array([contact_force_qp_draw[i],contact_gamma_qp_draw[i]])
        contact_pair_delta_draw[i] = np.array([contact_force_delta_draw[i],contact_gamma_delta_draw[i]])


    #plt.scatter(contact_force_qp_draw,contact_gamma_qp_draw)
    texts = []
    for i in range(ADMM_iter):

        plt.scatter(contact_force_qp_draw[i],contact_gamma_qp_draw[i],color="green")
        txt = plt.text(contact_force_qp_draw[i],contact_gamma_qp_draw[i],i,color="green")
        texts.append(txt)

        if (contact_force_delta_draw[i] == 0):

            if (contact_gamma_delta_draw[i]<= 1e-8):
                plt.scatter(contact_force_delta_draw[i],contact_force_delta_draw[i],color="red")
                txt = plt.text(contact_force_delta_draw[i],contact_force_delta_draw[i],i,color = "red")
                texts.append(txt)
            else:
                #plt.axhline(contact_gamma_delta_draw[i],color = "red",linestyle="--",linewidth = 1)
                plt.axvline(0,color = "red",linestyle="--",linewidth = 1)
                txt = plt.text(0,contact_gamma_qp_draw[i],i,color = "red")
                texts.append(txt)
        else:
            #plt.axvline(contact_force_delta_draw[i],color = "blue",linestyle="--",linewidth = 1,label=i)
            plt.axhline(0,color = "blue",linestyle="--",linewidth = 1)
            txt = plt.text(contact_force_qp_draw[i],0,i,color = "blue")
            texts.append(txt)

        print(f"ADMM iter: {i}, projection step: {contact_force_delta_draw[i],contact_gamma_delta_draw[i]}")

    plt.scatter(contact_force_final,contact_gamma_final,color="black")
    txt = plt.text(contact_force_final,contact_gamma_final,"final")
    texts.append(txt)
    print(f"final: {contact_force_final,contact_gamma_final}")

    if (N == 0):
        plt.scatter(output_force_draw,output_distance_draw,color="brown")
        txt = plt.text(output_force_draw,output_distance_draw,"Output",color = "brown")
        texts.append(txt)
        print(f"output: {output_force_draw,output_distance_draw}")


    #plt.legend(title='Lines')
    plt.xlabel(r"$\lambda$ value")
    plt.ylabel(r"$\gamma$ value")

    plt.title(f"Time frame: {time_frame}, Predict step: {N+1}, contact_index: {contact_index}, ground")

    adjust_text(texts,
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
                expand_text=(1.2, 1.2),
                expand_points=(1.2, 1.2))

    plt.show()




def main(N_,ADMM_iter):
    N = N_

    #x,v_x,y,v_y,alpha,v_alpha,f1,v_f1, f2, v_f2
    x0 = np.array([0,0,1.36,0,0.2,0,-0.3,0,-0.7,0])
    target = init_pivoting(x0,N)

    n = target.num_states()

    x_d_init = np.zeros(n)
    x_d_init[2] = math.sqrt(2)
    x_d_init[4] = math.pi/4
    x_d_init[6] = 0.9
    x_d_init[8] = 0.9

    xd = [x_d_init for _ in range(N + 1)]

    options = C3Options()
    options.admm_iter = ADMM_iter
    options.rho_scale = 2
    options.num_threads = 5
    options.delta_option = 0


    system_iter = 500

    x = np.zeros((n, system_iter + 1))

    x[:, 0] = x0.ravel()
    solve_times = []
    sdf_sol = []
    delta_sol = []
    qp_sol =[]
    qp_debug = []
    delta_debug = []

    height_lowest = []

    output_force = []
    output_distance = []

    for i in range(system_iter):
        target = init_pivoting(x0,N)
        costs = make_pivoting_cost(target)
        opt = ImprovedC3(target, costs, xd, options)


        start_time = time.perf_counter()
        opt.Solve(x[:, i])
        solve_times.append(time.perf_counter() - start_time)
        sdf_sol.append(opt.GetSDFSolution())
        delta_sol.append(opt.GetDualDeltaSolution())
        qp_sol.append(opt.GetFullSolution())

        qp_debug.append(opt.GetPrimalZAfterQP())
        delta_debug.append(opt.GetDualDeltaAfterProjection())

        u_opt = opt.GetInputSolution()[0]
        [prediction,force,distance]= target.Simulate_debug(x[:, i], u_opt)

        x[:, i + 1] = prediction
        x0 = prediction
        height = x0[2] - np.sqrt(2) * np.sin(x0[4] + math.pi/4)
        height_lowest.append(height)

        output_force.append(force)
        output_distance.append(distance)

        print(i*0.01)
        print(f"x0,{x0}")
        print(f"force: {force}")
        print(f"distance: {distance}")
        print(f"u,{u_opt}")
        print(f"height: {height}")
    sdf_sol = np.array(sdf_sol)
    delta_sol = np.array(delta_sol)
    qp_sol = np.array(qp_sol)
    height_lowest = np.array(height_lowest)

    output_force = np.array(output_force)
    output_distance = np.array(output_distance)

    print("qp_size",qp_sol.shape)

    np.save("qp_data",qp_sol)
    np.save("delta_data",delta_sol)

    np.save("qp_debug",qp_debug)
    np.save("delta_debug", delta_debug)

    np.save("output_force",output_force)
    np.save("output_distance",output_distance)



    print(x.T[-1])
    dt = target.dt()


    time_x = np.arange(0, system_iter * dt + dt, dt)

    np.save("dt",time_x)


    print(
        f"Average solve time: {np.mean(solve_times)}, equivalent to {1 / np.mean(solve_times)} Hz"
    )

    fig, ax = plt.subplots(1, 1, figsize=(8, 10))

    #ax.plot(time_x, x.T[:,0],label="x_obj")

    ax.plot(time_x, x.T[:,2],label="y_obj")

    ax.plot(time_x, x.T[:,4],label="alpha_obj")

    ax.plot(time_x, x.T[:,6],label="f1")

    ax.plot(time_x, x.T[:,8],label="f2")

    #ax.plot(time_x, height_lowest)

    ax.legend(["y_obj","alpha_obj","f1","f2"])
    #ax[0].legend(["Cart Position", "Pole Angle", "Cart Velocity", "Pole Velocity"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("State")
    ax.set_title("init pivoting""delta_data")


    plt.show()


if __name__ == "__main__":
    N_ = 10
    ADMM_iter = 10
    state_index = 4
    contact_index =3
    #main(N_,ADMM_iter)
    #drawgraph_t_pred_lambda_gamma(N_,ADMM_iter,contact_index,1)
    #drawgraph_t_pred_gamma(N_,ADMM_iter,contact_index)
    #drawgraph_t_pred_state(N_,4)

    #drawgraph_att_atADMM_iter(10,N_,9,4)
    #qp = np.asarray(np.load("qp_data.npy"))
    #print(np.shape(qp))


    #drawgraph_N_ADMM_lambda_gamma(0,ADMM_iter,contact_index)
    #drawgraph_t_pred_ground_lambda_gamma(N_,ADMM_iter,contact_index)
    #drawgraph_atN_ADMM_lambda_gamma(0,0,ADMM_iter,contact_index,1)
    drawgraph_ground_atN_ADMM_lambda_gamma(250,9,ADMM_iter,contact_index)