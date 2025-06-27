import time
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import math

from init_pivoting import init_pivoting, make_pivoting_cost

from pyc3 import (
    LCS,
    C3Options,
    ImprovedC3,
    ImprovedC3CostMatrices,
)
from matplotlib.patches import Patch

def drawgraph():
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

    n_pred = np.arange(1,11,1)

    print(f1_lambda_delta.shape)

    f1_lambda_qp_draw = np.zeros((len(time_x),len(n_pred),3))
    f1_lambda_qp_draw[1:] = f1_lambda_qp

    f1_lambda_delta_draw = np.zeros((len(time_x),len(n_pred),3))
    f1_lambda_delta_draw[1:] = f1_lambda_delta


    T, A = np.meshgrid(time_x, n_pred, indexing="ij")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5),
                             subplot_kw={"projection": "3d"})

    k = 0
    Z_qp = f1_lambda_qp_draw[:, :, k]
    Z_delta = f1_lambda_delta_draw[:, :, k]

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
    ax[0].set_ylabel("ADMM iteration")
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
    ax[1].set_ylabel("ADMM iteration")
    ax[1].set_zlabel(r"$\lambda$ value")


    plt.tight_layout()
    plt.show()



    #print(qp_lambda)
    #print(delta_lambda)
def main():
    N = 10

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
    options.admm_iter = 10
    options.rho_scale = 2
    options.num_threads = 5
    options.delta_option = 0


    system_iter = 100

    x = np.zeros((n, system_iter + 1))

    x[:, 0] = x0.ravel()
    solve_times = []
    sdf_sol = []
    delta_sol = []
    qp_sol =[]

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
        u_opt = opt.GetInputSolution()[0]
        prediction = target.Simulate(x[:, i], u_opt)
        x[:, i + 1] = prediction
        x0 = prediction
        print(x0)
        print(i)

    sdf_sol = np.array(sdf_sol)
    delta_sol = np.array(delta_sol)
    qp_sol = np.array(qp_sol)

    print("qp_size",qp_sol.shape)

    np.save("qp_data",qp_sol)
    np.save("delta_data",delta_sol)


    print(x.T[-1])
    dt = target.dt()


    time_x = np.arange(0, system_iter * dt + dt, dt)

    np.save("dt",time_x)


    print(
        f"Average solve time: {np.mean(solve_times)}, equivalent to {1 / np.mean(solve_times)} Hz"
    )

    fig, ax = plt.subplots(1, 1, figsize=(8, 10))

    ax.plot(time_x, x.T[:,0],label="x_obj")

    ax.plot(time_x, x.T[:,2],label="y_obj")

    ax.plot(time_x, x.T[:,4],label="alpha_obj")

    ax.plot(time_x, x.T[:,6],label="f1")

    ax.plot(time_x, x.T[:,8],label="f2")

    ax.legend(["x_obj","y_obj","alpha_obj","f1","f2"])
    #ax[0].legend(["Cart Position", "Pole Angle", "Cart Velocity", "Pole Velocity"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("State")
    ax.set_title("init pivoting""delta_data")


    plt.show()


if __name__ == "__main__":
    main()
    #drawgraph()