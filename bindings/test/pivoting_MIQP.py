import matplotlib.pyplot as plt
import math
import numpy as np
import time
import gurobipy as gp
from gurobipy import GRB,quicksum
from tqdm import tqdm
from lemke_lcp import lemkelcp

from pydrake.all import (
    ChooseBestSolver,
    MakeSolver,
    MathematicalProgram,
    Solve
)

class piovting_MIQP:
    def __init__(self,x0,xd,u0,N):
        self.x0 = x0
        self.u0 = u0
        self.N = N
        self.n = 10
        self.m = 10
        self.k = 4
        self.M = 10e4
        self.xd = np.asarray(xd)


    def pivoting(self):
        n = 10
        m = 10
        k = 4

        mu1 = 0.4
        mu2 = 0.4
        mu3 = 0.8

        g = 9.81
        dt = 0.01
        h = 1
        w = 1
        mm = 1
        y = self.x0[2]
        alpha = self.x0[4]
        f1 = self.x0[6]
        f2 = self.x0[8]
        N1 = self.u0[2]
        N2 = self.u0[3]

        rt = np.sqrt(h * h + w * w)
        sinn = math.sin(alpha)
        coss = math.cos(alpha)
        I_inv = (3 / 2)
        z = (-1) * (w * sinn + h * coss)
        # x4 = 0
        # x6 = 0
        # x8 = 0
        #
        # x5 = 0
        # x7 = 0

        Ainit = np.array([[1, dt, 0, 0, dt * dt * (N1 * sinn + N2 * coss), 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, dt * (N1 * sinn + N2 * coss), 0, 0, 0, 0, 0],
                          [0, 0, 1, dt, dt * dt * (N1 * coss - N2 * sinn), 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, dt * (N1 * coss - N2 * sinn), 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, dt, - I_inv * dt * dt * N1, 0, I_inv * dt * dt * N2, 0],
                          [0, 0, 0, 0, 0, 1, -I_inv * dt * N1, 0, I_inv * dt * N2, 0],
                          [0, 0, 0, 0, 0, 0, 1, dt, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1, dt],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

        Binit = np.array([[0, 0, -dt * dt * coss, dt * dt * sinn],
                          [0, 0, -dt * coss, dt * sinn],
                          [0, 0, dt * dt * sinn, dt * dt * coss],
                          [0, 0, dt * sinn, dt * coss],
                          [0, 0, dt * dt * I_inv * (-f1), dt * dt * I_inv * (f2)],
                          [0, 0, dt * I_inv * (-f1), dt * I_inv * (f2)],
                          [dt * dt, 0, 0, 0],
                          [dt, 0, 0, 0],
                          [0, dt * dt, 0, 0, ],
                          [0, dt, 0, 0]])

        Einit = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, dt * (N1 * sinn + N2 * coss), z, z * dt * I_inv * (-N1), 0,
                           z * dt * I_inv * (N2), 0],
                          [0, -1, 0, 0, -dt * (N1 * sinn + N2 * coss), -z, -z * dt * I_inv * (-N1), 0,
                           -z * dt * I_inv * (N2), 0],
                          [0, 0, 1, dt, dt * dt * (N1 * coss - N2 * sinn) + (-rt * coss + h * sinn),
                           dt * (-rt * coss + h * sinn), (-rt * coss + h * sinn) * dt * dt * I_inv * (-N1), 0,
                           (-rt * coss + h * sinn) * dt * dt * I_inv * (N2), 0]])

        Finit = np.array([[0, -1, -1, 0, 0, 0, 0, 0, 0, 0],
                          [1, dt, -dt, 0, 0, 0, 0, 0, 0, 0],
                          [1, -dt, dt, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, -1, -1, 0, 0, 0, 0],
                          [0, 0, 0, 1, dt, -dt, 0, 0, 0, 0],
                          [0, 0, 0, 1, -dt, dt, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, -1, -1, mu3],
                          [0, dt * sinn + z * dt * I_inv * (-rt), -dt * sinn - z * dt * I_inv * (-rt), 0,
                           dt * (-coss) + z * dt * I_inv * rt, dt * coss - z * dt * I_inv * rt, 1,
                           dt + dt * z * z * I_inv, -dt - dt * z * z * I_inv, dt * z * I_inv * (-rt * coss + h * sinn)],
                          [0, -dt * sinn - z * dt * I_inv * (-rt), dt * sinn + z * dt * I_inv * (-rt), 0,
                           dt * coss - z * dt * I_inv * rt, dt * (-coss) + z * dt * I_inv * rt, 1,
                           -dt - dt * z * z * I_inv, dt + dt * z * z * I_inv,
                           -dt * z * I_inv * (-rt * coss + h * sinn)],
                          [0, dt * dt * coss + (-rt * coss + h * sinn) * dt * dt * I_inv * (-rt),
                           -coss * dt * dt + (-rt * coss + h * sinn) * dt * dt * I_inv * rt, 0,
                           dt * dt * sinn + (-rt * coss + h * sinn) * dt * dt * I_inv * rt,
                           -dt * dt * sinn - (-rt * coss + h * sinn) * dt * dt * I_inv * rt, 0,
                           (-rt * coss + h * sinn) * dt * dt * I_inv * z,
                           -(-rt * coss + h * sinn) * dt * dt * I_inv * z,
                           dt * dt + (-rt * coss + h * sinn) * dt * dt * I_inv * (-rt * coss + h * sinn)]
                          ])

        Dinit = np.array(
            [[0, dt * dt * sinn, dt * dt * (-sinn), 0, dt * dt * (-coss), dt * dt * coss, 0, dt * dt, -dt * dt, 0],
             [0, dt * sinn, dt * (-sinn), 0, dt * (-coss), dt * coss, 0, dt, -dt, 0],
             [0, dt * dt * coss, dt * dt * (-coss), 0, dt * dt * sinn, dt * dt * (-sinn), 0, 0, 0, dt * dt],
             [0, dt * coss, dt * (-coss), 0, dt * sinn, dt * (-sinn), 0, 0, 0, dt],
             [0, I_inv * dt * dt * (-rt), I_inv * dt * dt * rt, 0, I_inv * dt * dt * rt, I_inv * dt * dt * (-rt), 0,
              I_inv * dt * dt * z, -I_inv * dt * dt * z, I_inv * dt * dt * (-rt * coss + h * sinn)],
             [0, I_inv * dt * (-rt), I_inv * dt * rt, 0, I_inv * dt * rt, I_inv * dt * (-rt), 0, I_inv * dt * z,
              -z * I_inv * dt, I_inv * dt * (-rt * coss + h * sinn)],
             [0, -dt * dt, dt * dt, 0, 0, 0, 0, 0, 0, 0],
             [0, -dt, dt, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, -dt * dt, dt * dt, 0, 0, 0, 0],
             [0, 0, 0, 0, -dt, dt, 0, 0, 0, 0]])

        c = np.array([[0],
                      [0],
                      [0],
                      [0],
                      [0],
                      [0],
                      [0],
                      [-alpha * dt * (N1 * sinn + N2 * coss) + dt * I_inv * z * (N1 * f1 - N2 * f2)],
                      [alpha * dt * (N1 * sinn + N2 * coss) - dt * I_inv * z * (N1 * f1 - N2 * f2)],
                      [-coss * h + dt * dt * (-alpha * (N1 * coss - N2 * sinn) - g) - sinn * w + (
                              -alpha + dt * dt * I_inv * (N1 * f1 - N2 * f2)) * (-rt * coss + h * sinn)]])

        d = np.array([[-alpha * dt * dt * (N1 * sinn + N2 * coss)],
                      [-dt * alpha * (N1 * sinn + N2 * coss)],
                      [dt * dt * (-alpha * (N1 * coss - N2 * sinn) - g)],
                      [dt * (-alpha * (N1 * coss - N2 * sinn) - g)],
                      [dt * dt * I_inv * (f1 * N1 - f2 * N2)],
                      [dt * I_inv * (f1 * N1 - f2 * N2)],
                      [0],
                      [0],
                      [0],
                      [0]])

        Hinit = np.array([[0, 0, mu1, 0],
                          [-dt, 0, 0, 0],
                          [dt, 0, 0, 0],
                          [0, 0, 0, mu2],
                          [0, -dt, 0, 0],
                          [0, dt, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, dt * (-coss) + z * dt * I_inv * (-f1), dt * sinn + z * dt * I_inv * f2],
                          [0, 0, dt * coss + (-z) * dt * I_inv * (-f1), -dt * sinn + (-z) * dt * I_inv * f2],
                          [0, 0, sinn * dt * dt + (-rt * coss + h * sinn) * dt * dt * I_inv * (-f1),
                           dt * dt * coss + (-rt * coss + h * sinn) * dt * dt * I_inv * f2]])

        d_col = d.reshape(10,)  # now shape is (10, 1)
        c_col = c.reshape(10,)  # now shape is (10, 1)

        self.A = [Ainit for _ in range(self.N)]
        self.A = np.asarray(self.A)
        self.B = [Binit for _ in range(self.N)]
        self.B = np.asarray(self.B)
        self.D = [Dinit for _ in range(self.N)]
        self.D = np.asarray(self.D)
        self.E = [Einit for _ in range(self.N)]
        self.E = np.asarray(self.E)
        self.H = [Hinit for _ in range(self.N)]
        self.H = np.asarray(self.H)
        self.F = [Finit for _ in range(self.N)]
        self.F = np.asarray(self.F)
        self.c = [c_col for _ in range(self.N)]
        self.c = np.asarray(self.c)
        self.d = [d_col for _ in range(self.N)]
        self.d = np.asarray(self.d)


        Rinit = 0.007 * np.eye(k)

        self.R = [Rinit for _ in range(self.N)]
        self.R = np.asarray(self.R)


        Qinit = np.eye(n)
        Qinit[4, 4] = 350
        Qinit[2, 2] = 100
        Qinit[0, 0] = 60
        Qinit[6, 6] = 50
        Qinit[8, 8] = 50
        Qinit[5, 5] = 11
        Qinit[3, 3] = 9
        Qinit[1, 1] = 11

        # Qinit[4, 4] = 300
        # Qinit[2, 2] = 100
        # Qinit[0, 0] = 70
        # Qinit[6, 6] = 50
        # Qinit[8, 8] = 50
        # Qinit[5, 5] = 11
        # Qinit[3, 3] = 9
        # Qinit[1, 1] = 11

        # Qinit[4,4] = 350
        # Qinit[2,2] = 100
        # Qinit[0,0] = 100
        # Qinit[6,6] = 50
        # Qinit[8,8] = 50
        # Qinit[5,5] = 11
        # Qinit[3,3] = 9
        # Qinit[1,1] = 11

        #self.Q = [Qinit for _ in range(self.N + 1)]
        self.Q = [Qinit for _ in range(self.N + 1)]
        self.Q = np.asarray(self.Q)

        # X0 = dare_slycot(lcs.A()[0], lcs.B()[0], Q[0], R[0])
        # Q.append(X0)

        # Q.append(DiscreteAlgebraicRiccatiEquation(lcs.A()[0], lcs.B()[0], Q[0], R[0]))

        Ginit = np.zeros((n + 2 * m + k, n + 2 * m + k))
        Ginit[n + m + k: n + m + k + m, n + m + k: n + m + k + m] = 0.65 * np.eye(m)
        Ginit[n: n + m, n: n + m] = 0.065 * np.eye(m)
        self.G = [Ginit for _ in range(self.N)]

        U = np.zeros((n + 2 * m + k, n + 2 * m + k))
        U[n: n + m, n: n + m] = np.eye(m)  # u2
        U[n + m + k: n + m + k + m, n + m + k: n + m + k + m] = 10000 * np.eye(m)  # u1
        self.U = [U for _ in range(self.N)]


    def solve(self):
        self.pivoting()
        obj_terms = []

        env = gp.Env(empty=True)

        # 2) set your WLS license params on that Env
        env.setParam("WLSACCESSID", "04c94f1f-f72b-41d6-bc06-cceab12fcfed")
        env.setParam("WLSSECRET", "398627c6-c1a4-4b7d-8e16-70f4970849f3")
        env.setParam("LICENSEID", 2682766)
        env.setParam("OutputFlag", 0)
        # 3) start it and build your model on it
        env.start()


        model = gp.Model(env=env)

        x = model.addMVar(
            (self.N + 1, self.n),
            ub=gp.GRB.INFINITY,lb=-gp.GRB.INFINITY,
            name="x"
        )

        u = model.addMVar((self.N,self.k), ub=gp.GRB.INFINITY,lb=-gp.GRB.INFINITY,name = "u")
        lambda_ = model.addMVar((self.N,self.m),ub=gp.GRB.INFINITY,lb=-gp.GRB.INFINITY,name = "lambda")
        s = model.addMVar((self.N,self.m),vtype=GRB.BINARY,name="s")
        ones = np.ones(self.m)


        model.addConstr(x[0] == self.x0)

        #print(self.A[0] @ x [0] + self.B[0] @ u[0] + self.D[0] @ lambda_[0] + self.d[0])

        for i in range(self.N):
            #print(self.d[i])
            model.addConstr(self.A[i] @ x[i] + self.B[i] @ u[i] + self.D[i] @ lambda_[i] + self.d[i] == x[i+1],name = f"dyn_{i}")
            model.addConstr(self.M * s[i] >= self.E[i] @ x[i] + self.F[i] @ lambda_[i] + self.H[i] @ u[i] + self.c[i])
            model.addConstr(self.E[i] @ x[i] + self.F[i] @ lambda_[i] + self.H[i] @ u[i] + self.c[i] >=0)
            model.addConstr(lambda_[i] >= 0)
            model.addConstr(self.M *(ones - s[i]) >= lambda_[i])

            model.addConstr(u[i][2] >= 0)
            model.addConstr(u[i][3] >= 0)

            #self.Q[i] = self.Q[i].tolist()


            obj_terms.append((x[i].T-self.xd.T) @ self.Q[i] @ (x[i]-self.xd))

            #obj_terms.append(-2*x[i,:].T @ right_matrix)

            obj_terms.append(u[i].T @ self.R[i] @ u[i])


        #right_matrix = np.asarray(self.Q[self.N] @ self.xd)

        obj_terms.append((x[self.N].T-self.xd.T) @ self.Q[self.N] @ (x[self.N]-self.xd))
        model.setObjective(quicksum(obj_terms), GRB.MINIMIZE)
        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            print("Model is infeasible")
            model.computeIIS()
            model.write("debug.ilp")
        elif model.Status == GRB.UNBOUNDED:
            print("Model is unbounded")
        # else:
        #     print("Status:", model.Status)



        self.x_opt = x.X
        self.u_opt = u.X
        self.lam_opt = lambda_.X

        #print(f"complem,{self.E[0] @ self.x_opt[0] + self.H[0] @ self.u_opt[0] + self.c[0]+self.F[0] @ self.lam_opt[0]}")

        return self.u_opt[0]

    def simulate(self):

        #print(f"ex,{self.E[0] @ self.x_opt[0] + self.H[0] @ self.u_opt[0] + self.c[0]}")
        #print(f"u_opt,{self.u_opt[0]}")
        #print(f"self.c,{self.c[0]}")

        prog = MathematicalProgram()
        force = prog.NewContinuousVariables(self.m)
        prog.AddLinearComplementarityConstraint(self.F[0], self.E[0] @ self.x_opt[0] + self.H[0] @ self.u_opt[0] + self.c[0], force)
        moby_id = ChooseBestSolver(prog)
        moby = MakeSolver(moby_id)

        result = Solve(prog)

        force_opt = result.GetSolution(force)

        #print(f"force_opt:{force_opt}")
        #print(f"complementary:{self.E[0] @ self.x_opt[0] + self.H[0] @ self.u_opt[0] + self.c[0] + self.F[0]@force_opt}")

        prediction = self.A[0] @ self.x_opt[0] + self.B[0] @ self.u_opt[0] + self.D[0] @ force_opt + self.d[0]
        dis = self.E[0] @ self.x_opt[0] + self.H[0] @ self.u_opt[0] + self.c[0] + self.F[0]@force_opt

        return prediction,force_opt,dis



if __name__ == "__main__":
    N = 10
    u_curr = np.zeros(4)
    dt = 0.01
    pre_save = []
    force_save = []
    u_save = []
    solve_times = []
    distance = []

    #x,v_x,y,v_y,alpha,v_alpha,f1,v_f1, f2, v_f2
    x0 = np.array([0,0,1.36,0,0.2,0,-0.3,0,-0.7,0])
    x_d_init = np.zeros(10)
    x_d_init[2] = 1.36602
    x_d_init[4] = 1.0471975512
    x_d_init[6] = 0.8
    x_d_init[8] = 0.8

    system_iter = 250

    is_normal_1 = np.zeros(system_iter+1)
    is_normal_2 = np.zeros(system_iter+1)
    pre_save.append(x0)


    friction_finger_1 = np.zeros((1, system_iter + 1))
    friction_finger_2 = np.zeros((1,system_iter+1))
    friction_finger_ground = np.zeros((1, system_iter+1))
    normal_finger_1 = np.zeros((1, system_iter + 1))
    normal_finger_2 = np.zeros((1, system_iter + 1))
    normal_ground = np.zeros((1, system_iter + 1))


    for i in tqdm(range(system_iter)):

        pivot = piovting_MIQP(x0, x_d_init, u_curr, N)
        start_time = time.perf_counter()
        u_opt = pivot.solve()
        solve_times.append(time.perf_counter() - start_time)

        prediction, force_opt, dis= pivot.simulate()

        pre_save.append(prediction)
        force_save.append(force_opt)
        u_save.append(u_opt)
        distance.append(dis)
        x0 = prediction
        u_curr = u_opt


        friction_finger_1[:,i+1] = force_opt[1] - force_opt[2]
        friction_finger_2[:,i+1] = force_opt[4] - force_opt[5]
        friction_finger_ground[:,i+1] = force_opt[7] - force_opt[8]
        normal_finger_1[:,i+1] = u_opt[2]
        normal_finger_2[:,i+1] = u_opt[3]
        normal_ground[:,i+1] = force_opt[9]



        #print(f"prediction: {x0}")
        #print(f"force_opt: {force_opt}")
        #print(f"distance: {dis}")

        if (u_opt[2] > 0.001):
            is_normal_1[i+1] = 1

        if (u_opt[3] > 0.001):
            is_normal_2[i+1] = 1


    pre_save = np.asarray(pre_save)
    force_save = np.asarray(force_save)
    u_save = np.asarray(u_save)


    np.save("prediction_MIQP.npy",pre_save)
    np.save("force_MIQP.npy",force_save)
    np.save("u_MIQP.npy",u_save)
    np.save("is_normal_1.npy",is_normal_1)
    np.save("is_normal_2.npy",is_normal_2)


    print(
        f"Average solve time: {np.mean(solve_times)}, equivalent to {1 / np.mean(solve_times)} Hz"
    )

    time_x = np.arange(0, system_iter * dt + dt, dt)
    np.save("dt", time_x)

    fig, ax = plt.subplots(1, 1, figsize=(8, 10))

    bottom_red,   top_red   = -0.25, 0.75
    bottom_blue, top_blue  =  0.75, 1.75
    ax.fill_between(time_x,
                    bottom_red, top_red,
                    where=(is_normal_1>0),
                    step='post',
                    facecolor='red',   alpha=0.3,
                    edgecolor='none',
                    zorder=0)

    ax.fill_between(time_x,
                    bottom_blue, top_blue,
                    where=(is_normal_2>0),
                    step='post',
                    facecolor='blue',  alpha=0.3,
                    edgecolor='none',
                    zorder=0)

    ax.plot(time_x, pre_save[:,0],label="x_obj")

    ax.plot(time_x, pre_save[:,2],label="y_obj")

    ax.plot(time_x, pre_save[:,4],label="alpha_obj")

    ax.plot(time_x, pre_save[:,6],label="f1")

    ax.plot(time_x, pre_save[:,8],label="f2")

    #ax.plot(time_x, height_lowest)

    ax.legend(["finger 1 normal force > 0.001","finger 2 normal force > 0.001","x_obj","y_obj","alpha_obj","f1","f2"])
    #ax[0].legend(["Cart Position", "Pole Angle", "Cart Velocity", "Pole Velocity"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("State")
    ax.set_title("MIQP pivoting example")
    plt.show()



    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 10))
    ax2.plot(time_x, friction_finger_1.T)
    ax2.plot(time_x, friction_finger_2.T)
    ax2.plot(time_x, friction_finger_ground.T)
    ax2.plot(time_x, normal_finger_1.T)
    ax2.plot(time_x, normal_finger_2.T)
    ax2.plot(time_x, normal_ground.T)

    ax2.legend(["friction_finger_1","friction_finger_2","friction_finger_ground","normal_finger_1","normal_finger_2","normal_ground"])

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Force")
    ax2.set_title("MIQP pivoting example")
    plt.show()