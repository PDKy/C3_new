import time
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import math
from adjustText import adjust_text

from pyc3 import LCS, C3MIQP, C3Options, CostMatrices


def make_cartpole_with_soft_walls_dynamics(N: int) -> LCS:
    g = 9.81
    mp = 0.411
    mc = 0.978
    len_p = 0.6
    len_com = 0.4267
    d1 = 0.35
    d2 = -0.35
    ks = 100
    dt = 0.01
    A = np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, g * mp / mc, 0, 0],
            [0, g * (mc + mp) / (len_com * mc), 0, 0],
        ]
    )
    A = np.eye(A.shape[0]) + dt * A
    B = dt * np.array([[0], [0], [1 / mc], [1 / (len_com * mc)]])
    D = dt * np.array(
        [
            [0, 0],
            [0, 0],
            [(-1 / mc) + (len_p / (mc * len_com)), (1 / mc) - (len_p / (mc * len_com))],
            [
                (-1 / (mc * len_com))
                + (len_p * (mc + mp)) / (mc * mp * len_com * len_com),
                -(
                    (-1 / (mc * len_com))
                    + (len_p * (mc + mp)) / (mc * mp * len_com * len_com)
                ),
            ],
        ]
    )
    E = np.array([[-1, len_p, 0, 0], [1, -len_p, 0, 0]])
    F = (1.0 / ks) * np.eye(2)
    c = np.array([[d1], [-d2]])
    d = np.zeros((4, 1))
    H = np.zeros((2, 1))

    return LCS(A, B, D, d, E, F, H, c, N, dt)


def make_cartpole_costs(lcs: LCS) -> CostMatrices:
    N = lcs.N()
    n = lcs.num_states()
    m = lcs.num_lambdas()
    k = lcs.num_inputs()

    R = [np.eye(k) for _ in range(N)]
    Q = [
        np.array([[10, 0, 0, 0], [0, 3, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        for _ in range(N)
    ]
    Q.append(linalg.solve_discrete_are(lcs.A()[0], lcs.B()[0], Q[0], R[0]))
    G = 0.1 * np.eye(n + m + k)
    G[6, 6] = 0
    G = [G for _ in range(N)]

    px = 1000
    plam = 1
    pu = 0
    U = np.block(
        [
            [px * np.eye(n), np.zeros((n, m + k))],
            [np.zeros((m + k, n)), plam * np.eye(m + k)],
        ]
    )
    U[-1, -1] = pu
    U = [U for _ in range(N)]

    return CostMatrices(Q, R, G, U)

def init_pivoting(x_curr:np.array,u_curr:np.array ,N:int,noise:np.array) -> LCS:

    n = 10
    m = 10
    k = 4

    mu1 = 0.4
    mu2 = 0.4
    mu3 = 0.8

    g = 9.81
    dt = 0.01
    h=1
    w=1
    mm=1
    y = x_curr[2]
    alpha = x_curr[4]
    f1 = x_curr[6]
    f2 = x_curr[8]
    N1 = u_curr[2]
    N2 = u_curr[3]

    rt = np.sqrt(h*h + w*w)
    sinn = math.sin(alpha)
    coss = math.cos(alpha)
    I_inv = (3/2)
    z = (-1)* (w*sinn + h*coss)
    # x4 = 0
    # x6 = 0
    # x8 = 0
    #
    # x5 = 0
    # x7 = 0



    Ainit = np.array([[1, dt, 0, 0, dt*dt*(N1*sinn+N2*coss), 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, dt*(N1*sinn+N2*coss), 0, 0, 0, 0, 0],
                      [0, 0, 1, dt, dt*dt*(N1*coss-N2*sinn), 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, dt*(N1*coss-N2*sinn), 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, dt, - I_inv*dt*dt*N1, 0, I_inv*dt*dt*N2, 0],
                      [0, 0, 0, 0, 0, 1, -I_inv*dt*N1, 0, I_inv*dt*N2, 0],
                      [0, 0, 0, 0, 0, 0, 1, dt, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, dt],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

    Binit = np.array([[0, 0, -dt*dt*coss, dt*dt*sinn],
                      [0, 0,  -dt*coss, dt*sinn],
                      [0, 0,  dt*dt*sinn, dt*dt*coss],
                      [0, 0,  dt*sinn, dt*coss],
                      [0, 0, dt*dt*I_inv*(-f1), dt*dt*I_inv*(f2)],
                      [0, 0, dt*I_inv*(-f1), dt*I_inv*(f2)],
                      [dt*dt, 0, 0, 0],
                      [dt, 0, 0, 0],
                      [ 0, dt*dt, 0, 0,],
                      [0, dt, 0, 0]])


    Einit = np.array([[0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,-1,0,0],
                      [0,0,0,0,0,0,0,1,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,-1],
                      [0,0,0,0,0,0,0,0,0,1],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0, 1, 0, 0, dt*(N1*sinn+N2*coss), z, z*dt*I_inv*(-N1), 0, z*dt*I_inv*(N2), 0],
                      [0, -1, 0, 0, -dt*(N1*sinn+N2*coss), -z, -z*dt*I_inv*(-N1), 0, -z*dt*I_inv*(N2), 0],
                      [0, 0, 1, dt, dt*dt*(N1*coss - N2*sinn)+(-rt*coss+h*sinn), dt*(-rt*coss+h*sinn), (-rt*coss+h*sinn)*dt*dt*I_inv*(-N1),0 , (-rt*coss+h*sinn)*dt*dt*I_inv*(N2), 0]])

    Finit = np.array([[0,-1,-1,0,0,0,0,0,0,0],
                      [1,dt,-dt,0,0,0,0,0,0,0],
                      [1,-dt,dt,0,0,0,0,0,0,0],
                      [0,0,0,0,-1,-1,0,0,0,0],
                      [0,0,0,1,dt,-dt,0,0,0,0],
                      [0,0,0,1,-dt,dt,0,0,0,0],
                      [0,0,0,0,0,0,0,-1,-1,mu3],
                      [0, dt*sinn+z*dt*I_inv*(-rt), -dt*sinn-z*dt*I_inv*(-rt), 0, dt*(-coss)+z*dt*I_inv*rt, dt*coss-z*dt*I_inv*rt, 1, dt+dt*z*z*I_inv, -dt-dt*z*z*I_inv, dt*z*I_inv*(-rt*coss+h*sinn)],
                      [0, -dt*sinn-z*dt*I_inv*(-rt), dt*sinn+z*dt*I_inv*(-rt), 0, dt*coss-z*dt*I_inv*rt, dt*(-coss)+z*dt*I_inv*rt, 1, -dt-dt*z*z*I_inv, dt+dt*z*z*I_inv, -dt*z*I_inv*(-rt*coss+h*sinn)],
                      [0, dt*dt*coss+(-rt*coss+h*sinn)*dt*dt*I_inv*(-rt), -coss*dt*dt+(-rt*coss+h*sinn)*dt*dt*I_inv*rt, 0, dt*dt*sinn+(-rt*coss+h*sinn)*dt*dt*I_inv*rt, -dt*dt*sinn-(-rt*coss+h*sinn)*dt*dt*I_inv*rt, 0, (-rt*coss+h*sinn)*dt*dt*I_inv*z, -(-rt*coss+h*sinn)*dt*dt*I_inv*z, dt*dt+(-rt*coss+h*sinn)*dt*dt*I_inv*(-rt*coss+h*sinn)]
                      ])


    Dinit = np.array([[0, dt*dt*sinn, dt*dt*(-sinn), 0, dt*dt*(-coss), dt*dt*coss, 0, dt*dt, -dt*dt, 0],
                      [0, dt*sinn, dt*(-sinn), 0, dt*(-coss), dt*coss, 0, dt, -dt, 0],
                      [0, dt*dt*coss, dt*dt*(-coss), 0, dt*dt*sinn, dt*dt*(-sinn), 0, 0, 0, dt*dt],
                      [0, dt*coss, dt*(-coss), 0, dt*sinn, dt*(-sinn), 0, 0, 0, dt],
                      [0, I_inv*dt*dt*(-rt), I_inv*dt*dt*rt, 0, I_inv*dt*dt*rt, I_inv*dt*dt*(-rt), 0, I_inv*dt*dt*z, -I_inv*dt*dt*z, I_inv*dt*dt*(-rt*coss+h*sinn)],
                      [0, I_inv*dt*(-rt), I_inv*dt*rt, 0, I_inv*dt*rt, I_inv*dt*(-rt), 0, I_inv*dt*z, -z*I_inv*dt, I_inv*dt*(-rt*coss+h*sinn)],
                      [0, -dt*dt, dt*dt, 0, 0, 0, 0, 0, 0, 0],
                      [0, -dt, dt, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, -dt*dt, dt*dt, 0, 0, 0, 0],
                      [0, 0, 0, 0, -dt, dt, 0, 0, 0, 0]])

    c = np.array([[0],
                  [0],
                  [0],
                  [0],
                  [0],
                  [0],
                  [0],
                  [-alpha*dt*(N1*sinn+N2*coss)+dt*I_inv*z*(N1*f1-N2*f2)],
                  [alpha*dt*(N1*sinn+N2*coss)-dt*I_inv*z*(N1*f1-N2*f2)],
                  [-coss*h +dt*dt*(-alpha*(N1*coss-N2*sinn)-g)-sinn*w+(-alpha+dt*dt*I_inv*(N1*f1-N2*f2))*(-rt*coss+h*sinn)]])


    # d = np.array([[-alpha*dt*dt*(N1*sinn+N2*coss)],
    #               [-dt*alpha*(N1*sinn+N2*coss)],
    #               [dt*dt*(-alpha*(N1*coss-N2*sinn)-g)],
    #               [dt*(-alpha*(N1*coss-N2*sinn)-g)],
    #               [dt*dt*I_inv*(f1*N1 - f2*N2)],
    #               [dt*I_inv*(f1*N1 - f2*N2)],
    #               [0],
    #               [0],
    #               [0],
    #               [0]])

    d = np.array([[-alpha*dt*dt*(N1*sinn+N2*coss) + noise[0]],
                  [-dt*alpha*(N1*sinn+N2*coss) + noise[1]],
                  [dt*dt*(-alpha*(N1*coss-N2*sinn)-g) + noise[2]],
                  [dt*(-alpha*(N1*coss-N2*sinn)-g) + noise[3]],
                  [dt*dt*I_inv*(f1*N1 - f2*N2) + noise[4]],
                  [dt*I_inv*(f1*N1 - f2*N2) + noise[5]],
                  [0 + noise[6]],
                  [0 + noise[7]],
                  [0 + noise[8]],
                  [0+ noise[9]]])


    Hinit = np.array([[0,0,mu1,0],
                      [-dt,0,0,0],
                      [dt,0,0,0],
                      [0,0,0,mu2],
                      [0,-dt,0,0],
                      [0,dt,0,0],
                      [0,0,0,0],
                      [0, 0, dt*(-coss)+z*dt*I_inv*(-f1), dt*sinn+z*dt*I_inv*f2],
                      [0, 0, dt*coss + (-z)*dt*I_inv*(-f1), -dt*sinn+(-z)*dt*I_inv*f2],
                      [0, 0, sinn*dt*dt+(-rt*coss+h*sinn)*dt*dt*I_inv*(-f1), dt*dt*coss+(-rt*coss+h*sinn)*dt*dt*I_inv*f2]])

    d_col = d.reshape(10, 1)   # now shape is (10, 1)
    c_col = c.reshape(10, 1)   # now shape is (10, 1)


    return LCS(Ainit, Binit, Dinit, d_col, Einit, Finit, Hinit, c_col,N, dt)


def make_pivoting_cost(lcs):


    N = lcs.N()
    n = lcs.num_states()
    m = lcs.num_lambdas()
    k = lcs.num_inputs()

    #work one
    #Rinit = 0.007*np.eye(k)

    #Rinit = 0.01 * np.eye(k)

    #Rinit = 0.0005 * np.eye(k)


    Rinit = 0.03 * np.eye(k)

    R = [Rinit for _ in range(N)]

    Qinit = np.eye(n)
    #work one
    Qinit[4,4] = 350
    Qinit[2,2] = 100
    Qinit[0,0] = 60
    Qinit[6,6] = 50
    Qinit[8,8] = 50
    Qinit[5,5] = 11
    Qinit[3,3] = 9
    Qinit[1,1] = 11

    #test one
    # Qinit[4,4] = 400
    # Qinit[2,2] = 100
    # Qinit[0,0] = 60
    # Qinit[6,6] = 50
    # Qinit[8,8] = 50
    # Qinit[5,5] = 11
    # Qinit[3,3] = 9
    # Qinit[1,1] = 11


    Q = [Qinit for _ in range(N+1)]


    #work one
    Ginit = 0.02*np.ones((n + m + k, n +  m + k))




    G = [Ginit for _ in range(N)]



    Us = np.zeros((n + m + k, n + m + k))
    Us[0:n, 0:n] = 1000 * np.eye(n)
    Us[n:n + m, n:n + m] = np.eye(m)
    Us[n + m:n + m + k, n + m:n + m + k] = np.eye(k)

    U = [Us for _ in range(N)]


    return CostMatrices(Q, R, G, U)


def drawgraph_atN_ADMM_lambda_gamma(time_frame,N,ADMM_iter,contact_index,finger,flag_truefriction):
    n_x = 10
    n_lambda = 10
    n_u = 4

    qp_debug = np.asarray(np.load("qp_debug_origin.npy"))[time_frame]

    delta_debug = np.asarray(np.load("delta_debug_origin.npy"))[time_frame]

    qp_final_step = np.asarray(np.load("qp_data_origin.npy"))[time_frame][N]

    output_force = np.asarray(np.load("output_force_origin.npy"))[time_frame]

    output_distance = np.asarray(np.load("output_distance_origin.npy"))[time_frame]


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

        contact_force_qp_draw_positive = f1_contact_force_qp[:,N,1]
        contact_force_qp_draw_negative = f1_contact_force_qp[:,N,2]

        contact_force_delta_draw_positive = f1_contact_force_delta[:,N,1]
        contact_force_delta_draw_negative = f1_contact_force_delta[:,N,2]


        output_force_draw = f1_output_force[contact_index]
    else:
        contact_force_qp_draw = f2_contact_force_qp_draw
        contact_force_delta_draw  = f2_contact_force_delta_draw
        contact_force_final = f2_contact_force_final_draw

        contact_force_qp_draw_positive = f2_contact_force_qp[:,N,1]
        contact_force_qp_draw_negative = f2_contact_force_qp[:,N,2]

        contact_force_delta_draw_positive = f2_contact_force_delta[:,N,1]
        contact_force_delta_draw_negative = f2_contact_force_delta[:,N,2]

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

        contact_gamma_qp_draw_positive = f1_contact_gamma_qp[:,N,1]
        contact_gamma_qp_draw_negative = f1_contact_gamma_qp[:,N,2]

        contact_gamma_delta_draw_positive = f1_contact_gamma_delta[:,N,1]
        contact_gamma_delta_draw_negative = f1_contact_gamma_delta[:,N,2]

    else:
        contact_gamma_qp_draw = f2_contact_gamma_qp_draw
        contact_gamma_delta_draw  = f2_contact_gamma_delta_draw
        contact_gamma_final = f2_contact_gamma_final_draw

        output_distance_draw = f2_output_distance[contact_index]

        contact_gamma_qp_draw_positive = f2_contact_gamma_qp[:,N,1]
        contact_gamma_qp_draw_negative = f2_contact_gamma_qp[:,N,2]

        contact_gamma_delta_draw_positive = f2_contact_gamma_delta[:,N,1]
        contact_gamma_delta_draw_negative = f2_contact_gamma_delta[:,N,2]



    contact_pair_qp_draw = np.zeros((ADMM_iter,2))
    contact_pair_delta_draw = np.zeros((ADMM_iter,2))

    for i in range(ADMM_iter):
        contact_pair_qp_draw[i] = np.array([contact_force_qp_draw[i],contact_gamma_qp_draw[i]])
        contact_pair_delta_draw[i] = np.array([contact_force_delta_draw[i],contact_gamma_delta_draw[i]])


    #plt.scatter(contact_force_qp_draw,contact_gamma_qp_draw)
    texts = []

    if (not flag_truefriction):
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
            print(f"ADMM iter:{i}, QP step:{contact_force_qp_draw[i],contact_gamma_qp_draw[i]}")

    else:
        for i in range(ADMM_iter):

            plt.scatter(contact_force_qp_draw_positive[i],contact_gamma_qp_draw_positive[i],color="green")
            txt = plt.text(contact_force_qp_draw_positive[i],contact_gamma_qp_draw_positive[i],i,color="green")
            texts.append(txt)

            plt.scatter(contact_force_qp_draw_negative[i],contact_gamma_qp_draw_negative[i],color="orange")
            txt = plt.text(contact_force_qp_draw_negative[i],contact_gamma_qp_draw_negative[i],i,color="orange")
            texts.append(txt)



            if (contact_force_delta_draw_positive[i] == 0):

                if (contact_gamma_delta_draw_positive[i]<= 1e-8):
                    plt.scatter(contact_force_delta_draw_positive[i],contact_force_delta_draw_positive[i],color="green")
                    txt = plt.text(contact_force_delta_draw_positive[i],contact_force_delta_draw_positive[i],i,color = "grenn")
                    texts.append(txt)
                else:
                    #plt.axhline(contact_gamma_delta_draw[i],color = "red",linestyle="--",linewidth = 1)
                    plt.axvline(0,color = "red",linestyle="--",linewidth = 1)
                    txt = plt.text(0,contact_gamma_qp_draw_positive[i],i,color = "red")
                    texts.append(txt)
            else:
                #plt.axvline(contact_force_delta_draw[i],color = "blue",linestyle="--",linewidth = 1,label=i)
                plt.axhline(0,color = "blue",linestyle="--",linewidth = 1)
                txt = plt.text(contact_force_qp_draw_positive[i],0,i,color = "blue")
                texts.append(txt)

            if (contact_force_delta_draw_negative[i] == 0):

                if (contact_gamma_delta_draw_negative[i]<= 1e-8):
                    plt.scatter(contact_force_delta_draw_negative[i],contact_force_delta_draw_negative[i],color="orange")
                    txt = plt.text(contact_force_delta_draw_negative[i],contact_force_delta_draw_negative[i],i,color = "orange")
                    texts.append(txt)
                else:
                    #plt.axhline(contact_gamma_delta_draw[i],color = "red",linestyle="--",linewidth = 1)
                    plt.axvline(0,color = "red",linestyle="--",linewidth = 1)
                    txt = plt.text(0,contact_gamma_qp_draw_negative[i],i,color = "red")
                    texts.append(txt)
            else:
                #plt.axvline(contact_force_delta_draw[i],color = "blue",linestyle="--",linewidth = 1,label=i)
                plt.axhline(0,color = "blue",linestyle="--",linewidth = 1)
                txt = plt.text(contact_force_qp_draw_negative[i],0,i,color = "blue")
                texts.append(txt)





            print(f"ADMM iter: {i}, projection step: {contact_force_delta_draw[i],contact_gamma_delta_draw[i]}")
            print(f"ADMM iter:{i}, QP step:{contact_force_qp_draw[i],contact_gamma_qp_draw[i]}")


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

    qp_debug = np.asarray(np.load("qp_debug_origin.npy"))[time_frame]

    delta_debug = np.asarray(np.load("delta_debug_origin.npy"))[time_frame]

    qp_final_step = np.asarray(np.load("qp_data_origin.npy"))[time_frame][N]

    output_force = np.asarray(np.load("output_force_origin.npy"))[time_frame]

    output_distance = np.asarray(np.load("output_distance_origin.npy"))[time_frame]


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


def drawgraph_input_atN_ADMM_lambda_gamma(time_frame,N,ADMM_iter,contact_index):
    n_x = 10
    n_lambda = 10
    n_u = 4

    qp_debug = np.asarray(np.load("qp_debug_origin.npy"))[time_frame]
    qp_final_step = np.asarray(np.load("qp_data_origin.npy"))[time_frame][N]

    state_qp = qp_debug[:,:,n_x+n_lambda:n_x+n_lambda+n_u]
    N1 = state_qp[:,N,2]
    N2 = state_qp[:,N,3]

    state_final = qp_final_step[n_x+n_lambda:n_x+n_lambda+n_u]
    N1_final = state_final[2]
    N2_final = state_final[3]


    #AMDD_iter_array = np.arange(0,ADMM_iter,1)

    #plt.scatter(N1,AMDD_iter_array)
    # plt.scatter(AMDD_iter_array,N2)
    # plt.scatter(AMDD_iter_array,N1)

    print(N1,N1_final)
    print(N2,N2_final)
    # print(AMDD_iter_array)

    plt.show()



def main(N_,ADMM_iter):
    N = N_
    u_curr = np.zeros(4)
    #x,v_x,y,v_y,alpha,v_alpha,f1,v_f1, f2, v_f2
    x0 = np.array([0,0,1.36,0,0.2,0,-0.3,0,-0.7,0])
    noise = np.zeros(10)
    target = init_pivoting(x0,u_curr,N,noise)

    n = target.num_states()


    x_d_init = np.zeros(n)
    x_d_init[2] = 1.36602
    x_d_init[4] = 1.0471975512
    x_d_init[6] = 0.8
    x_d_init[8] = 0.8

    xd = [x_d_init for _ in range(N + 1)]

    options = C3Options()
    options.admm_iter = ADMM_iter
    options.rho_scale = 1.2
    options.num_threads = 0
    options.delta_option = 1


    system_iter = 800

    is_normal_1 = np.zeros(system_iter+1)
    is_normal_2 = np.zeros(system_iter+1)

    x = np.zeros((n, system_iter + 1))

    friction_finger_1 = np.zeros((1, system_iter + 1))
    friction_finger_2 = np.zeros((1,system_iter+1))
    friction_finger_ground = np.zeros((1, system_iter+1))
    normal_finger_1 = np.zeros((1, system_iter + 1))
    normal_finger_2 = np.zeros((1, system_iter + 1))
    normal_ground = np.zeros((1, system_iter + 1))


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
        sigma = 0.5
        noise = np.zeros(10)
        target = init_pivoting(x[:,i],u_curr,N,noise)
        costs = make_pivoting_cost(target)
        opt = C3MIQP(target, costs, xd, options)


        start_time = time.perf_counter()
        opt.Solve(x[:, i])
        solve_times.append(time.perf_counter() - start_time)
        sdf_sol.append(opt.GetSDFSolution())
        delta_sol.append(opt.GetDualDeltaSolution())
        qp_sol.append(opt.GetFullSolution())

        qp_debug.append(opt.GetDebugInfo())
        delta_debug.append(opt.GetQPInfo())

        u_opt = opt.GetInputSolution()[0]
        u_curr = u_opt


        if (u_opt[2] > 1):
            is_normal_1[i+1] = 1

        if (u_opt[3] > 1):
            is_normal_2[i+1] = 1

        [prediction,force,distance]= target.Simulate_debug(x[:, i], u_opt)

        prediction = prediction - noise

        x[:, i + 1] = prediction

        friction_finger_1[:,i+1] = force[1] - force[2]
        friction_finger_2[:,i+1] = force[4] - force[5]
        friction_finger_ground[:,i+1] = force[7] - force[8]
        normal_finger_1[:,i+1] = u_opt[2]
        normal_finger_2[:,i+1] = u_opt[3]
        normal_ground[:,i+1] = force[9]

        #x0 = prediction
        height = x0[2] - np.sqrt(2) * np.sin(x0[4] + math.pi/4)
        height_lowest.append(height)

        output_force.append(force)
        output_distance.append(distance)

        print(i*0.01)
        print(f"x0,{prediction}")
        print(f"force: {force}")
        print(f"distance: {distance}")
        print(f"u,{u_opt}")
        print(f"height: {height}")
        print(f"normal force ground: {force[9]}")

    delta_sol = np.array(delta_sol)
    qp_sol = np.array(qp_sol)


    output_force = np.array(output_force)
    output_distance = np.array(output_distance)
    qp_debug = np.asarray(qp_debug)
    delta_debug = np.asarray(delta_debug)

    print(f"qp_debug shape: {qp_debug.shape}")



    np.save("qp_data_origin",qp_sol)
    np.save("delta_data_origin",delta_sol)

    np.save("qp_debug_origin",qp_debug)
    np.save("delta_debug_origin", delta_debug)

    np.save("output_force_origin",output_force)
    np.save("output_distance_origin",output_distance)


    print(x.T[-1])
    dt = target.dt()


    print(
        f"Average solve time: {np.mean(solve_times)}, equivalent to {1 / np.mean(solve_times)} Hz"
    )

    time_x = np.arange(0, system_iter * dt + dt, dt)
    np.save("dt",time_x)

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


    ax.plot(time_x, x.T[:,0],label="x_obj")

    ax.plot(time_x, x.T[:,2],label="y_obj")

    ax.plot(time_x, x.T[:,4],label="alpha_obj")

    ax.plot(time_x, x.T[:,6],label="f1")

    ax.plot(time_x, x.T[:,8],label="f2")

    #ax.plot(time_x, height_lowest)

    ax.legend(["finger 1 normal force > 1","finger 2 normal force > 1","x_obj","y_obj","alpha_obj","f1","f2"])
    #ax[0].legend(["Cart Position", "Pole Angle", "Cart Velocity", "Pole Velocity"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("State")
    ax.set_title("Improved C3 pivoting example ratio:0.001")


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
    ax2.set_title("Improved C3 pivoting example ratio:0.001")
    plt.show()




if __name__ == "__main__":
    N_ = 10
    ADMM_iter = 5
    state_index = 4
    contact_index = 3
    main(N_,ADMM_iter)

