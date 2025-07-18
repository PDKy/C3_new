#include <chrono>
#include <string>
#include <iostream>

#include "core/c3_miqp.h"

#include "drake/math/discrete_algebraic_riccati_equation.h"

using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using Eigen::VectorXd;
using std::vector;

using c3::C3Options;

void init_cartpole(int* n_, int* m_, int* k_, int* N_, vector<MatrixXd>* A_,
                   vector<MatrixXd>* B_, vector<MatrixXd>* D_,
                   vector<VectorXd>* d_, vector<MatrixXd>* E_,
                   vector<MatrixXd>* F_, vector<MatrixXd>* H_,
                   vector<VectorXd>* c_, vector<MatrixXd>* Q_,
                   vector<MatrixXd>* R_, vector<MatrixXd>* G_,
                   vector<MatrixXd>* U_, VectorXd* x0,
                   vector<VectorXd>* xdesired_, C3Options* options);

void init_fingergait(int* n_, int* m_, int* k_, int* N_, vector<MatrixXd>* A_,
                     vector<MatrixXd>* B_, vector<MatrixXd>* D_,
                     vector<VectorXd>* d_, vector<MatrixXd>* E_,
                     vector<MatrixXd>* F_, vector<MatrixXd>* H_,
                     vector<VectorXd>* c_, vector<MatrixXd>* Q_,
                     vector<MatrixXd>* R_, vector<MatrixXd>* G_,
                     vector<MatrixXd>* U_, VectorXd* x0,
                     vector<VectorXd>* xdesired_, C3Options* options);

void init_pivoting(VectorXd xcurrent, int* n_, int* m_, int* k_, int* N_,
                   vector<MatrixXd>* A_, vector<MatrixXd>* B_,
                   vector<MatrixXd>* D_, vector<VectorXd>* d_,
                   vector<MatrixXd>* E_, vector<MatrixXd>* F_,
                   vector<MatrixXd>* H_, vector<VectorXd>* c_,
                   vector<MatrixXd>* Q_, vector<MatrixXd>* R_,
                   vector<MatrixXd>* G_, vector<MatrixXd>* U_, VectorXd* x0,
                   vector<VectorXd>* xdesired_, C3Options* options);

namespace c3 {

int DoMain(int argc, char* argv[]) {
  int example = 1;  /// 0 for cartpole, 1 for finger gaiting, 2 for pivoting
  std::cout << "Running example : " << example << std::endl;

  /// dimensions (n: state dimension, m: complementarity variable dimension, k:
  /// input dimension, N: MPC horizon)
  int nd, md, kd, Nd;
  /// variables (LCS)
  vector<MatrixXd> Ad, Bd, Dd, Ed, Fd, Hd;
  vector<VectorXd> dd, cd;
  /// initial condition(x0), tracking (xdesired)
  VectorXd x0;
  vector<VectorXd> xdesired;
  /// variables (cost, C3)
  vector<MatrixXd> Qd, Rd, Gd, Ud;
  /// C3 options
  C3Options options;

  int stateconstraint = 1;
  int inputconstraint = 2;
  int forceconstraint = 3;

  if (example == 0) {
    init_cartpole(&nd, &md, &kd, &Nd, &Ad, &Bd, &Dd, &dd, &Ed, &Fd, &Hd, &cd,
                  &Qd, &Rd, &Gd, &Ud, &x0, &xdesired, &options);
  } else if (example == 1) {
    init_fingergait(&nd, &md, &kd, &Nd, &Ad, &Bd, &Dd, &dd, &Ed, &Fd, &Hd, &cd,
                    &Qd, &Rd, &Gd, &Ud, &x0, &xdesired, &options);
  } else if (example == 2) {
    VectorXd xcurrent = VectorXd::Zero(10);
    init_pivoting(xcurrent, &nd, &md, &kd, &Nd, &Ad, &Bd, &Dd, &dd, &Ed, &Fd,
                  &Hd, &cd, &Qd, &Rd, &Gd, &Ud, &x0, &xdesired, &options);
  } else {
    std::cout << "ERROR:Set example parameter to 1, 2 or 3" << std::endl;
  }

  /// set parameters as const
  const vector<MatrixXd> A = Ad;
  const vector<MatrixXd> B = Bd;
  const vector<MatrixXd> D = Dd;
  const vector<MatrixXd> E = Ed;
  const vector<MatrixXd> F = Fd;
  const vector<MatrixXd> H = Hd;
  const vector<VectorXd> d = dd;
  const vector<VectorXd> c = cd;
  const vector<MatrixXd> Q = Qd;
  const vector<MatrixXd> R = Rd;
  const vector<MatrixXd> G = Gd;
  const vector<MatrixXd> U = Ud;
  const int N = Nd;
  const int n = nd;
  const int m = md;
  const int k = kd;
  const double dt = 0.01;

  LCS system(A, B, D, d, E, F, H, c, dt);
  C3::CostMatrices cost(Q, R, G, U);
  C3MIQP opt(system, cost, xdesired, options);

  if (example == 1) {
    /// clear all user constraints
    opt.RemoveConstraints();

    /// add user constraints on input
    RowVectorXd LinIneq = RowVectorXd::Zero(4);
    LinIneq(2) = 1;
    double lowerbound = 0;
    double upperbound = 10000;
    opt.AddLinearConstraint(LinIneq, lowerbound, upperbound, inputconstraint);
    LinIneq(2) = 0;
    LinIneq(3) = 1;
    opt.AddLinearConstraint(LinIneq, lowerbound, upperbound, inputconstraint);

    /// add user constraints on state
    lowerbound = 1;
    upperbound = 3;
    RowVectorXd LinIneq2 = VectorXd::Zero(6);
    LinIneq2(2) = 1;
    opt.AddLinearConstraint(LinIneq2, lowerbound, upperbound, stateconstraint);

    LinIneq2(2) = 0;
    LinIneq2(4) = 1;
    lowerbound = 3;
    upperbound = 5;
    opt.AddLinearConstraint(LinIneq2, lowerbound, upperbound, stateconstraint);
  }


  /// initialize ADMM variables (delta, w)
  std::vector<VectorXd> delta(N, VectorXd::Zero(n + m + k));
  std::vector<VectorXd> w(N, VectorXd::Zero(n + m + k));

  /// initialize ADMM reset variables (delta, w are reseted to these values)
  std::vector<VectorXd> delta_reset(N, VectorXd::Zero(n + m + k));
  std::vector<VectorXd> w_reset(N, VectorXd::Zero(n + m + k));

  int timesteps = 10;  // number of timesteps for the simulation

  /// create state and input arrays
  std::vector<VectorXd> x(timesteps, VectorXd::Zero(n));
  std::vector<VectorXd> input(timesteps, VectorXd::Zero(k));

  /// initialize at x0
  x[0] = x0;

  double total_time = 0;
  for (int i = 0; i < timesteps - 1; i++) {
    if (options.delta_option == 1) {
      /// reset delta and w (option 1)
      delta = delta_reset;
      w = w_reset;
      for (int j = 0; j < N; j++) {
        delta[j].head(n) = x[i];
      }
    } else {
      /// reset delta and w (default option)
      delta = delta_reset;
      w = w_reset;
    }


    if (example == 2) {
      init_pivoting(x[i], &nd, &md, &kd, &Nd, &Ad, &Bd, &Dd, &dd, &Ed, &Fd, &Hd,
                    &cd, &Qd, &Rd, &Gd, &Ud, &x0, &xdesired, &options);
      LCS system(A, B, D, d, E, F, H, c, dt);
      C3::CostMatrices cost(Q, R, G, U);
      C3MIQP opt(system, cost, xdesired, options);
    }


    auto start = std::chrono::high_resolution_clock::now();
    /// calculate the input given x[i]
    opt.Solve(x[i]);
    input[i] = opt.GetInputSolution()[0];


        auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Solve time:" << elapsed.count() << std::endl;
    total_time = total_time + elapsed.count();

    /// simulate the LCS
    x[i + 1] = system.Simulate(x[i], input[i]);

    /// print the state
    //std::cout << "state: " << x[i + 1] << std::endl;
  }
  std::cout << "Average time: " << total_time / (timesteps - 1) << std::endl;
  return 0;
}

}  // namespace c3

int main(int argc, char* argv[]) {
  return c3::DoMain(argc, argv);
}

/// initialize LCS parameters for cartpole
void init_cartpole(int* n_, int* m_, int* k_, int* N_, vector<MatrixXd>* A_,
                   vector<MatrixXd>* B_, vector<MatrixXd>* D_,
                   vector<VectorXd>* d_, vector<MatrixXd>* E_,
                   vector<MatrixXd>* F_, vector<MatrixXd>* H_,
                   vector<VectorXd>* c_, vector<MatrixXd>* Q_,
                   vector<MatrixXd>* R_, vector<MatrixXd>* G_,
                   vector<MatrixXd>* U_, VectorXd* x0,
                   vector<VectorXd>* xdesired_, C3Options* options) {
  int n = 4;
  int m = 2;
  int k = 1;
  int N = 10;

  float g = 9.81;
  float mp = 0.411;
  float mc = 0.978;
  float len_p = 0.6;
  float len_com = 0.4267;
  float d1 = 0.35;
  float d2 = -0.35;
  float ks = 100;
  float Ts = 0.01;

  /// initial condition(cartpole)
  VectorXd x0init(n);
  x0init << 0.1, 0, 0.3, 0;

  MatrixXd Ainit(n, n);
  Ainit << 0, 0, 1, 0, 0, 0, 0, 1, 0, g * mp / mc, 0, 0, 0,
      (g * (mc + mp)) / (len_com * mc), 0, 0;

  MatrixXd Binit(n, k);
  Binit << 0, 0, 1 / mc, 1 / (len_com * mc);

  MatrixXd Dinit(n, m);
  Dinit << 0, 0, 0, 0, (-1 / mc) + (len_p / (mc * len_com)),
      (1 / mc) - (len_p / (mc * len_com)),
      (-1 / (mc * len_com)) +
          (len_p * (mc + mp)) / (mc * mp * len_com * len_com),
      -((-1 / (mc * len_com)) +
        (len_p * (mc + mp)) / (mc * mp * len_com * len_com));

  MatrixXd Einit(m, n);
  Einit << -1, len_p, 0, 0, 1, -len_p, 0, 0;

  MatrixXd Finit(m, m);
  Finit << 1 / ks, 0, 0, 1 / ks;

  VectorXd cinit(m);
  cinit << d1, -d2;

  VectorXd dinit(n);
  dinit << 0, 0, 0, 0;

  MatrixXd Hinit = MatrixXd::Zero(m, k);

  MatrixXd Qinit(n, n);
  Qinit << 10, 0, 0, 0, 0, 3, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;

  MatrixXd Rinit(k, k);
  Rinit << 1;

  MatrixXd Ginit(n + m + k, n + m + k);
  Ginit = 0.1 * MatrixXd::Identity(n + m + k, n + m + k);
  Ginit(n + m + k - 1, n + m + k - 1) = 0;

  MatrixXd QNinit = drake::math::DiscreteAlgebraicRiccatiEquation(
      Ainit * Ts + MatrixXd::Identity(n, n), Ts * Binit, Qinit, Rinit);

  std::vector<MatrixXd> Qsetup(N + 1, Qinit);
  Qsetup.at(N) = QNinit;  // Switch to QNinit, solution of Algebraic Ricatti

  std::vector<MatrixXd> A(N, Ainit * Ts + MatrixXd::Identity(n, n));
  std::vector<MatrixXd> B(N, Ts * Binit);
  std::vector<MatrixXd> D(N, Ts * Dinit);
  std::vector<VectorXd> d(N, Ts * dinit);
  std::vector<MatrixXd> E(N, Einit);
  std::vector<MatrixXd> F(N, Finit);
  std::vector<VectorXd> c(N, cinit);
  std::vector<MatrixXd> H(N, Hinit);
  std::vector<MatrixXd> Q = Qsetup;
  std::vector<MatrixXd> R(N, Rinit);
  std::vector<MatrixXd> G(N, Ginit);

  VectorXd xdesiredinit;
  xdesiredinit = VectorXd::Zero(n);
  std::vector<VectorXd> xdesired(N + 1, xdesiredinit);

  MatrixXd Us(n + m + k, n + m + k);
  Us << 1000, 0, 0, 0, 0, 0, 0, 0, 1000, 0, 0, 0, 0, 0, 0, 0, 1000, 0, 0, 0, 0,
      0, 0, 0, 1000, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0;
  vector<MatrixXd> U(N, Us);

  C3Options optionsinit;
  optionsinit.admm_iter = 10;
  optionsinit.rho_scale = 2;
  optionsinit.num_threads = 0;
  optionsinit.delta_option = 0;

  *options = optionsinit;
  *x0 = x0init;
  *n_ = n;
  *m_ = m;
  *k_ = k;
  *N_ = N;
  *A_ = A;
  *B_ = B;
  *D_ = D;
  *d_ = d;
  *c_ = c;
  *E_ = E;
  *F_ = F;
  *H_ = H;
  *Q_ = Q;
  *R_ = R;
  *G_ = G;
  *U_ = U;
  *xdesired_ = xdesired;
};

void init_fingergait(int* n_, int* m_, int* k_, int* N_, vector<MatrixXd>* A_,
                     vector<MatrixXd>* B_, vector<MatrixXd>* D_,
                     vector<VectorXd>* d_, vector<MatrixXd>* E_,
                     vector<MatrixXd>* F_, vector<MatrixXd>* H_,
                     vector<VectorXd>* c_, vector<MatrixXd>* Q_,
                     vector<MatrixXd>* R_, vector<MatrixXd>* G_,
                     vector<MatrixXd>* U_, VectorXd* x0,
                     vector<VectorXd>* xdesired_, C3Options* options) {
  int n = 6;
  int m = 6;
  int k = 4;
  int N = 10;

  float h = 0.1;
  float g = 9.81;
  float mu = 1.0;

  /// initial condition(fingergait)
  VectorXd x0init(n);
  x0init << -8, 0, 3, 0, 4, 0;

  MatrixXd Ainit(n, n);
  Ainit << 1, h, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, h, 0, 0, 0, 0, 0, 1, 0,
      0, 0, 0, 0, 0, 1, h, 0, 0, 0, 0, 0, 1;

  // MatrixXd Binit = MatrixXd::Zero(n,k);

  MatrixXd Binit(n, k);
  Binit << 0, 0, 0, 0, 0, 0, 0, 0, h * h, 0, 0, 0, h, 0, 0, 0, 0, h * h, 0, 0,
      0, h, 0, 0;

  MatrixXd Dinit(n, m);
  Dinit << 0, h * h, -h * h, 0, h * h, -h * h, 0, h, -h, 0, h, -h, 0, -h * h,
      h * h, 0, 0, 0, 0, -h, h, 0, 0, 0, 0, 0, 0, 0, -h * h, h * h, 0, 0, 0, 0,
      -h, h;

  MatrixXd Einit(m, n);
  Einit << 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, -1, 0, -1, 0, 0, 0, 1;

  MatrixXd Finit(m, m);
  Finit << 0, -1, -1, 0, 0, 0, 1, 2 * h, -2 * h, 0, h, -h, 1, -2 * h, 2 * h, 0,
      -h, h, 0, 0, 0, 0, -1, -1, 0, h, -h, 1, 2 * h, -2 * h, 0, -h, h, 1,
      -2 * h, 2 * h;

  VectorXd cinit(m);
  cinit << 0, -h * g, h * g, 0, -h * g, h * g;

  VectorXd dinit(n);
  dinit << -g * h * h, -g * h, 0, 0, 0, 0;

  MatrixXd Hinit(m, k);
  Hinit << 0, 0, mu, 0, -h, 0, 0, 0, h, 0, 0, 0, 0, 0, 0, mu, 0, -h, 0, 0, 0, h,
      0, 0;

  // MatrixXd Hinit = MatrixXd::Zero(m,k);

  MatrixXd Qinit(n, n);
  Qinit << 6000, 0, 0, 0, 0, 0,  // 5000
      0, 10, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 10,
      0, 0, 0, 0, 0, 0, 10;

  MatrixXd Rinit(k, k);
  Rinit = 1 * MatrixXd::Identity(k, k);

  MatrixXd Ginit(n + m + k, n + m + k);
  Ginit = 1 * MatrixXd::Identity(n + m + k, n + m + k);

  std::vector<MatrixXd> A(N, Ainit);
  std::vector<MatrixXd> B(N, Binit);
  std::vector<MatrixXd> D(N, Dinit);
  std::vector<VectorXd> d(N, dinit);
  std::vector<MatrixXd> E(N, Einit);
  std::vector<MatrixXd> F(N, Finit);
  std::vector<VectorXd> c(N, cinit);
  std::vector<MatrixXd> H(N, Hinit);
  std::vector<MatrixXd> Q(N + 1, Qinit);
  std::vector<MatrixXd> R(N, Rinit);
  std::vector<MatrixXd> G(N, Ginit);

  VectorXd xdesiredinit;
  xdesiredinit = VectorXd::Zero(n);
  std::vector<VectorXd> xdesired(N + 1, xdesiredinit);

  MatrixXd Us(n + m + k, n + m + k);
  Us << 1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1000, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1000,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1000, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;

  std::vector<MatrixXd> U(N, Us);

  C3Options optionsinit;
  optionsinit.admm_iter = 10;
  optionsinit.rho_scale = 1.2;
  optionsinit.num_threads = 0;
  optionsinit.delta_option = 1;

  *options = optionsinit;
  *x0 = x0init;
  *n_ = n;
  *m_ = m;
  *k_ = k;
  *N_ = N;
  *A_ = A;
  *B_ = B;
  *D_ = D;
  *d_ = d;
  *c_ = c;
  *E_ = E;
  *F_ = F;
  *H_ = H;
  *Q_ = Q;
  *R_ = R;
  *G_ = G;
  *U_ = U;
  *xdesired_ = xdesired;
};

void init_pivoting(VectorXd xcurrent, int* n_, int* m_, int* k_, int* N_,
                   vector<MatrixXd>* A_, vector<MatrixXd>* B_,
                   vector<MatrixXd>* D_, vector<VectorXd>* d_,
                   vector<MatrixXd>* E_, vector<MatrixXd>* F_,
                   vector<MatrixXd>* H_, vector<VectorXd>* c_,
                   vector<MatrixXd>* Q_, vector<MatrixXd>* R_,
                   vector<MatrixXd>* G_, vector<MatrixXd>* U_, VectorXd* x0,
                   vector<VectorXd>* xdesired_, C3Options* options) {
  int n = 10;
  int m = 10;
  int k = 4;
  int N = 10;

  float mu1 = 0.1;
  float mu2 = 9.81;
  float mu3 = 1.0;
  float g = 9.81;
  float dt = 0.01;
  float h = 1;
  float w = 1;
  float mm = 1;

  float x4 = xcurrent(4);
  float x6 = xcurrent(6);
  float x8 = xcurrent(8);

  float x5 = xcurrent(6);  // intentional
  float x7 = xcurrent(8);  // intentional

  float rt = (-1) * sqrt(h * h + w * w);
  float sinn = sin(x4);
  float coss = cos(x4);
  float z = w * sinn - h * coss;

  /// initial condition(pivoting)
  VectorXd x0init(n);
  x0init << 0, 0, 1.36, 0, 0.2, 0, -0.3, 0, -0.7, 0;

  MatrixXd Ainit(n, n);
  Ainit << 1, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
      dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, dt, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, dt, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, dt, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1;

  MatrixXd Binit(n, k);
  Binit << 0, 0, -dt * dt * coss, dt * dt * sinn, 0, 0, -dt * coss, dt * sinn,
      0, 0, dt * dt * sinn, dt * dt * coss, 0, 0, dt * sinn, dt * coss, 0, 0,
      -dt * dt * x6, dt * dt * x8, 0, 0, -dt * x6, dt * x8, dt * dt, 0, 0, 0,
      dt, 0, 0, 0, 0, dt * dt, 0, 0, 0, dt, 0, 0;

  MatrixXd Dinit(n, m);
  Dinit << 0, dt * dt * sinn, -dt * dt * sinn, 0, -coss * dt * dt,
      dt * dt * coss, 0, dt * dt, -dt * dt, 0, 0, dt * sinn, -dt * sinn, 0,
      -coss * dt, dt * coss, 0, dt, -dt, 0, 0, dt * dt * coss, -dt * dt * coss,
      0, dt * dt * sinn, -dt * dt * sinn, 0, 0, 0, dt * dt, 0, dt * coss,
      -dt * coss, 0, dt * dt * sinn, -dt * sinn, 0, 0, 0, dt, 0, -dt * dt * h,
      dt * dt * h, 0, dt * dt * w, -dt * dt * w, 0,
      -dt * dt * coss * w - dt * dt * sinn * h,
      dt * dt * coss * w + dt * dt * sinn * h,
      dt * dt * sinn * w - h * coss * dt * dt, 0, -dt * h, dt * h, 0, dt * w,
      -dt * w, 0, -dt * coss * w - dt * sinn * h, dt * coss * w + dt * sinn * h,
      dt * sinn * w - h * coss * dt, 0, -dt * dt, dt * dt, 0, 0, 0, 0, 0, 0, 0,
      0, -dt, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -dt * dt, dt * dt, 0, 0, 0,
      0, 0, 0, 0, 0, -dt, dt, 0, 0, 0, 0;

  MatrixXd Einit(m, n);
  Einit << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1,
      0, 0, 0, -rt, 0, 0, 0, 0, 0, 1, 0, 0, 0, rt, 0, 0, 0, 0, 0, 0, 1, dt,
      -h * sinn + w * coss + z, z * dt, 0, 0, 0, 0;

  MatrixXd Finit(m, m);
  Finit << 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 1, dt, -dt, 0, 0, 0, 0, 0, 0, 0, 1,
      -dt, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 1,
      dt, -dt, 0, 0, 0, 0, 0, 0, 0, 1, -dt, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      -1, -1, mu3, 0, -(dt * sinn - rt * dt * h), -(-dt * sinn + rt * dt * h),
      0, -(-dt * coss + rt * dt * w), -(dt * coss - dt * rt * w), -(-1),
      -(dt - rt * dt * coss * w - rt * dt * sinn * h),
      -(-dt + rt * dt * coss * w + rt * dt * sinn * h),
      -(sinn * w * rt * dt - h * coss * rt * dt), 0,
      -(-dt * sinn + rt * dt * h), -(dt * sinn - rt * dt * h), 0,
      -(dt * coss - rt * dt * w), -(-dt * coss + rt * dt * w), -(-1),
      -(-dt + rt * dt * coss * w + rt * dt * sinn * h),
      -(dt - rt * dt * coss * w - rt * dt * sinn * h),
      -(-sinn * w * rt * dt + h * coss * rt * dt), 0,
      dt * dt * coss - z * dt * dt * h, -dt * dt * coss + z * dt * dt * h, 0,
      dt * dt * sinn + z * dt * dt * w, -dt * dt * sinn - dt * dt * w * z, 0,
      -z * dt * dt * coss * w - z * dt * dt * sinn * h,
      z * dt * dt * coss * w + z * dt * dt * sinn * h,
      dt * dt + z * dt * dt * sinn * w - z * h * coss * dt * dt;

  VectorXd cinit(m);
  cinit << 0, -h * g, h * g, 0, -h * g, h * g;

  VectorXd dinit(n);
  dinit << 0, 0, -dt * dt * mm * g, -dt * mm * g, 0, 0, 0, 0, 0, 0;

  MatrixXd Hinit(m, k);
  Hinit << 0, 0, mu1, 0, -dt, 0, 0, 0, dt, 0, 0, 0, 0, 0, 0, mu2, 0, -dt, 0, 0,
      0, dt, 0, 0, 0, 0, 0, 0, 0, 0, -(-dt * coss - rt * dt * x5),
      -(dt * sinn + dt * rt * x7), 0, 0, -(dt * coss + rt * dt * x5),
      -(-dt * sinn - dt * rt * x7), 0, 0, dt * dt * sinn - dt * dt * x5 * z,
      dt * dt * coss + dt * dt * x7 * z;

  // MatrixXd Hinit = MatrixXd::Zero(m,k);

  MatrixXd Qinit(n, n);
  Qinit = MatrixXd::Identity(n, n);
  Qinit(4, 4) = 100;
  Qinit(2, 2) = 100;
  Qinit(0, 0) = 100;
  Qinit(6, 6) = 50;
  Qinit(8, 8) = 50;
  Qinit(5, 5) = 11;
  Qinit(3, 3) = 9;
  Qinit(1, 1) = 11;

  MatrixXd Rinit(k, k);
  Rinit = 0.01 * MatrixXd::Identity(k, k);

  std::vector<MatrixXd> A(N, Ainit);
  std::vector<MatrixXd> B(N, Binit);
  std::vector<MatrixXd> D(N, Dinit);
  std::vector<VectorXd> d(N, dinit);
  std::vector<MatrixXd> E(N, Einit);
  std::vector<MatrixXd> F(N, Finit);
  std::vector<VectorXd> c(N, cinit);
  std::vector<MatrixXd> H(N, Hinit);
  std::vector<MatrixXd> Q(N + 1, Qinit);
  std::vector<MatrixXd> R(N, Rinit);

  VectorXd xdesiredinit;
  xdesiredinit = VectorXd::Zero(n);
  std::vector<VectorXd> xdesired(N + 1, xdesiredinit);

  MatrixXd Us(n + m + k, n + m + k);
  Us.block(0, 0, n, n) = 1000 * MatrixXd::Identity(n, n);
  Us.block(n, n, m, m) = MatrixXd::Identity(m, m);
  Us.block(n + m, n + m, k, k) = MatrixXd::Identity(k, k);
  std::vector<MatrixXd> U(N, Us);

  C3Options optionsinit;
  optionsinit.admm_iter = 5;
  optionsinit.rho_scale = 1.1;
  optionsinit.num_threads = 0;
  optionsinit.delta_option = 1;

  MatrixXd Ginit(n + m + k, n + m + k);
  Ginit = 0.02 * MatrixXd::Identity(n + m + k, n + m + k);
  std::vector<MatrixXd> G(N, Ginit);

  *options = optionsinit;
  *x0 = x0init;
  *n_ = n;
  *m_ = m;
  *k_ = k;
  *N_ = N;
  *A_ = A;
  *B_ = B;
  *D_ = D;
  *d_ = d;
  *c_ = c;
  *E_ = E;
  *F_ = F;
  *H_ = H;
  *Q_ = Q;
  *R_ = R;
  *G_ = G;
  *U_ = U;
  *xdesired_ = xdesired;
};
