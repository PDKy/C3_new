#pragma once

#include "drake/common/yaml/yaml_io.h"
#include "drake/common/yaml/yaml_read_archive.h"

namespace c3 {

struct C3Options {
  // Hyperparameters
  int admm_iter = 3;    // total number of ADMM iterations
  float rho_scale = 10; // scaling of rho parameter (/rho = rho_scale * /rho)
  int num_threads = 10; // 0 is dynamic, greater than 0 for a fixed count
  int delta_option = 1; // different options for delta update
  std::string projection_type;
  std::string contact_model;
  double M = 1000; // big M value for MIQP
  bool warm_start = false;
  bool use_predicted_x0;
  bool end_on_qp_step = true;
  bool use_robust_formulation;
  double solve_time_filter_alpha;
  double publish_frequency;

  std::vector<double> u_horizontal_limits;
  std::vector<double> u_vertical_limits;
  std::vector<Eigen::VectorXd> workspace_limits;
  double workspace_margins;

  int N;
  double gamma;
  std::vector<double> mu;
  double dt;
  double solve_dt;
  int num_friction_directions;
  int num_contacts;

  // See comments below for how we parse the .yaml into the cost matrices
  Eigen::MatrixXd Q;
  Eigen::MatrixXd R;
  Eigen::MatrixXd G;
  Eigen::MatrixXd U;

  // Uniform scaling of the cost matrices.
  // Q = w_Q * diag(q_vector)
  // R = w_R * diag(r_vector)
  // G = w_G * diag(g_vector)
  // U = w_U * diag(u_vector)
  double w_Q;
  double w_R;
  double w_G;
  double w_U;

  // Unused except when parsing the costs from a yaml
  // We assume a diagonal Q, R, G, U matrix, so we can just specify the diagonal
  // terms as *_vector. To make indexing even easier, we split the parsing of
  // the g_vector and u_vector into the x, lambda, and u terms. The Stewart and
  // Trinkle contact model uses *_gamma, *_lambda_n, *_lambda_t while the
  // Anitescu model uses *_lambda.
  std::vector<double> q_vector;
  std::vector<double> r_vector;

  std::vector<double> g_vector;
  std::vector<double> g_x;
  std::vector<double> g_gamma;
  std::vector<double> g_lambda_n;
  std::vector<double> g_lambda_t;
  std::vector<double> g_lambda;
  std::vector<double> g_u;

  std::vector<double> u_vector;
  std::vector<double> u_x;
  std::vector<double> u_gamma;
  std::vector<double> u_lambda_n;
  std::vector<double> u_lambda_t;
  std::vector<double> u_lambda;
  std::vector<double> u_u;

  template <typename Archive> void Serialize(Archive *a) {
    a->Visit(DRAKE_NVP(admm_iter));
    a->Visit(DRAKE_NVP(rho_scale));
    a->Visit(DRAKE_NVP(num_threads));
    a->Visit(DRAKE_NVP(delta_option));
    a->Visit(DRAKE_NVP(contact_model));
    a->Visit(DRAKE_NVP(projection_type));
    if (projection_type == "QP") {
      DRAKE_DEMAND(contact_model == "anitescu");
    }
    a->Visit(DRAKE_NVP(warm_start));
    a->Visit(DRAKE_NVP(use_predicted_x0));
    a->Visit(DRAKE_NVP(end_on_qp_step));
    a->Visit(DRAKE_NVP(use_robust_formulation));
    a->Visit(DRAKE_NVP(solve_time_filter_alpha));
    a->Visit(DRAKE_NVP(publish_frequency));

    a->Visit(DRAKE_NVP(workspace_limits));
    a->Visit(DRAKE_NVP(u_horizontal_limits));
    a->Visit(DRAKE_NVP(u_vertical_limits));
    a->Visit(DRAKE_NVP(workspace_margins));

    a->Visit(DRAKE_NVP(mu));
    a->Visit(DRAKE_NVP(dt));
    a->Visit(DRAKE_NVP(solve_dt));
    a->Visit(DRAKE_NVP(num_friction_directions));
    a->Visit(DRAKE_NVP(num_contacts));

    a->Visit(DRAKE_NVP(N));
    a->Visit(DRAKE_NVP(gamma));
    a->Visit(DRAKE_NVP(w_Q));
    a->Visit(DRAKE_NVP(w_R));
    a->Visit(DRAKE_NVP(w_G));
    a->Visit(DRAKE_NVP(w_U));
    a->Visit(DRAKE_NVP(q_vector));
    a->Visit(DRAKE_NVP(r_vector));
    a->Visit(DRAKE_NVP(g_x));
    a->Visit(DRAKE_NVP(g_gamma));
    a->Visit(DRAKE_NVP(g_lambda_n));
    a->Visit(DRAKE_NVP(g_lambda_t));
    a->Visit(DRAKE_NVP(g_lambda));
    a->Visit(DRAKE_NVP(g_u));
    a->Visit(DRAKE_NVP(u_x));
    a->Visit(DRAKE_NVP(u_gamma));
    a->Visit(DRAKE_NVP(u_lambda_n));
    a->Visit(DRAKE_NVP(u_lambda_t));
    a->Visit(DRAKE_NVP(u_lambda));
    a->Visit(DRAKE_NVP(u_u));

    g_vector = std::vector<double>();
    g_vector.insert(g_vector.end(), g_x.begin(), g_x.end());
    if (contact_model == "stewart_and_trinkle") {
      g_vector.insert(g_vector.end(), g_gamma.begin(), g_gamma.end());
      g_vector.insert(g_vector.end(), g_lambda_n.begin(), g_lambda_n.end());
      g_vector.insert(g_vector.end(), g_lambda_t.begin(), g_lambda_t.end());
    } else {
      g_vector.insert(g_vector.end(), g_lambda.begin(), g_lambda.end());
    }

    g_vector.insert(g_vector.end(), g_u.begin(), g_u.end());
    u_vector = std::vector<double>();
    u_vector.insert(u_vector.end(), u_x.begin(), u_x.end());
    if (contact_model == "stewart_and_trinkle") {
      u_vector.insert(u_vector.end(), u_gamma.begin(), u_gamma.end());
      u_vector.insert(u_vector.end(), u_lambda_n.begin(), u_lambda_n.end());
      u_vector.insert(u_vector.end(), u_lambda_t.begin(), u_lambda_t.end());
    } else {
      u_vector.insert(u_vector.end(), u_lambda.begin(), u_lambda.end());
    }
    u_vector.insert(u_vector.end(), u_u.begin(), u_u.end());

    Eigen::VectorXd q = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(
        this->q_vector.data(), this->q_vector.size());
    Eigen::VectorXd r = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(
        this->r_vector.data(), this->r_vector.size());
    Eigen::VectorXd g = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(
        this->g_vector.data(), this->g_vector.size());
    Eigen::VectorXd u = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(
        this->u_vector.data(), this->u_vector.size());

    DRAKE_DEMAND(static_cast<int>(g_lambda.size()) ==
                 num_contacts * num_friction_directions * 2);
    DRAKE_DEMAND(static_cast<int>(u_lambda.size()) ==
                 num_contacts * num_friction_directions * 2);
    DRAKE_DEMAND(static_cast<int>(mu.size()) == num_contacts);
    DRAKE_DEMAND(g.size() == u.size());

    Q = w_Q * q.asDiagonal();
    R = w_R * r.asDiagonal();
    G = w_G * g.asDiagonal();
    U = w_U * u.asDiagonal();
  }
};

inline C3Options LoadC3Options(const std::string &filename) {
  auto options = drake::yaml::LoadYamlFile<C3Options>(filename);
  return options;
}

} // namespace c3