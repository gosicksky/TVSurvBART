/*
 * distribution.cpp — log-linear BART helpers.
 *
 * Pure C++ implementation (no Eigen-via-R callbacks).
 * All random-variate generation routes through the BART `rn&` abstraction
 * so the MCMC chain is a single coherent random stream.
 */

#include <Rcpp.h>
#include <Rmath.h>
#include "distribution.h"
#include "rn.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

//--------------------------------------------------
// log-sum-exp of two values (stable)
double logsumexp_two(double u, double v) {
  double m = std::max(u, v);
  if (!std::isfinite(m)) {
    return m;
  }
  return m + std::log(std::exp(u - m) + std::exp(v - m));
}

//--------------------------------------------------
// Overflow-safe log K_nu(x)  (modified Bessel function of the second kind)
//
// R::bessel_k(x, nu, expon.scaled = 2) returns e^x K_nu(x) and overflows to
// +Inf SILENTLY (no warning) for large ORDER, e.g. K_200(1) or K_800(10).
// In logZ() the order is |eta| = |r_hk - c| (events in the leaf minus the
// prior shape) and the argument is omega = 2 sqrt(d s_hk), so this happens
// whenever the order is large relative to the argument: the Murray leaf prior
// with c >~ 200 (a_B <= 0.5 at m=50), or -- at ANY c -- a leaf holding many
// events (|r - c| >~ 400-800 at the default c = m/6.125 + 0.5).  The previous
// fallback used the LARGE-ARGUMENT asymptotic sqrt(pi/(2x)) e^{-x}, which is
// wrong by hundreds to thousands of nats in that regime; it silently dropped
// the inverse-gamma component of the leaf prior (logZ1 << logZ2) in bd.cpp
// (birth/death marginal likelihood) and in sampleMixtureGIG (mixture weight).
//
// Strategy (bit-identical to the old code whenever Rmath is finite):
//   * Rmath finite            : log(kn) - x                     (unchanged)
//   * overflow and nu >= 10   : Debye / uniform asymptotic expansion,
//                               DLMF 10.41.4 with u_0..u_5 -- valid uniformly
//                               in x, relative error O(nu^-6) (<~1e-10 at
//                               nu = 10, far below anything else in the chain)
//   * overflow and nu <  10   : the overflow can only come from x tiny
//                               (x < ~1e-14): leading small-argument term
//                               K_nu(x) ~ Gamma(nu)/2 (2/x)^nu, exact to
//                               O(x^2) there.
//
// DLMF 10.41.4:  K_nu(nu z) ~ sqrt(pi/(2 nu)) (1+z^2)^{-1/4} e^{-nu eta}
//                              * sum_k (-1)^k u_k(p) / nu^k,
//   eta = sqrt(1+z^2) + log( z / (1 + sqrt(1+z^2)) ),   p = (1+z^2)^{-1/2},
//   u_k(p) = Debye polynomials, DLMF 10.41.10 / A&S 9.3.9-9.3.14.
static double logK_debye(double x, double nu) {
  const double z = x / nu, s = std::sqrt(1.0 + z * z), p = 1.0 / s, p2 = p * p;
  const double eta = s + std::log(z) - std::log1p(s);
  const double u1 = p * (3.0 - 5.0 * p2) / 24.0;
  const double u2 = p2 * (81.0 - 462.0 * p2 + 385.0 * p2 * p2) / 1152.0;
  const double u3 = p * p2 * (30375.0 - 369603.0 * p2 + 765765.0 * p2 * p2
                                - 425425.0 * p2 * p2 * p2) / 414720.0;
  const double u4 = p2 * p2 * (4465125.0 - 94121676.0 * p2 + 349922430.0 * p2 * p2
                                 - 446185740.0 * p2 * p2 * p2
                                 + 185910725.0 * p2 * p2 * p2 * p2) / 39813120.0;
                                 const double u5 = p * p2 * p2 * (1519035525.0 - 49286948607.0 * p2
                                                                    + 284499769554.0 * p2 * p2
                                                                    - 614135872350.0 * p2 * p2 * p2
                                                                    + 566098157625.0 * p2 * p2 * p2 * p2
                                                                    - 188699385875.0 * p2 * p2 * p2 * p2 * p2) / 6688604160.0;
                                                                    const double in = 1.0 / nu;
                                                                    const double S = 1.0 + in * (-u1 + in * (u2 + in * (-u3 + in * (u4 - in * u5))));
                                                                    return 0.5 * std::log(M_PI / (2.0 * nu)) - nu * eta - 0.25 * std::log1p(z * z) + std::log(S);
}

// log K_nu(x) for x > 0, any real nu (K_{-nu} = K_nu).
double logK_safe(double x, double nu) {
  double kn = R::bessel_k(x, nu, /*expon.scaled=*/2);   // e^x K_nu(x); takes |nu| itself
  if (std::isfinite(kn) && kn > 0.0) {
    return std::log(kn) - x;
  }
  nu = std::fabs(nu);
  if (nu >= 10.0) {
    return logK_debye(x, nu);                            // large-order overflow
  }
  // nu < 10 overflows only for x < ~1e-14: leading small-argument term
  return R::lgammafn(nu) - M_LN2 + nu * std::log(2.0 / x);
}

//--------------------------------------------------
// Z(eta, chi, psi) — GIG normalizing constant
double Z(double eta, double chi, double psi) {
  return std::exp(logZ(eta, chi, psi));
}

double logZ(double eta, double chi, double psi) {
  if (!std::isfinite(eta) || !std::isfinite(chi) || !std::isfinite(psi) ||
      chi < 0.0 || psi < 0.0 ||
      (chi == 0.0 && eta <= 0.0) ||
      (psi == 0.0 && eta >= 0.0)) {
    Rcpp::stop("logZ: invalid parameters eta=%g chi=%g psi=%g", eta, chi, psi);
  }
  
  constexpr double epsilon = 10.0 * std::numeric_limits<double>::epsilon();
  
  if (eta > 0.0 && chi < epsilon && psi > 0.0) {
    return R::lgammafn(eta) + eta * std::log(2.0 / psi);
  }
  if (eta < 0.0 && chi > 0.0 && psi < epsilon) {
    return R::lgammafn(-eta) + (-eta) * std::log(2.0 / chi);
  }
  
  double omega = std::sqrt(psi * chi);
  
  return std::log(2.0) + logK_safe(omega, eta) - 0.5 * eta * std::log(psi / chi);
}

//==================================================
// GIG sampler — pure C++
//==================================================

static double _sampleGIG_standard(double lambda, double omega, rn& gen);

double sampleGIG(double lambda, double chi, double psi, rn& gen) {
  if (!std::isfinite(lambda) || !std::isfinite(chi) || !std::isfinite(psi) ||
      chi < 0.0 || psi < 0.0) {
    Rcpp::stop("sampleGIG: invalid parameters lambda=%g chi=%g psi=%g",
               lambda, chi, psi);
  }
  
  constexpr double epsilon = 10.0 * std::numeric_limits<double>::epsilon();
  
  if (chi < epsilon) {
    if (lambda <= 0.0) {
      Rcpp::stop("sampleGIG: Gamma limit requires lambda > 0 (got %g)", lambda);
    }
    return gen.gamma(lambda, psi / 2.0);
  }
  
  if (psi < epsilon) {
    if (lambda >= 0.0) {
      Rcpp::stop("sampleGIG: InvGamma limit requires lambda < 0 (got %g)",
                 lambda);
    }
    return 1.0 / gen.gamma(-lambda, chi / 2.0);
  }
  
  bool invert = false;
  if (lambda < 0.0) {
    lambda = -lambda;
    std::swap(chi, psi);
    invert = true;
  }
  
  double beta  = std::sqrt(chi / psi);
  double omega = std::sqrt(chi * psi);
  
  double y = _sampleGIG_standard(lambda, omega, gen);
  double x = beta * y;
  return invert ? (1.0 / x) : x;
}

// Ratio-of-uniforms without mode shift for
//     g(y) ∝ y^{lambda-1} exp(-(omega/2)(y + 1/y)),    y > 0
// with lambda ≥ 0 and omega > 0.
static double _sampleGIG_standard(double lambda, double omega, rn& gen) {
  auto logf = [lambda, omega](double y) {
    return (lambda - 1.0) * std::log(y) - 0.5 * omega * (y + 1.0 / y);
  };
  
  double sq1  = std::sqrt((lambda - 1.0) * (lambda - 1.0) + omega * omega);
  double y_m  = (lambda - 1.0 + sq1) / omega;
  double log_fm = logf(y_m);
  
  double sq2  = std::sqrt((lambda + 1.0) * (lambda + 1.0) + omega * omega);
  double y_v  = ((lambda + 1.0) + sq2) / omega;
  double log_g_at_yv = logf(y_v) - log_fm;
  double v_max = y_v * std::exp(0.5 * log_g_at_yv);
  
  for (int trial = 0; trial < 10000; ++trial) {
    double U = gen.uniform();
    if (U == 0.0) continue;
    double V = gen.uniform() * v_max;
    double Y = V / U;
    double log_g_at_Y = logf(Y) - log_fm;
    if (2.0 * std::log(U) <= log_g_at_Y) {
      return Y;
    }
  }
  
  Rcpp::stop("sampleGIG: ratio-of-uniforms failed to converge in 10000 "
               "trials (lambda=%g omega=%g)", lambda, omega);
  return y_m;  // unreachable
}

//--------------------------------------------------
// Mixture sampler for the Pλ full conditional (Murray 2021, Prop / Step 4)
double sampleMixtureGIG(double c, double d, double rht, double sht, rn& gen) {
  // Robustness guard for a ZERO-EXPOSURE terminal node.  sht = sum(vij*fnoh) can
  // underflow to ~0 (a leaf isolated by fine splits on continuous covariates, or
  // allfit underflow) while rht = sum(ui) stays > 0.  Then the P1 component is
  // GIG(lambda=-c+rht, chi=2d, psi=2*sht~0): for lambda >= 0 this is IMPROPER
  // (logZ1 -> +inf and sampleGIG would abort with "psi=0").  Draw from the proper
  // Gamma limit P2 = Gamma(c+rht, d+sht) instead -- the prior-regularized node
  // value, identical to the else-branch below, just forced.  (When lambda < 0, P1
  // is the proper inverse-gamma limit and the normal path below handles it.)
  constexpr double epsilon = 10.0 * std::numeric_limits<double>::epsilon();
  if (2.0 * sht < epsilon && (rht - c) >= 0.0) {
    return sampleGIG(c + rht, 0.0, 2.0 * (d + sht), gen);
  }
  
  double logZ1 = logZ(-c + rht, 2.0 * d,        2.0 * sht);
  double logZ2 = logZ( c + rht, 0.0,            2.0 * (d + sht));
  double logZ_denom = logsumexp_two(logZ1, logZ2);
  double pi_ht = std::exp(logZ1 - logZ_denom);
  
  double lambda, chi, psi;
  if (gen.uniform() < pi_ht) {
    lambda = -c + rht;
    chi    = 2.0 * d;
    psi    = 2.0 * sht;
  } else {
    lambda = c + rht;
    chi    = 0.0;
    psi    = 2.0 * (d + sht);
  }
  
  return sampleGIG(lambda, chi, psi, gen);
}

//==================================================
// Multivariate normal / inverse-Wishart — native Eigen, no R callbacks
//==================================================

//--------------------------------------------------------------------------------------------
// Multivariate normal sampling via Cholesky.
//   covar = L L^T,  z ~ N(0, I),  x = mean + L z
Eigen::VectorXd multivariate_normal_sample(Eigen::VectorXd& mean,
                                           Eigen::MatrixXd& covar,
                                           rn& gen) {
  const int p = static_cast<int>(mean.size());
  if (covar.rows() != p || covar.cols() != p) {
    Rcpp::stop("multivariate_normal_sample: dimension mismatch "
                 "(mean=%d, covar=%dx%d)", p, (int)covar.rows(), (int)covar.cols());
  }
  Eigen::LLT<Eigen::MatrixXd> llt(covar);
  if (llt.info() != Eigen::Success) {
    Rcpp::stop("multivariate_normal_sample: covariance not positive definite");
  }
  Eigen::VectorXd z(p);
  for (int i = 0; i < p; ++i) z(i) = gen.normal();
  return mean + llt.matrixL() * z;
}

//--------------------------------------------------------------------------------------------
// Log-density of N(mean, covar) at params, computed from the Cholesky factor.
//   log p = -0.5 [ p log(2π) + log|Σ| + (x-μ)^T Σ^{-1} (x-μ) ]
double log_prior_density(Eigen::VectorXd& params,
                         Eigen::VectorXd& mean,
                         Eigen::MatrixXd& covar) {
  const int p = static_cast<int>(params.size());
  if (mean.size() != p || covar.rows() != p || covar.cols() != p) {
    Rcpp::stop("log_prior_density: dimension mismatch");
  }
  Eigen::LLT<Eigen::MatrixXd> llt(covar);
  if (llt.info() != Eigen::Success) {
    Rcpp::stop("log_prior_density: covariance not positive definite");
  }
  Eigen::VectorXd diff = params - mean;
  Eigen::VectorXd z = llt.matrixL().solve(diff);
  double quad = z.squaredNorm();
  
  Eigen::MatrixXd L = llt.matrixL();
  double log_det = 0.0;
  for (int i = 0; i < p; ++i) log_det += std::log(L(i, i));
  log_det *= 2.0;
  
  return -0.5 * (p * std::log(2.0 * M_PI) + log_det + quad);
}

//--------------------------------------------------------------------------------------------
double prior_density(Eigen::VectorXd& params,
                     Eigen::VectorXd& mean,
                     Eigen::MatrixXd& covar) {
  return std::exp(log_prior_density(params, mean, covar));
}

//--------------------------------------------------------------------------------------------
// Inverse-Wishart sampler via Bartlett decomposition.
//
// To draw X ~ IW(Ψ, ν) with Ψ p×p:
//   1. Chol Ψ = L L^T   (L lower triangular)
//   2. Build lower-triangular A with
//        A(i,i) = sqrt(χ²_{ν-i})     for i = 0, …, p-1
//        A(i,j) ~ N(0,1)             for i > j
//      so A A^T ~ W(I, ν) — standard Bartlett.
//   3. X = L A^{-T} A^{-1} L^T.  Equivalently X = T^T T with T = A^{-1} L^T.
//
// All randomness routes through `gen`.
Eigen::MatrixXd sampleInverseWishart(Eigen::MatrixXd& scaleMatrix,
                                     double dof, rn& gen) {
  // dof is REAL-valued: the Bartlett chi-square draws below use gamma(k/2, 1/2)
  // with k = dof - i, which is defined for any dof > p - 1. (Was int, which
  // silently rounded the Huang-Wand df for non-integer nu.)
  const int p = static_cast<int>(scaleMatrix.rows());
  if (scaleMatrix.cols() != p) {
    Rcpp::stop("sampleInverseWishart: scale matrix must be square");
  }
  if (dof <= p - 1) {
    Rcpp::stop("sampleInverseWishart: dof=%g must exceed p-1=%d", dof, p - 1);
  }
  
  Eigen::LLT<Eigen::MatrixXd> llt(scaleMatrix);
  if (llt.info() != Eigen::Success) {
    Rcpp::stop("sampleInverseWishart: scale matrix not positive definite");
  }
  Eigen::MatrixXd L = llt.matrixL();
  
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(p, p);
  for (int i = 0; i < p; ++i) {
    double k = dof - static_cast<double>(i);
    A(i, i) = std::sqrt(gen.gamma(k / 2.0, 0.5));  // χ²_k
    for (int j = 0; j < i; ++j) {
      A(i, j) = gen.normal();
    }
  }
  
  Eigen::MatrixXd T = A.triangularView<Eigen::Lower>().solve(L.transpose());
  return T.transpose() * T;
}
