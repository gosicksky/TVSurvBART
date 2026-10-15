/*
 * distribution.h — log-linear BART helpers
 *
 * All functions live in C++ — no Rcpp::Function callbacks, no Eigen.
 * RNG is drawn from the BART `rn&` abstraction so the MCMC chain is a
 * single coherent stream (reproducible via R's global RNG state).
 */

#ifndef LOGLINBART_DISTRIBUTION_H_
#define LOGLINBART_DISTRIBUTION_H_

#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <cmath>
#include <random>
#include <iostream>
#include <vector>
#include <random>
#include <RcppEigen.h>
// #include <eigen3/Eigen/Cholesky>
// #include <eigen3/Eigen/Dense>
#include <Rcpp.h>

// Forward declaration keeps this header independent of rn.h's Rcpp/vector
// dependencies.  The full definition is pulled in by distribution.cpp.
class rn;

//--------------------------------------------------
// numerical helpers
double logsumexp_two(double u, double v);

// Overflow-safe log K_nu(x), x > 0, any real nu.  Uses R::bessel_k when it is
// finite (bit-identical to the historical code path) and falls back to the
// Debye uniform asymptotic expansion (nu >= 10) / the leading small-argument
// term (nu < 10) where Rmath silently overflows for large order.
double logK_safe(double x, double nu);

//--------------------------------------------------
// GIG normalizing constant  Z(eta, chi, psi)  (Murray 2021 Sec 4.1)
//
//   eta > 0, chi = 0, psi > 0 : Gamma(eta, psi/2)   → Γ(eta) (2/psi)^eta
//   eta < 0, chi > 0, psi = 0 : Inverse Gamma       → Γ(-eta)(2/chi)^-eta
//   chi > 0, psi > 0          : 2 K_eta(sqrt(psi*chi)) / (psi/chi)^(eta/2)
double Z(double eta, double chi, double psi);
double logZ(double eta, double chi, double psi);

//--------------------------------------------------
// Pure-C++ GIG sampler driven by BART's rn&.
//
// Sample X ~ GIG(lambda, chi, psi) with density
//     f(x) ∝ x^{lambda - 1} exp(-(1/2)(chi/x + psi*x)),  x > 0
//
// Degenerate cases:
//   chi = 0 & lambda > 0 : Gamma(lambda, psi/2)
//   psi = 0 & lambda < 0 : 1 / Gamma(-lambda, chi/2)
//
// General case uses the mirror transformation (lambda < 0 ⇒ invert) plus
// ratio-of-uniforms without mode shift for GIG(lambda, omega, omega).
double sampleGIG(double lambda, double chi, double psi, rn& gen);

//--------------------------------------------------
// Mixture sampler: draws λ_ht from the Pλ full conditional (Murray 2021,
// Section 4.2, step 4).  Uses only `gen` — no global RNG, no R callbacks.
double sampleMixtureGIG(double c, double d, double rht, double sht, rn& gen);

Eigen::VectorXd multivariate_normal_sample(Eigen::VectorXd& mean, Eigen::MatrixXd& covar, rn& gen);

double prior_density(Eigen::VectorXd& params, Eigen::VectorXd& mean, Eigen::MatrixXd& covar);

double log_prior_density(Eigen::VectorXd& params, Eigen::VectorXd& mean, Eigen::MatrixXd& covar);

//--------------------------------------------------------------------------------------------
//inverse wishart distribution
Eigen::MatrixXd sampleInverseWishart(Eigen::MatrixXd& scaleMatrix, double dof, rn& gen);

#endif // LOGLINBART_DISTRIBUTION_H_

