/*
 *  BART: Bayesian Additive Regression Trees
 *  Copyright (C) 2017 Robert McCulloch and Rodney Sparapani
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, a copy is available at
 *  https://www.R-project.org/Licenses/GPL-2
 */

/*** R
library(GIGrvg)
library(mvtnorm)
library(MCMCpack)
library(LaplacesDemon)
library(MASS)
*/

#include "tree.h"
#include "treefuns.h"
#include "gk61.h"

#include <ctime>
#include <vector>
#include <RcppEigen.h>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <unordered_map>

// #include <Rcpp.h>

typedef std::vector<tree> vtree;

// Per-draw memo of segment integrals shared by UNSEEN subjects. Their
// Gauss-Hermite integrand at a given (draw, GH node) does not depend on the
// subject, and every subject walks the same visit grid and target time, so the
// same (node, a, b) segment recurs across subjects. The key is the node index
// and the exact bit patterns of a and b; integ() is deterministic in (f, a, b),
// so a hit returns the very double a recomputation would give.
struct PredSegKey {
  int q; uint64_t a, b;
  bool operator==(const PredSegKey& o) const { return q == o.q && a == o.a && b == o.b; }
};
struct PredSegHash {
  size_t operator()(const PredSegKey& k) const {
    uint64_t h = k.a * 0x9E3779B97F4A7C15ULL;
    h ^= k.b + 0x632BE59BD9B4E019ULL + (h << 6) + (h >> 2);
    h ^= (uint64_t)(uint32_t)k.q * 0xC2B2AE3D27D4EB4FULL;
    return (size_t)(h ^ (h >> 29));
  }
};
static inline uint64_t pred_bits(double x) { uint64_t u; std::memcpy(&u, &x, sizeof(u)); return u; }

#ifdef _OPENMP
void local_getpred(size_t nd, size_t p, size_t m, size_t np, xinfo& xi, std::vector<vtree>& tmat, double *px, Rcpp::NumericMatrix& yhat);
#endif

void getpred(int beg, int end, size_t p, size_t m, size_t np, xinfo& xi, std::vector<vtree>& tmat, double *px, Rcpp::NumericMatrix& yhat);


// Natural cubic spline k-th basis at a scalar x.  Truncated-power
// (Royston-Parmar) form; see the full derivation and the boundary-weight
// caveat in survivalBART.h's ns_kth_basis_scalar.  MUST stay bit-identical to
// the training-time basis there.
double ns_kth_basis_scalar(double x,
                           double* knots,
                           int K,
                           int degree,
                           int which,
                           bool include_intercept)
{
  // Extra marginal-curvature basis: index K (when intercept included) is t^2,
  // matching survivalBART.h's training-time basis when a random slope is present.
  if (include_intercept && which == K)
    return x * x;
  
  // Total basis functions:
  // include_intercept = FALSE -> total = K - 1
  // include_intercept = TRUE  -> total = K
  int total = include_intercept ? K : (K - 1);
  
  // Out of range
  if (which < 0 || which >= total)
    return 0.0;
  
  double xmin = knots[0];
  double xmax = knots[K - 1];
  
  // Internal knots = knots[1] ... knots[K-2]
  int K_int = K - 2;
  double* internal = knots + 1;  // pointer to knots[1]
  
  // Truncated power cubic
  auto tp3 = [&](double z) {
    return (z > 0.0) ? z * z * z : 0.0;
  };
  
  // ------------------------------------------------------------
  // Case 1: include_intercept = TRUE
  // ------------------------------------------------------------
  if (include_intercept) {
    
    // which = 0 -> intercept
    if (which == 0)
      return 1.0;
    
    // which = 1 -> x
    if (which == 1)
      return x;
    
    // cubic basis for internal knots
    int j = which - 2;           // j in {0, ..., K_int-1}
    if (j >= K_int)
      return 0.0;
    
    double k_j = internal[j];
    
    double val =
      tp3(x - k_j)
      - tp3(x - xmin) * (xmax - k_j) / (xmax - xmin)
      - tp3(x - xmax) * (k_j - xmin) / (xmax - xmin);
      
      return val;
  }
  
  // ------------------------------------------------------------
  // Case 2: include_intercept = FALSE (default)
  // ------------------------------------------------------------
  // which = 0 -> linear term
  if (which == 0)
    return (x - xmin);
  
  // cubic basis for internal knots
  int j = which - 1;     // j in {0, ..., K_int-1}
  if (j >= K_int)
    return 0.0;
  
  double k_j = internal[j];
  
  double val =
    tp3(x - k_j)
    - tp3(x - xmin) * (xmax - k_j) / (xmax - xmin)
    - tp3(x - xmax) * (k_j - xmin) / (xmax - xmin);
    
    return val;
}


// Piecewise baseline bin index: bins (knots[k], knots[k+1]], bin 0 closed on
// the left, last bin extended to +inf. Matches survivalBART::pwBin exactly.
static int pw_bin(double t, const double* knots, int K){
  const int J = K - 1;
  int k = 0;
  while(k < J - 1 && t > knots[k+1]) k++;
  return k;
}

// Log-baseline A(t): spline (centered basis) or piecewise (log lambda_{bin(t)}).
static double baseline_log(double t, const double *nsCoef, double *nsKnots, int nKnots,
                           int nNsBasis, const double *ns_mean, int baselineType){
  if(baselineType == 1) return nsCoef[pw_bin(t, nsKnots, nKnots)];
  double s = 0.0;
  for(int i=0; i<nNsBasis; i++){
    s += nsCoef[i] * (ns_kth_basis_scalar(t, nsKnots, nKnots, 3, i, true) - ns_mean[i]);
  }
  return s;
}

double integrand(double t, double *nsCoef, double *reCoef, int nKnots, double *nsKnots,
                 int nNsBasis, const int *rePowers, int nReCoef, double *ns_mean,
                 double reTimeScale, int baselineType){
  double integrand = baseline_log(t, nsCoef, nsKnots, nKnots, nNsBasis, ns_mean, baselineType);
  for(int i=0; i<nReCoef; i++){
    integrand += reCoef[i] * pow(t / reTimeScale, rePowers[i]);
  }
  return exp(integrand);
}

// Gauss-Hermite nodes/weights for the STANDARD NORMAL weight (probabilists'
// convention): int g(x) phi(x) dx  ~=  sum_q w_q g(x_q),  sum_q w_q = 1.
// Golub-Welsch on the Jacobi matrix of the orthonormal Hermite recurrence
// x p_k = sqrt(k+1) p_{k+1} + sqrt(k) p_{k-1}.
static void gauss_hermite_normal(int Q, std::vector<double>& nodes,
                                 std::vector<double>& wts) {
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(Q, Q);
  for (int k = 1; k < Q; ++k) {
    double v = std::sqrt((double)k);
    J(k, k - 1) = v;
    J(k - 1, k) = v;
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(J);
  nodes.resize(Q);
  wts.resize(Q);
  for (int q = 0; q < Q; ++q) {
    nodes[q] = es.eigenvalues()(q);
    double v0 = es.eigenvectors()(0, q);
    wts[q] = v0 * v0;
  }
}



//[[Rcpp::export]]
RcppExport SEXP cpwbart(
    SEXP sexp_iNsKnot,       //Ns knots
    SEXP sexp_iNsMean,       //Ns mean
    SEXP sexp_itrainId,      //training ID for each subject
    SEXP sexp_itestId,       //test ID for each subject
    SEXP sexp_itrees,        //treedraws list from fbart
    SEXP sexp_iNs,           //Ns draws
    SEXP sexp_ireCoef,       //random effects draws
    SEXP sexp_ireCoefCov,    //random effects covariance
    SEXP sexp_ireDegree,     //random-effect polynomial degree
    SEXP sexp_irePowers,     //random-effect polynomial powers
    SEXP sexp_ireTimeScale,  //time standardization for the RE design: z(t) = (t/scale)^power
    SEXP sexp_inGH,          //Gauss-Hermite nodes per RE dimension for unseen subjects (<=1: single MC draw, eq. 25 legacy)
    SEXP sexp_itargetTime,   //target time for each subject
    SEXP sexp_ixObsTime,     //observed time for each subject
    SEXP sexp_ixIndex,       //index of last observation in the subject
    SEXP sexp_ix,            //x matrix to predict at
    SEXP sexp_itc,           //thread count
    SEXP sexp_ibaselineType  //0 = spline baseline, 1 = piecewise-constant
)
{
  Rprintf("*****In main of C++ for bart prediction\n");
  
  //--------------------------------------------------
  //get threadcount
  int tc = Rcpp::as<int>(sexp_itc);
  cout << "tc (threadcount): " << tc << endl;
  //--------------------------------------------------
  //process trees
  Rcpp::List trees(sexp_itrees);
  
  Rcpp::CharacterVector itrees(Rcpp::wrap(trees["trees"]));
  std::string itv(itrees[0]);
  std::stringstream ttss(itv);
  
  size_t nd,m,p;
  ttss >> nd >> m >> p;
  cout << "number of bart draws: " << nd << endl;
  cout << "number of trees in bart sum: " << m << endl;
  cout << "number of x columns: " << p << endl;
  //--------------------------------------------------
  //process cutpoints (from trees)
  Rcpp::List  ixi(Rcpp::wrap(trees["cutpoints"]));
  size_t pp = ixi.size();
  if(p!=pp) cout << "WARNING: p from trees and p from x don't agree\n";
  xinfo xi;
  xi.resize(p);
  for(size_t i=0;i<p;i++) {
    Rcpp::NumericVector cutv(ixi[i]);
    xi[i].resize(cutv.size());
    std::copy(cutv.begin(),cutv.end(),xi[i].begin());
  }
  //--------------------------------------------------
  //process x
  cout << "processing x" << endl;
  Rcpp::NumericMatrix xpred(sexp_ix);
  
  cout << "Ns" << endl;
  Rcpp::NumericVector NsKnot(sexp_iNsKnot);
  double *iNsKnot = &NsKnot[0];
  int nKnots = NsKnot.size();
  
  Rcpp::NumericVector NsDraws(sexp_iNs);
  double *iNsDraws = &NsDraws[0];
  
  Rcpp::NumericVector NsMean(sexp_iNsMean);
  double *iNsMean = &NsMean[0];
  for (int k = 0; k < NsMean.size(); ++k) cout << "NsMean[" << k << "] = " << iNsMean[k] << endl;
  
  const int baselineType = Rcpp::as<int>(sexp_ibaselineType);
  cout << "baselineType: " << baselineType << " (0=spline, 1=piecewise)" << endl;
  
  cout << "id" << endl;
  Rcpp::IntegerVector trainId(sexp_itrainId);
  int *itrainId = &trainId[0];
  Rcpp::IntegerVector testId(sexp_itestId);
  int *itestId = &testId[0];
  int trainNum = trainId.size();
  int testNum = testId.size();
  
  cout << "reCoef" << endl;
  Rcpp::NumericVector reCoefDraws(sexp_ireCoef);
  double *ireCoefDraws = &reCoefDraws[0];
  Rcpp::NumericVector reCoefCov(sexp_ireCoefCov);
  double *ireCoefCov = &reCoefCov[0];
  size_t reDegree = Rcpp::as<size_t>(sexp_ireDegree);
  Rcpp::IntegerVector rePowersVec(sexp_irePowers);
  int *irePowers = &rePowersVec[0];
  const int nReCoef = rePowersVec.size();
  const double reTimeScale = Rcpp::as<double>(sexp_ireTimeScale);
  const int nGH = Rcpp::as<int>(sexp_inGH);
  cout << "reTimeScale (z uses t/scale): " << reTimeScale
       << ", nGH (unseen-subject quadrature nodes/dim): " << nGH << endl;
  
  cout << "Time" << endl;
  Rcpp::NumericVector targetTime(sexp_itargetTime);
  Rcpp::NumericVector xObsTime(sexp_ixObsTime);
  Rcpp::IntegerVector xIndex(sexp_ixIndex);
  size_t np = xpred.ncol();
  cout << "from x,p,np: " << xpred.nrow() << ", " << xpred.ncol() << endl;
  //--------------------------------------------------
  //read in trees
  std::vector<vtree> tmat(nd);
  for(size_t i=0;i<nd;i++) tmat[i].resize(m);
  for(size_t i=0;i<nd;i++) {
    for(size_t j=0;j<m;j++) ttss >> tmat[i][j];
  }
  //--------------------------------------------------
  //get predictions
  
  Rcpp::NumericMatrix yhat(nd,np);
  std::fill(yhat.begin(), yhat.end(), 1.0);
  double *px = &xpred(0,0);
  
#ifndef _OPENMP
  cout << "***using serial code\n";
  getpred(0, nd-1, p, m, np,  xi,  tmat, px,  yhat);
#else
  if(tc==1) {cout << "***using serial code\n"; getpred(0, nd-1, p, m, np,  xi,  tmat, px,  yhat);}
  else {
    cout << "***using parallel code\n";
#pragma omp parallel num_threads(tc)
    local_getpred(nd,p,m,np,xi,tmat,px,yhat);
  }
#endif
  
  //--------------------------------------------------
  Rcpp::List ret;
  ret["yhat.test"] = yhat;
  
  //--------------------------------------------------
  // Per-draw cumulative hazard and hazard at the target time.
  //--------------------------------------------------
  // For each MCMC draw i and each test subject j:
  //     Lambda_{i,j}(t*) = integral_0^{t*} lambda_{i,j}(t) dt
  //     lambda_{i,j}(t*) = bart_i(interval containing t*) * f_{i,j}(t*)
  // with
  //     f_{i,j}(t) = exp( nsCoef_i . (ns_basis(t) - ns_mean)
  //                       + reCoef_{i,j} . poly(t) )
  // using draw i's spline coefficients and draw i's BART prediction directly
  // (NO averaging of parameters before plugging in).
  //
  // Random effect reCoef_{i,j}:
  //   * subject seen in training -> draw i's posterior RE for that subject;
  //   * unseen (out-of-sample) subject -> the RE is INTEGRATED OUT against
  //         b ~ N(0, Sigma_i),  Sigma_i = draw i's RE covariance,
  //     by Gauss-Hermite quadrature with nGH nodes per dimension:
  //         Sbar_i(t*) = sum_q w_q exp{ -Lambda_{i}(t* | b_q) }.
  //     exp(-H) is spiky in the slope through exp(b1 t/tau), so a single inner
  //     MC draw (legacy eq. 25) makes Sbar noisy; quadrature is deterministic.
  //     For the (intercept, slope) design the intercept factors out of the
  //     time integral (H = exp(b0) * C(b1, t*)), so only nGH time integrations
  //     are needed per (draw, subject), not nGH^2.
  //     The RE prior is assumed centered at 0 (reCoefMean = 0 in training).
  //     Set nGH <= 1 to recover the legacy single-draw behavior.
  //
  // For unseen subjects the stored value is the EFFECTIVE cumulative hazard
  //     -log Sbar_i(t*),
  // so that the R-side  colMeans(exp(-cumHazard.test))  remains the correct
  // posterior-mean marginal survival for both seen and unseen subjects.
  //--------------------------------------------------
  cout << "Calculating per-draw cumulative hazard and hazard at target time" << endl;
  cout << "nd: " << nd << ", trainNum: " << trainNum << ", testNum: " << testNum << endl;
  
  const int coef_per  = nReCoef;   // per-subject RE dimension
  // Baseline coefficient count: spline = nKnots basis functions (intercept +
  // linear + cubic; t^2 removed); piecewise = nKnots-1 bin heights. Matches
  // training nsDraws columns in both modes.
  const int coef_size = (baselineType == 1) ? (nKnots - 1) : nKnots;
  
  // Map each test subject to its training index (-1 if unseen / out-of-sample).
  std::vector<int> trainLocOfTest(testNum, -1);
  for (size_t j = 0; j < (size_t)testNum; ++j)
    for (size_t k = 0; k < (size_t)trainNum; ++k)
      if (itestId[j] == itrainId[k]) { trainLocOfTest[j] = (int)k; break; }
      
      bool anyUnseen = false;
      for (size_t j = 0; j < (size_t)testNum; ++j)
        if (trainLocOfTest[j] < 0) { anyUnseen = true; break; }
        
        Rcpp::NumericMatrix cumHazard(nd, testNum);   // per-draw cumulative hazard at targetTime
        Rcpp::NumericMatrix hazard(nd, testNum);      // per-draw hazard at targetTime
        
        Rcpp::RNGScope rngScope;   // manage R's RNG; MC fallback below is reproducible under set.seed()
        std::vector<double> reCoef_j(coef_per, 0.0);
        boost::math::quadrature::gauss_kronrod<double, 61> integrator;
        
        // Integrate f over (a,b): piecewise mode splits at interior cutpoints so the
        // quadrature never straddles a baseline step; spline mode is one GK61 call.
        auto integ = [&](const std::function<double(double)>& f, double a, double b) {
          if (baselineType != 1) return integrator.integrate(f, a, b);
          double total = 0.0, lo = a;
          for (int c = 1; c <= nKnots - 2; ++c) {
            double cp = iNsKnot[c];
            if (cp <= lo) continue;
            if (cp >= b) break;
            total += integrator.integrate(f, lo, cp);
            lo = cp;
          }
          if (b > lo) total += integrator.integrate(f, lo, b);
          return total;
        };
        
        // Gauss-Hermite nodes/weights (standard normal), shared across draws/subjects.
        const bool useGH = (nGH > 1);
        std::vector<double> ghx, ghw;
        if (anyUnseen && useGH) gauss_hermite_normal(nGH, ghx, ghw);
        
        // Locate the intercept (power 0) and slope (power 1) columns of the RE design;
        // the factored 2-D quadrature below requires exactly this {0,1} structure.
        int k0 = -1, k1 = -1;
        for (int k = 0; k < nReCoef; ++k) {
          if (irePowers[k] == 0) k0 = k;
          else if (irePowers[k] == 1) k1 = k;
        }
        const bool interceptSlope = (nReCoef == 2 && k0 >= 0 && k1 >= 0);
        
        // Segment memo for unseen subjects (cleared per draw); TVSB_PRED_MEMO_OFF=1 disables it.
        std::unordered_map<PredSegKey, double, PredSegHash> segMemo;
        const bool useMemo = (std::getenv("TVSB_PRED_MEMO_OFF") == nullptr);
        
        for (size_t i = 0; i < nd; ++i) {
          segMemo.clear();
          
          // Draw i's spline coefficients: contiguous block of length coef_size.
          double* nsCoef_i = &iNsDraws[i * (size_t)coef_size];
          
          // Draw i's RE covariance (and its Cholesky factor for the MC fallback).
          Eigen::MatrixXd Sigma, L;
          if (anyUnseen) {
            Sigma.resize(coef_per, coef_per);
            for (int r = 0; r < coef_per; ++r)
              for (int c = 0; c < coef_per; ++c)
                Sigma(r, c) = ireCoefCov[i * (size_t)coef_per * coef_per
              + (size_t)r * coef_per + c];
            Eigen::LLT<Eigen::MatrixXd> llt(Sigma);
            L = llt.matrixL();   // Sigma is an inverse-Wishart draw, so positive definite
          }
          
          for (size_t j = 0; j < (size_t)testNum; ++j) {
            
            const int trainLoc = trainLocOfTest[j];
            
            const double endTime          = targetTime[j];
            const int    start_time_index = xIndex[j]     + 1;
            const int    end_time_index   = xIndex[j + 1];
            
            // integ() through the segment memo while walking a subject-independent
            // integrand (memoQ >= 0: its GH node key); the plain integ() otherwise.
            int memoQ = -1;
            auto integM = [&](const std::function<double(double)>& f, double a, double b) -> double {
              if (memoQ < 0 || !useMemo) return integ(f, a, b);
              const PredSegKey key{memoQ, pred_bits(a), pred_bits(b)};
              auto it = segMemo.find(key);
              if (it != segMemo.end()) return it->second;
              const double v = integ(f, a, b);
              segMemo.emplace(key, v);
              return v;
            };
            
            // Interval walk shared by every integrand below: accumulates
            //   C      = sum_l bart_l * int_{interval l ^ [0, t*]} f(t) dt   (+ LOCF tail)
            //   hazfac = bart(interval containing t*) * f(t*)
            auto walk = [&](const std::function<double(double)>& f,
                            double& C_out, double& hazfac_out) {
              const double f_endTime = f(endTime);
              C_out = 0.0;
              hazfac_out = 0.0;   // stays 0 if endTime precedes the first interval
              for (int l = start_time_index; l < end_time_index; ++l) {
                double a = xObsTime[l];
                double b = xObsTime[l + 1];
                double bart_pred_l = yhat(i, l);
                if (a >= endTime) {
                  break;
                } else if (b > endTime) {
                  // Target time falls inside this interval: integrate up to endTime.
                  C_out += integM(f, a, endTime) * bart_pred_l;
                  hazfac_out = bart_pred_l * f_endTime;
                  break;
                } else {
                  C_out += integM(f, a, b) * bart_pred_l;
                  hazfac_out = bart_pred_l * f_endTime;
                }
              }
              // LOCF extension: if endTime is past the subject's last observed time,
              // carry the last observed covariate row forward (BART term frozen on
              // (last_obs_time, endTime]); the spline baseline and RE polynomial in
              // f(t) keep evolving with t.
              double last_obs_time = xObsTime[end_time_index];
              if (endTime > last_obs_time) {
                double bart_pred_locf = yhat(i, end_time_index);
                C_out += integM(f, last_obs_time, endTime) * bart_pred_locf;
                hazfac_out = bart_pred_locf * f_endTime;
              }
            };
            
            // Baseline-only part of the integrand (RE handled per quadrature node).
            auto f_spline = [&](double t) {
              return exp(baseline_log(t, nsCoef_i, iNsKnot, nKnots, coef_size, iNsMean,
                                      baselineType));
            };
            
            if (trainLoc >= 0) {
              // ---- Seen subject: plug in draw i's posterior RE coefficients. ----
              for (int k = 0; k < coef_per; ++k)
                reCoef_j[k] = ireCoefDraws[i * (size_t)trainNum * coef_per
              + (size_t)trainLoc * coef_per + k];
              auto f = [&](double t) {
                return integrand(t, nsCoef_i, reCoef_j.data(),
                                 nKnots, iNsKnot, coef_size, irePowers, nReCoef, iNsMean,
                                 reTimeScale, baselineType);
              };
              double cumHaz_ij, hazard_ij;
              walk(f, cumHaz_ij, hazard_ij);
              cumHazard(i, j) = cumHaz_ij;
              hazard(i, j)    = hazard_ij;
              
            } else if (useGH && interceptSlope) {
              // ---- Unseen subject, (intercept, slope) design: factored 2-D GH. ----
              // b1 = sig1 * u,  b0 = (s01/sig1) * u + s0c * v,  (u, v) iid N(0,1),
              // so H(b0, b1) = exp(b0) * C(b1): only nGH time integrations (over u).
              double s00 = Sigma(k0, k0), s01 = Sigma(k0, k1), s11 = Sigma(k1, k1);
              double Sbar = 0.0, hazNum = 0.0;
              if (s11 > 1e-12) {
                double sig1 = std::sqrt(s11);
                double b0_load_u = s01 / sig1;
                double s0c = std::sqrt(std::max(s00 - s01 * s01 / s11, 0.0));
                for (int qu = 0; qu < nGH; ++qu) {
                  double b1q = sig1 * ghx[qu];
                  auto fq = [&](double t) { return f_spline(t) * exp(b1q * (t / reTimeScale)); };
                  double Cq, hazq;
                  memoQ = qu;
                  walk(fq, Cq, hazq);
                  memoQ = -1;
                  for (int qv = 0; qv < nGH; ++qv) {
                    double eb0 = exp(b0_load_u * ghx[qu] + s0c * ghx[qv]);
                    double Sqr = exp(-eb0 * Cq);
                    double w = ghw[qu] * ghw[qv];
                    Sbar   += w * Sqr;
                    hazNum += w * Sqr * eb0 * hazq;   // mixture hazard numerator
                  }
                }
              } else {
                // Degenerate slope: 1-D quadrature over the intercept only.
                double sig0 = std::sqrt(std::max(s00, 0.0));
                double C0, haz0;
                memoQ = nGH;   // spline-only integrand: subject-independent as well
                walk(f_spline, C0, haz0);
                memoQ = -1;
                for (int qv = 0; qv < nGH; ++qv) {
                  double eb0 = exp(sig0 * ghx[qv]);
                  double S = exp(-eb0 * C0);
                  Sbar   += ghw[qv] * S;
                  hazNum += ghw[qv] * S * eb0 * haz0;
                }
              }
              cumHazard(i, j) = -log(std::max(Sbar, 1e-300));
              hazard(i, j)    = (Sbar > 0.0) ? hazNum / Sbar : 0.0;
              
            } else if (useGH && nReCoef == 1) {
              // ---- Unseen subject, single RE column (any power): 1-D GH. ----
              double sig = std::sqrt(std::max(Sigma(0, 0), 0.0));
              int pw = irePowers[0];
              double Sbar = 0.0, hazNum = 0.0;
              for (int q = 0; q < nGH; ++q) {
                double bq = sig * ghx[q];
                auto fq = [&](double t) {
                  return f_spline(t) * exp(bq * pow(t / reTimeScale, pw));
                };
                double Cq, hazq;
                memoQ = q;
                walk(fq, Cq, hazq);
                memoQ = -1;
                double S = exp(-Cq);
                Sbar   += ghw[q] * S;
                hazNum += ghw[q] * S * hazq;
              }
              cumHazard(i, j) = -log(std::max(Sbar, 1e-300));
              hazard(i, j)    = (Sbar > 0.0) ? hazNum / Sbar : 0.0;
              
            } else if (useGH) {
              // ---- Unseen subject, general RE design: nGH^2 MC draws of b. ----
              int K = nGH * nGH;
              double Sbar = 0.0, hazNum = 0.0;
              for (int kk = 0; kk < K; ++kk) {
                Eigen::VectorXd z(coef_per);
                for (int k = 0; k < coef_per; ++k) z(k) = R::rnorm(0.0, 1.0);
                Eigen::VectorXd bvec = L * z;
                for (int k = 0; k < coef_per; ++k) reCoef_j[k] = bvec(k);
                auto f = [&](double t) {
                  return integrand(t, nsCoef_i, reCoef_j.data(),
                                   nKnots, iNsKnot, coef_size, irePowers, nReCoef, iNsMean,
                                   reTimeScale, baselineType);
                };
                double Ck, hazk;
                walk(f, Ck, hazk);
                double S = exp(-Ck);
                Sbar   += S / K;
                hazNum += S * hazk / K;
              }
              cumHazard(i, j) = -log(std::max(Sbar, 1e-300));
              hazard(i, j)    = (Sbar > 0.0) ? hazNum / Sbar : 0.0;
              
            } else {
              // ---- Legacy eq. 25: single Monte-Carlo draw b ~ N(0, Sigma_i). ----
              Eigen::VectorXd z(coef_per);
              for (int k = 0; k < coef_per; ++k) z(k) = R::rnorm(0.0, 1.0);
              Eigen::VectorXd bvec = L * z;
              for (int k = 0; k < coef_per; ++k) reCoef_j[k] = bvec(k);
              auto f = [&](double t) {
                return integrand(t, nsCoef_i, reCoef_j.data(),
                                 nKnots, iNsKnot, coef_size, irePowers, nReCoef, iNsMean,
                                 reTimeScale, baselineType);
              };
              double cumHaz_ij, hazard_ij;
              walk(f, cumHaz_ij, hazard_ij);
              cumHazard(i, j) = cumHaz_ij;
              hazard(i, j)    = hazard_ij;
            }
          } // test subjects
        } // draws
        
        ret["cumHazard.test"] = cumHazard;   // nd x testNum : per-draw cumulative hazard at targetTime
        ret["hazard.test"]    = hazard;      // nd x testNum : per-draw hazard at targetTime
        return ret;
}

//--------------------------------------------------
// cpwbart_grid: cpwbart for a grid of target times shared by all subjects
// (predict_S_tvsb), in one call.
//
// cumHazard.test[i, j, k] (and hazard.test[i, j, k] when wantHazard) are bit for
// bit cpwbart(..., targetTime = rep(grid[k], n), ...)$cumHazard.test[i, j]
// (resp. $hazard.test[i, j]):
//   * the trees are parsed and predicted once for the whole grid;
//   * the interval walk of every (subject, grid time) depends on the data only,
//     so it is laid out once: the segments whose integrals enter
//       C = sum_l bart_l * int_{segment l} f(t) dt   (+ LOCF tail)
//     in cpwbart's summation order, and the row whose BART factor multiplies
//     f(t*) in the hazard;
//   * each distinct segment is integrated once per draw: per seen subject (the
//     posterior b_i makes the integrand subject-specific; every full interval
//     then serves all later grid times), and ONCE for all unseen subjects, whose
//     Gauss-Hermite integrands depend only on (draw, GH node);
//   * a shared segment is integrated for all GH nodes in one fused GK61 pass
//     (gk61::fused; boost recomputes every component it would subdivide) that
//     evaluates exp(baseline) once per quadrature node, from boost's node times,
//     the centred spline basis and the RE design factor cached per segment;
//   * integrands are handed to boost as their own types (inlined into the GK
//     loop) instead of through std::function;
//   * the per-draw GH constants exp(b0) and w_u w_v are tabulated with cpwbart's
//     expressions.
// The fused pass is plain arithmetic in boost's and baseline_log's expression
// order, and every other step keeps cpwbart's expressions; with the package's
// -ffp-contract=off that is exact on any IEEE-754 toolchain. Guards for builds
// whose own flags re-enable contraction, or any other surprise: a build that
// contracts floating-point multiply-adds (gk61::contractionDetected()) does not
// use this code at all, and startup checks compare the fused pass and the inlined
// integrands with the reference dispatch (boost through std::function, as in
// cpwbart) bit for bit on a sample of segments -- for the unseen subjects the
// first time each GH path (1: degenerate slope, 2: intercept + slope, 3: single
// RE column) is used, for the seen subjects on up to 64 segments of the first
// draw. In either case cpwbart_grid returns list(fallback = TRUE, reason) and
// pwbart_grid() computes the grid with one cpwbart call per time (the reference
// code) instead.
// No RNG is used: the Monte-Carlo RE paths of cpwbart (nGH <= 1, or a general RE
// design, for unseen subjects) are not available here -- call cpwbart per time.
//--------------------------------------------------

// One step of an interval walk: segment id and the row of its BART factor.
struct GridStep { int seg; int row; };
// Steps [begin, end) of one (subject, grid time) walk; hazRow = row whose BART
// factor multiplies f(t*) (-1: t* precedes the subject's first interval).
struct GridWalk { size_t begin, end; int hazRow; };

//[[Rcpp::export]]
RcppExport SEXP cpwbart_grid(
    SEXP sexp_iNsKnot,       //Ns knots
    SEXP sexp_iNsMean,       //Ns mean
    SEXP sexp_itrainId,      //training ID for each subject
    SEXP sexp_itestId,       //test ID for each subject
    SEXP sexp_itrees,        //treedraws list from fbart
    SEXP sexp_iNs,           //Ns draws
    SEXP sexp_ireCoef,       //random effects draws
    SEXP sexp_ireCoefCov,    //random effects covariance
    SEXP sexp_irePowers,     //random-effect polynomial powers
    SEXP sexp_ireTimeScale,  //time standardization for the RE design: z(t) = (t/scale)^power
    SEXP sexp_inGH,          //Gauss-Hermite nodes per RE dimension for unseen subjects (> 1)
    SEXP sexp_igrid,         //target times, shared by every subject
    SEXP sexp_ixObsTime,     //observed time for each subject
    SEXP sexp_ixIndex,       //index of last observation in the subject
    SEXP sexp_ix,            //x matrix to predict at
    SEXP sexp_ibaselineType, //0 = spline baseline, 1 = piecewise-constant
    SEXP sexp_iwantHazard    //also return hazard.test
)
{
  Rprintf("*****In main of C++ for bart prediction on a time grid\n");

  //--------------------------------------------------
  //trees, cutpoints and x (as cpwbart)
  Rcpp::List trees(sexp_itrees);
  Rcpp::CharacterVector itrees(Rcpp::wrap(trees["trees"]));
  std::string itv(itrees[0]);
  std::stringstream ttss(itv);
  size_t nd,m,p;
  ttss >> nd >> m >> p;
  cout << "number of bart draws: " << nd << endl;
  cout << "number of trees in bart sum: " << m << endl;
  cout << "number of x columns: " << p << endl;
  Rcpp::List  ixi(Rcpp::wrap(trees["cutpoints"]));
  size_t pp = ixi.size();
  if(p!=pp) cout << "WARNING: p from trees and p from x don't agree\n";
  xinfo xi;
  xi.resize(p);
  for(size_t i=0;i<p;i++) {
    Rcpp::NumericVector cutv(ixi[i]);
    xi[i].resize(cutv.size());
    std::copy(cutv.begin(),cutv.end(),xi[i].begin());
  }
  Rcpp::NumericMatrix xpred(sexp_ix);

  Rcpp::NumericVector NsKnot(sexp_iNsKnot);
  double *iNsKnot = &NsKnot[0];
  int nKnots = NsKnot.size();
  Rcpp::NumericVector NsDraws(sexp_iNs);
  double *iNsDraws = &NsDraws[0];
  Rcpp::NumericVector NsMean(sexp_iNsMean);
  double *iNsMean = &NsMean[0];
  const int baselineType = Rcpp::as<int>(sexp_ibaselineType);

  Rcpp::IntegerVector trainId(sexp_itrainId);
  int *itrainId = &trainId[0];
  Rcpp::IntegerVector testId(sexp_itestId);
  int *itestId = &testId[0];
  int trainNum = trainId.size();
  int testNum = testId.size();

  Rcpp::NumericVector reCoefDraws(sexp_ireCoef);
  double *ireCoefDraws = &reCoefDraws[0];
  Rcpp::NumericVector reCoefCov(sexp_ireCoefCov);
  double *ireCoefCov = &reCoefCov[0];
  Rcpp::IntegerVector rePowersVec(sexp_irePowers);
  int *irePowers = &rePowersVec[0];
  const int nReCoef = rePowersVec.size();
  const double reTimeScale = Rcpp::as<double>(sexp_ireTimeScale);
  const int nGH = Rcpp::as<int>(sexp_inGH);

  Rcpp::NumericVector grid(sexp_igrid);
  const int J = grid.size();
  Rcpp::NumericVector xObsTime(sexp_ixObsTime);
  Rcpp::IntegerVector xIndex(sexp_ixIndex);
  const bool wantHazard = Rcpp::as<bool>(sexp_iwantHazard);
  size_t np = xpred.ncol();
  cout << "grid times: " << J << ", subjects: " << testNum << ", rows: " << np << endl;

  //--------------------------------------------------
  //trees and BART predictions, once for the whole grid (as cpwbart)
  std::vector<vtree> tmat(nd);
  for(size_t i=0;i<nd;i++) tmat[i].resize(m);
  for(size_t i=0;i<nd;i++) {
    for(size_t j=0;j<m;j++) ttss >> tmat[i][j];
  }
  Rcpp::NumericMatrix yhat(nd,np);
  std::fill(yhat.begin(), yhat.end(), 1.0);
  double *px = &xpred(0,0);
  getpred(0, nd-1, p, m, np,  xi,  tmat, px,  yhat);

  //--------------------------------------------------
  //subjects, RE design and GH rule (as cpwbart)
  const int coef_per  = nReCoef;
  const int coef_size = (baselineType == 1) ? (nKnots - 1) : nKnots;
  std::vector<int> trainLocOfTest(testNum, -1);
  for (size_t j = 0; j < (size_t)testNum; ++j)
    for (size_t k = 0; k < (size_t)trainNum; ++k)
      if (itestId[j] == itrainId[k]) { trainLocOfTest[j] = (int)k; break; }
  bool anyUnseen = false;
  for (size_t j = 0; j < (size_t)testNum; ++j)
    if (trainLocOfTest[j] < 0) { anyUnseen = true; break; }
  const bool useGH = (nGH > 1);
  std::vector<double> ghx, ghw;
  if (anyUnseen && useGH) gauss_hermite_normal(nGH, ghx, ghw);
  int k0 = -1, k1 = -1;
  for (int k = 0; k < nReCoef; ++k) {
    if (irePowers[k] == 0) k0 = k;
    else if (irePowers[k] == 1) k1 = k;
  }
  const bool interceptSlope = (nReCoef == 2 && k0 >= 0 && k1 >= 0);
  if (anyUnseen && !(useGH && (interceptSlope || nReCoef == 1)))
    Rcpp::stop("cpwbart_grid: unseen subjects need Gauss-Hermite nodes (n.gh > 1) and an (intercept, slope) "
               "or single-column RE design; the Monte-Carlo paths are only in cpwbart (one call per target time)");

  // Hand the grid back to pwbart_grid() for per-time cpwbart calls (the reference
  // code): used when this build contracts floating-point multiply-adds, and on any
  // startup-check mismatch below.
  auto fallback = [&](const std::string& why) -> Rcpp::List {
    Rprintf("cpwbart_grid: %s -- falling back to one cpwbart call per grid time\n", why.c_str());
    return Rcpp::List::create(Rcpp::Named("fallback") = true, Rcpp::Named("reason") = why);
  };
  if (gk61::contractionDetected())
    return fallback("this build contracts floating-point multiply-adds (-ffp-contract is not off)");

  //--------------------------------------------------
  //interval walks (data only) and the segment registry: the segments of unseen
  //subjects are shared (owner -1), a seen subject owns its own
  std::unordered_map<PredSegKey, int, PredSegHash> segIndex;
  std::vector<double> segA, segB;
  std::vector<int> segOwner;
  auto segOf = [&](int owner, double a, double b) -> int {
    const PredSegKey key{owner, pred_bits(a), pred_bits(b)};
    auto it = segIndex.find(key);
    if (it != segIndex.end()) return it->second;
    const int id = (int)segA.size();
    segIndex.emplace(key, id);
    segA.push_back(a); segB.push_back(b); segOwner.push_back(owner);
    return id;
  };
  std::vector<GridStep> steps;
  std::vector<GridWalk> walks((size_t)testNum * J);
  for (int j = 0; j < testNum; ++j) {
    const int owner = (trainLocOfTest[j] >= 0) ? j : -1;
    const int start_time_index = xIndex[j] + 1;
    const int end_time_index   = xIndex[j + 1];
    const double last_obs_time = xObsTime[end_time_index];
    for (int kk = 0; kk < J; ++kk) {
      const double endTime = grid[kk];
      GridWalk w;
      w.begin = steps.size();
      w.hazRow = -1;
      for (int l = start_time_index; l < end_time_index; ++l) {
        double a = xObsTime[l];
        double b = xObsTime[l + 1];
        if (a >= endTime) {
          break;
        } else if (b > endTime) {
          steps.push_back(GridStep{segOf(owner, a, endTime), l});
          w.hazRow = l;
          break;
        } else {
          steps.push_back(GridStep{segOf(owner, a, b), l});
          w.hazRow = l;
        }
      }
      if (endTime > last_obs_time) {
        steps.push_back(GridStep{segOf(owner, last_obs_time, endTime), end_time_index});
        w.hazRow = end_time_index;
      }
      w.end = steps.size();
      walks[(size_t)j * J + kk] = w;
    }
  }
  const int nSeg = (int)segA.size();
  //value slots: nGH per shared segment (one per GH node), 1 per seen-subject segment
  const int nQmax = (anyUnseen && useGH) ? nGH : 1;
  std::vector<size_t> segOff(nSeg);
  std::vector<int> sharedSegs;
  std::vector<std::vector<int> > ownSegs(testNum);
  size_t nVal = 0;
  for (int s = 0; s < nSeg; ++s) {
    segOff[s] = nVal;
    if (segOwner[s] < 0) { sharedSegs.push_back(s); nVal += nQmax; }
    else { ownSegs[segOwner[s]].push_back(s); nVal += 1; }
  }
  std::vector<double> segVal(nVal);
  const size_t nShared = sharedSegs.size();
  cout << "distinct segments: " << nSeg << " (" << nShared << " shared by unseen subjects)" << endl;

  //--------------------------------------------------
  //node cache of the shared segments (spline baseline): boost's top-level GK61
  //node times, the RE design factor z(t) (t / tau for the slope design, else
  //pow(t / tau, power)) and the centred spline basis there; all draw-independent
  const int K = gk61::kNodes;
  bool fusedOK = GK61_REPLICA_SUPPORTED && baselineType != 1 && anyUnseen && useGH;
  std::vector<char> segNodes(nShared, 0);
  std::vector<double> nodeZ, nodeB;
  if (fusedOK) {
    nodeZ.assign(nShared * K, 0.0);
    nodeB.assign(nShared * K * coef_size, 0.0);
    std::vector<double> Tn(K);
    for (size_t u = 0; u < nShared; ++u) {
      const double a = segA[sharedSegs[u]], b = segB[sharedSegs[u]];
      if (!(b - a >= 1e-6) || !gk61::recordNodes(a, b, Tn.data())) continue;   // tiny / degenerate: boost
      segNodes[u] = 1;
      for (int r = 0; r < K; ++r) {
        nodeZ[u * K + r] = interceptSlope ? Tn[r] / reTimeScale : pow(Tn[r] / reTimeScale, irePowers[0]);
        for (int l = 0; l < coef_size; ++l)
          nodeB[(u * K + r) * coef_size + l] = ns_kth_basis_scalar(Tn[r], iNsKnot, nKnots, 3, l, true) - iNsMean[l];
      }
    }
  }

  //--------------------------------------------------
  //integration
  boost::math::quadrature::gauss_kronrod<double, 61> integrator;
  // cpwbart's integ(): piecewise mode splits at interior cutpoints, spline mode is
  // one GK61 call; the integrand keeps its own type (inlined into the GK loop)
  auto integT = [&](const auto& f, double a, double b) -> double {
    if (baselineType != 1) return integrator.integrate(f, a, b);
    double total = 0.0, lo = a;
    for (int c = 1; c <= nKnots - 2; ++c) {
      double cp = iNsKnot[c];
      if (cp <= lo) continue;
      if (cp >= b) break;
      total += integrator.integrate(f, lo, cp);
      lo = cp;
    }
    if (b > lo) total += integrator.integrate(f, lo, b);
    return total;
  };
  // the reference dispatch: the integrand through std::function, as in cpwbart
  auto integRef = [&](const auto& f, double a, double b) -> double {
    const std::function<double(double)> F(f);
    return integT(F, a, b);
  };

  Rcpp::NumericVector cumHazard((size_t)nd * testNum * J);
  cumHazard.attr("dim") = Rcpp::IntegerVector::create((int)nd, testNum, J);
  Rcpp::NumericVector hazard(wantHazard ? (size_t)nd * testNum * J : 0);

  std::vector<double> reCoef_j(coef_per, 0.0), bq(nQmax), eb0tab((size_t)nQmax * nQmax), wtab((size_t)nQmax * nQmax),
                      eb0d(nQmax), C(nQmax), Hf(nQmax), fe(nQmax), fsN(K), vA(nQmax), vB(nQmax);
  std::vector<char> okQ(nQmax);
  bool checkedPath[4] = {false, false, false, false};   // unseen-subject check done, per GH path
  bool checkedSeen = false;                             // seen-subject check done (first draw)
  long seenCmp = 0, seenBad = 0;

  for (size_t i = 0; i < nd; ++i) {

    // Draw i's spline coefficients: contiguous block of length coef_size.
    double* nsCoef_i = &iNsDraws[i * (size_t)coef_size];
    auto f_spline = [&](double t) {
      return exp(baseline_log(t, nsCoef_i, iNsKnot, nKnots, coef_size, iNsMean,
                              baselineType));
    };

    //---- unseen subjects: this draw's GH path and constants (cpwbart's expressions) ----
    // path 2: (intercept, slope), factored 2-D GH; path 1: degenerate slope, 1-D GH
    // over the intercept; path 3: single RE column (any power), 1-D GH
    int path = 0, nQ = 0;
    if (anyUnseen) {
      Eigen::MatrixXd Sigma(coef_per, coef_per);
      for (int r = 0; r < coef_per; ++r)
        for (int c = 0; c < coef_per; ++c)
          Sigma(r, c) = ireCoefCov[i * (size_t)coef_per * coef_per
        + (size_t)r * coef_per + c];
      if (interceptSlope) {
        double s00 = Sigma(k0, k0), s01 = Sigma(k0, k1), s11 = Sigma(k1, k1);
        if (s11 > 1e-12) {
          path = 2; nQ = nGH;
          double sig1 = std::sqrt(s11);
          double b0_load_u = s01 / sig1;
          double s0c = std::sqrt(std::max(s00 - s01 * s01 / s11, 0.0));
          for (int qu = 0; qu < nGH; ++qu) {
            bq[qu] = sig1 * ghx[qu];
            for (int qv = 0; qv < nGH; ++qv) {
              eb0tab[(size_t)qu * nGH + qv] = exp(b0_load_u * ghx[qu] + s0c * ghx[qv]);
              wtab[(size_t)qu * nGH + qv] = ghw[qu] * ghw[qv];
            }
          }
        } else {
          path = 1; nQ = 1;
          double sig0 = std::sqrt(std::max(s00, 0.0));
          for (int qv = 0; qv < nGH; ++qv) eb0d[qv] = exp(sig0 * ghx[qv]);
        }
      } else {
        path = 3; nQ = nGH;
        double sig = std::sqrt(std::max(Sigma(0, 0), 0.0));
        for (int q = 0; q < nGH; ++q) bq[q] = sig * ghx[q];
      }
    }
    // component c of the unseen-subject integrand: exactly cpwbart's lambdas
    const int pw = (nReCoef > 0) ? irePowers[0] : 0;
    auto fq2 = [&](int c) { const double b1q = bq[c];
      return [&, b1q](double t) { return f_spline(t) * exp(b1q * (t / reTimeScale)); }; };
    auto fq3 = [&](int c) { const double bqc = bq[c];
      return [&, bqc](double t) { return f_spline(t) * exp(bqc * pow(t / reTimeScale, pw)); }; };

    // integrals of shared segment u for every component into out[0 .. nQ-1]:
    // mode 0 = fused pass + inlined boost fallback, 1 = inlined boost only,
    // 2 = reference dispatch only
    auto sharedSeg = [&](size_t u, double* out, int mode) {
      const double a = segA[sharedSegs[u]], b = segB[sharedSegs[u]];
      for (int c = 0; c < nQ; ++c) okQ[c] = 0;
      if (mode == 0 && fusedOK && segNodes[u]) {
        // exp(baseline) at the cached nodes: baseline_log's multiply-add chain
        // (s = 0.0; s += nsCoef[i] * (B_i(t) - ns_mean[i]); ...), same form and order
        for (int r = 0; r < K; ++r) {
          const double* Bc = &nodeB[(u * K + r) * coef_size];
          double s = 0.0;
          for (int l = 0; l < coef_size; ++l) s += nsCoef_i[l] * Bc[l];
          fsN[r] = exp(s);
        }
        const double* Z = &nodeZ[u * K];
        const double scale = (b - a) / 2;
        bool ok[gk61::kMaxComp];
        for (int c0 = 0; c0 < nQ; c0 += gk61::kMaxComp) {
          const int ncc = std::min(gk61::kMaxComp, nQ - c0);
          gk61::fused(scale, ncc, [&](int r, double* o) {
            if (path == 1) { o[0] = fsN[r]; return; }
            for (int c = 0; c < ncc; ++c) o[c] = fsN[r] * exp(bq[c0 + c] * Z[r]);
          }, out + c0, ok);
          for (int c = 0; c < ncc; ++c) okQ[c0 + c] = ok[c];
        }
      }
      const bool ref = (mode == 2);
      for (int c = 0; c < nQ; ++c) {
        if (okQ[c]) continue;
        if (path == 1)      out[c] = ref ? integRef(f_spline, a, b) : integT(f_spline, a, b);
        else if (path == 2) out[c] = ref ? integRef(fq2(c), a, b) : integT(fq2(c), a, b);
        else                out[c] = ref ? integRef(fq3(c), a, b) : integT(fq3(c), a, b);
      }
    };

    //---- check of the shared-segment shortcuts against the reference, the first
    //     time each GH path is used ----
    if (anyUnseen && path > 0 && !checkedPath[path] && nShared > 0) {
      checkedPath[path] = true;
      long nCmp = 0, badInline = 0, badFused = 0;
      const size_t stride = std::max<size_t>(1, nShared / 64);
      for (size_t u = 0; u < nShared; u += stride) {
        sharedSeg(u, vA.data(), 2);
        sharedSeg(u, vB.data(), 1);
        for (int c = 0; c < nQ; ++c) { nCmp++; if (!gk61::sameBits(vA[c], vB[c])) badInline++; }
      }
      if (fusedOK)   // off for the piecewise baseline / boost releases the replica was not checked against
        for (size_t u = 0; u < nShared; u += stride) {
          sharedSeg(u, vA.data(), 2);
          sharedSeg(u, vB.data(), 0);
          for (int c = 0; c < nQ; ++c) if (!gk61::sameBits(vA[c], vB[c])) badFused++;
        }
      Rprintf("cpwbart_grid check (unseen, GH path %d): %ld segment integrals; inlined integrands %s, fused GK61 pass %s\n",
              path, nCmp,
              badInline ? "MISMATCH" : "bit-identical",
              !fusedOK ? "not used" : badFused ? "MISMATCH" : "bit-identical");
      if (badInline || badFused) return fallback("startup check mismatch (unseen subjects)");
    }

    //---- this draw's segment integrals ----
    for (size_t u = 0; u < nShared; ++u) sharedSeg(u, &segVal[segOff[sharedSegs[u]]], 0);
    for (int j = 0; j < testNum; ++j) {
      if (ownSegs[j].empty()) continue;
      const int trainLoc = trainLocOfTest[j];
      for (int k = 0; k < coef_per; ++k)
        reCoef_j[k] = ireCoefDraws[i * (size_t)trainNum * coef_per
      + (size_t)trainLoc * coef_per + k];
      auto f = [&](double t) {
        return integrand(t, nsCoef_i, reCoef_j.data(),
                         nKnots, iNsKnot, coef_size, irePowers, nReCoef, iNsMean,
                         reTimeScale, baselineType);
      };
      size_t t0 = 0;
      if (!checkedSeen) {
        // check of the inlined seen-subject integrand on the first draw, over the
        // seen subjects' segments up to 64 in total; a checked segment takes the
        // reference value, so the check never lets an unverified value through
        for (; t0 < ownSegs[j].size() && seenCmp < 64; ++t0) {
          const int s = ownSegs[j][t0];
          const double vRef = integRef(f, segA[s], segB[s]);
          seenCmp++;
          if (!gk61::sameBits(integT(f, segA[s], segB[s]), vRef)) seenBad++;
          segVal[segOff[s]] = vRef;
        }
        if (seenBad) {
          Rprintf("cpwbart_grid check (seen): %ld segment integrals; inlined integrand MISMATCH\n", seenCmp);
          return fallback("startup check mismatch (seen subjects)");
        }
      }
      for (size_t t = t0; t < ownSegs[j].size(); ++t) {
        const int s = ownSegs[j][t];
        segVal[segOff[s]] = integT(f, segA[s], segB[s]);
      }
    }
    if (!checkedSeen && seenCmp > 0) {
      checkedSeen = true;
      Rprintf("cpwbart_grid check (seen): %ld segment integrals; inlined integrand bit-identical\n", seenCmp);
    }

    //---- walks: cumulative hazard (and hazard) at every grid time ----
    for (int j = 0; j < testNum; ++j) {
      const bool seen = (trainLocOfTest[j] >= 0);
      if (seen && wantHazard)
        for (int k = 0; k < coef_per; ++k)
          reCoef_j[k] = ireCoefDraws[i * (size_t)trainNum * coef_per
        + (size_t)trainLocOfTest[j] * coef_per + k];
      for (int kk = 0; kk < J; ++kk) {
        const GridWalk& w = walks[(size_t)j * J + kk];
        const double endTime = grid[kk];
        const size_t out = i + nd * ((size_t)j + (size_t)testNum * kk);
        if (seen) {
          // ---- Seen subject: draw i's posterior RE coefficients. ----
          double C_out = 0.0;
          for (size_t st = w.begin; st < w.end; ++st)
            C_out += segVal[segOff[steps[st].seg]] * yhat(i, steps[st].row);
          cumHazard[out] = C_out;
          if (wantHazard) {
            const double f_endTime = integrand(endTime, nsCoef_i, reCoef_j.data(),
                                               nKnots, iNsKnot, coef_size, irePowers, nReCoef, iNsMean,
                                               reTimeScale, baselineType);
            hazard[out] = (w.hazRow >= 0) ? yhat(i, w.hazRow) * f_endTime : 0.0;
          }
          continue;
        }
        // ---- Unseen subject: GH mixture over the RE, per component c. ----
        for (int c = 0; c < nQ; ++c) { C[c] = 0.0; Hf[c] = 0.0; }
        for (size_t st = w.begin; st < w.end; ++st) {
          const double* I = &segVal[segOff[steps[st].seg]];
          const double bart_pred_l = yhat(i, steps[st].row);
          for (int c = 0; c < nQ; ++c) C[c] += I[c] * bart_pred_l;
        }
        if (wantHazard && w.hazRow >= 0) {
          const double bl = yhat(i, w.hazRow);
          for (int c = 0; c < nQ; ++c)
            fe[c] = (path == 1) ? f_spline(endTime) : (path == 2) ? fq2(c)(endTime) : fq3(c)(endTime);
          for (int c = 0; c < nQ; ++c) Hf[c] = bl * fe[c];
        }
        double Sbar = 0.0, hazNum = 0.0;
        if (path == 2) {
          for (int qu = 0; qu < nGH; ++qu) {
            const double Cq = C[qu], hazq = Hf[qu];
            for (int qv = 0; qv < nGH; ++qv) {
              double eb0 = eb0tab[(size_t)qu * nGH + qv];
              double Sqr = exp(-eb0 * Cq);
              double w2 = wtab[(size_t)qu * nGH + qv];
              Sbar   += w2 * Sqr;
              hazNum += w2 * Sqr * eb0 * hazq;   // mixture hazard numerator
            }
          }
        } else if (path == 1) {
          const double C0 = C[0], haz0 = Hf[0];
          for (int qv = 0; qv < nGH; ++qv) {
            double eb0 = eb0d[qv];
            double S = exp(-eb0 * C0);
            Sbar   += ghw[qv] * S;
            hazNum += ghw[qv] * S * eb0 * haz0;
          }
        } else {
          for (int q = 0; q < nGH; ++q) {
            double S = exp(-C[q]);
            Sbar   += ghw[q] * S;
            hazNum += ghw[q] * S * Hf[q];
          }
        }
        cumHazard[out] = -log(std::max(Sbar, 1e-300));
        if (wantHazard) hazard[out] = (Sbar > 0.0) ? hazNum / Sbar : 0.0;
      }
    }
  } // draws

  Rcpp::List ret;
  ret["cumHazard.test"] = cumHazard;   // nd x testNum x J : per-draw cumulative hazard at each grid time
  if (wantHazard) {
    hazard.attr("dim") = Rcpp::IntegerVector::create((int)nd, testNum, J);
    ret["hazard.test"] = hazard;       // nd x testNum x J : per-draw hazard at each grid time
  }
  return ret;
}

void getpred(int beg, int end, size_t p, size_t m, size_t np, xinfo& xi, std::vector<vtree>& tmat, double *px, Rcpp::NumericMatrix& yhat)
{
  double *fptemp = new double[np];
  
  for(int i=beg;i<=end;i++) {
    for(size_t j=0;j<m;j++) {
      fit(tmat[i][j],xi,p,np,px,fptemp);
      for(size_t k=0;k<np;k++) yhat(i,k) *= fptemp[k];
    }
  }
  
  delete [] fptemp;
}

#ifdef _OPENMP
void local_getpred(size_t nd, size_t p, size_t m, size_t np, xinfo& xi, std::vector<vtree>& tmat, double *px, Rcpp::NumericMatrix& yhat)
{
  int my_rank = omp_get_thread_num();
  int thread_count = omp_get_num_threads();
  int h = nd/thread_count; int beg = my_rank*h; int end = beg+h-1;
  
  getpred(beg,end,p,m,np,xi,tmat,px,yhat);
}
#endif
