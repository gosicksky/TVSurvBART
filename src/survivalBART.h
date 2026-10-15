// this part is for ncs and random effect
// class should inherit bart class
/*** R
library(splines2)
*/

#include "bd.h"
#include "logbart.h"

#include <ctime>
#include <vector>
// #include <eigen3/Eigen/Dense>
#include <RcppEigen.h>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <Rcpp.h>
#include "gk61.h"

class survivalBART : public logBART{
public:
  
  friend bool bd(tree& x, xinfo& xi, dinfo& di, pinfo& pi, double sigma,
                 std::vector<size_t>& nv, std::vector<double>& pv, bool aug, rn& gen,
                 double *fnoh, double *vij, double *ui, tree::tree_cp *leaf);
  
  survivalBART() : logBART(), nsDegree(0), nsKnots(NULL), nKnots(0), nNsBasis(0), maxReDegree(0), nReCoef(0), rePowers(NULL), reTimeScale(1.0), nsCoef(NULL), reCoef(NULL), reCoefCov(NULL), reCovAux(NULL), vij(NULL), ui(NULL), reLoglik(NULL), reLoglikValid(false), cumHazard(NULL), reNcp(1), ncpScale(0.1), ncpShapeSet(false), ncpShapeInit(false), ncpAccept(0.0), ncpTried(0.0), ncpAcceptTotal(0.0), ncpTriedTotal(0.0), ncpCount(0.0), baselineType(0), pwShape(0.1), pwRate(0.1), gmStep(0.1), gmAccept(0.0) {}
  survivalBART(size_t m) : logBART(m), nsDegree(0), nsKnots(NULL), nKnots(0), nNsBasis(0), maxReDegree(0), nReCoef(0), rePowers(NULL), reTimeScale(1.0), nsCoef(NULL), reCoef(NULL), reCoefCov(NULL), reCovAux(NULL), vij(NULL), ui(NULL), reLoglik(NULL), reLoglikValid(false), cumHazard(NULL), reNcp(1), ncpScale(0.1), ncpShapeSet(false), ncpShapeInit(false), ncpAccept(0.0), ncpTried(0.0), ncpAcceptTotal(0.0), ncpTriedTotal(0.0), ncpCount(0.0), baselineType(0), pwShape(0.1), pwRate(0.1), gmStep(0.1), gmAccept(0.0) {}
  ~survivalBART() {
    if(ui) delete[] ui;
    if(vij) delete[] vij;
    if(nsCoef) delete[] nsCoef;
    if(reCoef) {
      for(int i=0; i<nSubject; i++){
        if (reCoef[i]) delete[] reCoef[i];
      }
      delete[] reCoef;
    }
    if(reCoefCov) {
      for(int i=0; i<nReCoef; i++){
        delete[] reCoefCov[i];
      }
      delete[] reCoefCov;
    }
    if(reCovAux) delete[] reCovAux;
    if(rePowers) delete[] rePowers;
    if(cumHazard) delete[] cumHazard;
    if(nsRate) delete[] nsRate;
    if(nsAccept) delete[] nsAccept;
    if(reRate) delete[] reRate;
    if(reAccept) delete[] reAccept;
    if(reLoglik) delete[] reLoglik;

    // if(reCoefCov) delete[] reCoefCov;
  }
  
  void setparam(int nsDegree, double *nsKnots, int nKnots, int maxReDegree,
                int *rePowersInput, int nReCoef, double reTimeScaleInput);
  void setparamPrior(double nsCoefPriorMean, double nsCoefPriorVar, double *reCoefPriorMean,
                     double *reCoefInitial, double *reCoefVarInitial,
                     double reHWnu, double *reHWA);
  void setu(const Rcpp::NumericVector& u);
  // Number of joint (Sigma_b, b) non-centred Metropolis proposals per sweep
  // (ncpSigmaUpdate); 0 disables the move (plain Gibbs Sigma_b | b only).
  void setNcp(int passes){
    reNcp = (passes > 0) ? passes : 0;
  }
  double getNcpAcceptRate() const {
    return (ncpTriedTotal > 0.0) ? ncpAcceptTotal / ncpTriedTotal : NA_REAL;
  }
  // Frozen proposal covariance in xi = (log L_11, L_21, ..., L_q1, log L_22, ...);
  // empty when the move is off.
  Eigen::MatrixXd getNcpProposalCov() const {
    if(reNcp == 0) return Eigen::MatrixXd(0, 0);
    Eigen::MatrixXd C = ncpScale * ncpScale * ncpShape;
    return 0.5 * (C + C.transpose());
  }
  // Baseline-hazard type. 0 = natural cubic spline on the log-hazard (default);
  // 1 = piecewise-constant ("Bayesian Breslow"): ns.knot become bin cutpoints,
  // nsCoef[k] = log lambda_k with lambda_k ~ Gamma(pwShape, pwRate) iid and a
  // conjugate Gibbs update (drawLambda). MUST be called before setparam().
  void setBaseline(int type, double shape, double rate){
    baselineType = type;
    pwShape = shape;
    pwRate = rate;
  }
  // Bin index for the piecewise baseline: bins are (knots[k], knots[k+1]] with
  // bin 0 closed on the left and the LAST bin extended to +inf (flat tail), so
  // an event at exactly a cutpoint falls in the bin it terminates (left-limit
  // convention, matching the exposure integrals).
  int pwBin(double t) const {
    const int J = nKnots - 1;
    int k = 0;
    while(k < J - 1 && t > nsKnots[k+1]) k++;
    return k;
  }
  // Integrate f over (a,b). In piecewise mode split at interior cutpoints so the
  // integrand is smooth on every piece (GK61 across a step jump is inaccurate);
  // in spline mode this is a single GK61 call, identical to the old code path.
  template <typename F>
  double integrateSeg(F&& f, double a, double b){
    boost::math::quadrature::gauss_kronrod<double, 61> integrator;
    if(baselineType != 1) return integrator.integrate(f, a, b);
    double total = 0.0, lo = a;
    for(int c = 1; c <= nKnots - 2; c++){
      double cp = nsKnots[c];
      if(cp <= lo) continue;
      if(cp >= b) break;
      total += integrator.integrate(f, lo, cp);
      lo = cp;
    }
    if(b > lo) total += integrator.integrate(f, lo, b);
    return total;
  }

  // ---- Preconditioned block MALA on the spline coefficients (ns.block) --------
  // on = true replaces the coordinatewise MALA (drawNs, one scalar step per
  // coefficient) by ONE block MALA step on the whole beta vector per sweep
  // (drawNsBlock); without this call (ns.block = FALSE) the sampler is exactly as
  // before. mSource selects the burn-in estimator of the preconditioner M:
  //   0 = empirical covariance of the beta draws ("draws"),
  //   1 = inverse of the secant (curvature) estimate of the negative Hessian of
  //       log pi(beta | rest), from the gradient pairs every step already computes
  //       ("secant", the R default).
  // nBurnIn fixes the adaptation schedule (see setNsBlock). Spline baseline only
  // (the piecewise baseline keeps its exact Gibbs drawLambda).
  // MUST be called after setparam() and setparamPrior() (needs nNsBasis, nsRate, v0).
  void setNsBlock(bool on, int mSource, size_t nBurnIn);
  bool getNsBlock() const { return nsBlock; }
  int getNsBlockMSource() const { return nsBlkMSource; }
  double getNsBlockAcceptRate() const {
    return (nsBlkTriedTotal > 0.0) ? nsBlkAccTotal / nsBlkTriedTotal : NA_REAL;
  }
  double getNsBlockH() const { return nsBlkH; }
  Eigen::MatrixXd getNsBlockM() const { return nsBlkM; }
  bool getNsBlockMAdapted() const { return nsBlkMSet; }
  int getNsBlockMUpdates() const { return nsBlkMUpdates; }
  double getNsBlockPairsDropped() const { return nsBlkPairsDropped; }
  Rcpp::IntegerVector getNsBlockSchedule() const;   // period, minStats, tail, first collected sweep
  Rcpp::NumericMatrix getNsBlockMLog() const;       // one row per M-update attempt
  
  //void setprior(double alpha, double beta, double tau, double c, double d);
  double b_spline_basis(int i, int k, double x, double *knots);
  void compute_ns_col_means(const std::vector<double>& x_train,
                            double* knots,
                            int K,
                            bool include_intercept);
  double ns_kth_basis_scalar(double x,
                             double* knots,
                             int K,
                             int degree,
                             int which,
                             bool include_intercept);
  double randomEffect(int i, double t, int subjectId);
  //void calculate_wi(double *y, double *xEvent, double *xIndex);
  void calculate_vij(double *xObsTime, double *xIndex);
  double* calCumHazard();
  void draw(double sigma, size_t nIteration, size_t nBurnIn, rn& gen); //for test
  void drawNs(rn& gen);
  void drawNsBlock(size_t nIteration, size_t nBurnIn, rn& gen);   // ns.block: one block MALA step
  void nsBlockAdapt(size_t nIteration, size_t nBurnIn);             // burn-in adaptation of h and M
  void drawLambda(rn& gen); //piecewise baseline: conjugate gamma Gibbs for lambda_k
  void drawRandomEffect(rn& gen);
  void drawRandomEffectCov(rn& gen);
  // Joint (Sigma_b, b) Metropolis move in log-Cholesky coordinates with the
  // standardized effects L^{-1}(b_i - mu_b) held fixed (non-centred move).
  double loglikelihood_re(int subjectId, double *nsCoef_arr, double *reCoef_arr);
  void ncpAdapt(size_t nIteration, size_t nBurnIn);
  void ncpSigmaUpdate(size_t nIteration, size_t nBurnIn, rn& gen);
  double integrand(double t, double *nsCoef, double *reCoef);
  double ns_integrand(double t, int i);
  double loglikelihood(double *y, double *xEvent, double *xObsTime, double *xIndex, int subjectId, double *nsCoef_temp, double *reCoef_temp);
  Eigen::VectorXd generate_proposal_eigen(Eigen::VectorXd& current_theta,double par_scale,Eigen::MatrixXd& cov_mat);
  // Metropolis-adjusted Langevin Algorithm (MALA) helpers.
  // loglikelihood_grad_ns: total log-lik (summed over subjects) and d/d nsCoef[k].
  // loglikelihood_grad_re: subject-i log-lik and d/d reCoef[i,:].
  double loglikelihood_grad_ns(int k, double *nsCoef_arr, double **reCoef_arr, double& grad_out);
  // total log-lik and ALL nNsBasis gradient components at one spline state (block
  // MALA): the drawNs memo / one fused pass with the node cache, else one uncached
  // pass whose per-subject arithmetic is that of loglikelihood_grad_ns (bit-identical).
  double loglikelihood_grad_ns_all(double *nsCoef_arr, double **reCoef_arr, double *grad_all);
  double loglikelihood_grad_re(int subjectId, double *nsCoef_arr, double *reCoef_arr, Eigen::VectorXd& grad_out);
  // MALA updates (replace random-walk Metropolis for ns and re).
  void mala_ns(int k, Eigen::VectorXd& nsCoef_temp, double **reCoef_temp_array,
               double step_sigma2, double prior_mean, double prior_var, rn& gen);
  void mala_re(int subjectId, double *nsCoef_arr, Eigen::MatrixXd& reCoef_temp,
               const Eigen::MatrixXd& proposal_covar, const Eigen::VectorXd& prior_mean,
               const Eigen::LLT<Eigen::MatrixXd>& llt_prior, rn& gen);
  
  double* getNsCoef();
  double** getReCoef();
  double** getReCoefCov();
  double* getReCovAux();   // Huang-Wand auxiliary scales a_k (length nReCoef)
  double getBARTMean();
  double* getSplineY();
  double* getNsMean();
  
protected:
  //parameters for ncs and random effects
  int nsDegree;
  double *nsKnots; //knots for ncs
  int nNsBasis; // number of ncs basis
  int nKnots; //number of knots
  int maxReDegree; //maximum degree of random effects
  int nReCoef; //number of random-effect coefficients
  int *rePowers; //powers included in the random-effect design
  double reTimeScale; //time standardization for the RE design: z(t) uses (t/reTimeScale)^power
  //coefficients for ncs and random effects
  double *nsCoef;
  double *splineY;
  double **reCoef;
  double **reCoefCov;
  double *reCovAux;   // Huang-Wand auxiliary scales a_k (length nReCoef); unused for plain IW
  double *ui; // event indicator
  double *vij; // integral for exp(ns + random effect)
  double *reLoglik;   // cached subject log-likelihood ell_i(b_i) at the current b_i
                      // (written by mala_re and by accepted ncpSigmaUpdate proposals)
  bool reLoglikValid; // set after drawRandomEffect (and by the ncpSigmaUpdate fallback);
                      // cleared at the start of draw and after the group move
  double *reAccept; // RWM accept times for random effect
  double *reRate; // RWM proposal rate for random effect
  double *nsAccept; // RWM accept times for ncs coefficients
  double *nsRate; // RWM proposal rate for ncs coefficients
  double **nsCoefAll; // store all ncs coefficients for each iteration
  double *cumHazard; // store cumulative hazard function for each subject
  double bartMean; // store the mean of BART predictions
  double **weights; // store weights of each people in each time region
  double *ns_mean; // store weighted mean of ns basis
  double bart_weighted_mean;
  double *mean_t;
  double cf_step;   // RW proposal sd for the BART-level (Move 1) recentering shift
  double cf_accept; // accept counter for cf_step adaptation (per burn-in batch)
  // Joint non-centred (Sigma_b, b) move. Proposal xi* = xi + zeta,
  // zeta ~ N(0, ncpScale^2 * ncpShape), in log-Cholesky coordinates xi of Sigma_b.
  // ncpScale and ncpShape are adapted during burn-in only, then frozen.
  int reNcp;               // proposals per sweep (0 = off)
  double ncpScale;         // global proposal scale
  Eigen::MatrixXd ncpShape;// proposal shape (identity until the empirical covariance is available)
  bool ncpShapeSet;        // true once ncpShape is the empirical burn-in covariance
  bool ncpShapeInit;       // true once ncpShape has been scaled to the starting Sigma_b
  double ncpAccept;        // accepts in the current burn-in adaptation batch
  double ncpTried;         // proposals in the current burn-in adaptation batch
  double ncpAcceptTotal;   // accepts after burn-in (reported)
  double ncpTriedTotal;    // proposals after burn-in (reported)
  double ncpCount;         // burn-in draws of xi accumulated for the empirical covariance
  Eigen::VectorXd ncpMean; // running mean of xi during burn-in (Welford)
  Eigen::MatrixXd ncpM2;   // running sum of squared deviations of xi during burn-in
  // ---- GK61 node cache (spline baseline) -----------------------------------
  // Every exposure integral of the sampler, int_{t_j}^{t_{j+1}} g(t) exp(A_i(t)) dt
  // with g = 1, a centred spline basis or an RE design column, is one boost GK61
  // integral that does not subdivide on real data. gkBuild() records boost's 61
  // top-level nodes on every exposure interval once and caches the centred basis
  // B_l(t) - ns_mean[l] and the RE design z_m(t) there; a likelihood pass then
  // evaluates exp(A_i) once per node and forms the value and every gradient
  // integral from it with gk61::fused(), a bit-exact replica of boost's top level.
  // Any component boost would subdivide, and every uncached interval, goes through
  // integrateSeg() as before, so all results are bit-identical to the uncached code
  // (gkSelfCheck() verifies this at the first sweep and switches the cache off on
  // any mismatch). Designs the replica has not been validated for keep the
  // uncached path (gkSupported()); TVSB_GK_OFF=1 forces it. The cache is only
  // built when its estimated size fits the memory cap (cacheCapBytes(),
  // TVSB_GK_MAXMB, default 2048 MB, shared with the leaf cache below) and the
  // build does not contract floating-point multiply-adds
  // (gk61::contractionDetected(): the package's -ffp-contract=off was overridden).
  bool gkOn = false;              // node cache in use
  double gkBytes = 0.0;           // estimated bytes held by the node + event caches (0 when off)
  static double cacheCapBytes();  // TVSB_GK_MAXMB in bytes (default 2048 MB)
  bool gkChecked = false;         // gkSelfCheck() has run
  int gkNiv = 0;                  // number of cached intervals
  std::vector<int> gkRowIv;       // row j (left end of an exposure interval) -> cached interval id,
                                  // -1 = uncached (boost), -2 = zero length (boost returns exactly 0)
  std::vector<double> gkB;        // [(iv*61 + q)*nNsBasis + l] = B_l(t_q) - ns_mean[l]
  std::vector<double> gkZ;        // [(iv*61 + q)*nReCoef + m]  = pow(t_q / reTimeScale, rePowers[m])
  std::vector<double> gkScale;    // (b - a) / 2 of each cached interval, as boost computes it
  // drawNs memo. mala_ns evaluates, for coordinate k, the current state (always the
  // state it evaluated last: the k-1 proposal if accepted, else the k-1 current
  // state) and one proposal. Each new spline state costs ONE fused pass that yields
  // the total log-lik, d/d nsCoef[k] for every k and the per-interval value
  // integrals; the two most recent states are kept. Valid only inside one drawNs()
  // (b_i and allfit fixed): drawNs() invalidates it on entry, calculate_vij() takes
  // the final state's value integrals as vij and invalidates it on exit.
  std::vector<double> gkMemoCoef[2], gkMemoGrad[2], gkMemoIv[2];
  double gkMemoLL[2] = {0.0, 0.0};
  bool gkMemoValid[2] = {false, false};
  int gkMemoLastUsed = 0;
  // ---- Preconditioned block MALA on beta = nsCoef (ns.block; see drawNsBlock) ----
  // Proposal beta* = beta + (h^2/2) M g(beta) + h L eps, M = L L' (Cholesky), eps ~ N(0, I_K),
  // g = grad log pi(beta | trees, b). h and M adapt during burn-in only, then stay frozen.
  bool nsBlock = false;          // block MALA replaces the coordinatewise drawNs
  int nsBlkMSource = 1;          // 0 = covariance of the beta draws, 1 = inverse secant Hessian
  double nsBlkH = 1.0;           // global step h
  // absolute cap on h: twice the MALA stability limit h = 2 for a preconditioner equal
  // to the target covariance (what the secant M estimates), about 3x the optimal
  // h = 1.65 K^(-1/6) (1.31 for K = 4); was 100
  static constexpr double nsBlkHMax = 4.0;
  Eigen::MatrixXd nsBlkM, nsBlkL;// preconditioner M (K x K SPD) and its lower Cholesky factor
  bool nsBlkMSet = false;        // M has been replaced by a burn-in estimate at least once
  int nsBlkMUpdates = 0;         // number of M replacements
  double nsBlkAcc = 0.0, nsBlkTried = 0.0;            // current 100-sweep burn-in batch
  double nsBlkAccTotal = 0.0, nsBlkTriedTotal = 0.0;  // after burn-in (reported)
  // Schedule (setNsBlock): M is replaced every nsBlkPeriod burn-in sweeps once at least
  // nsBlkMinStats statistics are available and at least nsBlkTail burn-in sweeps remain
  // (500 / 500 / 500 for nskip >= 1500, scaled down below; period 0 = M never replaced).
  int nsBlkPeriod = 500, nsBlkMinStats = 500, nsBlkTail = 500;
  size_t nsBlkBurn = 0;
  // burn-in statistics, from the second quarter of burn-in on:
  double nsBlkCount = 0.0;       // mSource 0: beta draws (Welford mean / M2)
  Eigen::VectorXd nsBlkMean;
  Eigen::MatrixXd nsBlkM2;
  double nsBlkPairs = 0.0;       // mSource 1: committed secant pairs x = beta* - beta, y = g(beta) - g(beta*)
  Eigen::MatrixXd nsBlkSxx, nsBlkSyx;                  // sum x x', sum y x' over committed batches
  double nsBlkPairsB = 0.0;      // pairs of the running 100-sweep batch, committed at its end
  Eigen::MatrixXd nsBlkSxxB, nsBlkSyxB;                // unless its acceptance was < 0.1
  double nsBlkPairsDropped = 0.0;// secant pairs dropped with low-acceptance batches
  std::vector<double> nsBlkLog;  // M-update attempts (nsBlockAdapt), 11 values each
  // step workspace
  Eigen::VectorXd nsBlkB0, nsBlkB1, nsBlkG0, nsBlkG1, nsBlkEps, nsBlkDF, nsBlkDB, nsBlkR;
  bool gkSupported() const;
  void gkBuild();
  void gkRelease();
  void gkSelfCheck();
  inline double gkExp(int iv, int q, const double* ns, const double* re) const;
  template <class Ex, class Fac, class Ref>
  void gkIntegrals(int j, int nc, Ex&& ex, Fac&& fac, Ref&& ref, double* out);
  double gkValue(int j, double* ns, double* re);
  // ---- RE and ncp moves: shared spline part of A_i(t) -----------------------
  // nsCoef is fixed from drawRandomEffect through ncpSigmaUpdate, so every b_i
  // state of both moves shares sum_l nsCoef[l] (B_l(t) - ns_mean[l]) at every
  // node: gkS[iv*61 + q] holds that partial sum, accumulated exactly as gkExp()
  // does before the RE terms, for the spline state gkSCoef (compared bit for bit
  // before use, so a stale gkS is never read).
  std::vector<double> gkS, gkSCoef;
  bool gkSValid = false;
  void gkFillS(const double* ns);
  inline bool gkSHolds(const double* ns) const;
  inline double gkExpS(int iv, int q, const double* re) const;
  double gkValueS(int j, double* ns, double* re);
  double gkGradNsAll(double* nsCoef_arr, double** reCoef_arr, double* grad_all, double* ivVal);
  // ---- per-subject event-time terms (with the node cache) ------------------
  // evD[i*nNsBasis + l] = B_l(y_i) - ns_mean[l] and evZ[i*nReCoef + m] =
  // pow(y_i / reTimeScale, rePowers[m]) are data-fixed; evLogF[i] =
  // log(pow(allfit[last row of i - 1], u_i)) depends on allfit and is recomputed
  // on first use after every change of allfit (evLogFValid is cleared wherever
  // allfit is written). They replace the per-call ns_kth_basis_scalar / pow / log
  // evaluations at y_i with the same values; the cached sums are formed with the
  // multiply-add expressions of the reference prologues (plain arithmetic, like
  // gkExp()), and they are used only together with the node cache, whose
  // self-check covers them.
  bool evOn = false;
  bool evLogFValid = false;
  std::vector<double> evD, evZ, evLogF;
  void evBuild();
  inline double evLogFAt(int i);
  inline double evSplineAt(int i, const double* ns) const;
  inline double evReAt(int i, double A, const double* re) const;
  // mala_re workspace, reused across subjects (Sigma_b is factored once per sweep
  // in drawRandomEffect); same Eigen kernels as fresh temporaries, no allocation.
  Eigen::LLT<Eigen::MatrixXd> mrLltPrior, mrLltProp;
  Eigen::MatrixXd mrPriorCov, mrPropCov;
  Eigen::VectorXd mrPriorMean, mrThetaCurr, mrThetaProp, mrGradLLCurr, mrGradLLProp, mrDiff, mrZ,
                  mrGradPrior, mrGradCurr, mrGradProp, mrDriftFwd, mrDriftBack, mrNoise,
                  mrDiffFwd, mrDiffBack;
  // ---- BART backfit: per-row bottom-node cache ----------------------------
  // leafCache[j*n + k] = bottom node of tree j that row k falls in, i.e.
  // t[j].bn(x_k). Tree structure changes only in bd(), which updates the cache on
  // an accepted birth or death (Move 1 and drmu only reset leaf values), so the
  // backfit reads fits and sufficient statistics from it instead of descending
  // every tree for every row four times per sweep. Built at the first sweep if
  // m*n pointers fit the memory cap left by the node cache (cacheCapBytes()) and
  // the build does not contract floating-point expressions
  // (gk61::contractionDetected()); otherwise leafOn = false and the backfit keeps
  // the reference tree-descending path.
  std::vector<tree::tree_cp> leafCache;
  bool leafValid = false;
  bool leafOn = true;
  int baselineType;   // 0 = natural cubic spline (default), 1 = piecewise-constant
  double pwShape;     // Gamma(shape, rate) prior on each piecewise height lambda_k
  double pwRate;
  double gmStep;      // RW step for the piecewise group move (b-intercept vs log-lambda level)
  double gmAccept;    // accept counter for gmStep adaptation (per burn-in batch)
};


void survivalBART::setu(const Rcpp::NumericVector& u) {
  // Free existing memory if already allocated
  if (ui) delete[] ui;
  
  // Allocate new memory and copy data
  size_t n = u.size();
  ui = new double[n];
  std::memcpy(ui, &u[0], n * sizeof(double));
  this->xEvent = ui;
}


double survivalBART::ns_integrand(double t, int i){
  return ns_kth_basis_scalar(t, nsKnots, nKnots, 3, i, true);
}

void survivalBART::setparam(int nsDegree, double *nsKnots, int nKnots, int maxReDegree,
                            int *rePowersInput, int nReCoef, double reTimeScaleInput){
  this->nsDegree = nsDegree;
  this->nsKnots = nsKnots;
  this->nKnots = nKnots;
  this->maxReDegree = maxReDegree;
  this->nReCoef = nReCoef;
  if(!(reTimeScaleInput > 0.0)) Rcpp::stop("setparam: reTimeScale must be > 0");
  this->reTimeScale = reTimeScaleInput;
  printf("reTimeScale: %f\n", this->reTimeScale);
  if(this->rePowers) delete[] this->rePowers;
  this->rePowers = new int[nReCoef];
  for(int i=0; i<nReCoef; i++){
    this->rePowers[i] = rePowersInput[i];
  }
  // Natural cubic spline with nKnots basis functions (intercept + linear + cubic
  // truncated powers). The extra t^2 basis was REMOVED: within the observed range
  // the cubic spline already absorbs the slope-variance-induced curvature, and
  // dropping it removes the most collinear (least stable) baseline coefficient.
  // (The which==K -> x*x branch in ns_kth_basis_scalar is now unreachable; restore
  //  nKnots+1 here to re-enable the explicit t^2 basis.)
  // Piecewise mode: the knots are bin CUTPOINTS, so there are nKnots-1 heights.
  if(baselineType == 1){
    if(nKnots < 2) Rcpp::stop("piecewise baseline needs >= 2 cutpoints (ns.knot)");
    this->nNsBasis = nKnots - 1;
  } else
    this->nNsBasis = nKnots;
  
  //print
  printf("nsDegree: %d\n", nsDegree);
  printf("Number of knots: %d\n", nKnots);
  for(int i=0; i<nKnots; i++){
    printf("%f ", nsKnots[i]);
  }
  printf("nNsBasis: %d\n", nNsBasis);
  printf("maxReDegree: %d\n", maxReDegree);
  printf("rePowers: ");
  for(int i=0; i<nReCoef; i++) printf("%d ", this->rePowers[i]);
  printf("\n");
  
  //initialize nsCoef and reCoef
  //initialize nsCoef
  if(nsCoef) delete[] nsCoef;
  nsCoef = new double[nNsBasis];
  for(int i=0; i<nNsBasis; i++){
    nsCoef[i] = 0;
  }
  
  //initialize reCoef
  if(reCoef) {
    for(int i=0; i<nSubject; i++){
      if (reCoef[i]) delete[] reCoef[i];
    }
    delete[] reCoef;
  }
  reCoef = new double*[nSubject];
  for(int i=0; i<nSubject; i++){
    reCoef[i] = new double[nReCoef];
  }
  for(int i=0; i<nSubject; i++){
    for(int j=0; j<nReCoef; j++){
      reCoef[i][j] = 0;
    }
  }
  
  //initialize cumHazard
  if(cumHazard) delete[] cumHazard;
  cumHazard = new double[n];
  
  if(vij) delete[] vij;
  vij = new double[n]; // beacuse 0,t1,t2,y should all be included in the data
  
  reAccept = new double[nSubject]; //initialize acceptance rate for random effect
  for(int i=0; i<nSubject; i++){
    reAccept[i] = 0.0;
  }
  nsAccept = new double[nNsBasis]; //initialize acceptance rate for ncs coefficients
  for(int i=0; i<nNsBasis; i++){
    nsAccept[i] = 0.0;
  }
  
  //initialize rate (interpreted as std-dev multiplier for the proposal)
  reRate = new double[nSubject];
  for(int i=0; i<nSubject; i++){
    reRate[i] = 0.1;
  }
  nsRate = new double[nNsBasis];
  for(int i=0; i<nNsBasis; i++){
    nsRate[i] = 0.1;
  }
  // BART-level (Move 1) random-walk step and its acceptance counter.
  cf_step = 0.1;
  cf_accept = 0.0;
  // Piecewise group move (b-intercept vs log-lambda level) RW step + counter.
  gmStep = 0.1;
  gmAccept = 0.0;
  // Joint non-centred (Sigma_b, b) move: dimension of the log-Cholesky vector.
  {
    const int dxi = nReCoef * (nReCoef + 1) / 2;
    ncpScale = 0.1;
    ncpShape = Eigen::MatrixXd::Identity(dxi, dxi);
    ncpShapeSet = false;
    ncpShapeInit = false;
    ncpAccept = 0.0;
    ncpTried = 0.0;
    ncpAcceptTotal = 0.0;
    ncpTriedTotal = 0.0;
    ncpCount = 0.0;
    ncpMean = Eigen::VectorXd::Zero(dxi);
    ncpM2 = Eigen::MatrixXd::Zero(dxi, dxi);
  }
  if(reLoglik) delete[] reLoglik;
  reLoglik = new double[nSubject];
  for(int i=0; i<nSubject; i++) reLoglik[i] = 0.0;
  reLoglikValid = false;

  splineY = new double[nSubject];
  for(int i=0; i<nSubject; i++){
    splineY[i] = 0.0;
  }
  
  //calculates weights
  weights = new double*[nSubject];
  for(int i=0; i<nSubject; i++){
    int intervalNum = xIndex[i+1] - xIndex[i];
    weights[i] = new double[intervalNum];
    for(int j=0; j<intervalNum; j++){
      weights[i][j] = 0.0;
    }
  }
  
  for(int i=0; i<nSubject; i++){
    int start_time_index = xIndex[i] + 1;
    int end_time_index = xIndex[i+1];
    double y_i = y[i];
    for(int j=start_time_index; j<end_time_index; j++){
      weights[i][j - start_time_index] = std::min(y_i, xObsTime[j+1]) - xObsTime[j];
    }
  }
  
  //calculate weighted mean of ns basis
  ns_mean = new double[nNsBasis];
  ns_mean[0] = 0.0; //the first basis is intercept
  if(baselineType == 1){
    // Piecewise mode: no centering. The level convention differs (there is no
    // separate intercept basis; the log-lambda level is identified against the
    // BART scale by Move 1 and the Gamma prior), so every ns_mean is 0 and the
    // "- ns_mean[k]" subtractions everywhere become no-ops.
    for(int ns_index = 0; ns_index < nNsBasis; ns_index++) ns_mean[ns_index] = 0.0;
  } else {
    boost::math::quadrature::gauss_kronrod<double, 61> integrator;
    for(int ns_index = 1; ns_index<nNsBasis; ns_index++) {
      double ns_weighted_sum = 0.0;
      double ns_total_weight = 0.0;
      auto f = [&, ns_index](double t) { return ns_integrand(t, ns_index); };
      for(int i=0; i<nSubject; i++){
        int start_time_index = xIndex[i] + 1;
        int end_time_index = xIndex[i+1];
        for(int j=start_time_index; j<end_time_index; j++){
          ns_weighted_sum += integrator.integrate(f, xObsTime[j], xObsTime[j+1]);
          ns_total_weight += weights[i][j - start_time_index];
        }
      }
      ns_mean[ns_index] = ns_weighted_sum / ns_total_weight;
    }
  }
  
  for(int i=0; i<nNsBasis; i++){
    printf("ns_mean[%d]: %f\n", i, ns_mean[i]);
  }
  
  //calculate mean t
  mean_t = new double[nReCoef];
  double t_weighted_sum = 0.0;
  double t_total_weight = 0.0;
  for(int i=0; i<nSubject; i++){
    int start_time_index = xIndex[i] + 1;
    int end_time_index = xIndex[i+1];
    for(int j=start_time_index; j<end_time_index; j++){
      double a = xObsTime[j];
      double b = xObsTime[j+1];
      t_weighted_sum += 0.5 * (a + b) * (b - a);
      t_total_weight += weights[i][j - start_time_index];
    }
  }
  double weighted_mean_t = t_weighted_sum / t_total_weight;
  for(int k=0; k<nReCoef; k++){
    mean_t[k] = (rePowers[k] == 0) ? 0.0 : weighted_mean_t;
  }

  gkBuild();   // node cache for the exposure integrals (spline baseline, supported designs)
}

// Designs the GK61 replica is validated for: at least one and at most three RE
// columns (the cached kernels index the RE design of every node and event time,
// and q <= 3 is what the end-to-end checks covered), the fused pass holds at most
// gk61::kMaxComp components, and only the random-effect powers the suite uses are
// admitted; everything else keeps the uncached boost path.
bool survivalBART::gkSupported() const {
  if(!GK61_REPLICA_SUPPORTED) return false;
  if(baselineType != 0) return false;
  if(std::getenv("TVSB_GK_OFF") != nullptr) return false;
  if(nNsBasis < 1 || nNsBasis + 1 > gk61::kMaxComp) return false;
  if(nReCoef < 1 || nReCoef > 3) return false;
  for(int m=0; m<nReCoef; m++) if(rePowers[m] < 0 || rePowers[m] > 3) return false;
  return true;
}

// Memory cap of the node cache and the leaf cache together, in bytes:
// TVSB_GK_MAXMB (MB; default 2048; 0 turns both caches off).
double survivalBART::cacheCapBytes(){
  double mb = 2048.0;
  const char* e = std::getenv("TVSB_GK_MAXMB");
  if(e != nullptr && *e != '\0'){
    char* end = nullptr;
    const double v = std::strtod(e, &end);
    if(end != e && std::isfinite(v) && v >= 0.0) mb = v;
  }
  return mb * 1048576.0;
}

void survivalBART::gkRelease(){
  gkOn = false;
  gkBytes = 0.0;
  evOn = false;
  evLogFValid = false;
  std::vector<double>().swap(evD);
  std::vector<double>().swap(evZ);
  std::vector<double>().swap(evLogF);
  gkNiv = 0;
  std::vector<int>().swap(gkRowIv);
  std::vector<double>().swap(gkB);
  std::vector<double>().swap(gkZ);
  std::vector<double>().swap(gkScale);
  for(int s=0; s<2; s++){
    std::vector<double>().swap(gkMemoCoef[s]);
    std::vector<double>().swap(gkMemoGrad[s]);
    std::vector<double>().swap(gkMemoIv[s]);
    gkMemoValid[s] = false;
  }
  std::vector<double>().swap(gkS);
  std::vector<double>().swap(gkSCoef);
  gkSValid = false;
}

// Record boost's top-level GK61 nodes on every exposure interval and cache the
// centred spline basis and the RE design there (once per fit, after ns_mean).
void survivalBART::gkBuild(){
  gkRelease();
  gkChecked = false;
  if(!gkSupported()) return;
  if(gk61::contractionDetected()){
    // the user's flags re-enabled contraction (-ffp-contract=on/fast): the replica
    // need not round like boost and the reference code, so keep the reference path
    printf("GK61 node cache: off (this build contracts floating-point multiply-adds; -ffp-contract is not off) -- using the uncached quadrature\n");
    return;
  }
  // Size estimate first: every interval that passes the cheap tests below may be
  // cached (recordNodes rarely rejects one), costing kNodes * (nNsBasis + nReCoef
  // + 1) doubles (gkB, gkZ, gkS) plus its scale and two memo slots; the row map
  // and the per-subject event caches come on top.
  size_t nCand = 0;
  for(int i=0; i<nSubject; i++)
    for(int j=(int)xIndex[i] + 1; j<(int)xIndex[i+1]; j++){
      const double a = xObsTime[j], b = xObsTime[j+1];
      if(a != b && b > a && std::isfinite(a) && std::isfinite(b) && !(b - a < 1e-6)) nCand++;
    }
  const double perIv = 8.0 * ((double)gk61::kNodes * (nNsBasis + nReCoef + 1) + 3.0);
  const double est = perIv * (double)nCand + 4.0 * (double)n
                   + 8.0 * (double)nSubject * (nNsBasis + nReCoef + 1);
  const double cap = cacheCapBytes();
  if(est > cap){
    printf("GK61 node cache: off (estimated %.1f MB for %zu intervals exceeds TVSB_GK_MAXMB = %.0f MB) -- using the uncached quadrature\n",
           est / 1048576.0, nCand, cap / 1048576.0);
    return;
  }
  gkRowIv.assign(n, -1);
  gkScale.reserve(nCand);
  gkB.reserve(nCand * gk61::kNodes * (size_t)nNsBasis);
  gkZ.reserve(nCand * gk61::kNodes * (size_t)nReCoef);
  double T[gk61::kNodes];
  for(int i=0; i<nSubject; i++){
    for(int j=(int)xIndex[i] + 1; j<(int)xIndex[i+1]; j++){
      const double a = xObsTime[j], b = xObsTime[j+1];
      if(a == b) { gkRowIv[j] = -2; continue; }
      if(!(b > a) || !std::isfinite(a) || !std::isfinite(b)) continue;
      // Below ~6e-8 boost's top-level test always subdivides (it compares the
      // unscaled [-1,1] error with the scaled tolerance): leave such intervals to boost.
      if(b - a < 1e-6) continue;
      if(!gk61::recordNodes(a, b, T)) continue;
      const int iv = gkNiv++;
      gkRowIv[j] = iv;
      gkScale.push_back((b - a) / 2);
      for(int q=0; q<gk61::kNodes; q++){
        for(int l=0; l<nNsBasis; l++)
          gkB.push_back(ns_kth_basis_scalar(T[q], nsKnots, nKnots, 3, l, true) - ns_mean[l]);
        for(int m=0; m<nReCoef; m++)
          gkZ.push_back(pow(T[q] / reTimeScale, rePowers[m]));
      }
    }
  }
  for(int s=0; s<2; s++){
    gkMemoCoef[s].assign(nNsBasis, 0.0);
    gkMemoGrad[s].assign(nNsBasis, 0.0);
    gkMemoIv[s].assign(gkNiv, 0.0);
    gkMemoValid[s] = false;
  }
  gkS.assign((size_t)gkNiv * gk61::kNodes, 0.0);
  gkSCoef.assign(nNsBasis, 0.0);
  gkSValid = false;
  gkOn = true;
  evBuild();
  gkBytes = 8.0 * (double)(gkB.capacity() + gkZ.capacity() + gkS.size() + gkScale.capacity()
                           + 2 * (size_t)gkNiv + evD.size() + evZ.size() + evLogF.size())
          + 4.0 * (double)gkRowIv.size();
}

void survivalBART::evBuild(){
  evD.assign((size_t)nSubject * nNsBasis, 0.0);
  evZ.assign((size_t)nSubject * nReCoef, 0.0);
  evLogF.assign(nSubject, 0.0);
  for(int i=0; i<nSubject; i++){
    const double ty = y[i];
    for(int l=0; l<nNsBasis; l++)
      evD[(size_t)i * nNsBasis + l] = ns_kth_basis_scalar(ty, nsKnots, nKnots, 3, l, true) - ns_mean[l];
    for(int m=0; m<nReCoef; m++)
      evZ[(size_t)i * nReCoef + m] = pow(ty / reTimeScale, rePowers[m]);
  }
  evLogFValid = false;
  evOn = true;
}

// log(pow(allfit[last_i - 1], u_i)) of subject i, refreshed for all subjects on
// first use after allfit changed.
inline double survivalBART::evLogFAt(int i){
  if(!evLogFValid){
    for(int s=0; s<nSubject; s++){
      const int e = (int)xIndex[s+1];
      evLogF[s] = log(pow(allfit[e - 1], xEvent[e]));
    }
    evLogFValid = true;
  }
  return evLogF[i];
}

// sum_l ns[l] * (B_l(y_i) - ns_mean[l]) and A + sum_m re[m] * z_m(y_i) from the
// cache: the multiply-add chains of the reference prologues
//   A_at_y = 0.0; A_at_y += nsCoef[k] * (B_k(ty) - ns_mean[k]); ...
//   A_at_y += reCoef[m] * pow(ty / reTimeScale, rePowers[m]); ...
// in the same expression form and order (the package builds with
// -ffp-contract=off, so neither side is contracted).
inline double survivalBART::evSplineAt(int i, const double* ns) const {
  const double* bD = &evD[(size_t)i * nNsBasis];
  double A = 0.0;
  for(int l=0; l<nNsBasis; l++) A += ns[l] * bD[l];
  return A;
}
inline double survivalBART::evReAt(int i, double A, const double* re) const {
  const double* bZ = &evZ[(size_t)i * nReCoef];
  for(int m=0; m<nReCoef; m++) A += re[m] * bZ[m];
  return A;
}

// exp(A_i(t_q)) at node q of cached interval iv: the multiply-add chain of
// integrand() (integrand = 0; spline terms, then RE terms), same form and order.
inline double survivalBART::gkExp(int iv, int q, const double* ns, const double* re) const {
  const double* B = &gkB[((size_t)iv * gk61::kNodes + q) * nNsBasis];
  const double* Z = &gkZ[((size_t)iv * gk61::kNodes + q) * nReCoef];
  double a = 0;
  for(int l=0; l<nNsBasis; l++) a += ns[l] * B[l];
  for(int m=0; m<nReCoef; m++) a += re[m] * Z[m];
  return exp(a);
}

// Fill gkS with the spline part of A(t_q) at state ns for every cached node.
void survivalBART::gkFillS(const double* ns){
  for(int iv=0; iv<gkNiv; iv++)
    for(int q=0; q<gk61::kNodes; q++){
      const double* B = &gkB[((size_t)iv * gk61::kNodes + q) * nNsBasis];
      double a = 0;
      for(int l=0; l<nNsBasis; l++) a += ns[l] * B[l];   // = the spline chain of gkExp()
      gkS[(size_t)iv * gk61::kNodes + q] = a;
    }
  std::memcpy(gkSCoef.data(), ns, sizeof(double) * nNsBasis);
  gkSValid = true;
}

inline bool survivalBART::gkSHolds(const double* ns) const {
  return gkSValid && std::memcmp(gkSCoef.data(), ns, sizeof(double) * nNsBasis) == 0;
}

// exp(A(t_q)) from the shared spline part gkS: the RE chain of gkExp().
inline double survivalBART::gkExpS(int iv, int q, const double* re) const {
  const size_t r = (size_t)iv * gk61::kNodes + q;
  const double* Z = &gkZ[r * nReCoef];
  double a = gkS[r];
  for(int m=0; m<nReCoef; m++) a += re[m] * Z[m];
  return exp(a);
}

// Integrals over the exposure interval that starts at row j of nc integrands
// g_c(t) exp(A(t)). ex(iv, q) gives exp(A(t_q)) at cached node q (gkExp, or
// gkExpS for the RE/ncp states); fac(iv, q, e, o) writes the nc cached-node
// values given e = exp(A(t_q)); ref(c, t) is the reference integrand of component
// c, used through integrateSeg() for uncached intervals and for every component
// boost would subdivide. out[c] is bit-identical to integrateSeg(ref_c, t_j, t_{j+1}).
template <class Ex, class Fac, class Ref>
void survivalBART::gkIntegrals(int j, int nc, Ex&& ex, Fac&& fac, Ref&& ref, double* out){
  const int iv = gkRowIv[j];
  if(iv == -2) { for(int c=0; c<nc; c++) out[c] = 0.0; return; }
  bool ok[gk61::kMaxComp];
  if(iv >= 0) {
    gk61::fused(gkScale[iv], nc, [&](int q, double* o) { fac(iv, q, ex(iv, q), o); }, out, ok);
  } else {
    for(int c=0; c<nc; c++) ok[c] = false;
  }
  for(int c=0; c<nc; c++) {
    if(ok[c]) continue;
    auto f = [&, c](double t) { return ref(c, t); };
    out[c] = integrateSeg(f, xObsTime[j], xObsTime[j+1]);
  }
}

// int_{t_j}^{t_{j+1}} exp(A(t)) dt at (ns, re) (the value integral / vij).
double survivalBART::gkValue(int j, double* ns, double* re){
  double out[1];
  gkIntegrals(j, 1, [this, ns, re](int iv, int q) { return gkExp(iv, q, ns, re); },
              [](int, int, double e, double* o) { o[0] = e; },
              [&](int, double t) { return integrand(t, ns, re); }, out);
  return out[0];
}

// gkValue for a spline state held by gkS (gkSHolds(ns)).
double survivalBART::gkValueS(int j, double* ns, double* re){
  double out[1];
  gkIntegrals(j, 1, [this, re](int iv, int q) { return gkExpS(iv, q, re); },
              [](int, int, double e, double* o) { o[0] = e; },
              [&](int, double t) { return integrand(t, ns, re); }, out);
  return out[0];
}

// Total log-likelihood and d/d nsCoef[k] for EVERY k at one spline state, one
// fused pass per interval (value + nNsBasis gradient integrands share exp(A)).
// Per-subject expressions and summation order are those of loglikelihood_grad_ns;
// ivVal[iv] receives each cached interval's value integral (the vij of this state).
double survivalBART::gkGradNsAll(double* nsCoef_arr, double** reCoef_arr, double* grad_all, double* ivVal){
  const int NB = nNsBasis;
  const int nc = 1 + NB;
  double loglik_total = 0.0;
  double grad_total[gk61::kMaxComp];
  double grad_i[gk61::kMaxComp];
  double out[gk61::kMaxComp];
  for(int k=0; k<NB; k++) grad_total[k] = 0.0;

  for(int i=0; i<nSubject; i++) {
    int start_time_index = (int)xIndex[i] + 1;
    int end_time_index = (int)xIndex[i+1];

    const double *bart_prediction = &allfit[start_time_index];   // BART factor of each interval

    double ui_event = xEvent[(int)xIndex[i+1]];

    // A_i(y_i)
    double A_at_y = evReAt(i, evSplineAt(i, nsCoef_arr), reCoef_arr[i]);
    double wi_event = ui_event * A_at_y;
    wi_event += evLogFAt(i);

    double loglik_i = wi_event;
    const double* bD = &evD[(size_t)i * NB];
    for(int k=0; k<NB; k++) grad_i[k] = ui_event * bD[k];

    double *re_i = reCoef_arr[i];
    for(int j=start_time_index; j<end_time_index; j++) {
      gkIntegrals(j, nc, [this, nsCoef_arr, re_i](int iv, int q) { return gkExp(iv, q, nsCoef_arr, re_i); },
                  [&](int iv, int q, double e, double* o) {
                    const double* B = &gkB[((size_t)iv * gk61::kNodes + q) * NB];
                    o[0] = e;
                    for(int l=0; l<NB; l++) o[1 + l] = B[l] * e;
                  },
                  [&](int c, double t) {
                    if(c == 0) return integrand(t, nsCoef_arr, re_i);
                    return (ns_kth_basis_scalar(t, nsKnots, nKnots, 3, c - 1, true) - ns_mean[c - 1])
                    * integrand(t, nsCoef_arr, re_i);
                  }, out);
      const int iv = gkRowIv[j];
      if(iv >= 0) ivVal[iv] = out[0];
      loglik_i -= out[0] * bart_prediction[j - start_time_index];
      for(int k=0; k<NB; k++) grad_i[k] -= out[1 + k] * bart_prediction[j - start_time_index];
    }

    loglik_total += loglik_i;
    for(int k=0; k<NB; k++) grad_total[k] += grad_i[k];
  }

  for(int k=0; k<NB; k++) grad_all[k] = grad_total[k];
  return loglik_total;
}

// Bit-for-bit comparison of the cached likelihood kernels with the uncached boost
// code at three deterministic synthetic states of increasing size (A(t) and the
// b_i of order 0.3, 1 and 3), with synthetic BART factors. No RNG is used and every
// member the likelihood functions write (splineY) or read (allfit) is restored.
// Any mismatch switches the cache off for the whole fit. Runs once, at the first
// sweep, when the data, event indicators and allfit are set.
void survivalBART::gkSelfCheck(){
  gkChecked = true;
  if(!gkOn) return;
  std::vector<double> allfitSave(allfit, allfit + n);
  std::vector<double> splineYSave(splineY, splineY + nSubject);
  // synthetic BART factors in [0.5, 2) so the multiply-adds with allfit are exercised
  for(size_t j=0; j<n; j++) allfit[j] = 0.5 + 1.5 * std::fmod(0.6180339887498949 * (double)(j + 1), 1.0);
  evLogFValid = false;
  const bool evBuilt = evOn;

  const int q = nReCoef;
  std::vector<double> ns(nNsBasis), reAll((size_t)nSubject * (q > 0 ? q : 1)), gRef(nNsBasis), gNew(nNsBasis), ivv(gkNiv);
  std::vector<double*> reRow(nSubject);
  for(int i=0; i<nSubject; i++) reRow[i] = &reAll[(size_t)i * (q > 0 ? q : 1)];
  long nCmp = 0, nBad = 0;
  const char* firstBad = "";
  auto cmp = [&](double x, double yv, const char* what) {
    nCmp++;
    if(!gk61::sameBits(x, yv)) { if(nBad == 0) firstBad = what; nBad++; }
  };
  const double amp[3] = {0.3, 1.0, 3.0};
  Eigen::VectorXd g0, g1;
  std::vector<double> v0(nSubject), w0(nSubject), y0(nSubject), g0All((size_t)nSubject * (q > 0 ? q : 1));
  for(int s=0; s<3 && nBad == 0; s++){
    for(int l=0; l<nNsBasis; l++)
      ns[l] = (l == 0 ? pi.nsCoefPriorMean : 0.0) + amp[s] * 0.5 * std::sin(1.7 * (l + 1) + 0.9 * s);
    for(int i=0; i<nSubject; i++)
      for(int m=0; m<q; m++) reRow[i][m] = amp[s] * 0.4 * std::sin(0.37 * (i + 1) + 1.3 * m + 2.1 * s);

    // subject log-likelihood (ncp move) and value + RE gradient (MALA on b_i):
    // reference, then the cached path with the full chain and with the shared
    // spline part (gkS)
    gkOn = false; evOn = false;
    for(int i=0; i<nSubject; i++){
      v0[i] = loglikelihood_re(i, ns.data(), reRow[i]);
      w0[i] = loglikelihood_grad_re(i, ns.data(), reRow[i], g0);
      y0[i] = splineY[i];
      for(int m=0; m<q; m++) g0All[(size_t)i * q + m] = g0(m);
    }
    gkOn = true; evOn = evBuilt;
    for(int pass=0; pass<2; pass++){
      if(pass == 1) gkFillS(ns.data());
      for(int i=0; i<nSubject; i++){
        cmp(v0[i], loglikelihood_re(i, ns.data(), reRow[i]), "loglikelihood_re");
        cmp(w0[i], loglikelihood_grad_re(i, ns.data(), reRow[i], g1), "loglikelihood_grad_re");
        cmp(y0[i], splineY[i], "loglikelihood_grad_re (splineY)");
        for(int m=0; m<q; m++) cmp(g0All[(size_t)i * q + m], g1(m), "loglikelihood_grad_re (gradient)");
      }
    }
    gkSValid = false;

    // total log-lik + spline gradient for every k (drawNs pass) and the vij hand-off
    if(s < 2){
      double llRef = 0.0;
      gkOn = false; evOn = false;
      for(int k=0; k<nNsBasis; k++){
        double g;
        const double ll = loglikelihood_grad_ns(k, ns.data(), reRow.data(), g);
        gRef[k] = g;
        if(k == 0) llRef = ll;
      }
      gkOn = true; evOn = evBuilt;
      const double llNew = gkGradNsAll(ns.data(), reRow.data(), gNew.data(), ivv.data());
      cmp(llRef, llNew, "loglikelihood_grad_ns");
      for(int k=0; k<nNsBasis; k++) cmp(gRef[k], gNew[k], "loglikelihood_grad_ns (gradient)");
      for(int i=0; i<nSubject; i++){
        auto f = [&](double t) { return integrand(t, ns.data(), reRow[i]); };
        for(int j=(int)xIndex[i] + 1; j<(int)xIndex[i+1]; j++)
          if(gkRowIv[j] >= 0) cmp(integrateSeg(f, xObsTime[j], xObsTime[j+1]), ivv[gkRowIv[j]], "vij hand-off");
      }
    }
  }

  std::copy(allfitSave.begin(), allfitSave.end(), allfit);
  std::copy(splineYSave.begin(), splineYSave.end(), splineY);
  evLogFValid = false;
  if(nBad > 0) {
    printf("GK61 node cache: self-check FAILED (%s; %ld of %ld comparisons differ) -- using the uncached quadrature\n",
           firstBad, nBad, nCmp);
    gkRelease();
  } else {
    printf("GK61 node cache: %d intervals (%.1f MB), self-check passed (%ld bit-identical comparisons)\n",
           gkNiv, gkBytes / 1048576.0, nCmp);
  }
}

void survivalBART::setparamPrior(double nsCoefPriorMean, double nsCoefPriorVar, double *reCoefPriorMean,
                                 double *reCoefInitial, double *reCoefVarInitial,
                                 double reHWnu, double *reHWA){
  // Spline coefficients share ONE prior N(nsCoefPriorMean, nsCoefPriorVar).
  // (A former per-coefficient curvature-shrinkage variance was removed: the
  // nsCurvVar sweep showed shrinkage relocates the slow ridge direction and
  // WORSENS sigma11 recovery when the true baseline genuinely curves.)
  pi.nsCoefPriorMean = nsCoefPriorMean;
  pi.nsCoefPriorVar = nsCoefPriorVar;
  
  // ---- Random-effect covariance: Huang-Wand hyperprior (the only option) ----
  pi.reHWnu = reHWnu;
  if(pi.reHWA) delete[] pi.reHWA;
  pi.reHWA = new double[nReCoef];
  for(int i=0; i<nReCoef; i++) pi.reHWA[i] = reHWA[i];
  if(reCovAux) delete[] reCovAux;
  reCovAux = new double[nReCoef];
  for(int i=0; i<nReCoef; i++) reCovAux[i] = 1.0;   // overwritten before first use
  printf("Huang-Wand nu: %f  A: ", pi.reHWnu);
  for(int i=0; i<nReCoef; i++) printf("%f ", pi.reHWA[i]);
  printf("\n");
  printf("maxReDegree: %d\n", maxReDegree);
  printf("nReCoef: %d\n", nReCoef);
  printf("nsCoefPriorMean: %f\n", nsCoefPriorMean);
  printf("nsCoefPriorVar: %f\n", nsCoefPriorVar);
  
  if(pi.reCoefPriorMean) delete[] pi.reCoefPriorMean;
  pi.reCoefPriorMean = new double[nReCoef];
  for(int i=0; i<nReCoef; i++){
    pi.reCoefPriorMean[i] = reCoefPriorMean[i];
  }
  for(int i=0; i<nReCoef; i++){
    printf("reCoefPriorMean[%d]: %f\n", i, reCoefPriorMean[i]);
  }
  
  //initialize nsCoef
  if(nsCoef) delete[] nsCoef; 
  nsCoef = new double[nNsBasis];
  for(int i=0; i<nNsBasis; i++){
    nsCoef[i] = pi.nsCoefPriorMean;
  }
  
  if(reCoef) {
    for(int i=0; i<nSubject; i++){
      if (reCoef[i]) delete[] reCoef[i];
    }
    delete[] reCoef;
  }
  reCoef = new double*[nSubject];
  for(int i=0; i<nSubject; i++){
    reCoef[i] = new double[nReCoef];
  }
  for(int i=0; i<nSubject; i++){
    for(int j=0; j<nReCoef; j++){
      reCoef[i][j] = reCoefInitial[i * nReCoef + j];
    }
  }
  
  if(reCoefCov) {
    for(int i=0; i<nReCoef; i++){
      delete[] reCoefCov[i];
    }
    delete[] reCoefCov;
  }
  printf("reCoefCov_temp:\n");
  for(int i=0; i<nReCoef; i++){
    for(int j=0; j<nReCoef; j++){
      printf("%f ", reCoefVarInitial[i * nReCoef + j]);
    }
    printf("\n");
  }
  reCoefCov = new double*[nReCoef];
  for(int i=0; i<nReCoef; i++){
    reCoefCov[i] = new double[nReCoef];
  }
  for(int i=0; i<nReCoef; i++){
    for(int j=0; j<nReCoef; j++){
      reCoefCov[i][j] = reCoefVarInitial[i * nReCoef + j];
    }
  }
  
  printf("reCoefCov:\n");
  for(int i=0; i<nReCoef; i++){
    for(int j=0; j<nReCoef; j++){
      printf("%f ", reCoefCov[i][j]);
    }
    printf("\n");
  }
}

double survivalBART::b_spline_basis(int i, int k, double x, double *knots) {
  if (k == 0) {
    return (x >= knots[i] && x <= knots[i+1]) ? 1.0 : 0.0;
  }
  
  double a = 0.0, b = 0.0;
  
  double denom1 = knots[i+k] - knots[i];
  if (denom1 > 1e-10) {
    a = (x - knots[i]) / denom1 * b_spline_basis(i, k-1, x, knots);
  }
  
  double denom2 = knots[i+k+1] - knots[i+1];
  if (denom2 > 1e-10) {
    b = (knots[i+k+1] - x) / denom2 * b_spline_basis(i+1, k-1, x, knots);
  }
  
  return a + b;
}

// Natural cubic spline basis (scalar), truncated-power (Royston-Parmar) form.
// Boundary knots a = knots[0], b = knots[K-1]; internal knots kappa_1..kappa_{K-2}.
// With include_intercept = true the K bases are
//
//   B_0(x) = 1,  B_1(x) = x,
//   B_{j+2}(x) = (x-kappa_j)_+^3 - lam_j (x-a)_+^3 - (1-lam_j)(x-b)_+^3,
//                lam_j = (b - kappa_j) / (b - a).
//
// The lam_j weights enforce BOTH natural boundary conditions f''(a) = f''(b) = 0,
// so the fit is linear outside [a,b].  This spans exactly the same space as
// cbind(1, splines::ns(x, knots, Boundary.knots = c(a,b)))  (all principal
// angles 0, verified numerically).
// CAUTION: lam_j multiplies the (x-a) term and (1-lam_j) the (x-b) term; the two
// are NOT interchangeable.  Swapping them (the pre-2026-08-20 form) leaves
// f''(b) = 6(a + b - 2 kappa_j) != 0 -- a quadratic right tail, a different
// K-dim spline space, and ~1.8x looser induced prior on log h0 near t = b.
double survivalBART::ns_kth_basis_scalar(double x,
                                         double* knots,
                                         int K,
                                         int degree,
                                         int which,
                                         bool include_intercept)
{
  // Piecewise-constant baseline: basis k is the indicator of bin k, so
  // A(t) = sum_k nsCoef[k] * 1{t in bin k} = log lambda_{bin(t)} (ns_mean is 0).
  if (baselineType == 1)
    return (which == pwBin(x)) ? 1.0 : 0.0;
  
  // Extra marginal-curvature basis: when the intercept is included, index K
  // (one past the natural-spline bases) is the t^2 term that absorbs the
  // quadratic curvature the random-slope variance induces in the marginal hazard.
  if (include_intercept && which == K)
    return x * x;
  
  int total = include_intercept ? K : (K - 1);
  
  if (which < 0 || which >= total)
    return 0.0;
  
  double xmin = knots[0];
  double xmax = knots[K - 1];
  
  int K_int = K - 2;
  double* internal = knots + 1;
  
  auto tp3 = [&](double z) {
    return (z > 0.0) ? z * z * z : 0.0;
  };
  
  if (include_intercept) {
    if (which == 0)
      return 1.0;
    
    if (which == 1)
      return x;
    
    int j = which - 2;
    if (j >= K_int)
      return 0.0;
    
    double k_j = internal[j];
    
    double val =
      tp3(x - k_j)
      - tp3(x - xmin) * (xmax - k_j) / (xmax - xmin)
      - tp3(x - xmax) * (k_j - xmin) / (xmax - xmin);
      
      return val;
  }
  
  if (which == 0)
    return (x - xmin);
  
  int j = which - 1;
  if (j >= K_int)
    return 0.0;
  
  double k_j = internal[j];
  
  double val =
    tp3(x - k_j)
    - tp3(x - xmin) * (xmax - k_j) / (xmax - xmin)
    - tp3(x - xmax) * (k_j - xmin) / (xmax - xmin);
    
    return val;
}



double survivalBART::randomEffect(int i, double t, int subjectId){
  double re = 0;
  for(int k=0; k<nReCoef; k++){
    re += reCoef[subjectId][k] * pow(t / reTimeScale, rePowers[k]);
  }
  return re;
}

double survivalBART::integrand(double t, double *nsCoef, double *reCoef){
  double integrand;
  if(baselineType == 1){
    // Fast path: the indicator-basis loop reduces to one bin lookup
    // (ns_mean is identically zero in piecewise mode).
    integrand = nsCoef[pwBin(t)];
  } else {
    integrand = 0;
    for(int i=0; i<nNsBasis; i++){
      integrand += nsCoef[i] * (ns_kth_basis_scalar(t, nsKnots, nKnots, 3, i, true) - ns_mean[i]);
    }
  }
  for(int i=0; i<nReCoef; i++){
    integrand += reCoef[i] * pow(t / reTimeScale, rePowers[i]);
  }
  return exp(integrand);
}

void survivalBART::calculate_vij(double *xObsTime, double *xIndex){
  if(gkOn){
    // drawNs already integrated exp(A) over every cached interval at the final
    // spline state (with the same reCoef and the same quadrature): take those values.
    int hit = -1;
    for(int s=0; s<2; s++)
      if(gkMemoValid[s] && std::memcmp(gkMemoCoef[s].data(), nsCoef, sizeof(double) * nNsBasis) == 0) { hit = s; break; }
    for(int i=0; i<nSubject; i++){
      int start_time_index = xIndex[i] + 1;
      int end_time_index = xIndex[i+1];
      for(int j=start_time_index; j<end_time_index; j++){
        const int iv = gkRowIv[j];
        vij[j] = (iv >= 0 && hit >= 0) ? gkMemoIv[hit][iv] : gkValue(j, nsCoef, reCoef[i]);
      }
      vij[end_time_index] = 0; //this is for the event time to infinity
    }
    gkMemoValid[0] = gkMemoValid[1] = false;   // trees and allfit change next
    return;
  }

  for(int i=0; i<nSubject; i++){
    int start_time_index = xIndex[i] + 1;
    int end_time_index = xIndex[i+1];
    
    auto f = [&](double t) { return integrand(t, nsCoef, reCoef[i]); };
    
    for(int j=start_time_index; j<end_time_index; j++){
      double a = xObsTime[j];
      double b = xObsTime[j+1];
      
      vij[j] = integrateSeg(f, a, b);
    }
    vij[end_time_index] = 0; //this is for the event time to infinity
  }
}

double* survivalBART::calCumHazard() {
  for (int i = 0; i < nSubject; i++) {
    int start_time_index = (int)xIndex[i] + 1;     // first event id of subject i  (= 0 when i==0)
    int end_time_index   = (int)xIndex[i + 1];     // last event id of subject i
    
    cumHazard[start_time_index] = 0.0;             // H = 0 at the subject's first time point
    
    // Snapshot the row pointer so the lambda doesn't depend on the loop variable `i`.
    double *re_i        = reCoef[i];
    double *ns_coef_ptr = nsCoef;
    auto f = [this, ns_coef_ptr, re_i](double t) {
      return integrand(t, ns_coef_ptr, re_i);
    };
    
    for (int j = start_time_index; j < end_time_index; j++) {
      double a            = xObsTime[j];
      double b            = xObsTime[j + 1];
      double bart_pred_j  = allfit[j];   // = allfit[xIndex[i] + 1 + (j - start_time_index)]
      double interval_int = gkOn ? gkValue(j, ns_coef_ptr, re_i) : integrateSeg(f, a, b);
      cumHazard[j + 1]    = cumHazard[j] + interval_int * bart_pred_j;
    }
  }
  return cumHazard;
}

// for test
void survivalBART::draw(double sigma, size_t nIteration, size_t nBurnIn, rn& gen)
{
  const bool verbose = (nIteration % 100 == 0);   // gate per-iter debug to every 100th sweep
  if(verbose) printf("survival draw\n");
  reLoglikValid = false;   // baseline and trees change below; ell_i cache is stale
  evLogFValid = false;     // log f at the event rows: recomputed from the current allfit
  if(gkOn && !gkChecked) gkSelfCheck();   // once, now that xEvent and allfit are set

  // (A former calculate_vij call here was removed: nothing reads the member vij
  // before the recomputation below -- drawNs/mala_ns and drawLambda build their
  // own integrals -- so it was one full dead GK61 pass per sweep in both modes.)
  
  if(verbose) printf("*****draw ns*****\n");
  if(baselineType == 1){
    // Piecewise baseline: exact conjugate gamma Gibbs for the bin heights
    // lambda_k -- no MALA, no step-size adaptation.
    drawLambda(gen);
  } else if(nsBlock) {
    // ns.block: one preconditioned block MALA step on the whole spline vector;
    // h and M adapt during burn-in only (nsBlockAdapt), then the kernel is fixed.
    nsBlockAdapt(nIteration, nBurnIn);
    drawNsBlock(nIteration, nBurnIn, gen);
  } else {
    // --- Adaptive tuning for nsRate (std-dev multiplier, MALA) ---
    // Only adapt during burn-in, in batches of 500, targeting ~0.574 (Roberts-Rosenthal MALA optimum).
    if(nIteration > 0 && nIteration <= nBurnIn && nIteration % 500 == 0) {
      for(int j=0; j<nNsBasis; j++){
        double rate = nsAccept[j] / 500.0;
        nsRate[j] *= exp(1.0 * (rate - 0.574));
        if(nsRate[j] < 1e-4) nsRate[j] = 1e-4;
        if(nsRate[j] > 1e2)  nsRate[j] = 1e2;
        nsAccept[j] = 0;
      }
      printf("ns rate adapted at iter %zu\n", nIteration);
      for(int j=0; j<nNsBasis; j++){
        printf("  nsRate[%d]: %f\n", j, nsRate[j]);
      }
    }
    drawNs(gen);
  }
  
  //printf nsCoef
  if(verbose) for(int i=0; i<nNsBasis; i++){
    printf("nsCoef[%d]: %f\n", i, nsCoef[i]);
  }
  
  calculate_vij(xObsTime, xIndex);
  
  //draw BART
  if(!leafValid && leafOn) {
    const double leafBytes = (double)sizeof(tree::tree_cp) * (double)m * (double)n;
    const double cap = cacheCapBytes();
    if(gk61::contractionDetected()) {
      // contracting build: keep the reference (tree-descending) sufficient statistics
      leafOn = false;
      printf("BART leaf cache: off (this build contracts floating-point multiply-adds; -ffp-contract is not off) -- descending the trees\n");
    } else if(leafBytes + gkBytes > cap) {
      leafOn = false;
      printf("BART leaf cache: off (%.1f MB + %.1f MB node cache exceeds TVSB_GK_MAXMB = %.0f MB) -- descending the trees\n",
             leafBytes / 1048576.0, gkBytes / 1048576.0, cap / 1048576.0);
    } else {
      leafCache.resize(m * n);
      for(size_t j=0;j<m;j++) fillleaf(t[j],xi,p,n,x,&leafCache[j * n]);
      leafValid = true;
    }
  }
  size_t i = 0;
  for(size_t j=0;j<m;j++) {
    // bottom node of every row in tree j (0: no leaf cache, descend the tree)
    tree::tree_cp *leaf = leafOn ? &leafCache[j * n] : 0;
    if(leaf) fit(leaf,n,ftemp); else fit(t[j],xi,p,n,x,ftemp);
    for(size_t k=0;k<n;k++) {
      if(allfit[k] == 0 && ftemp[k] == 0){
        fnoh[k] = 1;
        allfit[k] = 1;
      }else{
        fnoh[k] = allfit[k] / ftemp[k];
        allfit[k] /= ftemp[k];
      }
    }
    if(bd(t[j],xi,di,pi,sigma,nv,pv,aug,gen,fnoh,vij,ui,leaf)) i++;
    if(leaf) drmu(t[j],xi,di,pi,sigma,gen,fnoh,vij,ui,leaf);
    else drmu(t[j],xi,di,pi,sigma,gen,fnoh,vij,ui);
    if(leaf) fit(leaf,n,ftemp); else fit(t[j],xi,p,n,x,ftemp);
    for(size_t k=0;k<n;k++) allfit[k] *= ftemp[k];
  }
  evLogFValid = false;   // allfit changed
  
  // --- Move 1: BART overall-level <-> baseline-intercept recentering ---
  // EXACT random-walk Metropolis along the unidentified level direction.
  // The former proposal used a deterministic c_f = log(exposure-weighted mean f),
  // a STATE-DEPENDENT projection: its frozen Jacobian (-c_f*H/m) and implicit
  // q-ratio = 1 are only valid for a FIXED shift, so detailed balance failed.
  // A symmetric proposal a ~ N(0, cf_step^2) makes the SAME accept/reject -- leaf
  // GIG prior ratio + baseline N(m0,v0) prior ratio + Jacobian -a*H/m -- an exact
  // MH step (reverse proposes -a with the same density => q-ratio = 1). The shift
  // leaves eta_i(t) invariant: log f -= a (every leaf /= exp(a/m)), nsCoef[0] += a.
  // cf_step is adapted during burn-in (batches of 500) toward the 1-D RWM optimum ~0.44.
  if(nIteration > 0 && nIteration <= nBurnIn && nIteration % 500 == 0) {
    double rate = cf_accept / 500.0;
    cf_step *= exp(1.0 * (rate - 0.44));
    if(cf_step < 1e-4) cf_step = 1e-4;
    if(cf_step > 1e2)  cf_step = 1e2;
    cf_accept = 0.0;
    printf("cf_step adapted at iter %zu -> %f\n", nIteration, cf_step);
  }
  double c_f = cf_step * gen.normal();   // symmetric, state-independent RW shift
  if(verbose) printf("c_f (RW level-shift) proposed: %f\n", c_f);
  
  // Metropolis-Hastings acceptance step for the proposed level shift c_f.
  double per_tree_factor = exp(c_f / (double)m);
  
  std::vector<double> mus_curr;
  for(size_t j=0; j<m; j++) {
    tree::npv bots;
    t[j].getbots(bots);
    for(size_t k=0; k<bots.size(); k++) {
      mus_curr.push_back(bots[k]->gettheta());
    }
  }
  size_t H_total = mus_curr.size();
  
  auto log_pleaf_kernel = [&](double th) {
    if(th <= 0.0) return -std::numeric_limits<double>::infinity();
    double lt = std::log(th);
    double a = -(pi.c + 1.0) * lt - pi.d / th;
    double b =  (pi.c - 1.0) * lt - pi.d * th;
    return logsumexp_two(a, b);
  };
  
  double log_prior_diff_cf = 0.0;
  for(size_t k=0; k<H_total; k++) {
    double mu_new = mus_curr[k] / per_tree_factor;
    log_prior_diff_cf += log_pleaf_kernel(mu_new) - log_pleaf_kernel(mus_curr[k]);
  }
  
  if(baselineType == 1){
    // Piecewise: the level shift moves ALL log-lambda_k by c_f. State is
    // s_k = log lambda_k, whose prior density (Gamma(pwShape, pwRate) on
    // lambda, transformed) is  log p(s) = pwShape*s - pwRate*e^s + const,
    // so the shift needs no extra Jacobian on the lambda side.
    for(int kb=0; kb<nNsBasis; kb++){
      double lam_k = exp(nsCoef[kb]);
      log_prior_diff_cf += pwShape * c_f - pwRate * lam_k * (exp(c_f) - 1.0);
    }
  } else {
    double m0_cf = pi.nsCoefPriorMean;
    double v0_cf = pi.nsCoefPriorVar;
    double ns_old_cf = nsCoef[0];
    double ns_new_cf = nsCoef[0] + c_f;
    log_prior_diff_cf += ((ns_old_cf - m0_cf)*(ns_old_cf - m0_cf)
                            - (ns_new_cf - m0_cf)*(ns_new_cf - m0_cf)) / (2.0 * v0_cf);
  }
  
  double log_jac_cf = -c_f * (double)H_total / (double)m;
  double log_alpha_cf = log_prior_diff_cf + log_jac_cf;
  
  double u_cf = R::runif(0.0, 1.0);
  if(std::log(u_cf) < log_alpha_cf) {
    //accept, update leaf and fitted value
    for(size_t j=0; j<m; j++) {
      tree::npv bots;
      t[j].getbots(bots);
      for(size_t k=0; k<bots.size(); k++) {
        bots[k]->settheta(bots[k]->gettheta() / per_tree_factor);
      }
    }
    double scale = exp(c_f);
    for(int j=0; j<n; j++){
      allfit[j] = allfit[j] / scale;
    }
    evLogFValid = false;
    if(baselineType == 1){
      for(int kb=0; kb<nNsBasis; kb++) nsCoef[kb] += c_f;  // uniform log-lambda shift
    } else {
      nsCoef[0] += c_f;
    }
    bart_weighted_mean = c_f;
    bartMean = c_f;
    cf_accept += 1.0;
    if(verbose) printf("Move 1 (c_f) ACCEPT  c_f=%f  log_alpha=%f\n", c_f, log_alpha_cf);
  } else {
    //reject, do nothing
    bart_weighted_mean = 0.0;
    bartMean = 0.0;
    if(verbose) printf("Move 1 (c_f) REJECT  c_f=%f  log_alpha=%f\n", c_f, log_alpha_cf);
  }
  
  if(verbose) printf("draw random effect\n");
  // --- Adaptive tuning for reRate (std-dev multiplier, MALA) ---
  // MALA optimum ~0.574 (Roberts-Rosenthal 2001), dim-agnostic asymptotic.
  if(nIteration > 0 && nIteration <= nBurnIn && nIteration % 500 == 0) {
    for(int j=0; j<nSubject; j++){
      double rate = reAccept[j] / 500.0;
      reRate[j] *= exp(1.0 * (rate - 0.574));
      if(reRate[j] < 1e-4) reRate[j] = 1e-4;
      if(reRate[j] > 1e2)  reRate[j] = 1e2;
      reAccept[j] = 0;
    }
    printf("re rate adapted at iter %zu\n", nIteration);
  }
  drawRandomEffect(gen);
  
  if(verbose) printf("draw random effect covariance\n");
  drawRandomEffectCov(gen);
  
  // --- Joint non-centred (Sigma_b, b) Metropolis move ---
  // Proposes Sigma_b* in log-Cholesky coordinates and moves every b_i with it so
  // that the standardized effects L^{-1}(b_i - mu_b) stay fixed. Accepted with the
  // full-likelihood ratio times the prior ratio of Sigma_b in xi coordinates, it
  // leaves the joint conditional of (Sigma_b, b) given everything else invariant
  // (see ncpSigmaUpdate). Replaces the former b-marginalized (Gauss-Hermite)
  // Sigma_b passes and the extra MALA refresh, which were not joint-invariant.
  if(reNcp > 0) {
    ncpAdapt(nIteration, nBurnIn);
    for(int pass=0; pass<reNcp; pass++) ncpSigmaUpdate(nIteration, nBurnIn, gen);
  }
  if(verbose) {                       // print the sampled Sigma_b every 100th sweep
    printf("reCoefCov (Sigma_b):\n");
    for(int j=0; j<nReCoef; j++){
      for(int k=0; k<nReCoef; k++) printf("%f ", reCoefCov[j][k]);
      printf("\n");
    }
  }
  
  // Group move (random intercept + slope) -- EXACT Gibbs along the likelihood
  // orbit. REPLACES the former (c_b,c_s) Metropolis move, which targeted the
  // variance functionals 0.5*sigma00 / sigma01 (delta = change-in-functional)
  // and therefore went inert in steady state, never traversing the empirical-
  // mean identifiability ridge between the baseline and the random effects.
  //
  // The likelihood depends only on eta_i(t) and is invariant under the joint
  // shift by a common vector c shared by all subjects (RE design on t/tau,
  // tau = reTimeScale):
  //     reCoef[i] -= c                                        (all i)
  //     nsCoef[1] += c1/tau ;  nsCoef[0] += c0 + (c1/tau)*m1  (m1 = ns_mean[1] =
  //                                          weighted-mean t; centered-basis comp.)
  // i.e. beta += A c with A^T s(t) = z(t) = (1, t/tau). Because the spline prior
  // N(m0, v0 I) and the RE prior N(mu_b, Sigma_b) are both Gaussian, the shift's
  // conditional along the orbit is Gaussian (parameter expansion / interweaving):
  //     c ~ N(H^{-1} h, H^{-1}),
  //     H = A^T V_beta^{-1} A + N Sigma_b^{-1},
  //     h = Sigma_b^{-1} sum_i (b_i - mu_b) - A^T V_beta^{-1} (beta - m_beta).
  // Drawing c and applying the shift has acceptance probability 1 and leaves the
  // raw target invariant; it fixes the slow mixing along the intercept/slope
  // ridge. (Marginal "mean-one" hazard centering, which involves Sigma_b, is a
  // SEPARATE deterministic reporting transform done in post-processing, NOT here.)
  if(baselineType == 0) {
    const int q = nReCoef;
    const double m0 = pi.nsCoefPriorMean;
    const double v0 = pi.nsCoefPriorVar;
    const double m1 = (nNsBasis > 1) ? ns_mean[1] : 0.0;   // weighted mean of t
    
    bool group_supported = true;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2, q);
    for(int j=0; j<q; j++) {
      if(rePowers[j] == 0) {
        A(0, j) = 1.0;
      } else if(rePowers[j] == 1 && nNsBasis > 1) {
        A(0, j) = m1 / reTimeScale;
        A(1, j) = 1.0 / reTimeScale;
      } else {
        group_supported = false;
      }
    }
    
    if(group_supported) {
      // A^T V_beta^{-1} A and A^T V_beta^{-1} (beta - m_beta), with V_beta = v0*I.
      Eigen::MatrixXd AtVinvA = (A.transpose() * A) / v0;
      Eigen::VectorXd beta_minus_mean(2);
      beta_minus_mean(0) = nsCoef[0] - m0;
      beta_minus_mean(1) = (nNsBasis > 1) ? (nsCoef[1] - m0) : 0.0;
      Eigen::VectorXd AtVinv_bm = (A.transpose() * beta_minus_mean) / v0;
      
      Eigen::MatrixXd Sigma_re(q, q);
      for(int j=0; j<q; j++)
        for(int k=0; k<q; k++)
          Sigma_re(j, k) = reCoefCov[j][k];
      Eigen::LLT<Eigen::MatrixXd> llt_sigma(Sigma_re);
      if(llt_sigma.info() != Eigen::Success) {
        Rcpp::stop("group move: reCoefCov not positive definite");
      }
      Eigen::MatrixXd Sigma_inv =
        llt_sigma.solve(Eigen::MatrixXd::Identity(q, q));
      
      Eigen::VectorXd mu_re(q);
      for(int j=0; j<q; j++) mu_re(j) = pi.reCoefPriorMean[j];
      
      Eigen::VectorXd sum_bmu = Eigen::VectorXd::Zero(q);
      for(int i=0; i<nSubject; i++)
        for(int j=0; j<q; j++)
          sum_bmu(j) += reCoef[i][j] - mu_re(j);
      
      Eigen::MatrixXd H = AtVinvA + (double)nSubject * Sigma_inv;
      Eigen::VectorXd h = Sigma_inv * sum_bmu - AtVinv_bm;
      
      Eigen::LLT<Eigen::MatrixXd> llt_H(H);
      if(llt_H.info() != Eigen::Success) {
        Rcpp::stop("group move: H not positive definite");
      }
      Eigen::VectorXd mean_c = llt_H.solve(h);              // H^{-1} h
      Eigen::VectorXd zstd(q);
      for(int j=0; j<q; j++) zstd(j) = gen.normal();
      // c = mean_c + U^{-1} z with U = chol-upper of H  =>  Cov(c) = H^{-1}.
      Eigen::VectorXd c = mean_c + llt_H.matrixU().solve(zstd);
      
      // Apply the exact likelihood-invariant shift (acceptance probability 1).
      for(int i=0; i<nSubject; i++)
        for(int j=0; j<q; j++)
          reCoef[i][j] -= c(j);
      
      double shift0 = 0.0;
      double shift1 = 0.0;
      for(int j=0; j<q; j++) {
        if(rePowers[j] == 0) {
          shift0 += c(j);
        } else if(rePowers[j] == 1) {
          shift0 += c(j) * m1 / reTimeScale;
          shift1 += c(j) / reTimeScale;
        }
      }
      nsCoef[0] += shift0;
      if(nNsBasis > 1) nsCoef[1] += shift1;
      if(verbose) { printf("group move (Gibbs) powers:");
        for(int j=0; j<q; j++) printf(" p%d=%f", rePowers[j], c(j));
        printf("\n"); }
    }
  }
  else {
    // Piecewise group move. The exact Gibbs above relies on Gaussian priors for
    // the baseline coefficients; with Gamma priors on lambda_k the orbit
    // conditional is not Gaussian, so use a symmetric random-walk Metropolis
    // along the INTERCEPT orbit only:
    //     reCoef[i][j0] -= c (all i),   log lambda_k += c (all k),
    // which leaves eta_i(t) invariant because sum_k 1{t in bin k} = 1 for all t.
    // The slope orbit would need a linear-in-t baseline direction that a step
    // function cannot absorb exactly, so it is dropped in piecewise mode.
    // Acceptance = Gaussian b-prior ratio + Gamma prior ratio in log-lambda
    // coordinates (symmetric proposal, likelihood invariant, no Jacobian).
    int j0 = -1;
    for(int j=0; j<nReCoef; j++) if(rePowers[j] == 0){ j0 = j; break; }
    if(j0 >= 0){
      if(nIteration > 0 && nIteration <= nBurnIn && nIteration % 500 == 0) {
        double rate = gmAccept / 500.0;
        gmStep *= exp(1.0 * (rate - 0.44));
        if(gmStep < 1e-4) gmStep = 1e-4;
        if(gmStep > 1e2)  gmStep = 1e2;
        gmAccept = 0.0;
        printf("gmStep adapted at iter %zu -> %f\n", nIteration, gmStep);
      }
      const int q = nReCoef;
      Eigen::MatrixXd Sigma_re(q, q);
      for(int j=0; j<q; j++)
        for(int k=0; k<q; k++)
          Sigma_re(j, k) = reCoefCov[j][k];
      Eigen::LLT<Eigen::MatrixXd> llt_sigma(Sigma_re);
      if(llt_sigma.info() == Eigen::Success){
        Eigen::MatrixXd Sigma_inv = llt_sigma.solve(Eigen::MatrixXd::Identity(q, q));
        Eigen::VectorXd sum_bmu = Eigen::VectorXd::Zero(q);
        for(int i2=0; i2<nSubject; i2++)
          for(int j=0; j<q; j++)
            sum_bmu(j) += reCoef[i2][j] - pi.reCoefPriorMean[j];
        double c_g = gmStep * gen.normal();
        // b_i -> b_i - c e_{j0} (all i):
        //   c * (Sigma^{-1} sum_i (b_i - mu))_{j0} - 0.5 * N * c^2 * (Sigma^{-1})_{j0 j0}
        double log_alpha_g = c_g * Sigma_inv.row(j0).dot(sum_bmu)
          - 0.5 * (double)nSubject * c_g * c_g * Sigma_inv(j0, j0);
        for(int kb=0; kb<nNsBasis; kb++){
          double lam_k = exp(nsCoef[kb]);
          log_alpha_g += pwShape * c_g - pwRate * lam_k * (exp(c_g) - 1.0);
        }
        if(std::log(R::runif(0.0, 1.0)) < log_alpha_g){
          for(int i2=0; i2<nSubject; i2++) reCoef[i2][j0] -= c_g;
          for(int kb=0; kb<nNsBasis; kb++) nsCoef[kb] += c_g;
          gmAccept += 1.0;
          if(verbose) printf("piecewise group move ACCEPT c=%f\n", c_g);
        } else if(verbose) printf("piecewise group move REJECT c=%f\n", c_g);
      }
    }
  }
  
  reLoglikValid = false;   // the group move rewrote b_i and the baseline

  //print acceptance rates
  accept=i/(double)m;
  if(verbose) {
    printf("BART accept: %f\n", accept);
    printf("nIteration: %zu\n", nIteration);
    double mean_nsAcceptRate = 0;
    for(int j=0; j<nNsBasis; j++){
      mean_nsAcceptRate += (nsAccept[j] / nIteration);
    }
    printf("mean nsAcceptRate: %f\n", mean_nsAcceptRate / nNsBasis);
    double mean_reAcceptRate = 0;
    for(int j=0; j<nSubject; j++){
      mean_reAcceptRate += (reAccept[j] / nIteration);
    }
    printf("mean reRate: %f\n", mean_reAcceptRate / nSubject);
  }
}


//update Ns coefficients via MALA
void survivalBART::drawNs(rn& gen) {
  gkMemoValid[0] = gkMemoValid[1] = false;   // b_i and allfit changed since the last sweep
  Eigen::VectorXd nsCoef_temp(nNsBasis);
  for(int i=0; i<nNsBasis; i++){
    nsCoef_temp(i) = nsCoef[i];
  }
  
  // MALA step variance for basis k is nsRate[k]^2 (I preconditioning).
  // Interpretation B: prior on stored nsCoef[k] is N(pi.nsCoefPriorMean, pi.nsCoefPriorVar)
  // for all k, with no c_b adjustment.
  for(int i=0; i<nNsBasis; i++){
    double step_sigma2 = nsRate[i] * nsRate[i];
    // reCoef is only read while the spline coefficients move: passed as is
    // (no per-sweep deep copy)
    mala_ns(i, nsCoef_temp, reCoef, step_sigma2,
            pi.nsCoefPriorMean, pi.nsCoefPriorVar, gen);
  }
  
  return;
}

// ============================================================================
// ns.block: preconditioned block MALA on the whole spline vector beta = nsCoef.
//
// Target (the full conditional of beta given the trees, b and everything else):
//   log pi(beta) = sum_i ell_i(beta) + sum_k log N(beta_k; m0, v0) + const,
//   g(beta) = grad log pi(beta) = grad sum_i ell_i(beta) - (beta - m0) / v0.
// Proposal (M = L L' SPD, h > 0, both FIXED within a step):
//   beta* = beta + (h^2/2) M g(beta) + h L eps,  eps ~ N(0, I_K),
// i.e. q(beta* | beta) = N(beta*; beta + (h^2/2) M g(beta), h^2 M). Accept with
//   log alpha = log pi(beta*) - log pi(beta) + log q(beta | beta*) - log q(beta* | beta),
//   log q(y | x) = -|L^{-1} (y - x - (h^2/2) M g(x))|^2 / (2 h^2) + c(h, M),
// where c(h, M) is the same in both directions and cancels. This is a
// Metropolis-Hastings step for pi(beta | rest) for ANY SPD M and h > 0, so it
// leaves the full posterior invariant; M and h change only during burn-in
// (nsBlockAdapt), so the kept draws come from one fixed kernel.
// Cost: log pi and all K gradient components come from one fused likelihood pass
// per state (loglikelihood_grad_ns_all): two passes per sweep (current state and
// proposal) against K + 1 for the coordinatewise update. Both states stay in the
// drawNs memo, so calculate_vij takes vij from whichever one is kept.
// ============================================================================
void survivalBART::setNsBlock(bool on, int mSource, size_t nBurnIn) {
  nsBlock = on;
  nsBlkMSource = (mSource == 1) ? 1 : 0;
  if(nsBlock && baselineType != 0) {
    printf("ns.block: the block MALA needs the spline baseline -- ignored (the piecewise baseline keeps its conjugate Gibbs update)\n");
    nsBlock = false;
  }
  if(nsBlock && nNsBasis < 1) nsBlock = false;
  if(!nsBlock) return;
  const int K = nNsBasis;
  // Starting kernel: M = diag(nsRate[k]^2) (the coordinatewise MALA's starting
  // step variances, 0.1^2) and h = 1, i.e. the coordinatewise starting steps, joint.
  nsBlkH = 1.0;
  nsBlkM = Eigen::MatrixXd::Zero(K, K);
  for(int k=0; k<K; k++) nsBlkM(k, k) = nsRate[k] * nsRate[k];
  nsBlkL = Eigen::MatrixXd(nsBlkM.llt().matrixL());
  nsBlkMSet = false;
  nsBlkMUpdates = 0;
  nsBlkAcc = nsBlkTried = nsBlkAccTotal = nsBlkTriedTotal = 0.0;
  nsBlkCount = 0.0;
  nsBlkMean = Eigen::VectorXd::Zero(K);
  nsBlkM2 = Eigen::MatrixXd::Zero(K, K);
  nsBlkPairs = nsBlkPairsB = nsBlkPairsDropped = 0.0;
  nsBlkSxx = Eigen::MatrixXd::Zero(K, K);
  nsBlkSyx = Eigen::MatrixXd::Zero(K, K);
  nsBlkSxxB = Eigen::MatrixXd::Zero(K, K);
  nsBlkSyxB = Eigen::MatrixXd::Zero(K, K);
  nsBlkLog.clear();
  nsBlkB0.resize(K); nsBlkB1.resize(K); nsBlkG0.resize(K); nsBlkG1.resize(K);
  nsBlkEps.resize(K); nsBlkDF.resize(K); nsBlkDB.resize(K); nsBlkR.resize(K);
  // Adaptation schedule. h adapts every 100 burn-in sweeps. M is replaced every
  // nsBlkPeriod sweeps once nsBlkMinStats statistics (collected from sweep
  // ceil(nBurnIn/4) on) are available and while >= nsBlkTail burn-in sweeps remain,
  // so h is re-tuned to the final M before burn-in ends. nskip >= 1500 keeps the
  // original 500 / 500 / 500. Shorter burn-ins scale all three down to
  // P = 100 floor(nskip / 400) (a multiple of the h batch, P <= nskip / 4): with every
  // batch from c = ceil(nskip/4) on committed, the first update is due at the first
  // multiple m of P with m >= c + P, and m < c + 2P gives m + P < c + 3P <= nskip + 1,
  // so an M update is due at least once for every nskip in [400, 1500) (batches with
  // acceptance < 0.1 contribute no secant pairs and can delay it). nskip < 400: P = 0,
  // M stays the starting diagonal and only h adapts (wbart() warns).
  nsBlkBurn = nBurnIn;
  if(nBurnIn >= 1500) {
    nsBlkPeriod = nsBlkMinStats = nsBlkTail = 500;
  } else {
    nsBlkPeriod = nsBlkMinStats = nsBlkTail = 100 * (int)(nBurnIn / 400);
  }
  printf("ns.block: preconditioned block MALA on the %d spline coefficients (M from %s)\n",
         K, nsBlkMSource == 1 ? "the secant curvature" : "the beta draws");
  if(nsBlkPeriod > 0)
    printf("ns.block: M replaced every %d burn-in sweeps once >= %d statistics (collected from sweep %zu) while >= %d burn-in sweeps remain; h adapts every 100, h <= %g\n",
           nsBlkPeriod, nsBlkMinStats, (nBurnIn + 3) / 4, nsBlkTail, nsBlkHMax);
  else
    printf("ns.block: nskip = %zu < 400 is too short for the preconditioner schedule -- M stays the starting diagonal, only h adapts\n",
           nBurnIn);
}

Rcpp::IntegerVector survivalBART::getNsBlockSchedule() const {
  Rcpp::IntegerVector v = Rcpp::IntegerVector::create(
    Rcpp::Named("period") = nsBlkPeriod, Rcpp::Named("minStats") = nsBlkMinStats,
    Rcpp::Named("tail") = nsBlkTail, Rcpp::Named("collectFrom") = (int)((nsBlkBurn + 3) / 4));
  return v;
}

Rcpp::NumericMatrix survivalBART::getNsBlockMLog() const {
  const int nc = 11;
  const int nr = (int)(nsBlkLog.size() / nc);
  Rcpp::NumericMatrix L(nr, nc);
  for(int r=0; r<nr; r++)
    for(int c=0; c<nc; c++) L(r, c) = nsBlkLog[(size_t)r * nc + c];
  Rcpp::colnames(L) = Rcpp::CharacterVector::create(
    "iter", "status", "batchAccept", "hBatch", "hAdapted", "factorRaw", "factor", "hNew",
    "stats", "condRaw", "consistency");
  return L;
}

// Burn-in adaptation, every 100 burn-in sweeps (before that sweep's step):
//   h *= exp(accept rate of the last batch - 0.574), clamped to [1e-4, nsBlkHMax];
//   the secant pairs of that batch are committed if its acceptance was >= 0.1 and
//   dropped otherwise (a nearly stuck batch -- h and/or M badly off, proposals far
//   out in the tails -- misrepresents the curvature near the posterior bulk);
// every nsBlkPeriod burn-in sweeps while at least nsBlkTail burn-in sweeps remain
// (so h is re-tuned to the final M before burn-in ends), M is replaced by the
// estimate from the steps of the second quarter of burn-in onward (at least
// nsBlkMinStats), plus a relative ridge:
//   mSource 0: empirical covariance of the beta draws (state after each step);
//   mSource 1: M = H^{-1}, H the symmetrized least-squares fit of
//     y = g(beta) - g(beta*) ~ H (beta* - beta) over the committed steps (accepted or
//     not; both gradients are computed by the step anyway), i.e. a secant estimate of
//     the negative Hessian of log pi(beta | rest) averaged over burn-in; its
//     eigenvalues are floored at 1/v0 (the log-likelihood is concave in beta, so
//     the true negative Hessian is >= I/v0);
// and h is rescaled so that the change of M does not upset the acceptance rate
// (mSource 0: keep the mean proposal variance tr(h^2 M)/K; mSource 1: keep
// sum_i (h^2 lambda_i)^3, lambda_i the eigenvalues of L'HL), with the rescale
// factor clamped to [0.5, 2]. The previous M (and h) are kept when the estimate is
// non-finite, not SPD, numerically ill-conditioned (raw condition number > 1e8), or
// (secant) contradicts the acceptance just observed: after a batch with acceptance
// >= 0.1 under (h, M), a curvature estimate with h^2 lambda_max(L'HL) > 100 (MALA is
// unstable, acceptance ~0, once h^2 lambda > 4 in any direction) is rejected.
// Nothing changes after burn-in (nIteration > nBurnIn never adapts, and an M update
// needs nIteration + nsBlkTail <= nBurnIn).
void survivalBART::nsBlockAdapt(size_t nIteration, size_t nBurnIn) {
  if(!(nIteration > 0 && nIteration <= nBurnIn && nIteration % 100 == 0)) return;
  const int K = nNsBasis;
  const double rate = (nsBlkTried > 0.0) ? nsBlkAcc / nsBlkTried : NA_REAL;
  const double hBatch = nsBlkH;   // the step h of the batch that just ended
  if(nsBlkTried > 0.0) {
    nsBlkH *= exp(1.0 * (rate - 0.574));
    if(nsBlkH < 1e-4) nsBlkH = 1e-4;
    if(nsBlkH > nsBlkHMax) nsBlkH = nsBlkHMax;
  }
  if(nsBlkPairsB > 0.0) {
    if(nsBlkTried > 0.0 && rate >= 0.1) {
      nsBlkSxx += nsBlkSxxB;
      nsBlkSyx += nsBlkSyxB;
      nsBlkPairs += nsBlkPairsB;
    } else {
      nsBlkPairsDropped += nsBlkPairsB;
    }
    nsBlkSxxB.setZero();
    nsBlkSyxB.setZero();
    nsBlkPairsB = 0.0;
  }
  const bool checkpoint = (nsBlkPeriod > 0) ? (nIteration % nsBlkPeriod == 0) : (nIteration % 500 == 0);
  int status = 0;   // 0 no attempt; 1 M replaced; kept: -1 estimate failed / non-finite,
                    // -2 not SPD, -3 ill-conditioned, -4 inconsistent with the acceptance
  if(nsBlkPeriod > 0 && nIteration % nsBlkPeriod == 0 && nIteration + nsBlkTail <= nBurnIn) {
    const double stats = (nsBlkMSource == 1) ? nsBlkPairs : nsBlkCount;
    if(stats >= (double)nsBlkMinStats) {
      Eigen::MatrixXd Mn, Hf;   // new M; (secant) the floored curvature estimate H
      bool est = false;
      if(nsBlkMSource == 0) {
        Mn = nsBlkM2 / (nsBlkCount - 1.0);
        est = true;
      } else {
        Eigen::LLT<Eigen::MatrixXd> llt_x(nsBlkSxx);
        if(llt_x.info() == Eigen::Success) {
          const Eigen::MatrixXd Ht = llt_x.solve(nsBlkSyx.transpose());   // (Syx Sxx^{-1})'
          const Eigen::MatrixXd Hs = 0.5 * (Ht + Ht.transpose());
          Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hs);
          if(es.info() == Eigen::Success && es.eigenvalues().allFinite()) {
            const double lamFloor = 1.0 / pi.nsCoefPriorVar;
            Eigen::VectorXd lam(K), inv(K);
            for(int k=0; k<K; k++) {
              lam(k) = std::max(es.eigenvalues()(k), lamFloor);
              inv(k) = 1.0 / lam(k);
            }
            Mn = es.eigenvectors() * inv.asDiagonal() * es.eigenvectors().transpose();
            Hf = es.eigenvectors() * lam.asDiagonal() * es.eigenvectors().transpose();
            est = true;
          }
        }
      }
      double factorRaw = NA_REAL, factor = NA_REAL, condRaw = NA_REAL, consist = NA_REAL;
      const double hAdapted = nsBlkH;
      if(!est || !Mn.allFinite()) {
        status = -1;
      } else {
        Mn = 0.5 * (Mn + Mn.transpose());
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> em(Mn, Eigen::EigenvaluesOnly);
        if(em.info() == Eigen::Success && em.eigenvalues()(0) > 0.0)
          condRaw = em.eigenvalues()(K - 1) / em.eigenvalues()(0);
        if(!(condRaw <= 1e8)) {   // also NA / NaN / lambda_min <= 0
          status = -3;
        } else {
          const double ridge = 1e-6 * std::max(Mn.trace() / (double)K, 1e-12);
          Mn += ridge * Eigen::MatrixXd::Identity(K, K);
          Eigen::LLT<Eigen::MatrixXd> llt_M(Mn);
          if(llt_M.info() != Eigen::Success) {
            status = -2;
          } else {
            const Eigen::MatrixXd Lnew = llt_M.matrixL();
            double so = NA_REAL, sn = NA_REAL;
            if(nsBlkMSource == 1) {
              // eigenvalues of the preconditioned curvature under the current and the new M
              Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eo(nsBlkL.transpose() * Hf * nsBlkL, Eigen::EigenvaluesOnly);
              Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> en(Lnew.transpose() * Hf * Lnew, Eigen::EigenvaluesOnly);
              if(eo.info() == Eigen::Success && en.info() == Eigen::Success) {
                so = 0.0; sn = 0.0;
                for(int k=0; k<K; k++) {
                  const double a = std::max(eo.eigenvalues()(k), 0.0), b = std::max(en.eigenvalues()(k), 0.0);
                  so += a * a * a;
                  sn += b * b * b;
                }
                consist = hBatch * hBatch * std::max(eo.eigenvalues()(K - 1), 0.0);
                if(nsBlkTried > 0.0 && rate >= 0.1 && !(consist <= 100.0)) status = -4;
              }
            }
            if(status == 0) {
              if(nsBlkMSource == 1 && so > 0.0 && sn > 0.0 && std::isfinite(so) && std::isfinite(sn)) {
                // secant: keep sum_i (h^2 lambda_i)^3 over the eigenvalues lambda_i of the
                // preconditioned curvature L' H L (M = L L'); for a near-Gaussian target the
                // spread of MALA's log acceptance ratio grows like h^6 sum_i lambda_i^3
                factorRaw = std::pow(so / sn, 1.0 / 6.0);
              } else {
                // keep the mean proposal variance tr(h^2 M)/K
                const double oldSize = nsBlkM.trace() / (double)K;
                const double newSize = Mn.trace() / (double)K;
                if(oldSize > 0.0 && newSize > 0.0) factorRaw = std::sqrt(oldSize / newSize);
              }
              factor = std::isfinite(factorRaw) ? std::min(2.0, std::max(0.5, factorRaw)) : 1.0;
              nsBlkH *= factor;
              if(nsBlkH < 1e-4) nsBlkH = 1e-4;
              if(nsBlkH > nsBlkHMax) nsBlkH = nsBlkHMax;
              nsBlkM = Mn;
              nsBlkL = Lnew;
              nsBlkMSet = true;
              nsBlkMUpdates++;
              status = 1;
            }
          }
        }
      }
      const double rec[11] = {(double)nIteration, (double)status, rate, hBatch, hAdapted, factorRaw,
                              factor, nsBlkH, stats, condRaw, consist};
      nsBlkLog.insert(nsBlkLog.end(), rec, rec + 11);
    }
  }
  if(checkpoint) {
    printf("ns block MALA adapted at iter %zu: accept %.3f, h %f (M %s, %d %s%s)\n",
           nIteration, rate, nsBlkH,
           nsBlkMSet ? (nsBlkMSource == 1 ? "secant" : "empirical") : "initial diagonal",
           (int)(nsBlkMSource == 1 ? nsBlkPairs : nsBlkCount), nsBlkMSource == 1 ? "pairs" : "draws",
           status == 1 ? "; M replaced" : (status < 0 ? "; new M rejected, previous kept" : ""));
    printf("  nsBlockM diag:");
    for(int k=0; k<K; k++) printf(" %g", nsBlkM(k, k));
    printf("\n");
  }
  nsBlkAcc = 0.0;
  nsBlkTried = 0.0;
}

void survivalBART::drawNsBlock(size_t nIteration, size_t nBurnIn, rn& gen) {
  gkMemoValid[0] = gkMemoValid[1] = false;   // b_i and allfit changed since the last sweep
  const int K = nNsBasis;
  const double m0 = pi.nsCoefPriorMean, v0 = pi.nsCoefPriorVar;
  const double h = nsBlkH, h2 = h * h;
  Eigen::VectorXd &b0 = nsBlkB0, &b1 = nsBlkB1, &g0 = nsBlkG0, &g1 = nsBlkG1;

  // log pi and its gradient at the current state
  for(int k=0; k<K; k++) b0(k) = nsCoef[k];
  const double ll0 = loglikelihood_grad_ns_all(b0.data(), reCoef, g0.data());
  double lp0 = ll0;
  for(int k=0; k<K; k++) {
    const double d = b0(k) - m0;
    lp0 += -0.5 * d * d / v0;
    g0(k) -= d / v0;
  }

  // proposal beta* = beta + (h^2/2) M g(beta) + h L eps
  nsBlkDF = (0.5 * h2) * (nsBlkM * g0);
  for(int k=0; k<K; k++) nsBlkEps(k) = gen.normal();
  nsBlkR = nsBlkL.triangularView<Eigen::Lower>() * nsBlkEps;   // L eps ~ N(0, M)
  b1 = b0 + nsBlkDF + h * nsBlkR;

  // log pi and its gradient at the proposal
  const double ll1 = loglikelihood_grad_ns_all(b1.data(), reCoef, g1.data());
  double lp1 = ll1;
  for(int k=0; k<K; k++) {
    const double d = b1(k) - m0;
    lp1 += -0.5 * d * d / v0;
    g1(k) -= d / v0;
  }
  nsBlkDB = (0.5 * h2) * (nsBlkM * g1);

  // log q(beta* | beta) and log q(beta | beta*), same M, L and h (constants cancel)
  nsBlkR = b1 - b0 - nsBlkDF;
  const double qF = nsBlkL.triangularView<Eigen::Lower>().solve(nsBlkR).squaredNorm();
  nsBlkR = b0 - b1 - nsBlkDB;
  const double qB = nsBlkL.triangularView<Eigen::Lower>().solve(nsBlkR).squaredNorm();
  const double log_alpha = (lp1 - lp0) + (-0.5 * qB / h2) - (-0.5 * qF / h2);

  const bool accept = (std::log(R::runif(0.0, 1.0)) < log_alpha);   // NaN -> reject
  if(accept) for(int k=0; k<K; k++) nsCoef[k] = b1(k);

  if(nIteration < nBurnIn) {
    nsBlkTried += 1.0;
    if(accept) nsBlkAcc += 1.0;
    // burn-in statistics for M, from the second quarter of burn-in on
    if(4 * nIteration >= nBurnIn) {
      if(nsBlkMSource == 0) {
        const Eigen::VectorXd& bNow = accept ? b1 : b0;
        nsBlkCount += 1.0;
        const Eigen::VectorXd delta = bNow - nsBlkMean;
        nsBlkMean += delta / nsBlkCount;
        nsBlkM2 += delta * (bNow - nsBlkMean).transpose();
      } else if(b1.allFinite() && g1.allFinite() && std::isfinite(ll1)) {
        // secant pair of this step, held in the batch sums until the batch ends
        const Eigen::VectorXd x = b1 - b0;
        const Eigen::VectorXd yv = g0 - g1;
        nsBlkPairsB += 1.0;
        nsBlkSxxB += x * x.transpose();
        nsBlkSyxB += yv * x.transpose();
      }
    }
  } else {
    nsBlkTriedTotal += 1.0;
    if(accept) nsBlkAccTotal += 1.0;
  }
}

double survivalBART::loglikelihood_grad_ns_all(double *nsCoef_arr, double **reCoef_arr, double *grad_all) {
  const int K = nNsBasis;
  if(gkOn){
    // drawNs memo (see gkMemo*): a known state is a hit, a new one one fused pass
    for(int s=0; s<2; s++){
      if(gkMemoValid[s] && std::memcmp(gkMemoCoef[s].data(), nsCoef_arr, sizeof(double) * K) == 0){
        gkMemoLastUsed = s;
        for(int k=0; k<K; k++) grad_all[k] = gkMemoGrad[s][k];
        return gkMemoLL[s];
      }
    }
    const int slot = 1 - gkMemoLastUsed;   // keep the most recently used (current) state
    gkMemoLL[slot] = gkGradNsAll(nsCoef_arr, reCoef_arr, gkMemoGrad[slot].data(), gkMemoIv[slot].data());
    std::memcpy(gkMemoCoef[slot].data(), nsCoef_arr, sizeof(double) * K);
    gkMemoValid[slot] = true;
    gkMemoLastUsed = slot;
    for(int k=0; k<K; k++) grad_all[k] = gkMemoGrad[slot][k];
    return gkMemoLL[slot];
  }
  // Uncached: the per-subject expressions and summation order of the uncached
  // loglikelihood_grad_ns, for every k in one pass (the value integral once, then
  // the K gradient integrals), so the log-lik equals that of loglikelihood_grad_ns
  // and grad_all[k] its gradient for k, bit for bit.
  double loglik_total = 0.0;
  std::vector<double> grad_total(K, 0.0), grad_i(K, 0.0);
  for(int i=0; i<nSubject; i++) {
    int start_time_index = (int)xIndex[i] + 1;
    int end_time_index = (int)xIndex[i+1];
    int num_intervals = end_time_index - start_time_index;
    const double *bart_prediction = &allfit[start_time_index];   // BART factor of each interval

    double ty = y[i];
    double ui_event = xEvent[(int)xIndex[i+1]];

    // A_i(y_i)
    double A_at_y = 0.0;
    for(int l=0; l<K; l++) {
      A_at_y += nsCoef_arr[l] *
        (ns_kth_basis_scalar(ty, nsKnots, nKnots, 3, l, true) - ns_mean[l]);
    }
    for(int m=0; m<nReCoef; m++) {
      A_at_y += reCoef_arr[i][m] * pow(ty / reTimeScale, rePowers[m]);
    }
    double wi_event = ui_event * A_at_y;
    wi_event += log(pow(bart_prediction[num_intervals - 1], ui_event));

    double loglik_i = wi_event;
    for(int k=0; k<K; k++)
      grad_i[k] = ui_event * (ns_kth_basis_scalar(ty, nsKnots, nKnots, 3, k, true) - ns_mean[k]);

    double *re_i = reCoef_arr[i];
    auto f_lik = [&](double t) { return integrand(t, nsCoef_arr, re_i); };
    for(int j=start_time_index; j<end_time_index; j++) {
      double a = xObsTime[j];
      double b = xObsTime[j+1];
      double vij_j = integrateSeg(f_lik, a, b);
      loglik_i -= vij_j * bart_prediction[j - start_time_index];
      for(int k=0; k<K; k++) {
        auto f_grad = [&, k](double t) {
          return (ns_kth_basis_scalar(t, nsKnots, nKnots, 3, k, true) - ns_mean[k])
          * integrand(t, nsCoef_arr, re_i);
        };
        double vij_grad_j = integrateSeg(f_grad, a, b);
        grad_i[k] -= vij_grad_j * bart_prediction[j - start_time_index];
      }
    }

    loglik_total += loglik_i;
    for(int k=0; k<K; k++) grad_total[k] += grad_i[k];
  }
  for(int k=0; k<K; k++) grad_all[k] = grad_total[k];
  return loglik_total;
}

// ============================================================================
// Piecewise baseline: exact conjugate gamma Gibbs for the bin heights.
// Conditional on the trees (allfit) and the random effects b_i, the likelihood
// as a function of lambda_k is
//     lambda_k^{d_k} * exp(-lambda_k * E_k),
//     d_k = # events with T_i in bin k,
//     E_k = sum_{i,j} allfit[j] * int_{(t_j, t_{j+1}) cap bin_k} exp(b_i' z(u)) du
// (lambda_k factors out of every exposure segment because the baseline is
// constant on the bin, and the BART factor multiplies the hazard per interval).
// With the iid Gamma(pwShape, pwRate) prior the full conditional is
//     lambda_k | . ~ Gamma(pwShape + d_k, pwRate + E_k)   (rate parameterization)
// -- the same gamma kernel the tree leaves already use (rhk/shk, mixture-GIG).
// ============================================================================
void survivalBART::drawLambda(rn& gen) {
  const int J = nNsBasis;
  std::vector<double> dk(J, 0.0), Ek(J, 0.0);
  
  // Event counts per bin (data-fixed; O(nSubject), recomputed for simplicity).
  for(int i=0; i<nSubject; i++)
    if(xEvent[(int)xIndex[i+1]] > 0.0) dk[pwBin(y[i])] += 1.0;
    
    // Hazard-weighted exposures per bin: split each inter-observation interval at
    // the cutpoints; within a segment only exp(RE) varies (smooth), so GK61 is
    // effectively exact for any RE power design.
    boost::math::quadrature::gauss_kronrod<double, 61> integrator;
    for(int i=0; i<nSubject; i++){
      int start_time_index = (int)xIndex[i] + 1;
      int end_time_index   = (int)xIndex[i+1];
      double *re_i = reCoef[i];
      auto fre = [&](double u){
        double s = 0.0;
        for(int k2=0; k2<nReCoef; k2++) s += re_i[k2] * pow(u / reTimeScale, rePowers[k2]);
        return std::exp(s);
      };
      for(int j=start_time_index; j<end_time_index; j++){
        double a = xObsTime[j], b = xObsTime[j+1];
        double w = allfit[j];
        double lo = a;
        for(int c=1; c<=nKnots-2 && lo<b; c++){
          double cp = nsKnots[c];
          if(cp <= lo) continue;
          if(cp >= b) break;
          Ek[pwBin(0.5*(lo+cp))] += w * integrator.integrate(fre, lo, cp);
          lo = cp;
        }
        if(b > lo) Ek[pwBin(0.5*(lo+b))] += w * integrator.integrate(fre, lo, b);
      }
    }
    
    for(int k=0; k<J; k++){
      double lam = gen.gamma(pwShape + dk[k], pwRate + Ek[k]);
      if(!(lam > 0.0) || !std::isfinite(lam)) lam = 1e-12;   // guard RNG underflow
      nsCoef[k] = std::log(lam);
    }
}


//update random effect coefficients via MALA
void survivalBART::drawRandomEffect(rn& gen) {
  Eigen::VectorXd nsCoef_temp(nNsBasis);
  Eigen::MatrixXd reCoef_temp(nSubject, nReCoef);
  for(int i=0; i<nNsBasis; i++){
    nsCoef_temp(i) = nsCoef[i];
  }
  for(int i=0; i<nSubject; i++){
    for(int j=0; j<nReCoef; j++){
      reCoef_temp(i, j) = reCoef[i][j];
    }
  }
  
  //prior covariance matrix reCoefCov (shared across subjects), factored once
  mrPriorCov.resize(nReCoef, nReCoef);
  for(int j=0; j<nReCoef; j++){
    for(int k=0; k<nReCoef; k++){
      mrPriorCov(j, k) = reCoefCov[j][k];
    }
  }
  mrLltPrior.compute(mrPriorCov);
  if (mrLltPrior.info() != Eigen::Success) {
    Rcpp::stop("mala_re: prior covariance not positive definite");
  }
  
  double *nsCoef_arr = nsCoef_temp.data();
  
  // nsCoef stays fixed through ncpSigmaUpdate: share the spline part of A_i(t)
  // across every b_i state of the MALA and non-centred moves.
  if(gkOn) gkFillS(nsCoef_arr);
  
  // Interpretation B: prior on stored reCoef[i] is MVN(pi.reCoefPriorMean, reCoefCov),
  // no c_b adjustment.
  mrPriorMean.resize(nReCoef);
  for(int j=0; j<nReCoef; j++){
    mrPriorMean(j) = pi.reCoefPriorMean[j];
  }
  mrPropCov.resize(nReCoef, nReCoef);
  for(int i=0; i<nSubject; i++){
    // reRate[i] is a std-dev multiplier; MALA step covariance = reRate[i]^2 * I.
    mrPropCov.setIdentity();
    mrPropCov = mrPropCov * (reRate[i] * reRate[i]);
    
    mala_re(i, nsCoef_arr, reCoef_temp, mrPropCov, mrPriorMean, mrLltPrior, gen);
  }
  reLoglikValid = true;   // mala_re left ell_i(b_i) of every subject in reLoglik
  return;
}

//update random effect covariance matrix
void survivalBART::drawRandomEffectCov(rn& gen) {
  Eigen::VectorXd reCoef_prior_mean(nReCoef);
  for(int j=0; j<nReCoef; j++){
    reCoef_prior_mean(j) = pi.reCoefPriorMean[j];
  }
  // Interpretation B: IW conjugacy on stored reCoef directly (no c_b add-back).
  Eigen::MatrixXd reCoef_reCoefT(nReCoef, nReCoef);
  reCoef_reCoefT.setZero();
  for(int i=0; i<nSubject; i++){
    Eigen::VectorXd reCoef_temp(nReCoef);
    for(int j=0; j<nReCoef; j++){
      reCoef_temp(j) = reCoef[i][j] - reCoef_prior_mean(j);
    }
    reCoef_reCoefT += reCoef_temp * reCoef_temp.transpose();
  }
  
  // ============================================================
  // Huang-Wand (2013) hierarchical prior — two-block Gibbs (the only Sigma_b
  // prior; the plain single-scale inverse-Wishart option was removed).
  //   SD_k ~ half-t(nu, A_k); at nu=2 the correlations are marginal-uniform.
  //   Auxiliary a_k with the conjugate scheme:
  //     Sigma | b, a ~ IW(nu + q - 1 + N,  2*nu*diag(1/a_k) + S)
  //     a_k   | Sigma ~ InvGamma((nu+q)/2,  nu*(Sigma^{-1})_kk + 1/A_k^2)
  // The a_k let each variance choose its own scale, so a weakly identified
  // sigma_kk is NOT dragged to a fixed prior value.
  // ============================================================
  const int q = nReCoef;
  const double nu = pi.reHWnu;
  
  // Current Sigma^{-1} (reCoefCov holds the previous sweep's draw / init).
  Eigen::MatrixXd Sig(q, q);
  for(int j=0; j<q; j++)
    for(int k=0; k<q; k++)
      Sig(j, k) = reCoefCov[j][k];
  Eigen::LLT<Eigen::MatrixXd> llt_sig(Sig);
  if(llt_sig.info() != Eigen::Success)
    Rcpp::stop("drawRandomEffectCov (HW): current Sigma not positive definite");
  Eigen::MatrixXd SigInv = llt_sig.solve(Eigen::MatrixXd::Identity(q, q));
  
  // Block 1: a_k | Sigma ~ InvGamma((nu+q)/2, nu*(Sigma^{-1})_kk + 1/A_k^2).
  Eigen::MatrixXd Lambda = Eigen::MatrixXd::Zero(q, q);   // diag(1/a_k)
  for(int k=0; k<q; k++){
    double rate = nu * SigInv(k, k) + 1.0 / (pi.reHWA[k] * pi.reHWA[k]);
    double ak = 1.0 / gen.gamma((nu + q) / 2.0, rate);     // rate parameterization
    reCovAux[k] = ak;
    Lambda(k, k) = 1.0 / ak;
  }
  
  // Block 2: Sigma | b, a ~ IW(nu + q - 1 + N, 2*nu*Lambda + S).
  Eigen::MatrixXd newScaleMatrix = 2.0 * nu * Lambda + reCoef_reCoefT;
  // Exact (real-valued) df: sampleInverseWishart takes double, so non-integer
  // reHW.nu is not rounded here; ncpSigmaUpdate uses the same exact nu + q - 1.
  double newDf = nu + (double)q - 1.0 + (double)nSubject;
  Eigen::MatrixXd reCoefCov_temp = sampleInverseWishart(newScaleMatrix, newDf, gen);
  for(int j=0; j<q; j++)
    for(int k=0; k<q; k++)
      reCoefCov[j][k] = reCoefCov_temp(j, k);
  return;
}

// ============================================================================
// Subject-i log-likelihood ell_i(b_i), value only. Same expression as the value
// returned by loglikelihood_grad_re (including the b-free event-time terms), so
// the two can be differenced; the gradient integrals are skipped.
// ============================================================================
double survivalBART::loglikelihood_re(int subjectId, double *nsCoef_arr,
                                      double *reCoef_arr) {
  const int start_time_index = (int)xIndex[subjectId] + 1;
  const int end_time_index = (int)xIndex[subjectId+1];
  const double ty = y[subjectId];
  const double ui_event = xEvent[(int)xIndex[subjectId+1]];
  
  if(evOn) {
    double loglik = ui_event * evReAt(subjectId, evSplineAt(subjectId, nsCoef_arr), reCoef_arr);
    loglik += evLogFAt(subjectId);
    if(gkSHolds(nsCoef_arr)) {   // ncp move: spline part shared with the RE step
      for(int j=start_time_index; j<end_time_index; j++)
        loglik -= gkValueS(j, nsCoef_arr, reCoef_arr) * allfit[j];
    } else {
      for(int j=start_time_index; j<end_time_index; j++)
        loglik -= gkValue(j, nsCoef_arr, reCoef_arr) * allfit[j];
    }
    return loglik;
  }
  
  double A_at_y = 0.0;
  for(int k=0; k<nNsBasis; k++) {
    A_at_y += nsCoef_arr[k] *
      (ns_kth_basis_scalar(ty, nsKnots, nKnots, 3, k, true) - ns_mean[k]);
  }
  for(int m=0; m<nReCoef; m++) {
    A_at_y += reCoef_arr[m] * pow(ty / reTimeScale, rePowers[m]);
  }
  double loglik = ui_event * A_at_y;
  loglik += log(pow(allfit[end_time_index - 1], ui_event));
  
  auto f_lik = [&](double t) { return integrand(t, nsCoef_arr, reCoef_arr); };
  for(int j=start_time_index; j<end_time_index; j++) {
    if(gkOn) loglik -= gkValue(j, nsCoef_arr, reCoef_arr) * allfit[j];
    else     loglik -= integrateSeg(f_lik, xObsTime[j], xObsTime[j+1]) * allfit[j];
  }
  return loglik;
}

// ============================================================================
// Burn-in adaptation of the non-centred move, called once per sweep. Every 100
// burn-in sweeps the global scale is moved toward the random-walk optimum (0.44
// in one dimension, ~0.30 for q = 2, 0.25 beyond); every 500 burn-in sweeps the
// proposal shape is reset to the empirical covariance of the xi draws collected
// from the second quarter of burn-in onward (once at least 500 are available),
// but only while at least 500 burn-in sweeps remain, so the scale is re-tuned to
// the final shape before burn-in ends. Nothing is adapted after burn-in, so the
// retained draws come from a fixed Metropolis-Hastings kernel.
// ============================================================================
void survivalBART::ncpAdapt(size_t nIteration, size_t nBurnIn) {
  if(!(nIteration > 0 && nIteration <= nBurnIn && nIteration % 100 == 0)) return;
  const int dxi = nReCoef * (nReCoef + 1) / 2;
  if(ncpTried > 0.0) {
    const double rate = ncpAccept / ncpTried;
    const double target = (dxi == 1) ? 0.44 : ((dxi <= 3) ? 0.30 : 0.25);
    ncpScale *= exp(1.0 * (rate - target));
    if(ncpScale < 1e-4) ncpScale = 1e-4;
    if(ncpScale > 1e2)  ncpScale = 1e2;
  }
  if(nIteration % 500 == 0 && ncpCount >= 500.0 && nIteration + 500 <= nBurnIn) {
    Eigen::MatrixXd S = ncpM2 / (ncpCount - 1.0);
    S = 0.5 * (S + S.transpose());
    const double ridge = 1e-6 * std::max(1e-2, S.trace() / (double)dxi);
    S += ridge * Eigen::MatrixXd::Identity(dxi, dxi);
    Eigen::LLT<Eigen::MatrixXd> llt_S(S);
    if(llt_S.info() == Eigen::Success) {
      // Keep the proposal's overall size when the shape changes (the scale then
      // keeps adapting from there): rescale by the ratio of mean variances.
      const double oldSize = ncpShape.trace() / (double)dxi;
      const double newSize = S.trace() / (double)dxi;
      if(oldSize > 0.0 && newSize > 0.0) ncpScale *= std::sqrt(oldSize / newSize);
      if(ncpScale < 1e-4) ncpScale = 1e-4;
      if(ncpScale > 1e2)  ncpScale = 1e2;
      ncpShapeSet = true;
      ncpShape = S;
    }
  }
  if(nIteration % 500 == 0)
    printf("ncp move adapted at iter %zu: accept %.3f, scale %f (shape %s, %d draws)\n",
           nIteration, (ncpTried > 0.0) ? ncpAccept / ncpTried : NA_REAL, ncpScale,
           ncpShapeSet ? "empirical" : "identity", (int)ncpCount);
  ncpAccept = 0.0;
  ncpTried = 0.0;
}

// ============================================================================
// Joint non-centred Metropolis move for (Sigma_b, b_1, ..., b_N).
//
// Coordinates: Sigma_b = L L' with L lower triangular, positive diagonal, and
//   xi = (log L_11, L_21, ..., L_q1, log L_22, L_32, ..., log L_qq)
// (lower triangle by columns, diagonal on the log scale).
// Proposal: xi* = xi + zeta, zeta ~ N(0, ncpScale^2 ncpShape) (symmetric), and
//   b_i* = mu_b + L* L^{-1} (b_i - mu_b)      for every subject i,
// so the standardized effects L^{-1}(b_i - mu_b) stay fixed. The reverse move is
// -zeta, which maps (xi*, b*) back to (xi, b). The Jacobian of the b-map,
// (|L*| / |L|)^N, cancels the ratio of Gaussian random-effects densities
// prod_i N(b_i*; mu_b, Sigma*) / N(b_i; mu_b, Sigma) = (|L| / |L*|)^N, so
//   log alpha = sum_i [ ell_i(b_i*) - ell_i(b_i) ] + log p(xi*) - log p(xi),
// where p(xi) is the Huang-Wand conditional prior
//   Sigma_b | a ~ IW(m, Psi),  m = nu + q - 1,  Psi = 2 nu diag(1/a_k),
// written in xi coordinates:
//   log p(xi) = -((m + q + 1)/2) log|Sigma| - tr(Psi Sigma^{-1}) / 2 + log|J(xi)|,
//   log|Sigma| = 2 sum_k log L_kk,
//   |J(xi)| = |d vech(Sigma) / d xi| = 2^q prod_{k=1..q} L_kk^{q-k+2}.
// There is no Gaussian random-effects density in the ratio: it cancels against
// the Jacobian of the b-transformation. The move leaves the joint conditional of
// (Sigma_b, b) given the trees, baseline and a invariant, needs one likelihood
// evaluation per subject per proposal (ell_i at the current b_i is cached by
// mala_re), and no quadrature over b.
// ============================================================================
void survivalBART::ncpSigmaUpdate(size_t nIteration, size_t nBurnIn, rn& gen) {
  const int q = nReCoef;
  if(q < 1) return;
  const int dxi = q * (q + 1) / 2;
  const Eigen::MatrixXd Iq = Eigen::MatrixXd::Identity(q, q);
  
  Eigen::MatrixXd Sig(q, q);
  for(int j=0; j<q; j++)
    for(int k=0; k<q; k++)
      Sig(j, k) = reCoefCov[j][k];
  Eigen::LLT<Eigen::MatrixXd> llt_sig(Sig);
  if(llt_sig.info() != Eigen::Success)
    Rcpp::stop("ncpSigmaUpdate: current Sigma_b not positive definite");
  const Eigen::MatrixXd L = llt_sig.matrixL();
  
  // xi <-> L (lower triangle by columns, log diagonal)
  Eigen::VectorXd xi(dxi);
  for(int k=0, r=0; k<q; k++)
    for(int j=k; j<q; j++, r++)
      xi(r) = (j == k) ? std::log(L(j, k)) : L(j, k);
  
  if(!ncpShapeInit) {
    // Starting shape: unit variance for the log-diagonal coordinates and L_jj^2
    // for the off-diagonal L_jk (their natural scale); replaced by the empirical
    // burn-in covariance once available.
    for(int k=0, r=0; k<q; k++)
      for(int j=k; j<q; j++, r++)
        ncpShape(r, r) = (j == k) ? 1.0 : L(j, j) * L(j, j);
    ncpShapeInit = true;
  }
  
  Eigen::VectorXd zeta(dxi);
  for(int r=0; r<dxi; r++) zeta(r) = gen.normal();
  Eigen::LLT<Eigen::MatrixXd> llt_shape(ncpShape);
  const Eigen::VectorXd step = llt_shape.matrixL() * zeta;
  const Eigen::VectorXd xiStar = xi + ncpScale * step;
  
  Eigen::MatrixXd Ls = Eigen::MatrixXd::Zero(q, q);
  for(int k=0, r=0; k<q; k++)
    for(int j=k; j<q; j++, r++)
      Ls(j, k) = (j == k) ? std::exp(xiStar(r)) : xiStar(r);
  
  // Huang-Wand conditional prior of Sigma_b in xi coordinates (up to a constant).
  const double nu = pi.reHWnu;
  const double mIW = nu + (double)q - 1.0;
  auto log_prior_xi = [&](const Eigen::MatrixXd& Lm) {
    const Eigen::MatrixXd Linv = Lm.triangularView<Eigen::Lower>().solve(Iq);
    const Eigen::MatrixXd SigInv = Linv.transpose() * Linv;
    double lp = 0.0;
    for(int k=0; k<q; k++) {
      const double log_lkk = std::log(Lm(k, k));
      lp += -(mIW + q + 1.0) * log_lkk;           // -((m+q+1)/2) * 2 log L_kk
      lp += (double)(q - k + 1) * log_lkk;        // log|J(xi)|, k 0-based
      lp += -0.5 * (2.0 * nu / reCovAux[k]) * SigInv(k, k);   // Psi diagonal
    }
    return lp;
  };
  
  if(!reLoglikValid) {
    for(int i=0; i<nSubject; i++) reLoglik[i] = loglikelihood_re(i, nsCoef, reCoef[i]);
    reLoglikValid = true;
  }
  
  // Move every b_i with Sigma_b: b_i* - mu_b = L* L^{-1} (b_i - mu_b).
  const Eigen::MatrixXd Mmap = Ls * L.triangularView<Eigen::Lower>().solve(Iq);
  std::vector<double> bStar((size_t)nSubject * q), llStar(nSubject);
  double dLL = 0.0;
  for(int i=0; i<nSubject; i++) {
    double *bs = &bStar[(size_t)i * q];
    for(int j=0; j<q; j++) {
      double s = 0.0;
      for(int k=0; k<q; k++) s += Mmap(j, k) * (reCoef[i][k] - pi.reCoefPriorMean[k]);
      bs[j] = pi.reCoefPriorMean[j] + s;
    }
    llStar[i] = loglikelihood_re(i, nsCoef, bs);
    dLL += llStar[i] - reLoglik[i];
  }
  const double log_alpha = dLL + log_prior_xi(Ls) - log_prior_xi(L);
  const bool accept = (std::log(R::runif(0.0, 1.0)) < log_alpha);   // NaN -> reject
  if(accept) {
    for(int i=0; i<nSubject; i++) {
      for(int j=0; j<q; j++) reCoef[i][j] = bStar[(size_t)i * q + j];
      reLoglik[i] = llStar[i];
    }
    const Eigen::MatrixXd Ss = Ls * Ls.transpose();
    for(int j=0; j<q; j++)
      for(int k=0; k<q; k++)
        reCoefCov[j][k] = 0.5 * (Ss(j, k) + Ss(k, j));
  }
  
  if(nIteration < nBurnIn) {
    ncpTried += 1.0;
    if(accept) ncpAccept += 1.0;
    // Collect the post-move xi for the empirical proposal covariance (from the
    // second quarter of burn-in on, to skip the initial transient).
    if(4 * nIteration >= nBurnIn) {
      const Eigen::VectorXd& xiNow = accept ? xiStar : xi;
      ncpCount += 1.0;
      const Eigen::VectorXd delta = xiNow - ncpMean;
      ncpMean += delta / ncpCount;
      ncpM2 += delta * (xiNow - ncpMean).transpose();
    }
  } else {
    ncpTriedTotal += 1.0;
    if(accept) ncpAcceptTotal += 1.0;
  }
}


//for subejct i, the likelihood is
double survivalBART::loglikelihood(double *y, double *xEvent, double *xObsTime, double *xIndex, int subjectId, double *nsCoef_temp, double *reCoef_temp){
  int start_time_index = xIndex[subjectId] + 1;
  int end_time_index = xIndex[subjectId+1];
  int num_intervals = end_time_index - start_time_index;
  
  double *bart_prediction = new double[num_intervals];
  for(int j=0; j<num_intervals; j++){
    bart_prediction[j] = allfit[(int)xIndex[subjectId] + 1 + j];
  }
  
  double loglikelihood = 0;
  double ty = y[subjectId];
  double wi_temp = 0;
  for(int k=0; k<nNsBasis; k++){
    wi_temp += nsCoef_temp[k] * (ns_kth_basis_scalar(ty, nsKnots, nKnots, 3, k, true) - ns_mean[k]);
  }
  splineY[subjectId] = wi_temp;
  for(int j=0; j<nReCoef; j++){
    wi_temp += reCoef_temp[j] * pow(ty / reTimeScale, rePowers[j]);
  }
  wi_temp = wi_temp * xEvent[(int)xIndex[subjectId+1]];
  
  wi_temp += log(pow(bart_prediction[num_intervals - 1], xEvent[(int)xIndex[subjectId+1]]));
  
  double *vij_temp = new double[num_intervals];
  auto f = [&](double t) { return integrand(t, nsCoef_temp, reCoef_temp); };
  for(int j=start_time_index; j<end_time_index; j++){
    double a = xObsTime[j];
    double b = xObsTime[j+1];
    vij_temp[j - start_time_index] = integrateSeg(f, a, b);
  }
  
  loglikelihood = wi_temp;
  for (int j = 0; j < num_intervals; j++) {
    loglikelihood -= vij_temp[j] * bart_prediction[j];
  }
  
  delete[] vij_temp;
  delete[] bart_prediction;
  return loglikelihood;
}

// =====================================================================
// MALA helpers: log-likelihood with analytic gradient
// =====================================================================
//
// Notation: A_i(t) = sum_l nsCoef[l] * (phi_l(t) - ns_mean[l])
//                  + sum_m reCoef[i,m] * t^m
// Integrand is exp(A_i(t)); vij_ij = integral over (xObsTime[j], xObsTime[j+1]).
// Log-lik per subject i (as in loglikelihood()):
//   L_i = u_i * A_i(y_i) + u_i * log(bart_pred_last)
//        - sum_j bart_pred[j] * vij_ij
// with u_i = xEvent[xIndex[i+1]] the event indicator.

// d L_total / d nsCoef[k] (summed over all subjects, since nsCoef is shared)
double survivalBART::loglikelihood_grad_ns(int k, double *nsCoef_arr,
                                           double **reCoef_arr, double& grad_out) {
  if(gkOn){
    // drawNs memo (see gkMemo*): the current state is a hit, a proposal one pass.
    for(int s=0; s<2; s++){
      if(gkMemoValid[s] && std::memcmp(gkMemoCoef[s].data(), nsCoef_arr, sizeof(double) * nNsBasis) == 0){
        gkMemoLastUsed = s;
        grad_out = gkMemoGrad[s][k];
        return gkMemoLL[s];
      }
    }
    const int slot = 1 - gkMemoLastUsed;   // keep the most recently used (current) state
    gkMemoLL[slot] = gkGradNsAll(nsCoef_arr, reCoef_arr, gkMemoGrad[slot].data(), gkMemoIv[slot].data());
    std::memcpy(gkMemoCoef[slot].data(), nsCoef_arr, sizeof(double) * nNsBasis);
    gkMemoValid[slot] = true;
    gkMemoLastUsed = slot;
    grad_out = gkMemoGrad[slot][k];
    return gkMemoLL[slot];
  }
  double loglik_total = 0.0;
  double grad_total = 0.0;
  
  for(int i=0; i<nSubject; i++) {
    int start_time_index = (int)xIndex[i] + 1;
    int end_time_index = (int)xIndex[i+1];
    int num_intervals = end_time_index - start_time_index;
    
    double *bart_prediction = new double[num_intervals];
    for(int j=0; j<num_intervals; j++) {
      bart_prediction[j] = allfit[(int)xIndex[i] + 1 + j];
    }
    
    double ty = y[i];
    double ui_event = xEvent[(int)xIndex[i+1]];
    
    // A_i(y_i)
    double A_at_y = 0.0;
    for(int l=0; l<nNsBasis; l++) {
      A_at_y += nsCoef_arr[l] *
        (ns_kth_basis_scalar(ty, nsKnots, nKnots, 3, l, true) - ns_mean[l]);
    }
    for(int m=0; m<nReCoef; m++) {
      A_at_y += reCoef_arr[i][m] * pow(ty / reTimeScale, rePowers[m]);
    }
    double wi_event = ui_event * A_at_y;
    wi_event += log(pow(bart_prediction[num_intervals - 1], ui_event));
    
    double loglik_i = wi_event;
    double grad_i = ui_event *
      (ns_kth_basis_scalar(ty, nsKnots, nKnots, 3, k, true) - ns_mean[k]);
    
    auto f_lik = [&](double t) { return integrand(t, nsCoef_arr, reCoef_arr[i]); };
    auto f_grad = [&, k](double t) {
      return (ns_kth_basis_scalar(t, nsKnots, nKnots, 3, k, true) - ns_mean[k])
      * integrand(t, nsCoef_arr, reCoef_arr[i]);
    };
    
    for(int j=start_time_index; j<end_time_index; j++) {
      double a = xObsTime[j];
      double b = xObsTime[j+1];
      double vij_j      = integrateSeg(f_lik,  a, b);
      double vij_grad_j = integrateSeg(f_grad, a, b);
      loglik_i -= vij_j      * bart_prediction[j - start_time_index];
      grad_i   -= vij_grad_j * bart_prediction[j - start_time_index];
    }
    
    loglik_total += loglik_i;
    grad_total   += grad_i;
    delete[] bart_prediction;
  }
  
  grad_out = grad_total;
  return loglik_total;
}

// d L_i / d reCoef[i,:] (subject-specific — other subjects don't depend on reCoef[i,:])
double survivalBART::loglikelihood_grad_re(int subjectId, double *nsCoef_arr,
                                           double *reCoef_arr,
                                           Eigen::VectorXd& grad_out) {
  int start_time_index = (int)xIndex[subjectId] + 1;
  int end_time_index = (int)xIndex[subjectId+1];
  int num_intervals = end_time_index - start_time_index;
  
  const double *bart_prediction = &allfit[start_time_index];   // BART factor of each interval
  
  double ty = y[subjectId];
  double ui_event = xEvent[(int)xIndex[subjectId+1]];
  
  double loglik;
  if(evOn) {
    const double wi_ns = evSplineAt(subjectId, nsCoef_arr);
    splineY[subjectId] = wi_ns;
    double wi_event = ui_event * evReAt(subjectId, wi_ns, reCoef_arr);
    wi_event += evLogFAt(subjectId);
    loglik = wi_event;
    const double* bZ = &evZ[(size_t)subjectId * nReCoef];
    grad_out = Eigen::VectorXd::Zero(nReCoef);
    for(int k=0; k<nReCoef; k++) grad_out(k) = ui_event * bZ[k];
  } else {
    double wi_ns = 0.0;
    for(int k=0; k<nNsBasis; k++) {
      wi_ns += nsCoef_arr[k] *
        (ns_kth_basis_scalar(ty, nsKnots, nKnots, 3, k, true) - ns_mean[k]);
    }
    splineY[subjectId] = wi_ns;
    
    double A_at_y = wi_ns;
    for(int m=0; m<nReCoef; m++) {
      A_at_y += reCoef_arr[m] * pow(ty / reTimeScale, rePowers[m]);
    }
    double wi_event = ui_event * A_at_y;
    wi_event += log(pow(bart_prediction[num_intervals - 1], ui_event));
    
    loglik = wi_event;
    grad_out = Eigen::VectorXd::Zero(nReCoef);
    for(int k=0; k<nReCoef; k++) {
      grad_out(k) = ui_event * pow(ty / reTimeScale, rePowers[k]);
    }
  }
  
  auto f_lik = [&](double t) { return integrand(t, nsCoef_arr, reCoef_arr); };
  const bool useS = gkOn && gkSHolds(nsCoef_arr);   // MALA on b_i: spline part shared

  for(int j=start_time_index; j<end_time_index; j++) {
    if(gkOn) {
      // value + all RE-gradient integrals in one fused pass
      double out[gk61::kMaxComp];
      auto fac = [&](int iv, int q, double e, double* o) {
        const double* Z = &gkZ[((size_t)iv * gk61::kNodes + q) * nReCoef];
        o[0] = e;
        for(int k=0; k<nReCoef; k++) o[1 + k] = Z[k] * e;
      };
      auto ref = [&](int c, double t) {
        if(c == 0) return integrand(t, nsCoef_arr, reCoef_arr);
        return pow(t / reTimeScale, rePowers[c - 1]) * integrand(t, nsCoef_arr, reCoef_arr);
      };
      if(useS) gkIntegrals(j, 1 + nReCoef, [this, reCoef_arr](int iv, int q) { return gkExpS(iv, q, reCoef_arr); }, fac, ref, out);
      else     gkIntegrals(j, 1 + nReCoef, [this, nsCoef_arr, reCoef_arr](int iv, int q) { return gkExp(iv, q, nsCoef_arr, reCoef_arr); }, fac, ref, out);
      loglik -= out[0] * bart_prediction[j - start_time_index];
      for(int k=0; k<nReCoef; k++) {
        grad_out(k) -= out[1 + k] * bart_prediction[j - start_time_index];
      }
      continue;
    }
    double a = xObsTime[j];
    double b = xObsTime[j+1];
    double vij_j = integrateSeg(f_lik, a, b);
    loglik -= vij_j * bart_prediction[j - start_time_index];
    
    for(int k=0; k<nReCoef; k++) {
      auto f_grad_k = [&, k](double t) {
        return pow(t / reTimeScale, rePowers[k]) * integrand(t, nsCoef_arr, reCoef_arr);
      };
      double vij_grad_k = integrateSeg(f_grad_k, a, b);
      grad_out(k) -= vij_grad_k * bart_prediction[j - start_time_index];
    }
  }
  
  return loglik;
}

// =====================================================================
// MALA update for a single nsCoef[k] (scalar).
//   Proposal: theta* = theta + (sigma^2/2) * g(theta) + sigma * epsilon
//   Accept with log ratio
//     log pi(theta*) - log pi(theta) + log q(theta | theta*) - log q(theta* | theta)
//   where q(.|.) is Normal(theta + (sigma^2/2) g, sigma^2).
// =====================================================================
void survivalBART::mala_ns(int k, Eigen::VectorXd& nsCoef_temp,
                           double **reCoef_temp_array, double step_sigma2,
                           double prior_mean, double prior_var, rn& gen) {
  double theta_curr = nsCoef_temp(k);
  
  // --- Log-posterior + gradient at current state ---
  double grad_loglik_curr;
  double loglik_curr =
    loglikelihood_grad_ns(k, nsCoef_temp.data(), reCoef_temp_array, grad_loglik_curr);
  // 1-D Normal prior; normalization cancels between current and proposed.
  double log_prior_curr = -0.5 * (theta_curr - prior_mean) * (theta_curr - prior_mean) / prior_var;
  double grad_prior_curr = -(theta_curr - prior_mean) / prior_var;
  double logpost_curr = loglik_curr + log_prior_curr;
  double grad_curr = grad_loglik_curr + grad_prior_curr;
  
  // --- Propose theta* ---
  double drift_fwd = 0.5 * step_sigma2 * grad_curr;
  double theta_prop = theta_curr + drift_fwd + std::sqrt(step_sigma2) * gen.normal();
  
  // Place proposal in nsCoef_temp so the log-lik eval sees the updated value.
  nsCoef_temp(k) = theta_prop;
  
  // --- Log-posterior + gradient at proposed state ---
  double grad_loglik_prop;
  double loglik_prop =
    loglikelihood_grad_ns(k, nsCoef_temp.data(), reCoef_temp_array, grad_loglik_prop);
  double log_prior_prop = -0.5 * (theta_prop - prior_mean) * (theta_prop - prior_mean) / prior_var;
  double grad_prior_prop = -(theta_prop - prior_mean) / prior_var;
  double logpost_prop = loglik_prop + log_prior_prop;
  double grad_prop = grad_loglik_prop + grad_prior_prop;
  
  // --- Log proposal densities (normalization cancels) ---
  double drift_back = 0.5 * step_sigma2 * grad_prop;
  double diff_fwd  = theta_prop - theta_curr - drift_fwd;   // = sigma * epsilon
  double diff_back = theta_curr - theta_prop - drift_back;
  double log_q_fwd  = -0.5 * diff_fwd  * diff_fwd  / step_sigma2;
  double log_q_back = -0.5 * diff_back * diff_back / step_sigma2;
  
  // --- Accept / reject ---
  double log_alpha = logpost_prop - logpost_curr + log_q_back - log_q_fwd;
  double u = R::runif(0.0, 1.0);
  if (log(u) < log_alpha) {
    nsAccept[k] = nsAccept[k] + 1;
    nsCoef[k] = theta_prop;
    // nsCoef_temp(k) already holds theta_prop
  } else {
    nsCoef_temp(k) = theta_curr;   // restore
  }
}

// =====================================================================
// MALA update for reCoef[subjectId, :] (vector of dim nReCoef).
// Proposal covariance Sigma is passed in (here reRate[i]^2 * I); the prior
// covariance comes factored (llt_prior, once per sweep in drawRandomEffect).
// Vectors live in the mr* members, so a call allocates nothing; every Eigen
// expression evaluates with the same kernels as the former fresh temporaries.
// =====================================================================
void survivalBART::mala_re(int subjectId, double *nsCoef_arr,
                           Eigen::MatrixXd& reCoef_temp,
                           const Eigen::MatrixXd& proposal_covar,
                           const Eigen::VectorXd& prior_mean,
                           const Eigen::LLT<Eigen::MatrixXd>& llt_prior, rn& gen) {
  int p = nReCoef;
  
  mrLltProp.compute(proposal_covar);
  if (mrLltProp.info() != Eigen::Success) {
    Rcpp::stop("mala_re: proposal covariance not positive definite");
  }
  const Eigen::LLT<Eigen::MatrixXd>& llt_prop = mrLltProp;
  
  Eigen::VectorXd& theta_curr = mrThetaCurr;
  theta_curr = reCoef_temp.row(subjectId).transpose();
  
  // --- Current state ---
  double loglik_curr =
    loglikelihood_grad_re(subjectId, nsCoef_arr, theta_curr.data(), mrGradLLCurr);
  mrDiff = theta_curr - prior_mean;
  mrZ = llt_prior.matrixL().solve(mrDiff);
  double log_prior_curr = -0.5 * mrZ.squaredNorm();            // drop normalization
  mrGradPrior = llt_prior.solve(mrDiff);
  mrGradPrior = -mrGradPrior;                                  // -Sigma^{-1} (theta - mu)
  double logpost_curr = loglik_curr + log_prior_curr;
  mrGradCurr = mrGradLLCurr + mrGradPrior;
  
  // --- Propose: theta* = theta + (Sigma/2) g + L epsilon ---
  Eigen::VectorXd& drift_fwd = mrDriftFwd;
  drift_fwd.noalias() = 0.5 * proposal_covar * mrGradCurr;
  mrNoise.resize(p);
  for(int j=0; j<p; j++) mrNoise(j) = gen.normal();
  Eigen::VectorXd& theta_prop = mrThetaProp;
  // reference form: Eigen evaluates "xpr + Product" as dst = xpr; dst += L * noise
  theta_prop.noalias() = theta_curr + drift_fwd + llt_prop.matrixL() * mrNoise;
  
  // Place proposal in reCoef_temp for log-lik eval.
  for(int j=0; j<p; j++) reCoef_temp(subjectId, j) = theta_prop(j);
  
  // --- Proposed state ---
  double loglik_prop =
    loglikelihood_grad_re(subjectId, nsCoef_arr, theta_prop.data(), mrGradLLProp);
  mrDiff = theta_prop - prior_mean;
  mrZ = llt_prior.matrixL().solve(mrDiff);
  double log_prior_prop = -0.5 * mrZ.squaredNorm();
  mrGradPrior = llt_prior.solve(mrDiff);
  mrGradPrior = -mrGradPrior;
  double logpost_prop = loglik_prop + log_prior_prop;
  mrGradProp = mrGradLLProp + mrGradPrior;
  
  // --- Log proposal densities (normalization cancels) ---
  mrDriftBack.noalias() = 0.5 * proposal_covar * mrGradProp;
  mrDiffFwd = theta_prop - theta_curr - drift_fwd;
  mrDiffBack = theta_curr - theta_prop - mrDriftBack;
  mrZ = llt_prop.matrixL().solve(mrDiffFwd);
  double log_q_fwd  = -0.5 * mrZ.squaredNorm();
  mrZ = llt_prop.matrixL().solve(mrDiffBack);
  double log_q_back = -0.5 * mrZ.squaredNorm();
  
  // --- Accept / reject ---
  double log_alpha = logpost_prop - logpost_curr + log_q_back - log_q_fwd;
  double u = R::runif(0.0, 1.0);
  if (log(u) < log_alpha) {
    reAccept[subjectId] = reAccept[subjectId] + 1;
    for(int j=0; j<p; j++) {
      reCoef[subjectId][j] = theta_prop(j);
      // reCoef_temp(subjectId, j) already updated
    }
    reLoglik[subjectId] = loglik_prop;
  } else {
    for(int j=0; j<p; j++) {
      reCoef_temp(subjectId, j) = theta_curr(j);   // restore
    }
    reLoglik[subjectId] = loglik_curr;
  }
}

//return ns coefficients
double* survivalBART::getNsCoef() {
  return nsCoef;
}

//return random effect coefficients
double** survivalBART::getReCoef() {
  return reCoef;
}

//return random effect covariance matrix
double** survivalBART::getReCoefCov() {
  return reCoefCov;
}

//return Huang-Wand auxiliary scales a_k
double* survivalBART::getReCovAux() {
  return reCovAux;
}

//return BART mean predictions
double survivalBART::getBARTMean() {
  return bartMean;
}

//return splineY
double* survivalBART::getSplineY() {
  return splineY;
}

//return NsMeab
double* survivalBART::getNsMean() {
  return ns_mean;
}
