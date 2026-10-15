# TVSurvBART 0.0.3.1 (2026-10-14)

## Package renamed from CoxBARTserver to TVSurvBART

* The package is now called `TVSurvBART`: load it with `library(TVSurvBART)`. The sampler and the prediction code are those of the CoxBARTserver 0.0.3.1 engine: with the same call and seed, fits and predictions are identical bit for bit.
* The compiled routines are registered as `_TVSurvBART_cwbart`, `_TVSurvBART_cpwbart` and `_TVSurvBART_cpwbart_grid` (`Rcpp::compileAttributes()` regenerated `RcppExports`).
* `dplyr`, `mvtnorm` and `nnet` are declared in `Imports`, and the `dplyr` / `nnet` functions that `wbart()` and `pwbart()` use are imported in `NAMESPACE`. Users no longer need to attach any package before calling `wbart()`, and factor covariates no longer fail with `could not find function "class.ind"`.
* `BH` is declared in `LinkingTo`: the sampler includes Boost headers (special functions and Gauss-Kronrod quadrature), which were previously found only through a local include path. The package needs R >= 4.1 (C++14 or later).
* DESCRIPTION and the package help page carry the real title, authors and URLs; the unused `rcpp_hello_world` help page from the package skeleton was removed.
* License changed from GPL (>= 2) to GPL (>= 3), which the GPL (>= 2) code of the BART package allows; the full text is in `LICENSE.md`.
* Removed bundled code the package never used (the MCMC, BaseMatrixOps and GSL headers and the unused lambda, rtgamma and rtnorm sources) and the tibble dependency. `wbart()` no longer prints its R-side progress lines and no longer evaluates the unused Gaussian-BART and DART arguments (`sigest`, `sigdf`, `sigquant`, `k`, `sigmaf`, `lambda`, `rho`), so `lambda = NA` no longer stops with an error; the sampler's start-up printout shows fixed placeholder values for them. Fits and predictions are unchanged bit for bit.
* New help pages for `wbart()`, `pwbart()`, `pwbart_grid()`, `predict.wbart()` and `convert_start_stop()` document every argument and the returned components.

## Tree-proposal fix (CoxBARTserver 0.0.3.1, 2026-10-04)

* `bprop()` (`src/bartfuns.cpp`) decided whether the parent of the chosen leaf is a nog node with `ntype() == 'n'`. `ntype()` returns `'t'` for the root, so in a two-leaf tree the root was not recognised as a nog. The reverse-move probability of a 2 -> 3 leaf birth was then 1/2 instead of 1, which halved that move's Metropolis-Hastings ratio. The test is now `isnog()`. The bug is inherited from CRAN BART. Draws differ from 0.0.3 for the same seed.

# CoxBARTserver 0.0.3 (2026-09-30)

* Consistent thinning: with `keepevery > 1`, every per-draw output (tree draws, baseline, random-effect and covariance draws, cumulative hazards) is stored at the same sweeps, so prediction pairs each tree draw with the parameters of the same sweep. `pwbart()` refuses fits whose draw counts do not pair. Fits with `keepevery = 1` are unchanged.
* `ns.block = TRUE`: optional preconditioned block MALA update of the spline baseline coefficients (off by default; the environment variable `TVSB_NSBLOCK` also switches it on).

# CoxBARTserver 0.0.2 (2026-09-27)

* `re.ncp` (default `1L`): an exact joint non-centred Metropolis move on (Sigma_b, b) in log-Cholesky coordinates, run after the Huang-Wand Gibbs update. It replaces the approximate collapsed move of `re.marginal`; `re.marginal`, `re.marginal.gh` and `re.marginal.burnin` were removed.
* The unused residual-variance draw (`$sigma`) was removed, which changes the random-number stream; same-seed draws therefore differ from 0.0.1.
* Compiled with `-ffp-contract=off` (no fused multiply-add), so draws do not depend on whether the compiler contracts floating-point operations; exact speed-ups (Gauss-Kronrod node cache, leaf cache) that do not change the draws; new `pwbart_grid()` for prediction on a shared time grid.
