## =============================================================================
## sim1.R -- Simulation I of the paper
##
## Time-varying covariates and a random intercept + random slope (frailty) in
## the hazard. Two scenarios, "ph" (constant effect of X3) and "nonph"
## (step-function effect beta3(t)), each with 20% and 40% total censoring,
## 200 replicates. Methods: TVSurvBART-RIS, CoxPH, adapted frailty Cox and RSF.
## Every method predicts the survival of new (test) subjects from their
## observed covariate history, carried forward after the last observation
## (LOCF). Output: the paper's Tables 1-2, Web Table 2 and the data of Web
## Figure 1.
##
## Usage:  Rscript sim1.R [first_seed] [last_seed] [--quick] [--summary] [--no-rsf | --rsf-only]
##   Default seeds 1..200. One result file per (seed, scenario, censoring) is
##   written to results/ (results/quick/ with --quick). Existing files are
##   skipped, so runs can be resumed and several processes with different seed
##   ranges can run in parallel. --quick shortens the MCMC chains and the RSF
##   (for testing only). --summary only summarizes the existing result files.
##   --no-rsf / --rsf-only run every method except the RSF / only the RSF, into
##   separate result files that the summary combines (the RSF can crash R on
##   some clusters; it can then be run on another machine).
## Run time: one TVSurvBART fit (20,000 burn-in + 2,000 x 50 sweeps) took a
##   median of about 7 h on one cluster core in the paper's runs (about 30 min
##   on a recent desktop CPU); one seed has 4 fits. The other methods take about
##   15 s per cell. --quick: under 2 minutes per seed. Peak memory about 1.7 GB.
## Packages: TVSurvBART, survival, coxme, timeROC, LTRCforests, statmod, mvtnorm.
## =============================================================================

suppressPackageStartupMessages({
  library(TVSurvBART); library(survival); library(statmod); library(mvtnorm)
})  # coxme, timeROC and LTRCforests are called with :: (attaching LTRCforests masks print)

## -----------------------------------------------------------------------------
## 1. Settings
## -----------------------------------------------------------------------------
args  <- commandArgs(trailingOnly = TRUE)
flags <- args[startsWith(args, "--")]
num   <- suppressWarnings(as.integer(args[!startsWith(args, "--")]))
if (anyNA(num) || length(num) > 2 || !all(flags %in% c("--quick", "--summary", "--no-rsf", "--rsf-only")) ||
    all(c("--no-rsf", "--rsf-only") %in% flags))
  stop("usage: Rscript sim1.R [first_seed] [last_seed] [--quick] [--summary] [--no-rsf | --rsf-only]", call. = FALSE)
QUICK <- "--quick" %in% flags
SEEDS <- if (length(num)) num[1]:num[length(num)] else 1:200
OUT   <- if (QUICK) file.path("results", "quick") else "results"
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

N <- 1500; N_TRAIN <- 1200                  # subjects per replicate: 1200 train / 300 test
TAU <- 3                                    # end of follow-up = administrative censoring time
VIS <- seq(0, TAU, by = 0.5)                # visit times; interval k = (VIS[k], VIS[k+1]]
K <- length(VIS) - 1                        # 6 intervals
GRID <- seq(0, TAU, by = 0.1)               # prediction and evaluation grid
AUC_TIMES <- c(1, 2, 3)                     # AUC times reported in the paper
SCENARIOS <- c("ph", "nonph")
CENS <- c(0.20, 0.40)                       # total censoring (random + administrative at TAU)
N_POP <- 1e5                                # population sample for the true RMST, median and AUC
COV <- c("X1", "X2", "X3", "X4", "X5")      # X1-X3 continuous, X4 binary, X5 categorical 1-4 (static)
FACTORS <- c("X4", "X5")                    # entered as factors in CoxPH, frailty Cox and RSF
METHODS <- c("tvsurvbart", "cox", "fcox", "rsf")
TAG <- ""                                   # result-file suffix of a partial run
if ("--no-rsf" %in% flags)   { METHODS <- setdiff(METHODS, "rsf"); TAG <- "_norsf" }
if ("--rsf-only" %in% flags) { METHODS <- "rsf"; TAG <- "_rsf" }

## -----------------------------------------------------------------------------
## 2. Data-generating process
## -----------------------------------------------------------------------------
## hazard: lambda_i(t | b_i) = 3 LAM t^2 exp{eta_ik + b_0i + b_1i t/TAU}, t in interval k, where
## eta_ik = lp(X_i at the start of interval k) + beta3_k X3_ik - c0
LAM <- 0.05
SB0 <- 1; SB1 <- 2                          # SDs of the random intercept and slope (independent)
BETA_EDU <- c(0, 1, 2, 1)                   # effect of X5 = 1, ..., 4
BETA3 <- list(ph = rep(0, K), nonph = c(1, 1, 0, 0, -1, -1))
C0 <- c(ph = -1.238951650240403, nonph = -1.0860154274499196)   # calibrated: P(T < TAU) = 0.85 in the population
lp0 <- function(x1, x2, x3, x4, x5) -x1/3 - x2/3 + x4 + BETA_EDU[x5] - 0.5 * x3^2 + x1 * x2

## covariate processes (Web Table 1): X4 logistic and X1-X3 linear mixed models, quadratic in time
CP <- list(X4 = list(g0 = 0.2,  D = matrix(c(1, .05, .005, .05, .3, .005, .005, .005, .02), 3)),
           X1 = list(g0 = -0.5, D = matrix(c(.25, .05, .005, .05, .5, .005, .005, .005, .02), 3), sd = .30),
           X2 = list(g0 = 0.5,  D = matrix(c(.3, -.05, 0, -.05, .6, -.005, 0, -.005, .03), 3), sd = .35),
           X3 = list(g0 = 0,    D = matrix(c(.2, .03, .003, .03, .4, .003, .003, .003, .018), 3), sd = .25))

## n x 7 matrices (visits 0, 0.5, ..., 3) of X1-X4 and the static X5. byrow = TRUE draws the
## per-visit terms subject by subject (replicates), FALSE visit by visit (population sample);
## the law is the same, the order only fixes the random stream.
gen_cov <- function(n, byrow) {
  ord  <- function(m) c(if (byrow) t(m) else m)
  cell <- function(x) matrix(x, n, length(VIS), byrow = byrow)
  quad <- function(p) { u <- rmvnorm(n, sigma = p$D)
    sapply(VIS, function(t) p$g0 + u[, 1] + u[, 2] * t + u[, 3] * t^2) }
  lmm  <- function(p) { m <- quad(p); m + cell(rnorm(n * length(VIS), 0, p$sd)) }
  X5 <- sample(1:4, n, replace = TRUE)
  X4 <- cell(rbinom(n * length(VIS), 1, ord(plogis(quad(CP$X4)))))
  X1 <- lmm(CP$X1); X2 <- lmm(CP$X2); X3 <- lmm(CP$X3)
  list(X1 = X1, X2 = X2, X3 = X3, X4 = X4, X5 = X5)
}

## fixed part of the log hazard, n x K (covariates measured at the interval start)
eta_mat <- function(X, scen) sapply(1:K, function(k)
  lp0(X$X1[, k], X$X2[, k], X$X3[, k], X$X4[, k], X$X5) + BETA3[[scen]][k] * X$X3[, k] - C0[[scen]])

## int_a^m 3 LAM u^2 exp(c u) du (closed form; series for small |c|), vectorized over c
Ibase <- function(a, m, c) {
  out <- numeric(length(c)); if (m <= a) return(out)
  s <- abs(c) < 1e-3; cs <- c[s]; cc <- c[!s]
  out[s] <- LAM * ((m^3 - a^3) + .75 * cs * (m^4 - a^4) + .3 * cs^2 * (m^5 - a^5) + cs^3 / 12 * (m^6 - a^6))
  g <- function(x) exp(cc * x) * (x^2 / cc - 2 * x / cc^2 + 2 / cc^3)
  out[!s] <- 3 * LAM * (g(m) - g(a))
  out
}

## conditional cumulative hazard at time t (vector over subjects)
H_at <- function(eta, b0, b1, t)
  exp(b0) * rowSums(exp(eta) * sapply(1:K, function(k) Ibase(VIS[k], min(VIS[k + 1], t), b1 / TAU)))

## observed long data: one row per visit strictly before the exit time Y, plus an exit row
## at Y that carries the last visit's covariates (LOCF) and the event indicator
build_long <- function(X, Y, delta) {
  nv <- findInterval(Y, VIS, left.open = TRUE)          # number of visits before Y
  id <- rep(seq_along(Y), nv + 1); last <- !duplicated(id, fromLast = TRUE)
  j  <- pmin(sequence(nv + 1), rep(nv, nv + 1))         # visit index (exit row: the last visit)
  out <- data.frame(ID = id, time = ifelse(last, Y[id], VIS[j]), indicator = ifelse(last, delta[id], 0L))
  for (v in names(X)) out[[v]] <- if (is.matrix(X[[v]])) X[[v]][cbind(id, j)] else X[[v]][id]
  out
}

gen_data <- function(seed, scen, cens) {
  set.seed(seed)
  b0 <- rnorm(N, 0, SB0); b1 <- rnorm(N, 0, SB1)
  X <- gen_cov(N, byrow = TRUE)
  eta <- eta_mat(X, scen)
  ## latent event time: exact inversion of the cumulative hazard, interval by interval
  T <- rep(TAU, N); ev <- integer(N)
  for (i in 1:N) {
    E <- -log(runif(1)); H <- 0
    for (k in 1:K) {
      r <- exp(eta[i, k] + b0[i]); inc <- r * Ibase(VIS[k], VIS[k + 1], b1[i] / TAU)
      if (H + inc >= E) {
        T[i] <- uniroot(function(u) r * Ibase(VIS[k], u, b1[i] / TAU) - (E - H), VIS[k:(k + 1)], tol = 1e-9)$root
        ev[i] <- 1L; break
      }
      H <- H + inc
    }
  }
  ## exponential censoring; its rate makes the expected TOTAL censored fraction
  ## (random + administrative at TAU) equal to cens
  stopifnot(cens > mean(T >= TAU) + 1e-9)
  a <- uniroot(function(a) mean(ifelse(T < TAU, 1 - exp(-exp(a) * T), 1)) - cens,
               c(-15, 5), tol = 1e-10, extendInt = "yes")$root
  C <- pmin(rexp(N, exp(a)), TAU)
  Y <- pmin(T, C); delta <- as.integer(T < C)
  list(X = X, T = T, ev = ev, Y = Y, delta = delta, long = build_long(X, Y, delta))
}

split_ids <- function(seed) {
  set.seed(seed + 1000); tr <- sort(sample(1:N, N_TRAIN))
  list(train = tr, test = setdiff(1:N, tr))
}

## -----------------------------------------------------------------------------
## 3. True survival quantities
## -----------------------------------------------------------------------------
## covariate matrices restricted to the rows ids; with nv given, the visits after the nv-th are
## replaced by the nv-th (the frozen LOCF path a model sees)
take <- function(X, ids, nv = NULL) lapply(X, function(M) {
  if (!is.matrix(M)) return(M[ids])
  M <- M[ids, , drop = FALSE]
  if (is.null(nv)) M else matrix(M[cbind(c(row(M)), pmin(c(col(M)), nv))], nrow(M))
})

## frailty-marginalized survival on GRID given a covariate path (25 x 25 Gauss-Hermite)
S_true <- function(X, scen) {
  Ef <- exp(eta_mat(X, scen))
  g0 <- gauss.quad.prob(25, "normal", sigma = SB0); g1 <- gauss.quad.prob(25, "normal", sigma = SB1)
  sapply(GRID, function(t) {
    s <- 0
    for (q in seq_along(g1$nodes)) {
      Q <- drop(Ef %*% sapply(1:K, function(k) Ibase(VIS[k], min(VIS[k + 1], t), g1$nodes[q] / TAU)))
      s <- s + g1$weights[q] * colSums(g0$weights * exp(-outer(exp(g0$nodes), Q)))
    }
    s
  })
}

## population truths (seed 12345, N_POP subjects, full covariate paths, no censoring):
## RMST(TAU/2), RMST(TAU), median and the two AUC references of Table 2, "Marginal"
## (marker 1 - frailty-marginalized survival) and "Oracle" (marker 1 - survival given
## the realized random effects). Computed by the summary, about 25 s per scenario.
pop_truth <- function(scen) {
  set.seed(12345)
  b0 <- rnorm(N_POP, 0, SB0); b1 <- rnorm(N_POP, 0, SB1); X <- gen_cov(N_POP, byrow = FALSE)
  eta <- eta_mat(X, scen); E <- -log(runif(N_POP))
  ## event times: first crossing of E by H on a 0.01 grid, linear interpolation in between
  T <- rep(TAU + 1e-6, N_POP); done <- logical(N_POP); Hp <- numeric(N_POP); tp <- 0
  for (t in seq(0, TAU, by = 0.01)[-1]) {
    H <- H_at(eta, b0, b1, t); x <- !done & H >= E
    T[x] <- tp + (E[x] - Hp[x]) / (H[x] - Hp[x]) * (t - tp); done[x] <- TRUE
    Hp <- H; tp <- t
  }
  S_mg <- S_true(X, scen)
  auc <- function(marker) sapply(seq_along(GRID), function(j) if (GRID[j] == 0) NA else
    timeROC::timeROC(T = T, delta = as.integer(done), marker = marker(j), cause = 1, times = GRID[j], iid = FALSE)$AUC[[2]])
  list(rmst = c(half = mean(pmin(T, TAU / 2)), full = mean(pmin(T, TAU))),
       median = if (mean(done) >= 0.5 && median(T) <= TAU) median(T) else NA,
       auc_marginal = auc(function(j) 1 - S_mg[, j]),
       auc_oracle = auc(function(j) 1 - exp(-H_at(eta, b0, b1, GRID[j]))),
       event_rate = mean(done))
}

## -----------------------------------------------------------------------------
## 4. Methods (each returns the n_test x length(GRID) matrix of predicted survival,
##    LOCF mode: the last observed covariates are carried forward to TAU)
## -----------------------------------------------------------------------------
## TVSurvBART-RIS as in the paper: 100 trees with leaf prior kappa = log(1000)/qnorm(0.975);
## natural cubic spline baseline with knots 0:3; random intercept + slope z(t) = (1, t/TAU)
## with a Huang-Wand prior; one chain, 20,000 burn-in sweeps, 2,000 draws kept every 50th sweep.
KAPPA <- log(1000) / qnorm(0.975)
BART_ARGS <- list(ntree = 100L, c = 100 / (KAPPA^2 / 2) + 0.5, d = 100 / (KAPPA^2 / 2),
                  nskip = 20000L, ndpost = 2000L, keepevery = 50L,
                  baseline.type = "spline", ns.knot = 0:3, nsCoefMean = 0, nsCoefVar = 1,
                  ns.block = TRUE, ns.block.M = "secant",
                  re.max.power = 1, re.time.scale = TAU, reCoefMean = c(0, 0),
                  reCoefVarInit = diag(c(0.5, 1)^2),   # initial Sigma_b (also the spread of the initial b_i)
                  reHW.nu = 2, reHW.A = 5, re.ncp = 3L)
N_GH <- 16L                                 # Gauss-Hermite nodes per random effect for new subjects
RSF_NTREE <- 200L
if (QUICK) {                                # short test chains (ns.block needs nskip >= 400)
  BART_ARGS[c("nskip", "ndpost", "keepevery")] <- list(400L, 50L, 1L); RSF_NTREE <- 50L
}

## The trees get the time (interval start / TAU) only in the non-PH scenario.
time_input <- function(t) pmin(0.5 * floor(t / 0.5 + 1e-9), TAU - 0.5) / TAU

fit_tvsurvbart <- function(d, sp, seed, scen) {
  rows <- function(ids, cols) {
    x <- d$long[d$long$ID %in% ids, cols]
    if (scen == "nonph") x$Xt <- time_input(x$time)
    x
  }
  train <- rows(sp$train, c("ID", "time", "indicator", COV))
  test  <- rows(sp$test, c("ID", "time", COV))             # test IDs never occur in the training data
  set.seed(1000003 * seed + 7919 + 104729 * 3)              # chain seed of the paper's runs
  invisible(capture.output(fit <- do.call(wbart, c(list(train), BART_ARGS))))
  S <- matrix(NA_real_, length(sp$test), length(GRID))
  for (idx in split(seq_along(GRID), ceiling(seq_along(GRID) / 8))) {   # blocks of grid times cap memory
    invisible(capture.output(ch <- pwbart_grid(GRID[idx], test, fit, n.gh = N_GH)$cumHazard.test))
    S[, idx] <- colMeans(exp(-ch), dims = 1)               # posterior mean of S = exp(-H), draws x n x times
  }
  S
}

## counting-process (start, end] frame for CoxPH, frailty Cox and RSF: covariates measured
## at the interval start, factors with levels taken over the whole cohort
cp_frame <- function(long) {
  first <- !duplicated(long$ID); prev <- which(!first) - 1L
  cx <- long[!first, ]; cx$start <- long$time[prev]; cx$end <- cx$time
  for (v in COV) cx[[v]] <- long[[v]][prev]
  for (v in FACTORS) cx[[v]] <- factor(cx[[v]], levels = sort(unique(long[[v]])))
  cx
}
cox_formula <- function(extra = NULL)       # environment = caller (LTRCforests evaluates data there)
  reformulate(c(COV, extra), "Surv(start, end, indicator)", env = parent.frame())
lin_pred <- function(d, beta) drop(model.matrix(reformulate(COV), d)[, names(beta), drop = FALSE] %*% beta)

## cumulative hazard on GRID from a step baseline H0 and the relative risk of each interval;
## after the last interval its risk is carried forward (LOCF)
cum_haz <- function(cx, ids, risk, H0) {
  last <- !duplicated(cx$ID, fromLast = TRUE); key <- as.character(ids)
  sapply(GRID, function(t) {
    inside <- tapply(ifelse(cx$start < t, risk * pmax(0, H0(pmin(t, cx$end)) - H0(cx$start)), 0), cx$ID, sum)
    tail <- setNames(risk[last] * pmax(0, H0(t) - H0(cx$end[last])), cx$ID[last])
    unname(inside[key] + tail[key])
  })
}

## CoxPH with time-varying covariates, Breslow baseline
fit_cox <- function(tr, te, ids) {
  f <- coxph(cox_formula(), data = tr)
  bh <- basehaz(f, centered = FALSE)
  exp(-cum_haz(te, ids, exp(lin_pred(te, coef(f))), stepfun(bh$time, c(0, bh$hazard), right = FALSE)))
}

## adapted frailty Cox: coxme with a random intercept; Breslow baseline with posterior-mean
## (Laplace) risk sets; new subjects integrate the frailty out (41-node Gauss-Hermite)
fit_fcox <- function(tr, te, ids) {
  f <- coxme::coxme(cox_formula("(1 | ID)"), data = tr)
  beta <- coxme::fixef(f); m <- coxme::ranef(f)$ID[as.character(tr$ID)]
  s2b <- as.numeric(coxme::VarCorr(f)$ID)[1]; if (!is.finite(s2b) || s2b < 0) s2b <- 0
  tj <- sort(unique(tr$end[tr$indicator == 1]))
  breslow <- function(w) stepfun(tj, c(0, cumsum(vapply(tj, function(t)
    sum(tr$indicator == 1 & tr$end == t) / sum(w[tr$start < t & tr$end >= t]), 0))), right = FALSE)
  lp <- lin_pred(tr, beta)
  H0m <- breslow(exp(lp + m))                                          # mode plug-in
  Lam <- tapply(exp(lp + m) * pmax(0, H0m(tr$end) - H0m(tr$start)), tr$ID, sum)
  s2 <- 1 / (1 / max(s2b, 1e-8) + Lam)                                 # Laplace posterior variance of b_i
  H0 <- breslow(exp(lp + m + 0.5 * s2[as.character(tr$ID)]))          # posterior-mean risk sets
  H <- cum_haz(te, ids, exp(lin_pred(te, beta)), H0)
  gh <- if (s2b > 1e-8) gauss.quad.prob(41, "normal", sigma = sqrt(s2b)) else list(nodes = 0, weights = 1)
  Reduce(`+`, lapply(seq_along(gh$nodes), function(g) gh$weights[g] * exp(-H * exp(gh$nodes[g]))))
}

## random survival forest for left-truncated right-censored data (LTRCforests defaults otherwise)
fit_rsf <- function(tr, te, ids) {
  fit <- LTRCforests::ltrcrrf(cox_formula(), data = tr, id = ID, ntree = RSF_NTREE,
                              mtry = max(2L, ceiling(sqrt(length(COV)))), trace = FALSE)
  pr <- LTRCforests::predictProb(fit, newdata = te, newdata.id = ID, time.eval = GRID,
                                 time.tau = rep(TAU, length(ids)))
  t(pr$survival.probs)[match(ids, unique(pr$survival.id)), ]
}

## -----------------------------------------------------------------------------
## 5. Metrics for one predicted survival matrix S (test subjects x GRID)
##    o: observed (Tstar, delta) and latent (aT, ad) test outcomes; Strue: true S on GRID
## -----------------------------------------------------------------------------
evaluate <- function(S, o, Strue, widen = TRUE) {
  nG <- length(GRID)
  ## cumulative/dynamic AUC on the latent event times (non-events at TAU + 1e-6)
  auc <- sapply(seq_along(GRID), function(j) {
    mk <- 1 - S[, j]
    if (GRID[j] == 0 || length(unique(mk)) < 2) return(NA_real_)
    tryCatch(timeROC::timeROC(T = o$aT, delta = o$ad, marker = mk, cause = 1, times = GRID[j], iid = FALSE)$AUC[[2]],
             error = function(e) NA_real_)
  })
  ## IPCW Brier score on the observed outcomes; IBS = trapezoid over [0, TAU - 0.1] / (TAU - 0.1)
  G <- survfit(Surv(o$Tstar, 1 - o$delta) ~ 1); Gf <- stepfun(G$time, c(1, G$surv), right = FALSE)
  GT <- pmax(Gf(o$Tstar - 1e-8), 1e-4)
  BS <- sapply(seq_along(GRID), function(k) mean(ifelse(o$Tstar <= GRID[k] & o$delta == 1, S[, k]^2 / GT,
                 ifelse(o$Tstar > GRID[k], (1 - S[, k])^2 / max(Gf(GRID[k]), 1e-4), 0))))
  trap <- function(x, y) sum(diff(x) * (head(y, -1) + tail(y, -1)) / 2)
  ib <- GRID < TAU
  ## marginal curve (S(0) = 1): RMST by the trapezoid rule; median = first crossing of 0.5
  Sb <- colMeans(S); Sb[1] <- 1
  rmst <- function(h) trap(GRID[GRID <= h + 1e-9], Sb[GRID <= h + 1e-9])
  mo <- cummin(Sb); k <- which(mo <= 0.5)[1]
  med <- if (is.na(k)) NA else if (k == 1) 0 else
    GRID[k - 1] + (0.5 - mo[k - 1]) * (GRID[k] - GRID[k - 1]) / (mo[k] - mo[k - 1])
  ## test log score (total over subjects): event -> probability of the 0.1-grid cell containing
  ## Tstar per unit time (step curves: zero-mass cells widened symmetrically); censored -> S(Tstar)
  jh <- pmin(pmax(findInterval(o$Tstar, GRID, left.open = TRUE) + 1L, 2L), nG); jl <- jh - 1L
  lpd <- sapply(seq_along(o$Tstar), function(i) {
    a <- jl[i]; b <- jh[i]
    if (o$delta[i] == 1) {
      p <- S[i, a] - S[i, b]
      while (widen && is.finite(p) && p <= 0 && (a > 1 || b < nG)) {
        a <- max(1L, a - 1L); b <- min(nG, b + 1L); p <- S[i, a] - S[i, b]
      }
      p <- p / (GRID[b] - GRID[a])
    } else {
      w <- min(max((GRID[b] - o$Tstar[i]) / (GRID[b] - GRID[a]), 0), 1); p <- w * S[i, a] + (1 - w) * S[i, b]
    }
    log(if (is.finite(p) && p > 1e-12) p else 1e-12)
  })
  c(ibs = trap(GRID[ib], BS[ib]) / max(GRID[ib]), rmse_s = sqrt(mean((S - Strue)^2)),
    rmst_half = rmst(TAU / 2), rmst_tau = rmst(TAU), median = med, logscore = sum(lpd),
    setNames(auc, paste0("auc_", GRID)))
}

## -----------------------------------------------------------------------------
## 6. Main loop: one result file per (seed, scenario, censoring level)
## -----------------------------------------------------------------------------
run_cell <- function(seed, scen, cens) {
  d  <- gen_data(seed, scen, cens)
  sp <- split_ids(seed); te <- sp$test
  nv <- findInterval(d$Y, VIS, left.open = TRUE)            # observed visits per subject
  S_full <- S_true(take(d$X, te), scen)                      # truth on the full covariate path (RMSE_S)
  cx <- cp_frame(d$long); ctr <- cx[cx$ID %in% sp$train, ]; cte <- cx[cx$ID %in% te, ]
  ## RSF first: its random stream then continues from the split, as in the paper's runs
  fits <- list(rsf = function() fit_rsf(ctr, cte, te), cox = function() fit_cox(ctr, cte, te),
               fcox = function() fit_fcox(ctr, cte, te), tvsurvbart = function() fit_tvsurvbart(d, sp, seed, scen))
  S <- list(); secs <- c()
  for (m in intersect(names(fits), METHODS)) {
    t0 <- proc.time()[[3]]; S[[m]] <- fits[[m]](); secs[m] <- proc.time()[[3]] - t0
  }
  S$oracle_locf <- S_true(take(d$X, te, nv[te]), scen)      # reference: true model on the LOCF path
  o <- list(Tstar = d$Y[te], delta = d$delta[te], aT = ifelse(d$ev[te] == 1, d$T[te], TAU + 1e-6), ad = d$ev[te])
  ev <- t(sapply(names(S), function(m) evaluate(S[[m]], o, S_full, widen = m != "tvsurvbart")))
  list(seed = seed, scen = scen, cens = cens, cens_realized = mean(d$delta == 0),
       metrics = data.frame(method = names(S), ev, secs = unname(secs[names(S)]), row.names = NULL),
       S_tvsurvbart = S$tvsurvbart)          # TVSurvBART survival matrix (test subjects in ID order x GRID)
}

save_rds <- function(x, f) { tmp <- paste0(f, ".tmp", Sys.getpid()); saveRDS(x, tmp); file.rename(tmp, f) }

if (!"--summary" %in% flags) for (seed in SEEDS) for (scen in SCENARIOS) for (cens in CENS) {
  f <- file.path(OUT, sprintf("sim1_%s_cens%02d_seed%03d", scen, round(100 * cens), seed))
  ## done if the cell's files cover METHODS: the full file, or "_norsf" (all but RSF) and "_rsf" (RSF)
  have <- file.exists(paste0(f, c("", "_norsf", "_rsf"), ".rds"))
  if (have[1] || ((have[2] || identical(METHODS, "rsf")) && (have[3] || !"rsf" %in% METHODS))) next
  t0 <- Sys.time(); cat(sprintf("[%s] seed %d, %s, censoring %.2f ... ", format(t0, "%H:%M"), seed, scen, cens))
  res <- tryCatch(run_cell(seed, scen, cens), error = function(e) { cat("FAILED:", conditionMessage(e), "\n"); NULL })
  if (!is.null(res)) {
    save_rds(res, paste0(f, TAG, ".rds"))
    cat(sprintf("done in %.1f min\n", as.numeric(difftime(Sys.time(), t0, units = "mins"))))
  }
}

## -----------------------------------------------------------------------------
## 7. Summary over seeds: mean (Monte Carlo SE) per scenario x censoring x method
## -----------------------------------------------------------------------------
summarize <- function() {
  files <- list.files(OUT, "^sim1_.*\\.rds$", full.names = TRUE)
  if (!length(files)) { cat("No result files in ", OUT, "\n", sep = ""); return(invisible(NULL)) }
  R <- do.call(rbind, lapply(files, function(f) {
    r <- readRDS(f); data.frame(seed = r$seed, scen = r$scen, cens = r$cens, r$metrics) }))
  R <- R[!duplicated(R[c("seed", "scen", "cens", "method")]), ]   # a split cell has oracle_locf twice
  cat("Computing the population truths ...\n")
  truth <- lapply(setNames(nm = intersect(SCENARIOS, R$scen)), pop_truth)
  tv <- function(field, i) sapply(R$scen, function(s) truth[[s]][[field]][i])
  R$d_rmst_half <- R$rmst_half - tv("rmst", "half")
  R$d_rmst_tau  <- R$rmst_tau - tv("rmst", "full")
  R$d_median    <- R$median - tv("median", 1)
  ms <- function(x) { x <- x[is.finite(x)]; c(mean = mean(x), se = sd(x) / sqrt(length(x))) }
  bias <- function(x) { x <- x[is.finite(x)]; c(bias = mean(x), se = sd(x) / sqrt(length(x)), rmse = sqrt(mean(x^2))) }
  lev <- c("tvsurvbart", "cox", "fcox", "rsf", "oracle_locf", "ref_marginal", "ref_oracle")   # row order
  groups <- split(R, list(factor(R$method, lev), R$cens, factor(R$scen, SCENARIOS)), drop = TRUE)
  lab <- function(g) data.frame(scen = g$scen[1], cens = g$cens[1], method = g$method[1], n = nrow(g))

  ## Table 1: RMST(TAU/2), RMST(TAU) and median bias vs the population truth; IBS; RMSE_S; log score
  main <- do.call(rbind, lapply(groups, function(g) data.frame(lab(g), t(c(
    rmst_half = bias(g$d_rmst_half), rmst_tau = bias(g$d_rmst_tau), median = bias(g$d_median),
    ibs = ms(g$ibs), rmse_s = ms(g$rmse_s), logscore = ms(g$logscore))))))
  ## Table 2 / Web Figure 1: mean AUC(t) on the grid, plus the population references
  ac <- grep("^auc_", names(R), value = TRUE)
  auc <- do.call(rbind, lapply(groups, function(g) data.frame(lab(g), t = GRID,
    auc = colMeans(g[ac], na.rm = TRUE), se = apply(g[ac], 2, function(x) ms(x)[["se"]]))))
  for (s in names(truth)) for (cn in sort(unique(R$cens[R$scen == s]))) for (ref in c("marginal", "oracle"))
    auc <- rbind(auc, data.frame(scen = s, cens = cn, method = paste0("ref_", ref), n = NA, t = GRID,
                                 auc = truth[[s]][[paste0("auc_", ref)]], se = NA))
  auc <- auc[order(factor(auc$scen, SCENARIOS), auc$cens, factor(auc$method, lev), auc$t), ]
  ## Web Table 2: paired differences of the test log score (row minus column method)
  pairs <- list(c("rsf", "tvsurvbart"), c("cox", "tvsurvbart"), c("cox", "rsf"),
                c("fcox", "tvsurvbart"), c("fcox", "rsf"), c("fcox", "cox"))
  ls <- do.call(rbind, lapply(split(R, list(R$cens, factor(R$scen, SCENARIOS)), drop = TRUE), function(g) {
    w <- tapply(g$logscore, list(g$seed, g$method), mean)
    do.call(rbind, lapply(pairs, function(p) if (all(p %in% colnames(w)))
      data.frame(scen = g$scen[1], cens = g$cens[1], pair = paste(p, collapse = " - "),
                 n = sum(is.finite(w[, p[1]] - w[, p[2]])), t(ms(w[, p[1]] - w[, p[2]])))))
  }))
  rownames(main) <- rownames(auc) <- NULL; if (!is.null(ls)) rownames(ls) <- NULL
  write.csv(main, file.path(OUT, "summary_sim1_main.csv"), row.names = FALSE)
  write.csv(auc, file.path(OUT, "summary_sim1_auc.csv"), row.names = FALSE)
  if (!is.null(ls)) write.csv(ls, file.path(OUT, "summary_sim1_logscore.csv"), row.names = FALSE)
  op <- options(width = 200, digits = 4); on.exit(options(op))
  for (s in names(truth)) cat(sprintf("Population truth, %s: RMST(TAU/2) %.4f, RMST(TAU) %.4f, median %.4f\n",
                                      s, truth[[s]]$rmst[["half"]], truth[[s]]$rmst[["full"]], truth[[s]]$median))
  cat("\nRMST / median bias vs population truth, IBS, RMSE_S, log score (mean and MC SE over seeds):\n")
  print(main)
  cat("\nAUC at t =", AUC_TIMES, "\n")
  print(reshape(auc[sapply(auc$t, function(t) any(abs(t - AUC_TIMES) < 1e-9)), c("scen", "cens", "method", "t", "auc")],
                idvar = c("scen", "cens", "method"), timevar = "t", direction = "wide"), row.names = FALSE)
  cat("\nPaired log-score differences:\n")
  print(ls, row.names = FALSE)
}
summarize()
