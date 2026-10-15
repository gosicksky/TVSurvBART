## =============================================================================
## sim2.R -- Simulation II of the paper
##
## Time-dependent confounding: binary covariates L1-L3, continuous L4-L6 and a
## binary predictor A evolve over yearly visits and feed back on each other;
## the event hazard is piecewise constant per year, with nonlinear covariate
## effects and a U-shaped baseline. No frailty in the data-generating process.
## Total censoring 20% and 40%, 200 replicates. Methods: TVSurvBART-RIS,
## CoxPH, adapted frailty Cox and RSF. Every method predicts the survival of
## new (test) subjects from their observed covariate history, carried forward
## after the last observation (LOCF). Output: the paper's Tables 3-4, Web
## Table 3 and the data of Web Figure 2.
##
## Usage:  Rscript sim2.R [first_seed] [last_seed] [--quick] [--summary] [--no-rsf | --rsf-only]
##   Default seeds 1..200. One result file per (seed, censoring) is written to
##   results/ (results/quick/ with --quick). Existing files are skipped, so runs
##   can be resumed and several processes with different seed ranges can run in
##   parallel. --quick shortens the MCMC chains and the RSF (for testing only).
##   --summary only summarizes the existing result files. --no-rsf / --rsf-only
##   run every method except the RSF / only the RSF, into separate result files
##   that the summary combines (the RSF can crash R on some clusters; it can
##   then be run on another machine).
## Run time: one TVSurvBART fit (20,000 burn-in + 2,000 x 50 sweeps) took a
##   median of about 9 h on one cluster core in the paper's runs (about 40 min
##   on a recent desktop CPU); one seed has 2 fits. The other methods take about
##   15 s per cell. --quick: under a minute per seed. Peak memory about 1.3 GB.
## Packages: TVSurvBART, survival, coxme, timeROC, LTRCforests, statmod.
## =============================================================================

suppressPackageStartupMessages({
  library(TVSurvBART); library(survival); library(statmod)
})  # coxme, timeROC and LTRCforests are called with :: (attaching LTRCforests masks print)

## -----------------------------------------------------------------------------
## 1. Settings
## -----------------------------------------------------------------------------
args  <- commandArgs(trailingOnly = TRUE)
flags <- args[startsWith(args, "--")]
num   <- suppressWarnings(as.integer(args[!startsWith(args, "--")]))
if (anyNA(num) || length(num) > 2 || !all(flags %in% c("--quick", "--summary", "--no-rsf", "--rsf-only")) ||
    all(c("--no-rsf", "--rsf-only") %in% flags))
  stop("usage: Rscript sim2.R [first_seed] [last_seed] [--quick] [--summary] [--no-rsf | --rsf-only]", call. = FALSE)
QUICK <- "--quick" %in% flags
SEEDS <- if (length(num)) num[1]:num[length(num)] else 1:200
OUT   <- if (QUICK) file.path("results", "quick") else "results"
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

N <- 1500; N_TRAIN <- 1200                  # subjects per replicate: 1200 train / 300 test
TAU <- 5                                    # end of follow-up = administrative censoring time
VIS <- 0:TAU                                # visit times; interval j = (j-1, j]
J <- length(VIS) - 1                        # 5 intervals
GRID <- seq(0, TAU, by = 0.1)               # prediction and evaluation grid
AUC_TIMES <- 1:5                            # AUC times reported in the paper
SCENARIOS <- "general"                      # one scenario
CENS <- c(0.20, 0.40)                       # total censoring (random + administrative at TAU)
N_POP <- 1e5                                # population sample for the true RMST, median and AUC
COV <- c("L4", "L5", "L6", "L1", "L2", "L3", "A")   # L4-L6 continuous, L1-L3 and A binary
FACTORS <- c("L1", "L2", "L3", "A")         # entered as factors in CoxPH, frailty Cox and RSF
METHODS <- c("tvsurvbart", "cox", "fcox", "rsf")
TAG <- ""                                   # result-file suffix of a partial run
if ("--no-rsf" %in% flags)   { METHODS <- setdiff(METHODS, "rsf"); TAG <- "_norsf" }
if ("--rsf-only" %in% flags) { METHODS <- "rsf"; TAG <- "_rsf" }

## -----------------------------------------------------------------------------
## 2. Data-generating process
## -----------------------------------------------------------------------------
## event probability in interval j: p_ij = expit(beta0_j + g(A, L) at visit j-1), with
## g = -2A + L4/3 - L5/3 + L1 + 0.5 (L4 - L6)^2 + 0.5 (L5 - 3)^2; constant hazard -log(1 - p_ij)
## within the interval. U-shaped intercepts; their level is calibrated (50,000 paths, seed
## 12345) so that P(T < TAU) = 0.85.
U <- 0.4 * ((1:J) - 3)^2
BETA0 <- -3.1229938267594139 + (U - mean(U))
g_lp <- function(A, L1, L4, L5, L6) -2 * A + (L4/3 - L5/3 + L1 + 0.5 * (L4 - L6)^2 + 0.5 * (L5 - 3)^2)
hazards <- function(X) sapply(1:J, function(j)       # n x J interval hazards from the visit matrices
  -log1p(-plogis(BETA0[j] + g_lp(X$A[, j], X$L1[, j], X$L4[, j], X$L5[, j], X$L6[, j]))))

## covariate paths (visits 0..TAU, generated for every subject even after the event) and
## latent event times, subject by subject
gen_paths <- function(n) {
  Lb <- Lc <- array(NA_real_, c(n, J + 1, 3)); A <- matrix(NA_real_, n, J + 1)
  T <- rep(TAU + 1e-6, n); ev <- integer(n)               # no event by TAU: T = TAU + 1e-6
  for (i in 1:n) {
    xb <- rbinom(3, 1, 0.5); xc <- rnorm(3, 2, 1)         # visit 0: L1-L3 ~ Bern(0.5), L4-L6 ~ N(2, 1)
    a <- rbinom(1, 1, plogis(-1 + mean(xb) - 0.25 * mean(xc)))
    Lb[i, 1, ] <- xb; Lc[i, 1, ] <- xc; A[i, 1] <- a; alive <- TRUE
    for (j in 1:J) {
      if (alive) {                                        # event in (j-1, j]: exponential with the interval hazard
        E <- rexp(1, -log1p(-plogis(BETA0[j] + g_lp(a, xb[1], xc[1], xc[2], xc[3]))))
        if (E <= 1) { T[i] <- j - 1 + max(E, 1e-9); ev[i] <- 1L; alive <- FALSE }
      }
      ## transition to visit j: binaries and A are absorbing at 1
      xbn <- xb; for (k in 1:3) if (xb[k] == 0) xbn[k] <- rbinom(1, 1, plogis(-2 - 2 * a - 0.5 * xc[k]))
      xcn <- numeric(3); for (k in 1:3) xcn[k] <- rnorm(1, 2 + a - xb[k] + 0.5 * xc[k], 1)
      an <- if (a == 1) 1 else rbinom(1, 1, plogis(-1 + mean(xbn) - 0.25 * mean(xcn)))
      Lb[i, j + 1, ] <- xb <- xbn; Lc[i, j + 1, ] <- xc <- xcn; A[i, j + 1] <- a <- an
    }
  }
  list(X = list(L4 = Lc[, , 1], L5 = Lc[, , 2], L6 = Lc[, , 3], L1 = Lb[, , 1], L2 = Lb[, , 2], L3 = Lb[, , 3], A = A),
       T = T, ev = ev)
}

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

gen_data <- function(seed, cens) {
  set.seed(seed)
  p <- gen_paths(N)
  ## exponential censoring; its rate makes the expected TOTAL censored fraction
  ## (random + administrative at TAU) equal to cens
  stopifnot(cens > mean(p$ev == 0) + 1e-9)
  Tcap <- pmin(p$T, TAU)
  a <- uniroot(function(a) mean(ifelse(p$ev == 1, 1 - exp(log1p(-plogis(a)) * Tcap), 1)) - cens,
               c(-15, 6), tol = 1e-10, extendInt = "yes")$root
  C <- rexp(N, -log1p(-plogis(a)))
  obs <- p$ev == 1 & p$T <= pmin(C, TAU)
  Y <- ifelse(obs, p$T, pmin(C, TAU)); delta <- as.integer(obs)
  list(X = p$X, T = p$T, ev = p$ev, Y = Y, delta = delta, long = build_long(p$X, Y, delta))
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

## exact survival on GRID given a covariate path (piecewise-constant hazard)
S_true <- function(X) {
  h <- hazards(X); cH <- cbind(0, t(apply(h, 1, cumsum)))
  sapply(GRID, function(t) {
    j0 <- floor(t); H <- cH[, j0 + 1]
    if (j0 < J && t > j0) H <- H + (t - j0) * h[, j0 + 1]   # skipped at integer t (h may be Inf later)
    exp(-H)
  })
}

## population truths (seed 12345, N_POP subjects, full covariate paths, no censoring):
## RMST(TAU/2), RMST(TAU), median and the AUC reference of Table 4 (marker 1 - true
## survival). Computed by the summary, about 10 s.
pop_truth <- function() {
  set.seed(12345); p <- gen_paths(N_POP)
  S <- S_true(p$X)
  list(rmst = c(half = mean(pmin(p$T, TAU / 2)), full = mean(pmin(p$T, TAU))),
       median = if (mean(p$ev) >= 0.5 && median(p$T) <= TAU) median(p$T) else NA,
       auc_reference = sapply(seq_along(GRID), function(j) if (GRID[j] == 0) NA else
         timeROC::timeROC(T = p$T, delta = p$ev, marker = 1 - S[, j], cause = 1, times = GRID[j], iid = FALSE)$AUC[[2]]),
       event_rate = mean(p$ev))
}

## -----------------------------------------------------------------------------
## 4. Methods (each returns the n_test x length(GRID) matrix of predicted survival,
##    LOCF mode: the last observed covariates are carried forward to TAU)
## -----------------------------------------------------------------------------
## TVSurvBART-RIS as in the paper: 100 trees with leaf prior kappa = log(1000)/qnorm(0.975);
## natural cubic spline baseline with knots 0:5; random intercept + slope z(t) = (1, t/TAU)
## with a Huang-Wand prior; one chain, 20,000 burn-in sweeps, 2,000 draws kept every 50th sweep.
## The trees get covariates only (no time input).
KAPPA <- log(1000) / qnorm(0.975)
BART_ARGS <- list(ntree = 100L, c = 100 / (KAPPA^2 / 2) + 0.5, d = 100 / (KAPPA^2 / 2),
                  nskip = 20000L, ndpost = 2000L, keepevery = 50L,
                  baseline.type = "spline", ns.knot = 0:5, nsCoefMean = 0, nsCoefVar = 1,
                  ns.block = TRUE, ns.block.M = "secant",
                  re.max.power = 1, re.time.scale = TAU, reCoefMean = c(0, 0),
                  reCoefVarInit = diag(c(0.5, 0.5)^2),   # initial Sigma_b (also the spread of the initial b_i)
                  reHW.nu = 2, reHW.A = 5, re.ncp = 3L)
N_GH <- 16L                                 # Gauss-Hermite nodes per random effect for new subjects
RSF_NTREE <- 200L
if (QUICK) {                                # short test chains (ns.block needs nskip >= 400)
  BART_ARGS[c("nskip", "ndpost", "keepevery")] <- list(400L, 50L, 1L); RSF_NTREE <- 50L
}

fit_tvsurvbart <- function(d, sp, seed) {
  train <- d$long[d$long$ID %in% sp$train, c("ID", "time", "indicator", COV)]
  test  <- d$long[d$long$ID %in% sp$test, c("ID", "time", COV)]   # test IDs never occur in the training data
  set.seed(1000003 * seed + 7919 + 104729 * 3)                    # chain seed of the paper's runs
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
## 6. Main loop: one result file per (seed, censoring level)
## -----------------------------------------------------------------------------
run_cell <- function(seed, scen, cens) {
  d  <- gen_data(seed, cens)
  sp <- split_ids(seed); te <- sp$test
  nv <- findInterval(d$Y, VIS, left.open = TRUE)            # observed visits per subject
  S_full <- S_true(take(d$X, te))                            # truth on the full covariate path (RMSE_S)
  cx <- cp_frame(d$long); ctr <- cx[cx$ID %in% sp$train, ]; cte <- cx[cx$ID %in% te, ]
  ## RSF first: its random stream then continues from the split, as in the paper's runs
  fits <- list(rsf = function() fit_rsf(ctr, cte, te), cox = function() fit_cox(ctr, cte, te),
               fcox = function() fit_fcox(ctr, cte, te), tvsurvbart = function() fit_tvsurvbart(d, sp, seed))
  S <- list(); secs <- c()
  for (m in intersect(names(fits), METHODS)) {
    t0 <- proc.time()[[3]]; S[[m]] <- fits[[m]](); secs[m] <- proc.time()[[3]] - t0
  }
  S$oracle_locf <- S_true(take(d$X, te, nv[te]))            # reference: true model on the LOCF path
  o <- list(Tstar = d$Y[te], delta = d$delta[te], aT = d$T[te], ad = d$ev[te])
  ev <- t(sapply(names(S), function(m) evaluate(S[[m]], o, S_full, widen = m != "tvsurvbart")))
  list(seed = seed, scen = scen, cens = cens, cens_realized = mean(d$delta == 0),
       metrics = data.frame(method = names(S), ev, secs = unname(secs[names(S)]), row.names = NULL),
       S_tvsurvbart = S$tvsurvbart)          # TVSurvBART survival matrix (test subjects in ID order x GRID)
}

save_rds <- function(x, f) { tmp <- paste0(f, ".tmp", Sys.getpid()); saveRDS(x, tmp); file.rename(tmp, f) }

if (!"--summary" %in% flags) for (seed in SEEDS) for (scen in SCENARIOS) for (cens in CENS) {
  f <- file.path(OUT, sprintf("sim2_%s_cens%02d_seed%03d", scen, round(100 * cens), seed))
  ## done if the cell's files cover METHODS: the full file, or "_norsf" (all but RSF) and "_rsf" (RSF)
  have <- file.exists(paste0(f, c("", "_norsf", "_rsf"), ".rds"))
  if (have[1] || ((have[2] || identical(METHODS, "rsf")) && (have[3] || !"rsf" %in% METHODS))) next
  t0 <- Sys.time(); cat(sprintf("[%s] seed %d, censoring %.2f ... ", format(t0, "%H:%M"), seed, cens))
  res <- tryCatch(run_cell(seed, scen, cens), error = function(e) { cat("FAILED:", conditionMessage(e), "\n"); NULL })
  if (!is.null(res)) {
    save_rds(res, paste0(f, TAG, ".rds"))
    cat(sprintf("done in %.1f min\n", as.numeric(difftime(Sys.time(), t0, units = "mins"))))
  }
}

## -----------------------------------------------------------------------------
## 7. Summary over seeds: mean (Monte Carlo SE) per censoring level x method
## -----------------------------------------------------------------------------
summarize <- function() {
  files <- list.files(OUT, "^sim2_.*\\.rds$", full.names = TRUE)
  if (!length(files)) { cat("No result files in ", OUT, "\n", sep = ""); return(invisible(NULL)) }
  R <- do.call(rbind, lapply(files, function(f) {
    r <- readRDS(f); data.frame(seed = r$seed, scen = r$scen, cens = r$cens, r$metrics) }))
  R <- R[!duplicated(R[c("seed", "scen", "cens", "method")]), ]   # a split cell has oracle_locf twice
  cat("Computing the population truth ...\n")
  truth <- setNames(list(pop_truth()), SCENARIOS)
  tv <- function(field, i) sapply(R$scen, function(s) truth[[s]][[field]][i])
  R$d_rmst_half <- R$rmst_half - tv("rmst", "half")
  R$d_rmst_tau  <- R$rmst_tau - tv("rmst", "full")
  R$d_median    <- R$median - tv("median", 1)
  ms <- function(x) { x <- x[is.finite(x)]; c(mean = mean(x), se = sd(x) / sqrt(length(x))) }
  bias <- function(x) { x <- x[is.finite(x)]; c(bias = mean(x), se = sd(x) / sqrt(length(x)), rmse = sqrt(mean(x^2))) }
  lev <- c("tvsurvbart", "cox", "fcox", "rsf", "oracle_locf", "ref_population")   # row order
  groups <- split(R, list(factor(R$method, lev), R$cens, factor(R$scen, SCENARIOS)), drop = TRUE)
  lab <- function(g) data.frame(scen = g$scen[1], cens = g$cens[1], method = g$method[1], n = nrow(g))

  ## Table 3: RMST(TAU/2), RMST(TAU) and median bias vs the population truth; IBS; RMSE_S; log score
  main <- do.call(rbind, lapply(groups, function(g) data.frame(lab(g), t(c(
    rmst_half = bias(g$d_rmst_half), rmst_tau = bias(g$d_rmst_tau), median = bias(g$d_median),
    ibs = ms(g$ibs), rmse_s = ms(g$rmse_s), logscore = ms(g$logscore))))))
  ## Table 4 / Web Figure 2: mean AUC(t) on the grid, plus the population reference
  ac <- grep("^auc_", names(R), value = TRUE)
  auc <- do.call(rbind, lapply(groups, function(g) data.frame(lab(g), t = GRID,
    auc = colMeans(g[ac], na.rm = TRUE), se = apply(g[ac], 2, function(x) ms(x)[["se"]]))))
  for (s in names(truth)) for (cn in sort(unique(R$cens[R$scen == s])))
    auc <- rbind(auc, data.frame(scen = s, cens = cn, method = "ref_population", n = NA, t = GRID,
                                 auc = truth[[s]]$auc_reference, se = NA))
  auc <- auc[order(factor(auc$scen, SCENARIOS), auc$cens, factor(auc$method, lev), auc$t), ]
  ## Web Table 3: paired differences of the test log score (row minus column method)
  pairs <- list(c("rsf", "tvsurvbart"), c("cox", "tvsurvbart"), c("cox", "rsf"),
                c("fcox", "tvsurvbart"), c("fcox", "rsf"), c("fcox", "cox"))
  ls <- do.call(rbind, lapply(split(R, list(R$cens, factor(R$scen, SCENARIOS)), drop = TRUE), function(g) {
    w <- tapply(g$logscore, list(g$seed, g$method), mean)
    do.call(rbind, lapply(pairs, function(p) if (all(p %in% colnames(w)))
      data.frame(scen = g$scen[1], cens = g$cens[1], pair = paste(p, collapse = " - "),
                 n = sum(is.finite(w[, p[1]] - w[, p[2]])), t(ms(w[, p[1]] - w[, p[2]])))))
  }))
  rownames(main) <- rownames(auc) <- NULL; if (!is.null(ls)) rownames(ls) <- NULL
  write.csv(main, file.path(OUT, "summary_sim2_main.csv"), row.names = FALSE)
  write.csv(auc, file.path(OUT, "summary_sim2_auc.csv"), row.names = FALSE)
  if (!is.null(ls)) write.csv(ls, file.path(OUT, "summary_sim2_logscore.csv"), row.names = FALSE)
  op <- options(width = 200, digits = 4); on.exit(options(op))
  for (s in names(truth)) cat(sprintf("Population truth: RMST(TAU/2) %.4f, RMST(TAU) %.4f, median %.4f\n",
                                      truth[[s]]$rmst[["half"]], truth[[s]]$rmst[["full"]], truth[[s]]$median))
  cat("\nRMST / median bias vs population truth, IBS, RMSE_S, log score (mean and MC SE over seeds):\n")
  print(main)
  cat("\nAUC at t =", AUC_TIMES, "\n")
  print(reshape(auc[sapply(auc$t, function(t) any(abs(t - AUC_TIMES) < 1e-9)), c("scen", "cens", "method", "t", "auc")],
                idvar = c("scen", "cens", "method"), timevar = "t", direction = "wide"), row.names = FALSE)
  cat("\nPaired log-score differences:\n")
  print(ls, row.names = FALSE)
}
summarize()
