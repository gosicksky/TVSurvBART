
## BART: Bayesian Additive Regression Trees
## Copyright (C) 2017-2021 Robert McCulloch and Rodney Sparapani
## wbart.R

## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 2 of the License, or
## (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program; if not, a copy is available at
## https://www.R-project.org/Licenses/GPL-2


### This is for Cox Bart

## ----------------------------------------------------------------------------
## convert_start_stop()
## Convert survival-style (start, stop] counting-process input -- like
## Surv(time1, time2, death) in the `survival` package -- into the internal long
## format wbart() consumes.
##
## Input columns (BY POSITION):
##   1: subject id
##   2: time1  (interval start / left boundary; left-truncation if > 0)
##   3: time2  (interval stop  / right boundary; the subject's last time2 is the
##              event/censoring time)
##   4: death  (event indicator for that interval; 1 only on the subject's last
##              interval if the event occurred, else 0)
##   5+: time-varying covariate(s), value held constant over (time1, time2]
##
## Internal format produced (by position): id, time, event, covariates...
##   - one row per interval placed at its START time1, event = 0, carrying that
##     interval's covariate (used as the piecewise-constant value over the
##     interval [time1, time2]);
##   - one final row per subject at the last stop time2, carrying the event
##     indicator and a copy of the last interval's covariates.  This row adds no
##     exposure, but the event term of the likelihood (the log hazard at time2) is
##     evaluated at its covariates, i.e. at the value carried forward from the
##     last interval start (left-continuous).  In long-format input
##     (start.stop = FALSE) the final row must therefore repeat the previous row.
## This reproduces exactly the interval/covariate structure the C++ sampler
## integrates over (intervals [obsTime[j], obsTime[j+1]] with the start covariate;
## last row = event time + indicator).
## ----------------------------------------------------------------------------
convert_start_stop <- function(data) {
  data <- as.data.frame(data)
  if (ncol(data) < 4)
    stop("start.stop input needs >= 4 columns: subject, time1, time2, death, [covariates...]")
  names(data)[1:4] <- c("id", "t1", "t2", "event")
  covcols <- if (ncol(data) >= 5) names(data)[5:ncol(data)] else character(0)
  
  ## order by subject, then interval start (then stop, to be safe)
  data <- data[order(data$id, data$t1, data$t2), , drop = FALSE]
  if (any(data$t2 <= data$t1))
    warning("convert_start_stop: some intervals have time2 <= time1")
  
  is_last <- !duplicated(data$id, fromLast = TRUE)   # subject's final interval
  
  ## A: one internal row per interval, placed at its start, event = 0
  A <- data.frame(id = data$id, time = data$t1, event = 0, check.names = FALSE)
  if (length(covcols)) A <- cbind(A, data[, covcols, drop = FALSE])
  
  ## B: final row per subject at the last stop time, carrying the event
  dl <- data[is_last, , drop = FALSE]
  B <- data.frame(id = dl$id, time = dl$t2, event = as.numeric(dl$event),
                  check.names = FALSE)
  if (length(covcols)) B <- cbind(B, dl[, covcols, drop = FALSE])
  
  out <- rbind(A, B)
  out <- out[order(out$id, out$time), , drop = FALSE]
  rownames(out) <- NULL
  out
}

wbart=function(
    data, test.data=NULL, start.stop=FALSE,
    ns.degree=1, ns.knot=c(0,1,2),
    baseline.type="spline", pw.shape=0.1, pw.rate=0.1,
    re.max.power=1, re.powers=NULL, re.time.scale=1, c=1,d=1,
    nsCoefMean=0, nsCoefVar=1,
    reCoefMean=c(0,0), reCoefVarInit=NULL,
    reHW.nu=2, reHW.A=NULL,
    re.ncp=1L,
    ns.block=NULL, ns.block.M=NULL,
    sparse=FALSE, theta=1, omega=1,
    a=0.5, b=1, augment=FALSE, rho=NULL,
    xinfo=matrix(0.0,0,0), usequants=FALSE,
    cont=FALSE, rm.const=TRUE,
    sigest=NA, sigdf=3, sigquant=.90,
    k=2.0, power=2.0, base=.95,
    sigmaf=NA, lambda=1,
    ntree=200L, numcut=100L,
    ndpost=1000L, nskip=100L, keepevery=1L,
    nkeeptrain=ndpost, nkeeptest=ndpost,
    nkeeptestmean=ndpost, nkeeptreedraws=ndpost,
    printevery=100L, transposed=FALSE
)
{
  #--------------------------------------------------
  
  # ---- Baseline hazard type ----
  # "spline"    : natural cubic spline on the log-hazard (legacy default).
  # "piecewise" : piecewise-constant ("Bayesian Breslow"): ns.knot are the bin
  #               CUTPOINTS (K cutpoints -> K-1 heights), each height
  #               lambda_k ~ Gamma(pw.shape, pw.rate) iid with an exact
  #               conjugate gamma Gibbs update. Times beyond the last cutpoint
  #               use the last bin (flat tail).
  baseline.type <- match.arg(baseline.type, c("spline", "piecewise"))
  if (baseline.type == "piecewise") {
    ns.knot <- sort(unique(unlist(ns.knot)))
    if (length(ns.knot) < 2)
      stop("Error: baseline.type='piecewise' needs >= 2 distinct ns.knot cutpoints.")
    pw.shape <- as.numeric(pw.shape); pw.rate <- as.numeric(pw.rate)
    if (!is.finite(pw.shape) || pw.shape <= 0 || !is.finite(pw.rate) || pw.rate <= 0)
      stop("Error: pw.shape and pw.rate must be positive (Gamma prior on lambda_k).")
  }
  
  if (is.null(re.powers)) {
    re.powers <- 0:re.max.power
  } else {
    re.powers <- as.integer(re.powers)
    if (!length(re.powers) || any(is.na(re.powers)) || any(re.powers < 0))
      stop("Error: re.powers must contain one or more non-negative integer powers.")
    if (any(duplicated(re.powers)))
      stop("Error: re.powers contains duplicate powers.")
    re.max.power <- max(re.powers)
  }
  q <- length(re.powers)
  
  if(q != length(reCoefMean)) {
    stop("Error: re.powers and reCoefMean do not match!")
  }
  
  re.time.scale <- as.numeric(re.time.scale)
  if (length(re.time.scale) != 1L || !is.finite(re.time.scale) || re.time.scale <= 0)
    stop("Error: re.time.scale must be a single positive number (tau for z(t) = (t/tau)^power).")
  
  # Joint non-centred (Sigma_b, b) Metropolis move: proposals per sweep (0 = off,
  # leaving the conjugate Gibbs draw of Sigma_b | b as the only Sigma_b update).
  if (length(re.ncp) != 1L || !is.finite(re.ncp) || re.ncp < 0 || re.ncp != round(re.ncp) ||
      re.ncp > 1000)
    stop("Error: re.ncp must be a single integer in 0..1000 (proposals per sweep).")

  # ---- Posterior storage and thinning (ndpost, keepevery) ----
  # The chain runs nskip burn-in sweeps, then ndpost * keepevery KEPT sweeps. keepevery
  # never changes the chain (same RNG stream, same states, same .Random.seed afterwards):
  # it only selects what is stored. ndpost is the number of STORED draws. Draw d
  # (d = 1..ndpost) is the state at the end of kept sweep d * keepevery, i.e. 0-based
  # sweep index nskip + d * keepevery - 1 (every keepevery-th kept sweep, the last kept
  # sweep included). EVERY per-draw output is stored at exactly these sweeps, so row / block
  # d of each one is the same state of the chain:
  #   treedraws$trees (draw d = the d-th set of ntree trees), varcount[d, ], varprob[d, ],
  #   nsDraws[d, ], reCoefCovDraws block d (rows (d-1) q + 1:q), reCovAuxDraws[d, ],
  #   reCoefDraws block d (rows (d-1) nSubject + 1:nSubject, subjects in trainId order),
  #   BartMeanDraws[d, ], splineYDraws[d, ], cumHazard[d, ], yhat.train[d, ], yhat.test[d, ].
  # cpwbart / pwbart / pwbart_grid / predict.wbart pair tree draw d with row / block d of
  # nsDraws, reCoefCovDraws and reCoefDraws, so every predicted draw is one state. The fit
  # with keepevery = k is the keepevery = 1 fit of the same chain (ndpost * k draws)
  # restricted to draws k, 2k, ..., ndpost * k.
  # nkeeptrain / nkeeptest / nkeeptestmean / nkeeptreedraws (each 0 or a divisor of ndpost;
  # otherwise reset to ndpost below) sub-sample the ndpost stored draws: e.g. yhat.train
  # holds draws r, 2r, ..., ndpost with r = ndpost / nkeeptrain. Prediction needs
  # nkeeptreedraws = ndpost (the default): pwbart_inputs() refuses a fit whose tree draws do
  # not match the rows of nsDraws / reCoefCovDraws / reCoefDraws.
  # Chain summaries use every kept sweep: yhat.train.mean averages all ndpost * keepevery
  # kept sweeps (the same value for every keepevery), reNcpAccept / nsBlockAccept count every
  # kept sweep; yhat.test.mean averages the nkeeptestmean stored draws.
  # keepevery = 1 stores every kept sweep, exactly as before this storage fix; with
  # keepevery > 1 the per-sweep series were previously stored UNthinned (ndpost * keepevery
  # rows) next to thinned tree draws, which mispaired draws in prediction.
  if (length(keepevery) != 1L || !is.numeric(keepevery) || !is.finite(keepevery) ||
      keepevery < 1 || keepevery != round(keepevery))
    stop("Error: keepevery must be a single integer >= 1 (the state is stored at every keepevery-th kept sweep).")
  keepevery <- as.integer(keepevery)

  # Spline-coefficient update. ns.block = FALSE (default): coordinatewise MALA, one
  # scalar step per coefficient per sweep (the sampler exactly as before).
  # ns.block = TRUE: ONE preconditioned block MALA step on the whole coefficient
  # vector per sweep, beta* = beta + (h^2/2) M grad log pi(beta) + h M^(1/2) eps,
  # accepted with the full MALA ratio; h and M adapt during burn-in only, then stay
  # frozen (fit outputs nsBlockAccept, nsBlockH, nsBlockM, nsBlockMLog, ...). NULL =
  # environment variable TVSB_NSBLOCK (TRUE/FALSE or 1/0; default FALSE), so a caller
  # that does not pass the argument (the simulation suite) can switch it on.
  # ns.block.M = burn-in estimator of the preconditioner M: "secant" (default: the
  # inverse of the secant estimate of the negative Hessian of log pi(beta | rest),
  # i.e. the conditional covariance of beta given the trees and b, from the gradient
  # pairs every step computes anyway) or "draws" (empirical covariance of the beta
  # draws; their marginal spread includes the Move-1 / group-move drift of the level
  # and slope coefficients, far wider than their conditional spread, so h is then
  # throttled by beta_0 -- much slower on the production data); NULL = env
  # TVSB_NSBLOCK_M (default "secant").
  # M is re-estimated every 500 burn-in sweeps for nskip >= 1500; shorter burn-ins
  # use a proportionally shorter schedule (period 100 floor(nskip/400)), and below
  # nskip = 400 M cannot be adapted at all (warning). Spline baseline only: ignored
  # for "piecewise".
  if (is.null(ns.block)) {
    ev <- Sys.getenv("TVSB_NSBLOCK", "")
    ns.block <- if (nzchar(ev)) {
      v <- suppressWarnings(as.logical(ev))
      if (is.na(v)) v <- suppressWarnings(as.numeric(ev)) != 0
      v
    } else FALSE
  }
  if (length(ns.block) != 1L || !(is.logical(ns.block) || is.numeric(ns.block)) || is.na(ns.block))
    stop("Error: ns.block (or env TVSB_NSBLOCK) must be a single TRUE/FALSE (or 1/0).")
  ns.block <- as.logical(ns.block)
  if (is.null(ns.block.M)) {
    ev <- Sys.getenv("TVSB_NSBLOCK_M", "")
    ns.block.M <- if (nzchar(ev)) ev else "secant"
  }
  ns.block.M <- match.arg(ns.block.M, c("secant", "draws"))
  if (ns.block && baseline.type == "piecewise") {
    message("ns.block = TRUE is ignored for baseline.type = 'piecewise': the piecewise baseline keeps its exact conjugate Gibbs update (drawLambda).")
    ns.block <- FALSE
  }
  if (ns.block && nskip < 400)
    warning(sprintf(paste0("ns.block = TRUE with nskip = %d < 400: the block-MALA preconditioner M ",
                           "cannot be adapted (only the step h adapts), so the spline coefficients mix ",
                           "poorly; use nskip >= 400 (>= 1500 for the full schedule) or ns.block = FALSE."),
                    as.integer(nskip)), call. = FALSE)
  
  # ---- RE-covariance prior: Huang-Wand hyperprior (the only option) ----
  reHW.nu <- as.numeric(reHW.nu)
  if (length(reHW.nu) != 1L || !is.finite(reHW.nu) || reHW.nu <= 0)
    stop("Error: reHW.nu must be a single positive number (2 => marginal-uniform correlations).")
  # Half-t scale per RE dimension. Default: weakly informative (large A) so each
  # SD chooses its own scale instead of being tied to a single IW scale.
  if (is.null(reHW.A)) reHW.A <- rep(1e5, q)
  reHW.A <- as.numeric(reHW.A)
  if (length(reHW.A) == 1L) reHW.A <- rep(reHW.A, q)
  if (length(reHW.A) != q || any(!is.finite(reHW.A)) || any(reHW.A <= 0))
    stop("Error: reHW.A must be q positive numbers (half-t scale per RE dimension).")
  
  #longitudinal survival data preparation
  #x.train: first column is subject id, second column is time, third column is the event indicator and the rest of the columns are the covariates
  #each subject has multiple rows, and get the row id for last event of each subject which is y.train
  
  #train data
  data <- as.data.frame(data)
  # Accept survival-style (start, stop] counting-process input (subject, time1,
  # time2, death, covariates...) and convert to the internal long format.
  if(start.stop) data <- convert_start_stop(data)
  colnames(data)[1:3] <- c("id","time","event")
  colnames(data)[4:ncol(data)] <- paste0("X", 1:(ncol(data)-3))
  
  #give the row id for last event of each subject, need minus 1 because C++ starts from 0
  x.train.index <- unname(c(0,data %>% mutate(row_id = row_number()) %>%
                              group_by(id) %>% slice_max(time) %>% ungroup() %>% dplyr::select(row_id) %>% unlist()) - 1)
  
  y.train <- unname(data %>% group_by(id) %>% filter(row_number() == n()) %>% ungroup() %>% dplyr::select(time) %>% unlist())
  x.train.event <- data %>% dplyr::select(event) %>% unlist()
  x.train.obs.time <- unname(data$time %>% unlist())
  
  x.train <- data[, -c(1,2,3)]
  
  w=rep(1,length(y.train))
  
  subject_n = length(y.train)
  obs_n = nrow(x.train)
  
  ns.knot <- unlist(ns.knot)
  
  #test data
  x.test <- x.train[1:2,]
  
  if(!transposed) {
    temp = bartModelMatrix(x.train, numcut, usequants=usequants,
                           cont=cont, xinfo=xinfo, rm.const=rm.const)
    x.train = t(temp$X)
    numcut = temp$numcut
    xinfo = temp$xinfo
    if(length(x.test)>0) {
      x.test = bartModelMatrix(x.test)
      x.test = t(x.test[ , temp$rm.const])
    }
    rm(temp)
  }
  
  p = nrow(x.train)
  np = ncol(x.test)
  
  #--------------------------------------------------
  #set nkeeps for thinning: sub-samples of the ndpost stored draws (see the storage note above)
  if((nkeeptrain!=0) & ((ndpost %% nkeeptrain) != 0)) {
    nkeeptrain=ndpost
    cat('*****nkeeptrain set to ndpost\n')
  }
  if((nkeeptest!=0) & ((ndpost %% nkeeptest) != 0)) {
    nkeeptest=ndpost
    cat('*****nkeeptest set to ndpost\n')
  }
  if((nkeeptestmean!=0) & ((ndpost %% nkeeptestmean) != 0)) {
    nkeeptestmean=ndpost
    cat('*****nkeeptestmean set to ndpost\n')
  }
  if((nkeeptreedraws!=0) & ((ndpost %% nkeeptreedraws) != 0)) {
    nkeeptreedraws=ndpost
    cat('*****nkeeptreedraws set to ndpost\n')
  }
  #--------------------------------------------------
  # Initial Sigma_b (and initial b_i scatter). Default I preserves legacy
  # behavior; supply reCoefVarInit to start the chain at a scale-appropriate
  # value (the (b, Sigma) Gibbs coupling mixes slowly, so the start matters).
  if (is.null(reCoefVarInit)) reCoefVarInit <- diag(rep(1, q))
  reCoefVar <- as.matrix(reCoefVarInit)
  if (nrow(reCoefVar) != q || ncol(reCoefVar) != q)
    stop("Error: reCoefVarInit must be a q x q matrix matching re.powers.")
  reCoef <- mvtnorm::rmvnorm(n = subject_n, mean = reCoefMean, sigma = reCoefVar)
  if (q == 1) reCoef <- matrix(reCoef, ncol = 1)
  
  res = cwbart(
    obs_n,  #number of observations in training data
    subject_n,  #number of subjects in training data
    p,  #dimension of x
    np, #number of observations in test data
    x.train,   #pxn training data x
    x.train.obs.time,   #obs_n training data time
    x.train.index,   #n index of last event of each subject
    x.train.event,   #n event data x
    y.train,   #n training data y
    x.test,   #p*np test data x
    ns.degree,  #degree of natural spline
    ns.knot,  #knots of natural spline
    re.max.power, #max degree of random effect
    re.powers, #powers included in the random effect design
    re.time.scale, #time standardization for the RE design: z(t) = (t/scale)^power
    c,  #c parameter for the prior of the BART
    d,  #d parameter for the prior of the BART
    nsCoefMean,  #mean of the prior of the natural spline coefficients
    nsCoefVar,  #sigma of the prior of the natural spline coefficients (level + linear)
    reCoefMean,  #random-effect prior mean mu_b
    reHW.nu,   #Huang-Wand nu (df; 2 => uniform correlations)
    reHW.A,    #Huang-Wand half-t scale per RE dimension
    t(reCoef), #initial value of the random effect coefficients
    reCoefVar, #initial value of the random effect covariance matrix
    ntree,
    numcut,
    ndpost*keepevery,   #number of kept sweeps (ndpost stored draws, one every keepevery-th)
    nskip,
    power,
    base,
    #tau, nu, lambda, sigest: Gaussian-BART prior slots, not used by the survival sampler
    #(the engine only prints them; nu, lambda and sigest get their old default values)
    1,
    3,
    1,
    1,
    w,
    sparse,
    theta,
    omega,
    1:p,   #grp: DART slot, not used by the survival sampler
    a,
    b,
    p,     #rho: DART slot, not used by the survival sampler
    augment,
    nkeeptrain,
    nkeeptest,
    nkeeptestmean,
    nkeeptreedraws,
    printevery,
    xinfo,
    #joint non-centred (Sigma_b, b) Metropolis proposals per sweep (0 = off)
    as.integer(re.ncp),
    #baseline type (0 = spline, 1 = piecewise) + Gamma prior on the heights
    as.integer(baseline.type == "piecewise"),
    as.numeric(pw.shape),
    as.numeric(pw.rate),
    #spline coefficients: 1 = preconditioned block MALA, 0 = coordinatewise MALA; M estimator
    as.integer(ns.block),
    match(ns.block.M, c("draws", "secant")) - 1L,
    #thinning: every per-draw output is stored at every keepevery-th kept sweep
    keepevery
  )
  
  res$trainId <- unique(data$id)
  res$NsKnot <- ns.knot
  res$baselineType <- baseline.type   # "spline" | "piecewise"; prediction wrappers forward it
  return(res)
}
