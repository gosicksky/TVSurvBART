
## BART: Bayesian Additive Regression Trees
## Copyright (C) 2017-2018 Robert McCulloch and Rodney Sparapani

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

## cpwbart inputs that do not depend on the target times (shared by pwbart and
## pwbart_grid)
pwbart_inputs = function(
    data,		#x matrix to predict at: n length
    object,		#$treedraws from wbart
    transposed=FALSE
)
{
  data <- as.data.frame(data)
  colnames(data)[1:2] <- c("id","time")
  colnames(data)[3:ncol(data)] <- paste0("X", 1:(ncol(data)-2))

  x.test.index <- unname(c(0,data %>% mutate(row_id = row_number()) %>%
                             group_by(id) %>% slice_max(time) %>% ungroup() %>% dplyr::select(row_id) %>% unlist()) - 1)
  x.test.obs.time <- unname(data$time %>% unlist())

  x.test <- data[, -c(1,2)]

  if(!transposed) x.test <- t(bartModelMatrix(x.test))

  nsKnot <- object$NsKnot
  nsMean <- object$nsMean
  nsDraws <- t(object$nsDraws)
  treedraws <- object$treedraws
  reCoefDraws <- t(object$reCoefDraws)
  reCoefCovDraws <- t(object$reCoefCovDraws)
  trainId <- object$trainId # the id for training id
  testId <- unique(data$id) # the id for test data
  rePowers <- if (!is.null(object$rePowers)) as.integer(object$rePowers) else 0:(ncol(object$reCoefDraws) - 1)
  reDegree <- max(rePowers) # highest random-effect polynomial power
  # time standardization used during fitting: z(t) = (t/reTimeScale)^power
  reTimeScale <- if (!is.null(object$reTimeScale)) as.numeric(object$reTimeScale) else 1
  # baseline type used during fitting ("spline" default for fits predating the option)
  baselineType <- if (!is.null(object$baselineType)) object$baselineType else "spline"

  p <- length(treedraws$cutpoints)

  if(p!=nrow(x.test))
    stop(paste0('The number of columns in x.test must be equal to ', p))

  ## Draw pairing: cpwbart / cpwbart_grid pair tree draw d with row d of nsDraws and
  ## block d of reCoefCovDraws / reCoefDraws, which wbart stores at the same sweeps
  ## (every keepevery-th kept sweep; see wbart). Refuse objects whose counts differ
  ## (nkeeptreedraws < ndpost, a fit from an engine that stored the per-sweep series
  ## unthinned with keepevery > 1, or draws subset inconsistently): they would pair
  ## different states of the chain (or read past the end of the draws).
  nTreeDraws <- suppressWarnings(as.integer(strsplit(trimws(substr(treedraws$trees, 1, 64)), "[[:space:]]+")[[1]][1]))
  q <- NCOL(object$reCoefCovDraws)
  if(is.na(nTreeDraws) || NROW(object$nsDraws) != nTreeDraws || NROW(object$reCoefCovDraws) != nTreeDraws * q ||
     (NROW(object$reCoefDraws) != 0L && NROW(object$reCoefDraws) != nTreeDraws * length(trainId)))
    stop(sprintf(paste0("draw counts do not pair: %s tree draws, %d nsDraws rows, %g reCoefCovDraws blocks, ",
                        "%g reCoefDraws blocks (every per-draw output must hold one row / block per tree draw)"),
                 format(nTreeDraws), NROW(object$nsDraws), NROW(object$reCoefCovDraws) / q,
                 NROW(object$reCoefDraws) / length(trainId)))
  list(nsKnot = nsKnot, nsMean = nsMean, trainId = trainId, testId = testId, treedraws = treedraws,
       nsDraws = nsDraws, reCoefDraws = reCoefDraws, reCoefCovDraws = reCoefCovDraws, reDegree = reDegree,
       rePowers = rePowers, reTimeScale = reTimeScale, x.test.obs.time = x.test.obs.time,
       x.test.index = x.test.index, x.test = x.test, baselineType = baselineType)
}

pwbart = function(
    target_time, #a time vector for every subject: nSubject length
    data,		#x matrix to predict at: n length
    object,		#$treedraws from wbart
    mc.cores=1L,         #thread count
    transposed=FALSE,
    dodraws=TRUE,
    n.gh=16L,            #Gauss-Hermite nodes/dim for unseen-subject RE integration (<=1: legacy single draw)
    nice=19L             #mc.pwbart only
)
{
  a <- pwbart_inputs(data, object, transposed)
  res = cpwbart(
    a$nsKnot, #natural-spline knot
    a$nsMean, #natural-spline mean
    a$trainId,	#the id for training data
    a$testId, #the id for test data
    a$treedraws,	#trees list
    a$nsDraws, #natural-spline draws
    a$reCoefDraws,  #random effect coefficient draws
    a$reCoefCovDraws, #random effect covariance draws
    a$reDegree,
    a$rePowers,
    a$reTimeScale,
    as.integer(n.gh),
    target_time,
    a$x.test.obs.time, #the time vector for each observation
    a$x.test.index, #index of the last observation for each subject
    a$x.test,      #the test x
    mc.cores,   	#thread count
    as.integer(a$baselineType == "piecewise") #baseline type (0 = spline, 1 = piecewise)
  )
  return(res)
}

## pwbart for a grid of target times shared by every subject, in one engine call:
## cumHazard.test[, , k] (and hazard.test[, , k] with want.hazard) is identical to
## pwbart(rep(time_grid[k], n), data, object, n.gh = n.gh) $cumHazard.test (resp.
## $hazard.test), an ndraws x n x length(time_grid) array. The trees are parsed
## and predicted once and every distinct segment integral is computed once for the
## whole grid (see cpwbart_grid). Deterministic paths only: unseen subjects need
## n.gh > 1 and an (intercept, slope) or single-column RE design; pwbart keeps
## the Monte-Carlo paths.
pwbart_grid = function(
    time_grid,           #target times, shared by every subject
    data,		#x matrix to predict at: n length
    object,		#$treedraws from wbart
    n.gh=16L,            #Gauss-Hermite nodes/dim for unseen-subject RE integration (> 1)
    want.hazard=FALSE,   #also return hazard.test
    transposed=FALSE
)
{
  a <- pwbart_inputs(data, object, transposed)
  res = cpwbart_grid(
    a$nsKnot, #natural-spline knot
    a$nsMean, #natural-spline mean
    a$trainId,	#the id for training data
    a$testId, #the id for test data
    a$treedraws,	#trees list
    a$nsDraws, #natural-spline draws
    a$reCoefDraws,  #random effect coefficient draws
    a$reCoefCovDraws, #random effect covariance draws
    a$rePowers,
    a$reTimeScale,
    as.integer(n.gh),
    as.numeric(time_grid),
    a$x.test.obs.time, #the time vector for each observation
    a$x.test.index, #index of the last observation for each subject
    a$x.test,      #the test x
    as.integer(a$baselineType == "piecewise"), #baseline type (0 = spline, 1 = piecewise)
    as.logical(want.hazard)
  )
  if (isTRUE(res$fallback)) {
    ## the engine declined the shared-grid shortcut (a build that contracts
    ## floating-point multiply-adds, or a startup-check mismatch): one cpwbart call
    ## per grid time, i.e. exactly pwbart(rep(time_grid[k], n), ...)
    message("pwbart_grid: ", res$reason, "; using one cpwbart call per grid time")
    n <- length(a$testId); J <- length(time_grid)
    ch <- hz <- NULL
    for (k in seq_len(J)) {
      r <- cpwbart(a$nsKnot, a$nsMean, a$trainId, a$testId, a$treedraws, a$nsDraws,
                   a$reCoefDraws, a$reCoefCovDraws, a$reDegree, a$rePowers, a$reTimeScale,
                   as.integer(n.gh), rep(as.numeric(time_grid[k]), n), a$x.test.obs.time,
                   a$x.test.index, a$x.test, 1L, as.integer(a$baselineType == "piecewise"))
      if (is.null(ch)) {
        ch <- array(0, c(nrow(r$cumHazard.test), n, J))
        if (want.hazard) hz <- ch
      }
      ch[, , k] <- r$cumHazard.test
      if (want.hazard) hz[, , k] <- r$hazard.test
    }
    res <- list(cumHazard.test = ch)
    if (want.hazard) res$hazard.test <- hz
  }
  return(res)
}
