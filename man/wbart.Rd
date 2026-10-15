\name{wbart}
\alias{wbart}
\title{Fit TVSurvBART by MCMC}
\description{
Fits the TVSurvBART survival model to longitudinal data. The hazard is the product of a baseline hazard (natural cubic spline on the log scale, or piecewise constant), a BART ensemble of the covariates, and subject-level random effects \eqn{\exp\{b_i^\top z(t)\}} with \eqn{z(t) = ((t/\tau)^{p_1}, \dots, (t/\tau)^{p_q})} and a Huang-Wand prior on \eqn{\Sigma_b}.
}
\usage{
wbart(data, test.data = NULL, start.stop = FALSE,
      ns.degree = 1, ns.knot = c(0, 1, 2),
      baseline.type = "spline", pw.shape = 0.1, pw.rate = 0.1,
      re.max.power = 1, re.powers = NULL, re.time.scale = 1,
      c = 1, d = 1,
      nsCoefMean = 0, nsCoefVar = 1,
      reCoefMean = c(0, 0), reCoefVarInit = NULL,
      reHW.nu = 2, reHW.A = NULL, re.ncp = 1L,
      ns.block = NULL, ns.block.M = NULL,
      sparse = FALSE, theta = 1, omega = 1, a = 0.5, b = 1, augment = FALSE, rho = NULL,
      xinfo = matrix(0, 0, 0), usequants = FALSE, cont = FALSE, rm.const = TRUE,
      sigest = NA, sigdf = 3, sigquant = 0.9, k = 2,
      power = 2, base = 0.95, sigmaf = NA, lambda = 1,
      ntree = 200L, numcut = 100L,
      ndpost = 1000L, nskip = 100L, keepevery = 1L,
      nkeeptrain = ndpost, nkeeptest = ndpost, nkeeptestmean = ndpost, nkeeptreedraws = ndpost,
      printevery = 100L, transposed = FALSE)
}
\arguments{
\item{data}{Data frame or matrix, read by column position (names are ignored). With \code{start.stop = TRUE}: \code{id, start, stop, death, covariates...}, one row per interval \eqn{(start, stop]} with constant covariates. With \code{start.stop = FALSE}: \code{id, time, event, covariates...}, one row per measurement time with \code{event = 0}, then a final row per subject at its exit time with \code{event} 0/1 that repeats the previous row's covariates. Ids must be integer-valued; rows of the long format must be sorted by id and time. At least two non-constant covariate columns (numeric or factor) are needed.}
\item{test.data}{Not used.}
\item{start.stop}{Logical. \code{TRUE} converts counting-process data with \code{\link{convert_start_stop}}; rows are sorted for you, but gaps between a subject's intervals are not kept (exposure runs continuously).}
\item{ns.degree}{Not used: the spline baseline is always a natural cubic spline.}
\item{ns.knot}{On the time scale of the data. For the spline baseline: increasing knots, the first and last being the boundary knots (the spline is linear beyond them); \eqn{K} knots give \eqn{K} coefficients including the intercept. For the piecewise baseline: cutpoints (sorted and de-duplicated); \eqn{K} cutpoints give \eqn{K - 1} bins, the first bin extending down to \eqn{-\infty} and the last up to \eqn{+\infty}. Choose them to span follow-up.}
\item{baseline.type}{\code{"spline"} (natural cubic spline on the log baseline hazard) or \code{"piecewise"} (piecewise-constant baseline hazard).}
\item{pw.shape, pw.rate}{Gamma(shape, rate) prior on each piecewise baseline height; both positive. Piecewise baseline only.}
\item{re.max.power}{If \code{re.powers} is \code{NULL}, the random-effect powers are \code{0:re.max.power}. The default 1 gives a random intercept and slope; use 0 (with \code{reCoefMean = 0}) for a random intercept only.}
\item{re.powers}{Optional vector of distinct non-negative integer powers of \eqn{t/\tau}; overrides \code{re.max.power}.}
\item{re.time.scale}{\eqn{\tau} in \eqn{z(t) = (t/\tau)^p}; a positive number. The paper uses the maximum follow-up time.}
\item{c, d}{Leaf prior: each leaf value \eqn{\theta} is \eqn{W} or \eqn{1/W} with probability 1/2, \eqn{W \sim} Gamma(shape \code{c}, rate \code{d}). The defaults give a very diffuse prior; use \code{c = ntree/sF^2 + 0.5}, \code{d = ntree/sF^2}, which makes the prior SD of the log tree factor about \code{sF} (the paper uses \code{sF = log(1000)/(qnorm(0.975)*sqrt(2))}).}
\item{nsCoefMean, nsCoefVar}{Normal prior (mean, variance) on every spline coefficient, the intercept included. Spline baseline only.}
\item{reCoefMean}{Prior mean of \eqn{b_i}; its length must equal the number of random-effect powers. Keep it at 0: prediction for new subjects assumes a zero mean.}
\item{reCoefVarInit}{Starting value of \eqn{\Sigma_b} (a \eqn{q \times q} matrix); the initial \eqn{b_i} are drawn from a normal distribution with mean \code{reCoefMean} and covariance \code{reCoefVarInit}. \code{NULL} means the identity.}
\item{reHW.nu}{Huang-Wand degrees of freedom \eqn{\nu > 0}; \eqn{\nu = 2} gives uniform marginal correlations.}
\item{reHW.A}{Half-t scales \eqn{A_k} of the random-effect standard deviations, on the log-hazard scale; length 1 or \eqn{q}. \code{NULL} means \code{1e5} (nearly flat); the paper uses 5.}
\item{re.ncp}{Number of joint non-centred Metropolis proposals on \eqn{(\Sigma_b, b)} per sweep (an integer in 0..1000; 0 turns the move off). The paper uses 3.}
\item{ns.block}{\code{TRUE}: update the spline coefficients by one preconditioned block-MALA step per sweep instead of coordinatewise MALA. \code{NULL} reads the environment variable \code{TVSB_NSBLOCK} (default \code{FALSE}). Ignored for the piecewise baseline. Needs \code{nskip >= 400} (\code{>= 1500} for the full adaptation schedule).}
\item{ns.block.M}{Preconditioner of the block-MALA step: \code{"secant"} (inverse secant curvature, the default) or \code{"draws"} (empirical covariance of the draws). \code{NULL} reads \code{TVSB_NSBLOCK_M}.}
\item{sparse, theta, omega, a, b, rho}{Not used (DART arguments inherited from the BART package).}
\item{augment}{Logical. \code{TRUE} draws the split variable of a birth proposal only among the variables that can still split the node.}
\item{xinfo}{Optional cutpoints that replace the computed ones: a matrix with one row of increasing cutpoints per covariate column (one row per factor level), padded with \code{NA}. Do not pass a list: an element shorter than the computed set is silently merged with it.}
\item{usequants}{Logical. \code{TRUE} uses quantiles of each covariate as cutpoints instead of an equally spaced grid.}
\item{cont}{Logical. \code{TRUE} always uses \code{numcut} equally spaced cutpoints, even for covariates with few distinct values.}
\item{rm.const}{Logical. Drop covariates that are constant in \code{data}. Drop them from prediction data too.}
\item{sigest, sigdf, sigquant, k, sigmaf, lambda}{Not used (Gaussian-BART arguments inherited from the BART package).}
\item{power, base}{Tree prior: a node at depth \eqn{r} splits with probability \code{base * (1 + r)^(-power)}.}
\item{ntree}{Number of trees.}
\item{numcut}{Maximum number of cutpoints per covariate (covariates with fewer distinct values use the midpoints between them).}
\item{ndpost}{Number of stored posterior draws.}
\item{nskip}{Number of burn-in sweeps. Most step sizes adapt only at every 500th burn-in sweep, so use \code{nskip >= 500}.}
\item{keepevery}{Thinning: the chain runs \code{nskip + ndpost * keepevery} sweeps and stores every \code{keepevery}-th post-burn-in state.}
\item{nkeeptrain}{Number of stored rows of \code{yhat.train} (0 or a divisor of \code{ndpost}).}
\item{nkeeptest, nkeeptestmean}{Refer to an internal placeholder; leave at the default.}
\item{nkeeptreedraws}{Number of stored tree draws; must equal \code{ndpost} for prediction.}
\item{printevery}{Print a progress line every \code{printevery} sweeps. The sampler prints much more; wrap the call in \code{capture.output()} to silence it.}
\item{transposed}{Internal; leave at \code{FALSE}.}
}
\details{
Each sweep updates the baseline (MALA for the spline coefficients, conjugate Gibbs for the piecewise heights), the trees (birth/death proposals with the leaf values integrated out, then a draw of all leaves), a level shift between the trees and the baseline, and the random effects (MALA for each \eqn{b_i}, Huang-Wand Gibbs for \eqn{\Sigma_b}, the non-centred proposals and a group move). Call \code{set.seed()} before \code{wbart()} to reproduce a chain.

The trees use only the covariate columns. For covariate effects that change over time, add a column equal to each interval's start time divided by \eqn{\tau}.
}
\value{
A list (without a class). The main components, with \eqn{D} = \code{ndpost}, \eqn{n} = rows of the internal long data (one per interval start plus one final row per subject) and \eqn{N} = subjects:
\item{cumHazard}{\eqn{D \times n}: cumulative hazard at each row's time given that draw's \eqn{b_i} (0 at a subject's first row).}
\item{yhat.train}{\code{nkeeptrain} \eqn{\times n}: tree factor \eqn{f(x)} (hazard-ratio scale) at each row. \code{yhat.train.mean} averages it over all post-burn-in sweeps.}
\item{nsDraws, nsMean}{\eqn{D \times K} (spline) or \eqn{D \times (K-1)} (piecewise) baseline coefficients (spline: coefficients of the centred basis; piecewise: log heights) and the centring constants of the spline basis.}
\item{reCoefDraws}{\eqn{(D N) \times q}: \eqn{b_i}; block \eqn{d} holds the \eqn{N} subjects of draw \eqn{d} in \code{trainId} order.}
\item{reCoefCovDraws}{\eqn{(D q) \times q}: \eqn{\Sigma_b}, one \eqn{q \times q} block per draw.}
\item{reCovAuxDraws}{\eqn{D \times q}: Huang-Wand auxiliary variables.}
\item{reNcpAccept, reNcpProposalCov}{Acceptance rate and proposal covariance of the non-centred move.}
\item{varcount, varprob}{Number of splits on each covariate per draw, and the split probabilities (always uniform).}
\item{treedraws}{Cutpoints and trees, used by the prediction functions.}
\item{trainId, NsKnot, rePowers, reTimeScale, reHWnu, reHWA, baselineType, baselineTypeInt, pwShape, pwRate}{Training subject ids and the settings used.}
\item{BartMeanDraws, splineYDraws}{Internal diagnostics: the level shift accepted in that sweep (0 if rejected), and the baseline log hazard at each subject's exit time before that sweep's group move (so it does not match \code{nsDraws}).}
\item{yhat.test, yhat.test.mean}{Internal placeholder; ignore.}
With \code{ns.block = TRUE} the list also has the block-MALA diagnostics \code{nsBlock*}.
}
\seealso{
\code{\link{pwbart}}, \code{\link{pwbart_grid}}, \code{\link{predict.wbart}}, \code{\link{convert_start_stop}}
}
\examples{
set.seed(1)
sim_subject <- function(id) {
  v  <- seq(0, 2.5, by = 0.5)
  x1 <- rnorm(1) + 0.3 * v; x2 <- rbinom(1, 1, 0.5)
  Tev <- rexp(1, 0.3 * exp(0.5 * x1[1] + 0.5 * x2)); Y <- min(Tev, runif(1, 1, 4), 3)
  k <- sum(v < Y)
  data.frame(id = id, start = v[1:k], stop = c(v[1:k], Y)[-1],
             death = c(rep(0, k - 1), as.numeric(Tev <= Y)), x1 = x1[1:k], x2 = x2)
}
dat <- do.call(rbind, lapply(1:60, sim_subject))
log <- capture.output(
  fit <- wbart(dat, start.stop = TRUE, ntree = 20, c = 20 / 2.49^2 + 0.5, d = 20 / 2.49^2,
               ns.knot = 0:3, re.time.scale = 3, reHW.A = 5, nskip = 100, ndpost = 100)
)
dim(fit$cumHazard)

## survival of a new subject on a time grid
new <- data.frame(id = 999, time = c(0, 1, 2), x1 = c(0, 0.3, 0.6), x2 = 1)
log <- capture.output(pred <- pwbart_grid(c(1, 2, 3), new, fit))
colMeans(exp(-pred$cumHazard.test[, 1, ]))
}
