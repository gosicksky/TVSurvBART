\name{pwbart}
\alias{pwbart}
\title{Posterior predictive cumulative hazard at one time per subject}
\description{
Computes posterior draws of the cumulative hazard and the hazard of each subject in \code{data} at its own target time, from a \code{\link{wbart}} fit.
}
\usage{
pwbart(target_time, data, object, mc.cores = 1L, transposed = FALSE,
       dodraws = TRUE, n.gh = 16L, nice = 19L)
}
\arguments{
\item{target_time}{Numeric vector with one target time per subject, in ascending order of subject id. It is not recycled and its length is not checked.}
\item{data}{Prediction data, read by column position: \code{id, time, covariates...} (no event column), with the covariates in the same order as in training. One row per covariate measurement, sorted by id and time, with distinct times within a subject (not checked). The cumulative hazard is accumulated from each subject's first row, and after the last row its covariates are carried forward.}
\item{object}{A fit returned by \code{\link{wbart}}.}
\item{mc.cores}{No effect: the package is built without OpenMP.}
\item{transposed}{Internal; leave at \code{FALSE}.}
\item{dodraws, nice}{Not used.}
\item{n.gh}{Number of Gauss-Hermite nodes per random-effect dimension used to integrate out \eqn{b} for subjects not seen in training. Deterministic Gauss-Hermite quadrature is used for a single random-effect term or the (intercept, slope) design; other designs use \code{n.gh^2} Monte Carlo draws. \code{n.gh <= 1} uses a single Monte Carlo draw of \eqn{b} per posterior draw.}
}
\details{
Subjects are matched to the training data by id, after conversion to integer, so use integer-valued numeric ids. A subject seen in training uses its posterior \eqn{b_i}, draw by draw. For a new subject (an id not used in training), \eqn{b} is integrated out against \eqn{N(0, \Sigma_b)}, and \code{cumHazard.test} holds \eqn{-\log \bar S(t)}, where \eqn{\bar S} is the survival averaged over \eqn{b}. Monte Carlo paths use R's random numbers, so call \code{set.seed()} to reproduce them.
}
\value{
A list with
\item{cumHazard.test}{Draws \eqn{\times} subjects: cumulative hazard at \code{target_time}. Per-draw survival is \code{exp(-cumHazard.test)}.}
\item{hazard.test}{Draws \eqn{\times} subjects: hazard at \code{target_time}; for a new subject, the hazard of \eqn{\bar S}.}
\item{yhat.test}{Draws \eqn{\times} rows of \code{data}: tree factor \eqn{f(x)} of each row.}
}
\seealso{\code{\link{wbart}}, \code{\link{pwbart_grid}}, \code{\link{predict.wbart}}}
\examples{
## see the examples of wbart()
}
