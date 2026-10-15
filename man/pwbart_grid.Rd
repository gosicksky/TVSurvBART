\name{pwbart_grid}
\alias{pwbart_grid}
\title{Posterior predictive cumulative hazard on a time grid}
\description{
Like \code{\link{pwbart}}, but evaluates every subject at all times of a shared grid in one call.
}
\usage{
pwbart_grid(time_grid, data, object, n.gh = 16L, want.hazard = FALSE, transposed = FALSE)
}
\arguments{
\item{time_grid}{Numeric vector of target times shared by all subjects.}
\item{data}{Prediction data, as in \code{\link{pwbart}}.}
\item{object}{A fit returned by \code{\link{wbart}}.}
\item{n.gh}{Number of Gauss-Hermite nodes per random-effect dimension for subjects not seen in training. New subjects need \code{n.gh > 1} and a single random-effect term or the (intercept, slope) design; otherwise the call stops (use \code{\link{pwbart}}).}
\item{want.hazard}{Logical. Also return the hazard.}
\item{transposed}{Internal; leave at \code{FALSE}.}
}
\value{
A list with
\item{cumHazard.test}{Array draws \eqn{\times} subjects \eqn{\times} \code{length(time_grid)}. Slice \code{[, , k]} equals \code{pwbart(rep(time_grid[k], n), data, object, n.gh = n.gh)$cumHazard.test}, where \code{n} is the number of subjects.}
\item{hazard.test}{Same dimensions; only with \code{want.hazard = TRUE}.}
}
\seealso{\code{\link{wbart}}, \code{\link{pwbart}}}
\examples{
## see the examples of wbart()
}
