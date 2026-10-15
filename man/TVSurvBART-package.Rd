\name{TVSurvBART-package}
\alias{TVSurvBART-package}
\alias{TVSurvBART}
\docType{package}
\title{
  Bayesian Nonparametric Survival Regression with Time-Varying Covariates and Random Effects
}
\description{
  Fits TVSurvBART, a continuous-time Bayesian survival model for longitudinal data recorded in counting-process (start-stop) form. The hazard combines a baseline hazard (natural cubic spline or piecewise constant), a Bayesian additive regression tree ensemble of the time-varying covariates, and subject-level random effects (a random intercept, optionally with a random slope in time) with a Huang-Wand covariance prior.
}
\details{
  \code{\link{wbart}} fits the model by MCMC. \code{\link{pwbart}}, \code{\link{pwbart_grid}} and \code{\link{predict.wbart}} return posterior draws of the cumulative hazard for training and new subjects. See also \url{https://github.com/gosicksky/TVSurvBART}.
}
\author{
  Keming Zhang, Yudan Zou, Fan Li and Liangyuan Hu.

  Maintainer: Keming Zhang <keming_zhang@brown.edu>
}
\references{
  Zhang, K., Zou, Y., Li, F. and Hu, L. Bayesian nonparametric survival regression with time-varying covariates and random effects. Manuscript.
}
\keyword{ package }
