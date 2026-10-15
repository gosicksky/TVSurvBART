\name{TVSurvBART-internal}
\alias{bartModelMatrix}
\alias{pwbart_inputs}
\alias{cwbart}
\alias{cpwbart}
\alias{cpwbart_grid}
\alias{mc.cores.openmp}
\alias{mc_cores_openmp}
\title{Internal TVSurvBART functions}
\description{
Internal functions used by \code{\link{wbart}}, \code{\link{pwbart}} and \code{\link{pwbart_grid}}: covariate-matrix construction and cutpoints (\code{bartModelMatrix}), input preparation for prediction (\code{pwbart_inputs}), the compiled sampler and predictors (\code{cwbart}, \code{cpwbart}, \code{cpwbart_grid}) and an OpenMP query (\code{mc.cores.openmp}). They are not meant to be called directly.
}
\keyword{internal}
