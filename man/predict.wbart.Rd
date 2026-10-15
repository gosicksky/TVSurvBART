\name{predict.wbart}
\alias{predict.wbart}
\title{Predict from a TVSurvBART fit}
\description{
A wrapper around \code{\link{pwbart}}. The fit returned by \code{\link{wbart}} has no class, so call this function by its full name; \code{predict(fit, ...)} does not dispatch to it.
}
\usage{
predict.wbart(object, target_time, newdata, mc.cores = 1,
              openmp = (mc.cores.openmp() > 0), ...)
}
\arguments{
\item{object}{A fit returned by \code{\link{wbart}}.}
\item{target_time}{One target time per subject, as in \code{\link{pwbart}}.}
\item{newdata}{Prediction data, as \code{data} in \code{\link{pwbart}}.}
\item{mc.cores}{No effect: the package is built without OpenMP.}
\item{openmp}{Not used.}
\item{...}{Passed to \code{\link{pwbart}}, e.g. \code{n.gh}.}
}
\value{The value of \code{\link{pwbart}}.}
\seealso{\code{\link{pwbart}}}
