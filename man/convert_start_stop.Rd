\name{convert_start_stop}
\alias{convert_start_stop}
\title{Convert counting-process data to the long format of wbart()}
\description{
Converts start-stop data, as used by \code{Surv(start, stop, death)}, to the long format that \code{\link{wbart}} uses internally. \code{wbart(..., start.stop = TRUE)} calls it for you.
}
\usage{
convert_start_stop(data)
}
\arguments{
\item{data}{Data frame read by column position: \code{id, start, stop, death, covariates...}, one row per interval \eqn{(start, stop]} with constant covariates.}
}
\details{
The rows are sorted by id, start and stop. Each interval becomes a row at its start time with \code{event = 0}, and each subject gets a final row at its last stop time that carries \code{death} of its last interval and the covariates of that interval. \code{death} on earlier intervals is ignored. Gaps between intervals are not kept: exposure runs continuously from a subject's first start to its last stop. A warning is given if any interval has \code{stop <= start}.
}
\value{A data frame with columns \code{id, time, event} followed by the covariates.}
\seealso{\code{\link{wbart}}}
\examples{
d <- data.frame(id = c(1, 1, 2), start = c(0, 1, 0), stop = c(1, 2.5, 1.7),
                death = c(0, 1, 0), x1 = c(0.2, 0.4, -1), x2 = c(1, 1, 0))
convert_start_stop(d)
}
