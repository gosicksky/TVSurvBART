# TVSurvBART

Bayesian nonparametric survival regression with time-varying covariates and random effects.

TVSurvBART is a continuous-time survival model for longitudinal data in counting-process (start–stop) form. The hazard is the product of a baseline hazard (natural cubic spline or piecewise constant), a Bayesian additive regression tree (BART) ensemble of the time-varying covariates, and subject-level random effects (a random intercept, optionally with a random slope in time) with a Huang–Wand covariance prior. Inference is by MCMC. The prediction functions return posterior draws of the cumulative hazard $H(t)$, for training subjects and for new subjects; survival is $\exp(-H)$ per draw.

> Zhang, K., Zou, Y., Li, F. and Hu, L. *Bayesian Nonparametric Survival Regression with Time-Varying Covariates and Random Effects.* Manuscript.

## Installation

```r
# install.packages("remotes")
remotes::install_github("gosicksky/TVSurvBART")

# or from the source package in this repository (download it first; dependencies are not installed automatically)
install.packages(c("Rcpp", "RcppEigen", "BH", "dplyr", "mvtnorm", "nnet"))
install.packages("TVSurvBART_0.0.3.1.tar.gz", repos = NULL, type = "source")
```

You need R >= 4.1 and a C++ compiler. With `install_github` the dependencies (Rcpp, RcppEigen, BH, dplyr, mvtnorm, nnet) are installed automatically. The package has been built and tested on macOS (arm64). The same engine, under its earlier name CoxBARTserver 0.0.3.1, ran the paper's simulations on Linux. Each function has a help page (`?wbart`, `?pwbart`, `?pwbart_grid`, `?predict.wbart`, `?convert_start_stop`); the main arguments are summarised below.

## Quick start

```r
library(TVSurvBART)

## Simulated longitudinal data in start-stop form: id, start, stop, death, covariates
set.seed(2026)
tau <- 4
sim_subject <- function(id) {
  v   <- c(0, cumsum(runif(12, 0.2, 0.8))); v <- v[v < tau]   # irregular visits
  x1  <- rnorm(1) + 0.4 * v + rnorm(length(v), 0, 0.3)       # time-varying biomarker
  x2  <- rnorm(1); x3 <- rbinom(1, 1, 0.5)                   # baseline covariates
  haz <- 0.2 * exp(0.7 * x1 - 0.5 * x2^2 + 0.6 * x3 + rnorm(1, 0, 0.5))
  H   <- cumsum(haz * diff(c(v, Inf)))
  E   <- rexp(1); j <- which(H >= E)[1]
  Tev <- v[j] + (E - c(0, H)[j]) / haz[j]                    # event time
  Y   <- min(Tev, runif(1, 1, 8), tau)                       # observed time
  k   <- sum(v < Y)
  data.frame(id = id, start = v[1:k], stop = c(v[1:k], Y)[-1],
             death = c(rep(0, k - 1), as.numeric(Tev <= Y)),
             x1 = x1[1:k], x2 = x2, x3 = x3)
}
dat <- do.call(rbind, lapply(1:200, sim_subject))

## Fit: spline baseline, random intercept and slope on t / tau (short chain for illustration)
m  <- 50
sF <- log(1000) / (qnorm(0.975) * sqrt(2))                   # leaf-prior scale of the paper
log <- capture.output(                                       # silences the sampler's output
  fit <- wbart(dat, start.stop = TRUE, ntree = m, c = m / sF^2 + 0.5, d = m / sF^2,
               ns.knot = 0:4, re.max.power = 1, re.time.scale = tau, reHW.A = 5,
               nskip = 1000, ndpost = 1000)
)

## Survival curves of two new subjects (random effect integrated out)
new <- data.frame(id   = rep(c(901, 902), each = 3),
                  time = c(0, 1, 2,      0, 1, 2),
                  x1   = c(-1, -.6, -.2,  1, 1.4, 1.8),
                  x2   = rep(c(0, 1.5), each = 3),
                  x3   = rep(c(0, 1), each = 3))
log  <- capture.output(pred <- pwbart_grid(c(1, 2, 3, 4), new, fit))
S    <- exp(-pred$cumHazard.test)                            # draws x subjects x times
round(apply(S, 2:3, mean), 3)                                # posterior mean S(t)
apply(S, 2:3, quantile, c(.025, .975))                       # 95% credible bands
```

This takes about 15 s on a laptop. The posterior mean survival at t = 1, 2, 3, 4 is 0.936, 0.843, 0.752, 0.664 for subject 901 and 0.722, 0.542, 0.421, 0.340 for subject 902.

## Data format

- `wbart()` reads columns by position. Pass `start.stop = TRUE` for data with columns `id, start, stop, death, covariates...`, one row per interval with constant covariates; rows are sorted for you, and `death` is read from each subject's last interval. Without it, `wbart()` expects a long layout `id, time, event, covariates...`: one row per measurement time with `event = 0`, then a final row per subject at its exit time that repeats the previous row's covariates. Rows must be sorted by id and time.
- Use integer-valued numeric ids (not character or factor). Ids must be unique per subject: a new subject that reuses a training id gets that subject's random effect.
- Supply at least two non-constant covariate columns, numeric or factor. Constant columns are dropped at fit time, so drop them from prediction data too.
- The trees see only the supplied covariates. For time-varying (non-proportional) effects, as in the paper, add a column equal to each interval's start time divided by $\tau$, and give the prediction data the same column.
- Prediction data have columns `id, time, covariates...` (no event column, covariates in the training order), one row per covariate measurement. Sort them by id and time, with distinct times within a subject; this is not checked, and violating it silently gives wrong curves. The cumulative hazard is accumulated from each subject's first row, so start each subject at time 0. After the last row the covariates are carried forward. Factor covariates need the training levels, e.g. `factor(x, levels = levels(train$x))`.
- Round the times, e.g. with `round(x, 6)`: distinct values closer than about $10^{-7}$ make the numerical integration very slow.

## Paper settings

Several defaults of `wbart()` are not meant for inference (see `c`, `d`, `nskip` and `reHW.A` below). The paper uses:

```r
wbart(dat, start.stop = TRUE,
      ntree = 100, c = 100 / sF^2 + 0.5, d = 100 / sF^2,   # 16.601, 16.101 (sF as in the quick start)
      ns.knot = 0:tau, ns.block = TRUE,                     # spline knots 0, 1, ..., tau; block MALA
      re.max.power = 1, re.time.scale = tau,                # random intercept and slope on t / tau
      reHW.nu = 2, reHW.A = 5, re.ncp = 3,                  # Huang-Wand prior; 3 non-centred moves per sweep
      nskip = 20000, ndpost = 2000, keepevery = 50)         # 120,000 sweeps, 2,000 kept draws
```

For a piecewise-constant baseline use `baseline.type = "piecewise", pw.shape = 0.1, pw.rate = 0.1` with the cutpoints in `ns.knot` (the paper uses `seq(0, 3, 0.5)` in Simulation I and `0:5` in Simulation II). For a random intercept only use `re.max.power = 0, reCoefMean = 0`.

## Function reference

### `wbart()`: fit the model

| Argument | Default | Meaning |
|---|---|---|
| `data` | | Data frame read by column position; see [Data format](#data-format). |
| `start.stop` | `FALSE` | `TRUE` for counting-process data `id, start, stop, death, covariates...`. |
| `baseline.type` | `"spline"` | `"spline"` (natural cubic spline on the log baseline hazard) or `"piecewise"` (piecewise-constant baseline hazard). |
| `ns.knot` | `c(0, 1, 2)` | Spline knots (increasing; the first and last are the boundary knots) or piecewise cutpoints, on the time scale of the data. Choose them to span follow-up. |
| `nsCoefMean`, `nsCoefVar` | `0`, `1` | Normal prior (mean, variance) on the spline coefficients. |
| `pw.shape`, `pw.rate` | `0.1`, `0.1` | Gamma prior (shape, rate) on the piecewise heights. |
| `ns.block`, `ns.block.M` | `NULL`, `NULL` | `TRUE`: block MALA for the spline coefficients (needs `nskip >= 400`), with preconditioner `"secant"` or `"draws"`. `NULL` reads the environment variables `TVSB_NSBLOCK` (default `FALSE`) and `TVSB_NSBLOCK_M` (default `"secant"`). |
| `ntree` | `200` | Number of trees. |
| `c`, `d` | `1`, `1` | Leaf prior: each leaf value is $W$ or $1/W$ with $W \sim$ Gamma(shape `c`, rate `d`). The defaults are very diffuse; `c = ntree/sF^2 + 0.5`, `d = ntree/sF^2` gives the log tree factor a prior SD of about `sF` (`sF` as in the quick start). |
| `base`, `power` | `0.95`, `2` | Tree prior: a node at depth $r$ splits with probability `base * (1 + r)^(-power)`. |
| `numcut`, `usequants`, `cont`, `xinfo` | `100`, `FALSE`, `FALSE`, none | Cutpoints per covariate: at most `numcut` (midpoints if there are fewer distinct values), equally spaced or at quantiles (`usequants`); `cont = TRUE` always uses `numcut` equally spaced ones; or a matrix of cutpoints in `xinfo`. |
| `rm.const` | `TRUE` | Drop constant covariates. |
| `augment` | `FALSE` | Draw birth split variables only among variables that can still split the node. |
| `re.max.power`, `re.powers` | `1`, `NULL` | Random-effect design $(t/\tau)^p$ with powers `0:re.max.power` (intercept and slope by default) or the powers in `re.powers`. |
| `re.time.scale` | `1` | $\tau$; the paper uses the maximum follow-up. |
| `reCoefMean` | `c(0, 0)` | Prior mean of $b_i$; its length must equal the number of powers. Keep it at 0. |
| `reCoefVarInit` | identity | Starting value of $\Sigma_b$. |
| `reHW.nu`, `reHW.A` | `2`, `1e5` | Huang–Wand prior on $\Sigma_b$: degrees of freedom and half-t scales of the standard deviations (log-hazard scale; the paper uses 5). |
| `re.ncp` | `1` | Joint non-centred $(\Sigma_b, b)$ proposals per sweep; 0 turns them off. |
| `nskip`, `ndpost`, `keepevery` | `100`, `1000`, `1` | Burn-in sweeps, stored draws, thinning (`nskip + ndpost * keepevery` sweeps in total). Most step sizes adapt only at every 500th burn-in sweep, so use `nskip >= 500`. |
| `nkeeptrain`, `nkeeptreedraws` | `ndpost` | Stored rows of `yhat.train` and stored tree draws (the latter must equal `ndpost` for prediction). |
| `printevery` | `100` | Progress-line frequency. |

Not used: `test.data`, `ns.degree`, `sparse`, `theta`, `omega`, `a`, `b`, `rho`, `sigest`, `sigdf`, `sigquant`, `k`, `sigmaf`, `lambda`, `nkeeptest`, `nkeeptestmean`. `transposed` is internal; leave it at `FALSE`.

`wbart()` returns a plain list. `cumHazard` (given each draw's $b_i$) and `yhat.train` (the tree factor) are draws x rows of the internal long data, which has one row per interval start plus one final row per subject. It also holds `nsDraws` (baseline coefficients; log heights for the piecewise baseline), `reCoefDraws` ($b_i$, one block of subjects per draw), `reCoefCovDraws` ($\Sigma_b$, one block per draw) and `treedraws`. Call `set.seed()` before `wbart()` to reproduce a chain.

### Prediction

| Function | Arguments | Returns |
|---|---|---|
| `pwbart(target_time, data, object, n.gh = 16)` | `target_time`: one time per subject, in ascending id order (not recycled). `data`: prediction data. `object`: the fit. `n.gh`: Gauss–Hermite nodes per random-effect dimension for new subjects (pass by name). | `cumHazard.test` and `hazard.test` (draws x subjects), `yhat.test` (draws x rows). |
| `pwbart_grid(time_grid, data, object, n.gh = 16, want.hazard = FALSE)` | `time_grid`: times shared by all subjects. `want.hazard`: also return the hazard. | `cumHazard.test` (draws x subjects x times), plus `hazard.test`. |
| `predict.wbart(object, target_time, newdata, ...)` | As `pwbart()`; `...` is passed on (e.g. `n.gh`). | The value of `pwbart()`. |
| `convert_start_stop(data)` | Start-stop data `id, start, stop, death, covariates...`. | The long format `id, time, event, covariates...` used by `wbart()`. |

- Subjects seen in training use their posterior $b_i$. For new subjects, $b$ is integrated out and `cumHazard.test` holds $-\log \bar S(t)$: by Gauss–Hermite quadrature for a single random-effect term or for intercept and slope; otherwise, or with `n.gh <= 1`, by Monte Carlo in `pwbart()` (call `set.seed()` first), where `pwbart_grid()` stops.
- Per-draw survival is `exp(-cumHazard.test)`.
- The fit has no class, so call `pwbart()` or `predict.wbart()` directly; `predict(fit, ...)` does not work. The `mc.cores` arguments have no effect.

## Simulation code

[`simulation/`](simulation/) contains the paper's two simulation studies as plain R scripts, `sim1.R` (Simulation I) and `sim2.R` (Simulation II). Each one generates the data, fits TVSurvBART, CoxPH, the adapted frailty Cox model and a random survival forest, and computes the paper's metrics, e.g. `Rscript sim1.R 1 --quick` for a quick test of one replicate. The folder is not part of the R package; see [`simulation/README.md`](simulation/README.md).

## Version history

See [NEWS.md](NEWS.md). TVSurvBART 0.0.3.1 is the engine previously distributed as CoxBARTserver 0.0.3.1, renamed; results are identical.

## License

GPL (>= 3); see [LICENSE.md](LICENSE.md). The tree sampler and the R front end derive from the BART R package by Robert McCulloch and Rodney Sparapani (GPL >= 2).
