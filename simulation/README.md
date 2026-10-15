# Simulation studies

This folder contains the code of the two simulation studies in the paper. Each study is one self-contained R script that reads from top to bottom: settings, data-generating process, true survival quantities, methods, metrics, the loop over replicates, and the summary tables.

| Script | Paper | Design |
|---|---|---|
| `sim1.R` | Simulation I (Tables 1–2, Web Table 2, Web Figure 1) | Time-varying covariates with a random intercept and a random slope in the hazard; scenarios PH and non-PH; 20% and 40% total censoring |
| `sim2.R` | Simulation II (Tables 3–4, Web Table 3, Web Figure 2) | Time-dependent confounding with nonlinear covariate effects and a U-shaped baseline hazard; 20% and 40% total censoring |

Both scripts use N = 1500 subjects per replicate (1200 for training, 300 for testing) and 200 replicates. They compare TVSurvBART-RIS (random intercept and slope) with CoxPH, the adapted frailty Cox model and a random survival forest for left-truncated right-censored data. Every method predicts the survival of the test subjects from their observed covariate history, carried forward after the last observation (LOCF). For a given seed the scripts regenerate the paper's datasets and train/test splits, and TVSurvBART uses the paper's chain seeds.

## Requirements

- R (tested with 4.4.2) and the packages TVSurvBART, survival, coxme, timeROC, LTRCforests and statmod; `sim1.R` also needs mvtnorm.
- Tested versions: TVSurvBART 0.0.3.1, survival 3.8.3, coxme 2.2.22, timeROC 0.4, LTRCforests 0.7.0, statmod 1.5.0, mvtnorm 1.3.3.

## How to run

Run the scripts from this folder:

```sh
Rscript sim1.R 1 --quick   # quick test of seed 1: short MCMC chains, small RSF
Rscript sim1.R 7           # seed 7
Rscript sim1.R 1 50        # seeds 1 to 50
Rscript sim1.R             # all seeds, 1 to 200
Rscript sim1.R --summary   # only summarize the existing results (add --quick for the quick-test results)
```

`sim2.R` takes the same arguments. Each run ends by summarizing all result files found so far.

Runs are resumable: a cell whose result file exists is skipped, and a cell that fails writes no file, so it is redone at the next run. To run in parallel, start several processes with different seed ranges; each process uses one core and up to about 1.7 GB of memory. For example, four processes with 50 seeds each:

```sh
for s in 1 51 101 151; do nohup Rscript sim1.R $s $((s + 49)) > log_sim1_$s.txt 2>&1 & done
```

The random survival forest (LTRCforests) can crash R on some systems, which no error handler can catch. In that case run all other methods with `--no-rsf` and the forest alone with `--rsf-only`, for example on another machine, into the same `results/` folder; the two partial result files of a cell are combined by the summary. The forest is fitted directly after the train/test split, so its results do not depend on the other methods.

## Results

- `results/sim1_<ph|nonph>_cens<20|40>_seed<NNN>.rds` and `results/sim2_general_cens<20|40>_seed<NNN>.rds`: one file per replicate and censoring level, with the metrics of every method and the predicted survival matrix of TVSurvBART (test subjects × evaluation grid), so that further metrics can be computed without refitting. Partial runs add the suffix `_norsf` or `_rsf`.
- `results/summary_sim*_main.csv`: bias, Monte Carlo SE and RMSE of RMST(τ/2), RMST(τ) and the median against the population truth, and the mean IBS, RMSE of the survival curves and test log score.
- `results/summary_sim*_auc.csv`: mean time-dependent AUC on the whole evaluation grid (the paper reports t = 1, 2, 3 in Simulation I and t = 1, ..., 5 in Simulation II), with the population references.
- `results/summary_sim*_logscore.csv`: paired differences of the test log score between methods (row minus column method).
- The population truths (true RMST, median and AUC references from 100,000 simulated subjects) are recomputed by every summary and printed with it.
- `--quick` writes to `results/quick/` so that test runs never mix with production results.

## Run time

The cost is dominated by TVSurvBART. One fit with the paper's chain (20,000 burn-in sweeps, then 2,000 draws kept every 50th sweep) took a median of about 7 hours in Simulation I and 9 hours in Simulation II on one core of the computing cluster used for the paper; on a recent desktop CPU the same chain, including the prediction, takes roughly 30 minutes (Simulation I) and 40 minutes (Simulation II). A replicate (one seed) has 4 fits in Simulation I and 2 fits in Simulation II, so the full studies (800 and 400 fits) need several thousand core-hours on such a cluster. The other methods take about 15 seconds per cell. The population truths take about 50 seconds (Simulation I) and 10 seconds (Simulation II) per summary. A `--quick` run of one seed takes under 2 minutes (Simulation I) and under a minute (Simulation II).

## Naming in the paper and in the code

| Paper | Code |
|---|---|
| TVSurvBART-RIS | `tvsurvbart` |
| CoxPH, adapted frailty Cox, RSF | `cox`, `fcox`, `rsf` |
| AUC references "Marginal" and "Oracle" (Simulation I), "Reference" (Simulation II) | `ref_marginal`, `ref_oracle`, `ref_population` |
| True model on the observed (LOCF) covariate path, quoted in the Simulation II text | `oracle_locf` |
