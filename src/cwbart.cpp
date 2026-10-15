/*
 *  BART: Bayesian Additive Regression Trees
 *  Copyright (C) 2017 Robert McCulloch and Rodney Sparapani
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, a copy is available at
 *  https://www.R-project.org/Licenses/GPL-2
 */

#include "survivalBART.h"

#define TRDRAW(a, b) trdraw(a, b)
#define TEDRAW(a, b) tedraw(a, b)


//[[Rcpp::export]]
RcppExport SEXP cwbart(
    SEXP sexp_in,            //number of observations in training data
    SEXP sexp_isubjectN,		//number of subjects
    SEXP sexp_ip,		//dimension of x
    SEXP sexp_inp,		//number of observations in test data
    SEXP sexp_ix,		//x, train,  pxn (transposed so rows are contiguous in memory)
    SEXP sexp_ixObsTime, //observed time for each subject
    SEXP sexp_ixIndex, //index of last observation in the subject
    SEXP sexp_ixEvent, //event indicator for each observation
    SEXP sexp_iy,		//y, train,  nx1
    SEXP sexp_ixp,		//x, test, pxnp (transposed so rows are contiguous in memory)
    SEXP sexp_insDegree, //degree of the natural spline
    SEXP sexp_insKnots, //knots of the natural spline
    SEXP sexp_ireMaxPower, //maximum power of the random effects
    SEXP sexp_irePowers, //powers included in the random effects
    SEXP sexp_ireTimeScale, //time standardization for the RE design: z(t) = (t/scale)^power
    SEXP sexp_ic, //prior parameter for BART
    SEXP sexp_id, //prior parameter for BART
    SEXP sexp_insPriorMean, //prior mean for the natural spline coefficients
    SEXP sexp_insPriorVar, //prior variance for the natural spline coefficients (level + linear)
    SEXP sexp_ireCoefPriorMean, //random-effect prior mean mu_b
    SEXP sexp_ireHWnu, //Huang-Wand nu (df; 2 => uniform correlations)
    SEXP sexp_ireHWA, //Huang-Wand half-t scale per RE dimension
    SEXP sexp_ireCoef, //initial random effects coefficients
    SEXP sexp_ireCoefCov, //initial random effects covariance
    SEXP sexp_im,		//number of trees
    SEXP sexp_inc,		//number of cut points
    SEXP sexp_ind,		//number of kept draws (except for thinnning ..)
    SEXP sexp_iburn,		//number of burn-in draws skipped
    SEXP sexp_ipower,
    SEXP sexp_ibase,
    SEXP sexp_itau,
    SEXP sexp_inu,
    SEXP sexp_ilambda,
    SEXP sexp_isigest,
    SEXP sexp_iw,
    SEXP sexp_idart,
    SEXP sexp_itheta,
    SEXP sexp_iomega,
    SEXP sexp_igrp,
    SEXP sexp_ia,
    SEXP sexp_ib,
    SEXP sexp_irho,
    SEXP sexp_iaug,
    SEXP sexp_inkeeptrain,
    SEXP sexp_inkeeptest,
    SEXP sexp_inkeeptestme,
    SEXP sexp_inkeeptreedraws,
    SEXP sexp_inprintevery,
    SEXP sexp_Xinfo,
    SEXP sexp_ireNcp, //joint non-centred (Sigma_b, b) Metropolis proposals per sweep (0 = off)
    SEXP sexp_ibaselineType, //0 = natural cubic spline baseline, 1 = piecewise-constant (Breslow-type)
    SEXP sexp_ipwShape, //Gamma prior shape for piecewise heights lambda_k
    SEXP sexp_ipwRate, //Gamma prior rate for piecewise heights lambda_k
    SEXP sexp_insBlock, //1 = preconditioned block MALA on the spline coefficients, 0 = coordinatewise MALA
    SEXP sexp_insBlockM, //block-MALA preconditioner estimator: 0 = covariance of the beta draws, 1 = secant curvature (R default)
    SEXP sexp_ikeepevery //thinning k: store the state at every k-th kept sweep (sexp_ind = kept sweeps = stored draws * k)
)
{
  
  //--------------------------------------------------
  //process args
  printf("*****Rcpp into cwbart\n");
  size_t n = Rcpp::as<int>(sexp_in);
  size_t nSubject = Rcpp::as<int>(sexp_isubjectN);
  size_t p = Rcpp::as<int>(sexp_ip);
  size_t np = Rcpp::as<int>(sexp_inp);
  Rcpp::NumericVector  xv(sexp_ix);
  double *ix = &xv[0];
  
  Rcpp::NumericVector  xObsTime(sexp_ixObsTime);
  double *ixObsTime = &xObsTime[0];
  
  Rcpp::NumericVector  xIndex(sexp_ixIndex);
  double *ixIndex = &xIndex[0];
  
  Rcpp::NumericVector  xEvent(sexp_ixEvent);
  double *ixEvent = &xEvent[0];
  
  Rcpp::NumericVector  yv(sexp_iy);
  double *iy = &yv[0];
  
  Rcpp::NumericVector  xpv(sexp_ixp);
  double *ixp = &xpv[0];
  
  printf("*****ns data\n");
  
  // size_t nsDegree = Rcpp::as<int>(sexp_insDegree);
  
  // Rcpp::NumericVector  nsKnots(sexp_insKnots);
  // //repeat the first and last knots nsDegree times
  // Rcpp::NumericVector extendedKnots(nsKnots.size() - 2 + 2 * nsDegree);
  // for (int i = 0; i < nsDegree - 1; ++i)
  //   extendedKnots[i] = nsKnots[0];
  // for (int i = 0; i < nsKnots.size(); ++i)
  //   extendedKnots[i + nsDegree] = nsKnots[i];
  // for (int i = 0; i < nsDegree - 1; ++i)
  //   extendedKnots[nsKnots.size() - 1 + nsDegree + i] = nsKnots[nsKnots.size() - 1];
  // nsKnots = extendedKnots;
  // 
  // double *insKnots = &nsKnots[0];
  // 
  // size_t numNsKnot = nsKnots.size();
  
  //redefine nsKnots to be the extended knots
  Rcpp::NumericVector nsKnots(sexp_insKnots);
  double *insKnots = &nsKnots[0];
  size_t numNsKnot = nsKnots.size();
  size_t nsDegree = Rcpp::as<int>(sexp_insDegree);
  
  // // Open natural spline: repeat the boundary knots (degree + 1) times
  // size_t new_nKnots = nKnots + 2 * nsDegree; // total number of knots after padding
  // Rcpp::NumericVector openKnots(new_nKnots);
  // 
  // int idx = 0;
  // // Left boundary repeated (degree + 1) times
  // for (int j = 0; j < nsDegree + 1; ++j)
  //   openKnots[idx++] = nsKnots[0];
  // 
  // // Interior knots
  // for (size_t j = 1; j < nKnots - 1; ++j)
  //   openKnots[idx++] = nsKnots[j];
  // 
  // // Right boundary repeated (degree + 1) times
  // for (int j = 0; j < nsDegree + 1; ++j)
  //   openKnots[idx++] = nsKnots[nKnots - 1];
  // 
  // nsKnots = openKnots;
  // double *insKnots = &nsKnots[0];
  // size_t numNsKnot = nsKnots.size();
  
  size_t reMaxPower = Rcpp::as<int>(sexp_ireMaxPower);
  Rcpp::IntegerVector rePowersVec(sexp_irePowers);
  int *irePowers = &rePowersVec[0];
  size_t nReCoef = rePowersVec.size();
  double reTimeScale = Rcpp::as<double>(sexp_ireTimeScale);
  printf("*****Random Effects time scale (z uses t/scale): %lf\n", reTimeScale);
  
  // Baseline type: 0 = natural cubic spline (default), 1 = piecewise-constant
  // ("Bayesian Breslow": knots become bin cutpoints, one log-height per bin,
  // conjugate gamma Gibbs update with a Gamma(pwShape, pwRate) prior).
  int baselineType = Rcpp::as<int>(sexp_ibaselineType);
  double pwShape = Rcpp::as<double>(sexp_ipwShape);
  double pwRate = Rcpp::as<double>(sexp_ipwRate);
  printf("*****baselineType: %d (0=spline, 1=piecewise)\n", baselineType);
  if(baselineType == 1) printf("*****piecewise Gamma prior: shape %lf rate %lf\n", pwShape, pwRate);
  
  // Internal number of natural-spline basis coefficients. Must match
  // survivalBART::nNsBasis (set in setparam). The explicit t^2 basis was removed,
  // so this is simply the number of knots (intercept + linear + cubic truncated powers).
  // Piecewise mode: nKnots cutpoints -> nKnots-1 bin heights.
  size_t numNsBasis = (baselineType == 1) ? (numNsKnot - 1) : numNsKnot;
  
  double c = Rcpp::as<double>(sexp_ic);
  double d = Rcpp::as<double>(sexp_id);
  
  double nsPriorMean = Rcpp::as<double>(sexp_insPriorMean);
  double nsPriorVar = Rcpp::as<double>(sexp_insPriorVar);
  printf("*****Prior: nsPriorVar: %lf\n", nsPriorVar);
  
  printf("*****random effects prior data\n");
  
  Rcpp::NumericVector  reCoefPriorMean(sexp_ireCoefPriorMean);
  double *ireCoefPriorMean = &reCoefPriorMean[0];
  
  double reHWnu = Rcpp::as<double>(sexp_ireHWnu);
  Rcpp::NumericVector reHWAvec(sexp_ireHWA);
  double *ireHWA = &reHWAvec[0];
  printf("*****Huang-Wand nu: %lf  A: ", reHWnu);
  for(size_t i=0;i<nReCoef;i++) printf("%lf ", ireHWA[i]);
  printf("\n");
  
  printf("*****random effects initial data\n");
  
  //matrix
  Rcpp::NumericVector reCoef(sexp_ireCoef);
  double *ireCoef = &reCoef[0];
  
  printf("*****random effects initial covariance data\n");
  
  //matrix
  Rcpp::NumericVector reCoefCov(sexp_ireCoefCov);
  double *ireCoefCov = &reCoefCov[0];
  
  size_t m = Rcpp::as<int>(sexp_im);
  
  Rcpp::IntegerVector _nc(sexp_inc);
  
  int *numcut = &_nc[0];
  //size_t nc = Rcpp::as<int>(sexp_inc);
  size_t nd = Rcpp::as<int>(sexp_ind);
  size_t burn = Rcpp::as<int>(sexp_iburn);
  double mybeta = Rcpp::as<double>(sexp_ipower);
  double alpha = Rcpp::as<double>(sexp_ibase);
  double tau = Rcpp::as<double>(sexp_itau);
  double nu = Rcpp::as<double>(sexp_inu);
  double lambda = Rcpp::as<double>(sexp_ilambda);
  double sigma=Rcpp::as<double>(sexp_isigest);
  Rcpp::NumericVector  wv(sexp_iw);
  double *iw = &wv[0];
  bool dart;
  if(Rcpp::as<int>(sexp_idart)==1) dart=true;
  else dart=false;
  double a = Rcpp::as<double>(sexp_ia);
  double b = Rcpp::as<double>(sexp_ib);
  double rho = Rcpp::as<double>(sexp_irho);
  bool aug;
  if(Rcpp::as<int>(sexp_iaug)==1) aug=true;
  else aug=false;
  double theta = Rcpp::as<double>(sexp_itheta);
  double omega = Rcpp::as<double>(sexp_iomega);
  Rcpp::IntegerVector _grp(sexp_igrp);
  //   int *grp = &_grp[0];
  size_t nkeeptrain = Rcpp::as<int>(sexp_inkeeptrain);
  size_t nkeeptest = Rcpp::as<int>(sexp_inkeeptest);
  size_t nkeeptestme = Rcpp::as<int>(sexp_inkeeptestme);
  size_t nkeeptreedraws = Rcpp::as<int>(sexp_inkeeptreedraws);
  size_t printevery = Rcpp::as<int>(sexp_inprintevery);

  // Posterior storage (thinning). The chain runs burn + nd sweeps, nd = ndraw * keepevery
  // kept (post-burn-in) sweeps; keepevery never changes the chain (same RNG stream, same
  // states), only what is stored. Stored draw d (0-based, d = 0..ndraw-1) is the state at
  // the end of kept sweep (d+1)*keepevery, i.e. sweep index i = burn + (d+1)*keepevery - 1
  // ((i - burn + 1) % keepevery == 0). EVERY per-draw output (tree draws, varcount /
  // varprob, nsDraws, reCoefDraws, reCoefCovDraws, reCovAuxDraws, BartMeanDraws,
  // splineYDraws, cumHazard, yhat.train, yhat.test) is stored at these sweeps only, so
  // row / block d of each output is one and the same state and cpwbart / cpwbart_grid pair
  // tree draw d with row / block d of the rest. nkeeptrain, nkeeptest, nkeeptestme and
  // nkeeptreedraws (each 0 or a divisor of ndraw) sub-sample the stored draws: e.g.
  // yhat.train keeps stored draws d with (d+1) % (ndraw/nkeeptrain) == 0. Prediction pairs
  // correctly only with nkeeptreedraws == ndraw (R default; pwbart_inputs() refuses a fit
  // whose tree draws do not match nsDraws / reCoefCovDraws / reCoefDraws). Chain summaries use every
  // kept sweep: yhat.train.mean averages all nd kept sweeps (as before; the same value for
  // every keepevery), yhat.test.mean the nkeeptestme draws, and the acceptance rates count
  // every kept sweep. keepevery = 1 stores every kept sweep: the engine before this fix.
  const int keepeveryInt = Rcpp::as<int>(sexp_ikeepevery);
  if(keepeveryInt < 1) Rcpp::stop("cwbart: keepevery must be >= 1");
  const size_t keepevery = (size_t)keepeveryInt;
  if(nd % keepevery != 0)
    Rcpp::stop("cwbart: the number of kept sweeps (%d) is not a multiple of keepevery (%d)", (int)nd, keepeveryInt);
  const size_t ndraw = nd / keepevery;   // number of stored draws (wbart's ndpost)
  {
    const size_t nk[4] = {nkeeptrain, nkeeptest, nkeeptestme, nkeeptreedraws};
    const char* nm[4] = {"nkeeptrain", "nkeeptest", "nkeeptestmean", "nkeeptreedraws"};
    for(int j=0; j<4; j++)
      if(nk[j] && (nk[j] > ndraw || ndraw % nk[j] != 0))
        Rcpp::stop("cwbart: %s (%d) must be 0 or a divisor of the number of stored draws (%d)", nm[j], (int)nk[j], (int)ndraw);
  }
  //   int treesaslists = Rcpp::as<int>(_treesaslists);
  Rcpp::NumericMatrix Xinfo(sexp_Xinfo);
  Rcpp::IntegerMatrix varcount(nkeeptreedraws, p);
  
  //return data structures (using Rcpp)
  Rcpp::NumericVector trmean(n); //train
  Rcpp::NumericVector temean(np);
  Rcpp::NumericMatrix trdraw(nkeeptrain,n);
  Rcpp::NumericMatrix tedraw(nkeeptest,np);
  //   Rcpp::List list_of_lists(nkeeptreedraws*treesaslists);
  Rcpp::NumericMatrix varprb(nkeeptreedraws,p);
  Rcpp::IntegerMatrix varcnt(nkeeptreedraws,p);
  // per-draw outputs: one row (block) per STORED draw (ndraw = nd / keepevery), see above
  Rcpp::NumericMatrix cumHazard(ndraw, n); //cumulative hazard for each subject at each stored draw
  //number of b-spline basis: N = length(knots) - degree - 1
  // Rcpp::NumericMatrix nsDraws(nd, numNsKnot - nsDegree - 2); //natural spline draws for each iteration
  Rcpp::NumericMatrix nsDraws(ndraw, numNsBasis); //natural cubic spline draws for each stored draw
  Rcpp::NumericVector nsMean(numNsBasis);
  Rcpp::NumericMatrix reCoefDraws(ndraw * nSubject, nReCoef); //random effects coefficient draws, one block of nSubject rows per stored draw
  Rcpp::NumericMatrix reCoefCovDraws(ndraw * nReCoef, nReCoef); //random effects covariance matrix, one q x q block per stored draw
  Rcpp::NumericMatrix reCovAuxDraws(ndraw, nReCoef); //Huang-Wand auxiliary scales a_k per stored draw (0 padded if IW)
  Rcpp::NumericMatrix BartMeanDraws(ndraw, 1); //BART mean draws for each stored draw
  Rcpp::NumericMatrix splineYDraws(ndraw, nSubject); //natural spline draws for each subject and each stored draw
  
  
  //random number generation
  arn gen;
  
  //print m
  printf("*****m: %zu\n",m);
  
  survivalBART bm(m);
  
  printf("*****ifndef into main of wbart\n");
  
  if(Xinfo.size()>0) {
    xinfo _xi;
    _xi.resize(p);
    for(size_t i=0;i<p;i++) {
      _xi[i].resize(numcut[i]);
      //Rcpp::IntegerVector cutpts(Xinfo[i]);
      for(int j=0;j<numcut[i];j++) _xi[i][j]=Xinfo(i, j);
      //for(size_t j=0;j<numcut[i];j++) _xi[i][j]=Xinfo(i, j);
    }
    bm.setxinfo(_xi);
  }
  // 
  for(size_t i=0;i<n;i++) trmean[i]=0.0;
  for(size_t i=0;i<np;i++) temean[i]=0.0;
  
  printf("*****Into main of wbart\n");
  //-----------------------------------------------------------
  
  // sub-sampling periods in units of STORED draws (see the storage note above); in kept
  // sweeps they are keepevery times these, the same sweeps as before this fix
  size_t skiptr,skipte,skipteme,skiptreedraws;
  if(nkeeptrain) {skiptr=ndraw/nkeeptrain;}
  else skiptr = ndraw+1;
  if(nkeeptest) {skipte=ndraw/nkeeptest;}
  else skipte=ndraw+1;
  if(nkeeptestme) {skipteme=ndraw/nkeeptestme;}
  else skipteme=ndraw+1;
  if(nkeeptreedraws) {skiptreedraws = ndraw/nkeeptreedraws;}
  else skiptreedraws=ndraw+1;
  
  //--------------------------------------------------
  //print args
  printf("*****Data:\n");
  printf("data:n,p,np: %zu, %zu, %zu\n",n,p,np);
  // print y
  for(size_t i=0;i<nSubject;i++) printf("%lf ",iy[i]);
  printf("\nx1,x[n*p]: %lf, %lf\n",ix[0],ix[n*p-1]);
  if(np) printf("xp1,xp[np*p]: %lf, %lf\n",ixp[0],ixp[np*p-1]);
  printf("*****natural spline: degree: %zu\n",nsDegree);
  printf("*****number of knots: %zu\n",nsKnots.size());
  for(int i=0; i<nsKnots.size(); i++){
    printf("%lf ", insKnots[i]);
  }
  printf("\n*****Random Effects: max power: %zu\n",reMaxPower);
  printf("*****Random Effects powers: ");
  for(size_t i=0;i<nReCoef;i++) printf("%d ", irePowers[i]);
  printf("\n");
  printf("*****Prior: c,d: %lf,%lf\n",c,d);
  printf("*****Prior: nsPriorMean,nsPriorVar: %lf,%lf\n",nsPriorMean,nsPriorVar);
  printf("*****Prior: ireCoefPriorMean: %lf ... %lf\n",ireCoefPriorMean[0],ireCoefPriorMean[nReCoef - 1]);
  printf("*****Number of Trees: %zu\n",m);
  printf("*****Number of Cut Points: %d ... %d\n", numcut[0], numcut[p-1]);
  printf("*****burn and ndpost: %zu, %zu\n",burn,nd);
  printf("*****keepevery, stored draws: %zu, %zu (state stored at every %zu-th kept sweep)\n",keepevery,ndraw,keepevery);
  printf("*****Prior:beta,alpha,tau,nu,lambda: %lf,%lf,%lf,%lf,%lf\n",
         mybeta,alpha,tau,nu,lambda);
  printf("*****sigma: %lf\n",sigma);
  printf("*****w (weights): %lf ... %lf\n",iw[0],iw[nSubject-1]);   // w has nSubject entries; iw[n-1] read past its end
  cout << "*****Dirichlet:sparse,theta,omega,a,b,rho,augment: "
       << dart << ',' << theta << ',' << omega << ',' << a << ','
       << b << ',' << rho << ',' << aug << endl;
  printf("*****nkeeptrain,nkeeptest,nkeeptestme,nkeeptreedraws: %zu,%zu,%zu,%zu\n",
         nkeeptrain,nkeeptest,nkeeptestme,nkeeptreedraws);
  printf("*****printevery: %zu\n",printevery);
  printf("*****skiptr,skipte,skipteme,skiptreedraws: %zu,%zu,%zu,%zu\n",skiptr,skipte,skipteme,skiptreedraws);
  
  //--------------------------------------------------
  //bart bm(m);
  //bm.setdata(p,n,ix,iy,numcut);
  bm.setprior(alpha,mybeta,tau,c,d);
  bm.setdata(p,n,nSubject,ix,ixObsTime,ixIndex,iy,numcut);
  bm.setBaseline(baselineType,pwShape,pwRate);   // must precede setparam (sets nNsBasis)
  bm.setparam(nsDegree,insKnots,numNsKnot,reMaxPower,irePowers,nReCoef,reTimeScale);
  bm.setparamPrior(nsPriorMean,nsPriorVar,ireCoefPriorMean,ireCoef,ireCoefCov,
                   reHWnu,ireHWA);
  bm.setNcp(Rcpp::as<int>(sexp_ireNcp));
  bm.setdart(a,b,rho,aug,dart,theta,omega);
  bm.setu(xEvent);
  // ns.block: block MALA on the spline coefficients (after setparam/setparamPrior;
  // switched off in C++ too for the piecewise baseline); burn sets its schedule
  if(Rcpp::as<int>(sexp_insBlock) != 0) bm.setNsBlock(true, Rcpp::as<int>(sexp_insBlockM), burn);
  //--------------------------------------------------
  //sigma: the Gaussian-BART error sd is not part of the survival model. It is
  //no longer drawn or returned; 'sigma' keeps its initial value (sigest) and is
  //only passed on as the unused sigma argument of bm.draw()/bd()/drmu().
  
  //--------------------------------------------------
  
  std::stringstream treess;  //string stream to write trees to
  treess.precision(10);
  treess << nkeeptreedraws << " " << m << " " << p << endl;
  // dart iterations
  std::vector<double> ivarprb (p,0.);
  std::vector<size_t> ivarcnt (p,0);
  
  //--------------------------------------------------
  //temporary storage
  //out of sample fit
  double* fhattest=0; //posterior mean for prediction
  if(np) { fhattest = new double[np]; }
  
  
  //--------------------------------------------------
  //mcmc
  printf("\nMCMC\n");
  //size_t index;
  size_t trcnt=0; //count kept train draws
  size_t tecnt=0; //count kept test draws
  size_t temecnt=0; //count test draws into posterior mean
  size_t treedrawscnt=0; //count kept bart draws
  bool keeptest,keeptestme,keeptreedraw;
  
  time_t tp;
  int time1 = time(&tp);
  xinfo& xi = bm.getxinfo();
  
  for(size_t i=0;i<(nd+burn);i++) {
    if(i%printevery==0) printf("done %zu (out of %lu)\n",i,nd+burn);
    if(i==(burn/2)&&dart) bm.startdart();
    //draw bart
    bm.draw(sigma,i,burn,gen);
    double *nsMeanTemp = bm.getNsMean();
    for(size_t k=0; k<numNsBasis; k++) {
      nsMean(k) = nsMeanTemp[k];
    }
    if(i>=burn) {
      // chain summary over EVERY kept sweep (independent of keepevery)
      for(size_t k=0;k<n;k++) trmean[k]+=bm.f(k);
      // Everything below stores draw d, only at every keepevery-th kept sweep: sweep
      // i = burn + (d+1)*keepevery - 1. (This block ends the loop body: continue = skip.)
      if(((i-burn+1) % keepevery) != 0) continue;
      const size_t d = (i-burn)/keepevery;   // stored-draw index, 0..ndraw-1
      //store cumulative hazard
      double* cumHazardTemp = bm.calCumHazard();
      for(size_t k=0;k<n;k++) cumHazard(d,k) = cumHazardTemp[k];
      
      double* nsDrawsTemp = bm.getNsCoef();
      for(size_t k=0; k<numNsBasis; k++) {
        nsDraws(d, k) = nsDrawsTemp[k];
      }
      
      double** reCoefDrawsTemp = bm.getReCoef();
      for(size_t k=0; k<nSubject; k++) {
        for(size_t j=0; j<nReCoef; j++) {
          reCoefDraws(d * nSubject + k, j) = reCoefDrawsTemp[k][j];
        }
      }
      
      double** reCoefCovDrawsTemp = bm.getReCoefCov();
      for(size_t k=0; k<nReCoef; k++) {
        for(size_t j=0; j<nReCoef; j++) {
          reCoefCovDraws(d * nReCoef + k, j) = reCoefCovDrawsTemp[k][j];
        }
      }
      
      double* reCovAuxTemp = bm.getReCovAux();
      if(reCovAuxTemp) for(size_t k=0; k<nReCoef; k++) reCovAuxDraws(d, k) = reCovAuxTemp[k];
      
      BartMeanDraws(d, 0) = bm.getBARTMean();
      
      double* splineYDrawsTemp = bm.getSplineY();
      for(size_t k=0; k<nSubject; k++) {
        splineYDraws(d, k) = splineYDrawsTemp[k];
      }
      
      
      if(nkeeptrain && (((d+1) % skiptr) ==0)) {
        //index = trcnt*n;;
        //for(size_t k=0;k<n;k++) trdraw[index+k]=bm.f(k);
        for(size_t k=0;k<n;k++) TRDRAW(trcnt,k)=bm.f(k);
        trcnt+=1;
      }
      keeptest = nkeeptest && (((d+1) % skipte) ==0) && np;
      keeptestme = nkeeptestme && (((d+1) % skipteme) ==0) && np;
      if(keeptest || keeptestme) bm.predict(p,np,ixp,fhattest);
      if(keeptest) {
        //index=tecnt*np;
        //for(size_t k=0;k<np;k++) tedraw[index+k]=fhattest[k];
        for(size_t k=0;k<np;k++) TEDRAW(tecnt,k)=fhattest[k];
        tecnt+=1;
      }
      if(keeptestme) {
        for(size_t k=0;k<np;k++) temean[k]+=fhattest[k];
        temecnt+=1;
      }
      keeptreedraw = nkeeptreedraws && (((d+1) % skiptreedraws) ==0);
      if(keeptreedraw) {
        //	   #ifndef NoRcpp
        //	   Rcpp::List lists(m*treesaslists);
        //	   #endif
        
        for(size_t j=0;j<m;j++) {
          treess << bm.gettree(j);
          /*
#ifndef NoRcpp
           varcount.row(treedrawscnt)=varcount.row(treedrawscnt)+bm.gettree(j).tree2count(p);
           if(treesaslists) lists(j)=bm.gettree(j).tree2list(xi, 0., 1.);
#endif
           */
        }
#ifndef NoRcpp
        //	    if(treesaslists) list_of_lists(treedrawscnt)=lists;
        ivarcnt=bm.getnv();
        ivarprb=bm.getpv();
        size_t k=d/skiptreedraws;
        for(size_t j=0;j<p;j++){
          varcnt(k,j)=ivarcnt[j];
          //varcnt(i-burn,j)=ivarcnt[j];
          varprb(k,j)=ivarprb[j];
          //varprb(i-burn,j)=ivarprb[j];
        }
#else
        varcnt.push_back(bm.getnv());
        varprb.push_back(bm.getpv());
#endif
        
        treedrawscnt +=1;
      }
    }
  }
  int time2 = time(&tp);
  printf("time: %ds\n",time2-time1);
  for(size_t k=0;k<n;k++) trmean[k]/=nd;
  for(size_t k=0;k<np;k++) temean[k]/=temecnt;
  printf("check counts\n");
  printf("trcnt,tecnt,temecnt,treedrawscnt: %zu,%zu,%zu,%zu\n",trcnt,tecnt,temecnt,treedrawscnt);
  if(bm.getNsBlock())
    printf("nsBlock: post-burn-in acceptance %f, frozen h %f (M %s, %d replacements)\n", bm.getNsBlockAcceptRate(), bm.getNsBlockH(),
           bm.getNsBlockMAdapted() ? (bm.getNsBlockMSource() == 1 ? "secant" : "empirical") : "initial diagonal",
           bm.getNsBlockMUpdates());
  //--------------------------------------------------
  //PutRNGstate();
  
  if(fhattest) delete[] fhattest;
  
  //--------------------------------------------------
  //return
#ifndef NoRcpp
  Rcpp::List ret;
  ret["yhat.train.mean"]=trmean;
  ret["yhat.train"]=trdraw;
  ret["yhat.test.mean"]=temean;
  ret["yhat.test"]=tedraw;
  //ret["varcount"]=varcount;
  ret["varcount"]=varcnt;
  ret["varprob"]=varprb;
  ret["cumHazard"]=cumHazard;
  ret["nsDraws"]=nsDraws;
  ret["nsMean"]=nsMean;
  ret["reCoefDraws"]=reCoefDraws;
  ret["reCoefCovDraws"]=reCoefCovDraws;
  ret["reCovAuxDraws"]=reCovAuxDraws;
  ret["reNcpAccept"]=bm.getNcpAcceptRate();         //post-burn-in acceptance of the (Sigma_b, b) move
  {
    Eigen::MatrixXd ncpCov = bm.getNcpProposalCov();         //frozen proposal covariance in xi coordinates
    if(ncpCov.size() > 0) ret["reNcpProposalCov"]=Rcpp::wrap(ncpCov);
    else ret["reNcpProposalCov"]=R_NilValue;                 //move off
  }
  // ns.block outputs: only when the block MALA is on, so a coordinatewise fit
  // returns exactly the components (and values) of the engine without it.
  if(bm.getNsBlock()) {
    ret["nsBlock"]=true;
    ret["nsBlockAccept"]=bm.getNsBlockAcceptRate();   //post-burn-in acceptance of the block MALA
    ret["nsBlockH"]=bm.getNsBlockH();                 //frozen (end of burn-in) step h
    ret["nsBlockM"]=Rcpp::wrap(bm.getNsBlockM());     //frozen preconditioner M (K x K)
    ret["nsBlockMSource"]=(bm.getNsBlockMSource() == 1) ? "secant" : "draws";
    ret["nsBlockMAdapted"]=bm.getNsBlockMAdapted();   //FALSE: M is still the starting diagonal
    ret["nsBlockMUpdates"]=bm.getNsBlockMUpdates();   //number of M replacements in burn-in
    ret["nsBlockPairsDropped"]=bm.getNsBlockPairsDropped();   //secant pairs of batches with acceptance < 0.1
    ret["nsBlockSchedule"]=bm.getNsBlockSchedule();   //M-update period, min statistics, tail, first collected sweep
    ret["nsBlockMLog"]=bm.getNsBlockMLog();           //one row per M-update attempt
  }
  ret["reHWnu"]=reHWnu;
  ret["reHWA"]=reHWAvec;
  ret["rePowers"]=rePowersVec;
  ret["reTimeScale"]=reTimeScale;
  ret["baselineTypeInt"]=baselineType;
  ret["pwShape"]=pwShape;
  ret["pwRate"]=pwRate;
  ret["BartMeanDraws"]=BartMeanDraws;
  ret["splineYDraws"]=splineYDraws;
  
  //for(size_t i=0;i<m;i++) {
  //  bm.gettree(i).pr();
  //}
  
  Rcpp::List xiret(xi.size());
  for(size_t i=0;i<xi.size();i++) {
    Rcpp::NumericVector vtemp(xi[i].size());
    std::copy(xi[i].begin(),xi[i].end(),vtemp.begin());
    xiret[i] = Rcpp::NumericVector(vtemp);
  }
  
  Rcpp::List treesL;
  //treesL["nkeeptreedraws"] = Rcpp::wrap<int>(nkeeptreedraws); //in trees
  //treesL["ntree"] = Rcpp::wrap<int>(m); //in trees
  //treesL["numx"] = Rcpp::wrap<int>(p); //in cutpoints
  treesL["cutpoints"] = xiret;
  treesL["trees"]=Rcpp::CharacterVector(treess.str());
  //   if(treesaslists) treesL["lists"]=list_of_lists;
  ret["treedraws"] = treesL;
  
  return ret;
#else
  
#endif
  
}
